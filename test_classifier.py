"""Run `python classifier.py download`, then `python -m unittest -v`."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from classifier import build_pipeline, download, load_dataset, predict, train

DATA = Path(__file__).parent / "data/labeled_data.csv"


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA.exists():
            raise RuntimeError(
                "Run `python classifier.py download` before testing."
            )
        cls.data, cls.cleaning = load_dataset(DATA)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=DATA.parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_real_data_cleaning_and_checksum(self):
        self.assertEqual(download(DATA), DATA)
        self.assertEqual(len(self.data), 24770)
        self.assertEqual(self.cleaning["conflicting_rows_removed"], 2)
        self.assertEqual(self.cleaning["duplicate_rows_removed"], 11)
        keys = self.data.tweet.str.lower().str.split().str.join(" ")
        self.assertTrue(keys.is_unique)
        source = pd.read_csv(DATA, dtype={"Unnamed: 0": str}).set_index(
            "Unnamed: 0"
        )
        expected = source.loc[self.data.id, "class"].eq(0).astype(int)
        self.assertEqual(self.data.label.tolist(), expected.tolist())
        altered = self.folder / "altered.csv"
        altered.write_bytes(DATA.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "checksum"):
            download(altered)

    def test_invalid_input_is_rejected(self):
        source = pd.read_csv(DATA).head(5)
        invalid = self.folder / "invalid.csv"
        for column, value in [
            ("tweet", " "),
            ("class", -1),
            ("Unnamed: 0", ""),
        ]:
            with self.subTest(column=column):
                source.assign(**{column: value}).to_csv(invalid, index=False)
                with self.assertRaises(ValueError):
                    load_dataset(invalid)
        source.drop(columns="tweet").to_csv(invalid, index=False)
        with self.assertRaises(ValueError):
            load_dataset(invalid)

    def test_vocabulary_does_not_learn_from_holdout(self):
        sample = self.data.groupby("label").head(30)
        training, holdout = sample.iloc[::2], sample.iloc[1::2]
        model = build_pipeline().fit(training.tweet, training.label)
        vectorizer = model.named_steps["tfidf"]
        tokenize = vectorizer.build_analyzer()
        tokens = {token for text in training.tweet for token in tokenize(text)}
        unseen = {
            token for text in holdout.tweet for token in tokenize(text)
        } - tokens
        self.assertTrue(unseen)
        model.predict(holdout.tweet)
        self.assertTrue(set(vectorizer.get_feature_names_out()) <= tokens)

    def test_training_and_prediction_roundtrip(self):
        sample = pd.read_csv(DATA).groupby("class").head(30)
        sample_path = self.folder / "sample.csv"
        sample.to_csv(sample_path, index=False)
        report = train(sample_path, self.folder)
        saved = json.loads(
            (self.folder / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report, saved)
        holdout = pd.read_csv(self.folder / "holdout_predictions.csv")
        self.assertAlmostEqual(
            report["holdout"]["macro_f1"],
            f1_score(holdout.label, holdout.prediction, average="macro"),
        )
        cleaned, _ = load_dataset(sample_path)
        model_path = self.folder / "model.joblib"
        model = joblib.load(model_path)
        self.assertEqual(
            model.named_steps["classifier"].class_count_.sum(), len(cleaned)
        )
        inputs = pd.concat([cleaned.iloc[:8][::-1], cleaned.iloc[:1]])[
            ["id", "tweet"]
        ]
        input_path, output_path = (
            self.folder / "input.csv",
            self.folder / "output.csv",
        )
        inputs.to_csv(input_path, index=False)
        predict(model_path, input_path, output_path)
        output = pd.read_csv(output_path, dtype={"id": str})
        self.assertEqual(output.id.tolist(), inputs.id.tolist())
        self.assertEqual(
            output.label.tolist(), model.predict(inputs.tweet).tolist()
        )
        with self.assertRaises(ValueError):
            predict(model_path, input_path, input_path)


if __name__ == "__main__":
    unittest.main()
