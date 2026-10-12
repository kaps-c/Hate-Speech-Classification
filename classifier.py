"""Gaussian Naive Bayes baseline on the Davidson dataset."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

SEED = 42
MAX_FEATURES = 2000
LABEL_NAMES = ["non_hate", "hate"]
SMOOTHING_GRID = np.logspace(0, -9, num=10).tolist()
DATASET_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)
DATASET_SHA256 = (
    "fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281"
)


def download(path: str | Path = "data/labeled_data.csv") -> Path:
    """Download or reuse the CSV after verifying its checksum."""
    path = Path(path)
    if path.exists():
        payload = path.read_bytes()
    else:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise ValueError(
            "Dataset checksum mismatch; existing files have not been changed."
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return path


def build_pipeline() -> Pipeline:
    """Keep text features inside cross-validation and saved models."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=MAX_FEATURES, dtype=np.float32),
            ),
            (
                "dense",
                FunctionTransformer(csr_matrix.toarray, accept_sparse=True),
            ),
            ("classifier", GaussianNB()),
        ]
    )


def _validate_text_rows(data: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )
    if data.empty:
        raise ValueError("CSV must contain at least one row.")
    if (
        data["tweet"]
        .map(lambda text: not isinstance(text, str) or not text.strip())
        .any()
    ):
        raise ValueError("Every tweet must contain nonblank text.")


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Deduplicate text, then map original class 0 to hate speech."""
    source = pd.read_csv(
        path, dtype={"Unnamed: 0": str}, keep_default_na=False
    )
    _validate_text_rows(source, {"Unnamed: 0", "tweet", "class"})
    if not source["class"].isin([0, 1, 2]).all():
        raise ValueError("Davidson class values must be 0, 1, or 2.")
    ids = source["Unnamed: 0"]
    if ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError("Source IDs must be nonblank and unique.")
    keys = source.tweet.str.lower().str.split().str.join(" ")
    conflicts = source.groupby(keys)["class"].transform("nunique").gt(1)
    remaining = source.loc[~conflicts]
    duplicates = keys.loc[~conflicts].duplicated()
    clean = remaining.loc[~duplicates].copy()
    clean["label"] = clean["class"].eq(0).astype(int)
    clean = clean.rename(columns={"Unnamed: 0": "id"})
    report = {
        "input_rows": len(source),
        "conflicting_groups": int(keys[conflicts].nunique()),
        "conflicting_rows_removed": int(conflicts.sum()),
        "duplicate_rows_removed": int(duplicates.sum()),
        "output_rows": len(clean),
    }
    return clean[["id", "tweet", "label"]].reset_index(drop=True), report


def _metrics(labels, predictions) -> dict:
    scores = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": scores["accuracy"],
        "macro_f1": scores["macro avg"]["f1-score"],
        "classification_report": scores,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def _predict_labels(model: Pipeline, texts: pd.Series) -> np.ndarray:
    # Bound dense prediction allocations even for large input CSVs.
    return np.concatenate(
        [
            model.predict(texts.iloc[start : start + 512])
            for start in range(0, len(texts), 512)
        ]
    )


def train(data_path: str | Path, output_dir: str | Path = "artifacts") -> dict:
    """Save holdout results, then refit on all cleaned rows."""
    data, cleaning = load_dataset(data_path)
    if (
        set(data.label.unique()) != {0, 1}
        or data.label.value_counts().min() < 10
    ):
        raise ValueError(
            "Training requires at least 10 cleaned rows per binary class."
        )
    training, holdout = train_test_split(
        data,
        test_size=0.2,
        stratify=data.label,
        random_state=SEED,
    )
    search = GridSearchCV(
        build_pipeline(),
        {"classifier__var_smoothing": SMOOTHING_GRID},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(training.tweet, training.label)
    predictions = _predict_labels(search.best_estimator_, holdout.tweet)
    majority = int(training.label.mode().iloc[0])
    report = {
        "dataset": {
            "sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "cleaning": cleaning,
        },
        "split": {
            "seed": SEED,
            "holdout_fraction": 0.2,
            "train_rows": len(training),
            "holdout_rows": len(holdout),
        },
        "selection": {
            "var_smoothing": float(
                search.best_params_["classifier__var_smoothing"]
            ),
            "cv_macro_f1": float(search.best_score_),
        },
        "holdout": _metrics(holdout.label, predictions),
        "majority_baseline": _metrics(
            holdout.label, np.full(len(holdout), majority)
        ),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                )
            },
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    holdout[["id", "label"]].assign(prediction=predictions).to_csv(
        destination / "holdout_predictions.csv",
        index=False,
    )
    final_model = clone(search.best_estimator_).fit(data.tweet, data.label)
    joblib.dump(final_model, destination / "model.joblib")
    return report


def predict(
    model_path: str | Path, input_path: str | Path, output_path: str | Path
) -> None:
    """Predict every CSV row with a trusted model."""
    output_path = Path(output_path)
    if output_path.resolve() in {
        Path(input_path).resolve(),
        Path(model_path).resolve(),
    }:
        raise ValueError(
            "Prediction output must be different from input and model paths."
        )
    data = pd.read_csv(
        input_path, dtype={"id": str, "tweet": str}, keep_default_na=False
    )
    _validate_text_rows(data, {"id", "tweet"})
    if data.id.str.strip().eq("").any():
        raise ValueError("Prediction IDs must be nonblank.")
    model = joblib.load(model_path)
    labels = _predict_labels(model, data.tweet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": data.id, "label": labels}).to_csv(
        output_path, index=False
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "download", help="Download and verify the Davidson CSV"
    )
    fetch.add_argument(
        "--output", type=Path, default=Path("data/labeled_data.csv")
    )
    fit = commands.add_parser("train", help="Evaluate and save a model")
    fit.add_argument(
        "--data", type=Path, default=Path("data/labeled_data.csv")
    )
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    infer = commands.add_parser("predict", help="Predict an id,tweet CSV")
    infer.add_argument(
        "--model", type=Path, default=Path("artifacts/model.joblib")
    )
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(f"Verified dataset: {download(args.output)}")
        elif args.command == "train":
            print(
                "Tuning 10 candidates across 5 folds, "
                "then evaluating and refitting...",
                flush=True,
            )
            report = train(args.data, args.output_dir)
            print(f"Holdout macro F1: {report['holdout']['macro_f1']:.4f}")
            print(f"Results and fitted model: {args.output_dir}")
        else:
            predict(args.model, args.input, args.output)
            print(f"Predictions: {args.output}")
    except (OSError, ValueError, EOFError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
