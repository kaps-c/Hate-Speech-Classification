# Hate Speech Classification

A small text-classification baseline from 50.007 coursework: TF-IDF (word-importance features) and Gaussian Naive Bayes. `classifier.py` contains the implementation; `naive-bayes.ipynb` explains the experiment.

## Run

Python **3.13**, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python classifier.py download
python classifier.py train
```

On macOS/Linux, activate with `source .venv/bin/activate`. On Windows, you can also use `.\.venv\Scripts\python.exe` directly. For the notebook, select this environment in your notebook editor.

Download verifies and caches the dataset. Training takes a few minutes on CPU and saves metrics, holdout predictions, and the fitted model under `artifacts/`. Rerunning replaces these outputs; use `--output-dir` to keep separate runs.

```powershell
python classifier.py predict --input posts.csv --output artifacts/predictions.csv
python -m unittest -v
```

Prediction input: `id,tweet`; output: `id,label`, preserving IDs and row order. **1 = hate speech; 0 = non-hate, including offensive language.** Only load trusted model files. Tests use the downloaded real dataset, with no extra test dependencies.

## Method and results

Uses the [Davidson et al. dataset](https://github.com/t-davidson/hate-speech-and-offensive-language), replacing the unavailable Kaggle competition data. Cleaning removes 11 repeated rows and 2 conflicting rows, leaving **24,770 posts**. Original class `0` maps to hate; classes `1` and `2` map to non-hate.

A stratified 80/20 split preserves class proportions (seed `42`). Five-fold cross-validation selects smoothing by **macro F1**, the average F1 across classes. TF-IDF learns only from training folds, with a 2,000-word vocabulary. The final model is refitted on all cleaned data after evaluation.

Measured on **4,954 held-out posts**:

| Model | Accuracy | Macro F1 | Hate precision | Hate recall |
| --- | ---: | ---: | ---: | ---: |
| Gaussian Naive Bayes | 0.9231 | 0.6938 | 0.3753 | 0.5000 |
| Always predict non-hate | 0.9423 | 0.4851 | 0.0000 | 0.0000 |

The model catches half the labeled hate speech, but most flags are false positives. This historical dataset contains offensive text and [documented bias](https://arxiv.org/abs/1905.12516). Treat this as a baseline, not a validated moderation system.

Code: [MIT](LICENSE). Data: [upstream MIT license](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/LICENSE). Reference: Davidson et al. (2017), [*Automated Hate Speech Detection and the Problem of Offensive Language*](https://arxiv.org/abs/1703.04009).
