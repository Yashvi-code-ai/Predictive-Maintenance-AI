# Predictive Maintenance AI

Machine-failure prediction using explainable machine learning on the UCI AI4I 2020 Predictive Maintenance Dataset.

## Project goals

- Explore the dataset and document data-quality findings.
- Train and compare Logistic Regression, Decision Tree, and Random Forest classifiers.
- Prevent target leakage and fit preprocessing only on training data.
- Evaluate minority-class performance and explain model predictions with permutation importance.

## Dataset setup

1. Download the CSV from the [UCI AI4I 2020 dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Save it locally as `data/ai4i2020.csv`.

The raw dataset is intentionally **not tracked by Git**. The `.gitignore` excludes local CSVs in `data/` and generated files in `outputs/`.

## Requirements and setup

Use Python 3.10 or newer and create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the project

From the repository root:

```bash
# Unit tests (synthetic test data; real CSV not required)
python -m unittest discover -s tests -v

# Dataset cleaning report and stratified train/test split summary
python -m src.preprocessing

# Exploratory analysis and baseline classifiers
python -m src.eda
python -m src.baseline_models

# Random Forest evaluation and permutation importance
python -m src.random_forest
```

All scripts that load the real dataset expect `data/ai4i2020.csv`. Preprocessing also accepts a custom path:

```bash
python -m src.preprocessing --csv-path /path/to/ai4i2020.csv
```

Generated charts, summaries, and metrics are written to `outputs/` and are excluded from version control. Run the scripts to regenerate them.

## Preprocessing and leakage prevention

The shared utilities in `src/preprocessing.py`:

- Validate the expected AI4I schema and binary `Machine failure` target.
- Remove exact duplicate rows and rows with a missing target.
- Report missing feature values and IQR-based outliers; do not automatically delete outliers because unusual sensor readings may be meaningful.
- Exclude row identifiers (`UDI`, `Product ID`) and failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from predictors.
- Use a reproducible, stratified 80/20 split with random seed 42.
- Impute missing values, scale numeric features, and one-hot encode categorical features inside the model pipeline so the transformers are fitted on training data only.

## Models and evaluation

| Model | Script |
|---|---|
| Logistic Regression | `src/baseline_models.py` |
| Decision Tree | `src/baseline_models.py` |
| Random Forest | `src/random_forest.py` |

Because failures are rare, do not rely on accuracy alone. Compare precision, recall, F1-score, ROC-AUC, and the confusion matrix; pay particular attention to false negatives and false positives. The held-out test set should not be used to tune hyperparameters.

Permutation importance describes how shuffling an input feature affects the chosen evaluation score; it does not establish causation.

## Team contributions

- **Member 1 — Yashvi:** exploratory data analysis and baseline classifiers.
- **Member 2 — project teammate:** Random Forest model and permutation-based feature importance.
- **Shared:** preprocessing, leakage prevention, tests, documentation, and model comparison.

Both members should use the shared preprocessing module, random seed, split strategy, and feature exclusions.

## Limitations and future scope

- The AI4I dataset is synthetic and may not represent every real industrial setting.
- Failure classes are imbalanced, and the best decision threshold depends on maintenance costs.
- The dataset does not supply the operational context needed to estimate real-world downtime or intervention costs.
- Future work could add threshold tuning on validation data, calibration, model persistence, a Streamlit demo, and evaluation on real sensor data.
