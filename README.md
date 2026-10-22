# Predictive-Maintenance-AI

Machine Failure Prediction using Explainable Machine Learning.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Download the CSV and save it as `data/ai4i2020.csv`. The dataset is not committed to this repository.

## Preprocessing

The shared preprocessing utilities are in `src/preprocessing.py`. They:

- Validate the expected AI4I column schema and binary `Machine failure` target.
- Remove exact duplicate rows and rows with a missing target.
- Report missing feature values and IQR-based outlier counts. Outliers are reported rather than automatically deleted because extreme sensor readings may be meaningful.
- Exclude row identifiers (`UDI`, `Product ID`) and failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from predictors to avoid target leakage.
- Create a reproducible, stratified 80/20 train-test split (seed 42).
- Return a scikit-learn preprocessing transformer that imputes missing values, scales numeric features, and one-hot encodes categorical features. Fit it only through a model pipeline on the training set.

### Setup and run

Use Python 3.10+ in a virtual environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.preprocessing
```

If your CSV is stored elsewhere, pass `--csv-path /path/to/ai4i2020.csv` to the preprocessing command. The unit tests use a synthetic fixture and do not require the real CSV.

### Shared model-training interface

```python
from src.preprocessing import load_dataset, prepare_data

df = load_dataset()
data = prepare_data(df)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
preprocessor = data["preprocessor"]
```

For every model, put the preprocessor inside a scikit-learn `Pipeline` so transformations are fitted on training data only. Because failures are rare, report precision, recall, F1-score and confusion matrix alongside accuracy; ROC-AUC is also useful.

## Team ownership and current branches

- **Member 1 — Yashvi:** exploratory data analysis and baseline models (Logistic Regression and Decision Tree). Branch: `member1-eda-baselines`.
- **Member 2 — project teammate:** Random Forest training and explainability via permutation importance. Branch: `member2-random-forest`.
- Both members should use the same `src/preprocessing.py`, seed (42), stratified 80/20 split and leakage exclusions. Do not commit the downloaded dataset or generated output files unless the team agrees.

## Member 1 commands

```bash
python -m src.eda
python -m src.baseline_models
```

EDA figures and baseline metrics are written to `outputs/`.

## Member 2 commands

```bash
python -m src.random_forest
```

The script writes held-out metrics and permutation importance to `outputs/`. Permutation importance is a model explanation, not proof of causation.

## Evaluation notes

- The held-out test set must not be used for tuning model parameters.
- Accuracy alone can hide poor failure detection because the target is imbalanced.
- Compare precision, recall, F1 and ROC-AUC; inspect false negatives and false positives in the confusion matrix.
- Record limitations: synthetic-to-real-world domain gap, class imbalance, dataset age, and lack of deployment/maintenance cost data.
