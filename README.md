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
python -m src.preprocessing
```

If your CSV is stored elsewhere:

```bash
python -m src.preprocessing --csv-path /path/to/ai4i2020.csv
```

Run preprocessing unit tests (they use a small synthetic AI4I-shaped fixture and do not require downloading the dataset):

```bash
python -m unittest discover -s tests -v
```

### Shared model-training interface

```python
from src.preprocessing import load_dataset, prepare_data

df = load_dataset()
data = prepare_data(df)

# Fit transformations only on training data, as part of each model pipeline.
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
preprocessor = data["preprocessor"]
```

For each model, use `Pipeline([("preprocessor", preprocessor), ("model", estimator)])`. Do not fit the preprocessor on the complete dataset before splitting. Since failures are rare, report precision, recall, F1-score and confusion matrix alongside accuracy; ROC-AUC can also be useful.
