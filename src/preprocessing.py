"""Shared preprocessing utilities for the AI4I 2020 predictive-maintenance project.

Expected raw file:
    data/ai4i2020.csv

Download the AI4I 2020 Predictive Maintenance Dataset from:
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Important: the failure-mode flags (TWF, HDF, PWF, OSF, RNF) describe causes
of a failure and must not be used to predict the overall Machine failure
target. They would leak the answer into the features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# These are labels describing how a failure happened, so they leak the target.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
# UDI is a row identifier; Product ID is an identifier that can encourage
# memorisation rather than generalisation. Keep Type as the meaningful category.
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET_COLUMN,
    *LEAKAGE_COLUMNS,
]


def load_dataset(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw AI4I CSV without silently changing its schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the AI4I 2020 CSV from "
            "the UCI repository and save it as data/ai4i2020.csv, or pass "
            "the file path with --csv-path."
        )

    df = pd.read_csv(path)
    # Some copies of the dataset have a UTF-8 BOM or whitespace in headers.
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "The input does not look like the expected AI4I 2020 dataset. "
            f"Missing columns: {missing_columns}. Found: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"The dataset at {path} contains no rows.")

    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove exact duplicates and invalid targets; report missingness and outliers.

    Missing feature values are deliberately left for the training-only
    imputers in the sklearn pipeline. This avoids calculating imputation values
    using the held-out test set. IQR outliers are reported, not automatically
    deleted, because extreme sensor readings may be meaningful for maintenance.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")

    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()

    missing_target_rows = int(cleaned[TARGET_COLUMN].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).copy()

    # Ensure the target is a binary 0/1 value. Do not silently coerce unknowns.
    target_values = set(pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").dropna().unique())
    if not target_values.issubset({0, 1}) or cleaned[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{TARGET_COLUMN!r} must contain only 0 and 1 after removing missing "
            f"targets. Observed values: {sorted(map(str, cleaned[TARGET_COLUMN].unique()))}"
        )
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("int64")

    feature_columns = [
        column
        for column in cleaned.columns
        if column not in {TARGET_COLUMN, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS}
    ]
    numeric_columns = cleaned[feature_columns].select_dtypes(include="number").columns
    outlier_counts: dict[str, int] = {}
    for column in numeric_columns:
        values = cleaned[column].dropna()
        if values.empty:
            outlier_counts[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[column] = int(((values < lower_bound) | (values > upper_bound)).sum())

    report: dict[str, Any] = {
        "rows_before_cleaning": original_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "rows_with_missing_target_removed": missing_target_rows,
        "rows_after_cleaning": len(cleaned),
        "missing_values_by_column": cleaned.isna().sum().astype(int).to_dict(),
        "target_distribution": cleaned[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "iqr_outlier_counts_by_numeric_feature": outlier_counts,
        "dropped_identifier_columns": [c for c in IDENTIFIER_COLUMNS if c in cleaned.columns],
        "excluded_target_leakage_columns": [c for c in LEAKAGE_COLUMNS if c in cleaned.columns],
    }
    return cleaned, report


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable transformer; fit it only on X_train, never on all data."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Create a reproducible stratified split and an unfitted preprocessing step.

    Returns raw X_train/X_test and y_train/y_test plus a preprocessor. Each model
    should use Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    so preprocessing is fitted on training rows only during model.fit().
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    cleaned, cleaning_report = clean_dataset(df)
    X = cleaned.drop(
        columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS],
        errors="ignore",
    )
    y = cleaned[TARGET_COLUMN]

    if y.nunique() != 2:
        raise ValueError(
            f"Expected a binary target with classes 0 and 1; got {sorted(y.unique())}."
        )
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two rows for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": build_preprocessor(X_train),
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "random_state": random_state,
        "test_size": test_size,
        "cleaning_report": cleaning_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and prepare the AI4I predictive-maintenance dataset."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the raw AI4I CSV (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()

    raw_df = load_dataset(args.csv_path)
    prepared = prepare_data(raw_df)
    report = prepared["cleaning_report"]

    print("\nDataset cleaning summary")
    print("------------------------")
    print(f"Rows before cleaning: {report['rows_before_cleaning']}")
    print(f"Exact duplicates removed: {report['exact_duplicate_rows_removed']}")
    print(f"Rows removed for missing target: {report['rows_with_missing_target_removed']}")
    print(f"Rows after cleaning: {report['rows_after_cleaning']}")
    print(f"Training rows: {len(prepared['X_train'])}")
    print(f"Test rows: {len(prepared['X_test'])}")
    print(f"Features used ({len(prepared['feature_columns'])}): {prepared['feature_columns']}")
    print(f"Target distribution: {report['target_distribution']}")
    print(f"Missing values by column: {report['missing_values_by_column']}")
    print(f"IQR outlier counts (reported, not removed): {report['iqr_outlier_counts_by_numeric_feature']}")
    print(f"Excluded leakage columns: {report['excluded_target_leakage_columns']}")
    print("\nPreprocessing is ready. Fit the returned preprocessor only via a model pipeline.")


if __name__ == "__main__":
    main()
