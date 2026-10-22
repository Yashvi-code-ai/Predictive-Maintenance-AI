"""Unit tests for the shared preprocessing helpers.

Run from the repository root with:
    python -m unittest discover -s tests -v
"""

import unittest

import pandas as pd

from src.preprocessing import (
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    prepare_data,
)


def sample_dataset(rows: int = 100) -> pd.DataFrame:
    """Build a small AI4I-shaped frame without needing the real CSV."""
    return pd.DataFrame(
        {
            "UDI": range(1, rows + 1),
            "Product ID": [f"M{index:05d}" for index in range(rows)],
            "Type": ["L", "M", "H", "L"] * (rows // 4) + ["L"] * (rows % 4),
            "Air temperature [K]": [300.0 + (index % 7) for index in range(rows)],
            "Process temperature [K]": [310.0 + (index % 5) for index in range(rows)],
            "Rotational speed [rpm]": [1500 + index for index in range(rows)],
            "Torque [Nm]": [35.0 + (index % 11) for index in range(rows)],
            "Tool wear [min]": [index % 250 for index in range(rows)],
            TARGET_COLUMN: [1 if index % 10 == 0 else 0 for index in range(rows)],
            "TWF": [0] * rows,
            "HDF": [0] * rows,
            "PWF": [0] * rows,
            "OSF": [0] * rows,
            "RNF": [0] * rows,
        }
    )


class PreprocessingTests(unittest.TestCase):
    def test_removes_exact_duplicates_and_reports_them(self) -> None:
        df = sample_dataset()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        cleaned, report = clean_dataset(df)

        self.assertEqual(len(cleaned), 100)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)

    def test_drops_rows_with_missing_target(self) -> None:
        df = sample_dataset()
        df.loc[0, TARGET_COLUMN] = None

        cleaned, report = clean_dataset(df)

        self.assertEqual(len(cleaned), 99)
        self.assertEqual(report["rows_with_missing_target_removed"], 1)

    def test_split_is_stratified_and_excludes_leakage_and_ids(self) -> None:
        prepared = prepare_data(sample_dataset())

        self.assertEqual(len(prepared["X_train"]) + len(prepared["X_test"]), 100)
        self.assertNotIn(TARGET_COLUMN, prepared["X_train"].columns)
        self.assertNotIn("UDI", prepared["X_train"].columns)
        self.assertNotIn("Product ID", prepared["X_train"].columns)
        for column in LEAKAGE_COLUMNS:
            self.assertNotIn(column, prepared["X_train"].columns)

        train_rate = prepared["y_train"].mean()
        test_rate = prepared["y_test"].mean()
        self.assertLess(abs(train_rate - test_rate), 0.03)

    def test_preprocessor_handles_missing_features_and_unseen_categories(self) -> None:
        df = sample_dataset()
        df.loc[1, "Torque [Nm]"] = None
        prepared = prepare_data(df)

        transformer = prepared["preprocessor"]
        train_transformed = transformer.fit_transform(prepared["X_train"])
        test_transformed = transformer.transform(prepared["X_test"])

        self.assertEqual(train_transformed.shape[1], test_transformed.shape[1])
        self.assertEqual(train_transformed.shape[0], len(prepared["X_train"]))
        self.assertEqual(test_transformed.shape[0], len(prepared["X_test"]))

    def test_rejects_non_binary_target(self) -> None:
        df = sample_dataset()
        df.loc[0, TARGET_COLUMN] = 2

        with self.assertRaises(ValueError):
            clean_dataset(df)


if __name__ == "__main__":
    unittest.main()
