"""Smoke-test the six-model comparison without the real dataset."""
import unittest

import pandas as pd

from src.preprocessing import prepare_data
from src.model_comparison import compare_models


class ModelComparisonTests(unittest.TestCase):
    def test_all_six_models_return_expected_metrics(self):
        rows = 120
        frame = pd.DataFrame({
            "UDI": range(rows),
            "Product ID": [f"M{i:05d}" for i in range(rows)],
            "Type": (["L", "M", "H"] * (rows // 3)),
            "Air temperature [K]": [296.0 + (i % 11) * 0.3 for i in range(rows)],
            "Process temperature [K]": [306.0 + (i % 7) * 0.4 for i in range(rows)],
            "Rotational speed [rpm]": [1300 + (i % 19) * 17 for i in range(rows)],
            "Torque [Nm]": [25.0 + (i % 13) * 2.1 for i in range(rows)],
            "Tool wear [min]": [(i * 7) % 250 for i in range(rows)],
            "Machine failure": [1 if i % 9 == 0 else 0 for i in range(rows)],
            "TWF": [0] * rows, "HDF": [0] * rows, "PWF": [0] * rows,
            "OSF": [0] * rows, "RNF": [0] * rows,
        })
        results = compare_models(prepare_data(frame))

        self.assertEqual(len(results), 6)
        expected = {
            "model", "accuracy", "precision", "recall", "f1",
            "roc_auc", "confusion_matrix", "classification_report",
        }
        for result in results:
            self.assertTrue(expected.issubset(result))
            self.assertIsInstance(result["confusion_matrix"], list)
            self.assertIsNotNone(result["roc_auc"])
            self.assertGreaterEqual(result["roc_auc"], 0.0)
            self.assertLessEqual(result["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
