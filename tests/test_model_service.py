import unittest
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.model_service import FEATURE_COLUMNS, predict_machine_failure

class ModelServiceTests(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [298.1, 300.0, 299.0, 301.0],
            "Process temperature [K]": [308.0, 310.0, 309.0, 311.0],
            "Rotational speed [rpm]": [1500, 1400, 1600, 1300],
            "Torque [Nm]": [40.0, 50.0, 35.0, 55.0],
            "Tool wear [min]": [10, 100, 50, 200],
        })
        pre = ColumnTransformer([
            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("numeric", "passthrough", FEATURE_COLUMNS[1:]),
        ])
        self.model = Pipeline([("preprocessor", pre),
            ("model", DummyClassifier(strategy="prior"))]).fit(X, [0, 1, 0, 1])
        self.sample = dict(zip(FEATURE_COLUMNS, ["L", 298.5, 308.5, 1500, 42.0, 25]))

    def test_prediction_contract(self):
        result = predict_machine_failure(self.model, self.sample)
        self.assertIn(result["prediction"], (0, 1))
        self.assertIn("failure_probability", result)
        self.assertIn(result["label"], ("Potential failure detected", "No failure predicted"))

    def test_missing_features_raise_error(self):
        with self.assertRaisesRegex(ValueError, "Missing required input features"):
            predict_machine_failure(self.model, {"Type": "L"})

if __name__ == "__main__":
    unittest.main()
