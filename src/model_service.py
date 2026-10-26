"""Load the trained pipeline and run a single machine-failure prediction."""
from pathlib import Path
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "predictive_maintenance_pipeline.joblib"
FEATURE_COLUMNS = ["Type", "Air temperature [K]", "Process temperature [K]",
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the persisted sklearn Pipeline or raise a helpful error."""
    import joblib
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run 'python -m src.train_model' first."
        )
    return joblib.load(path)

def predict_machine_failure(model: Any, machine_input: dict[str, Any]) -> dict[str, Any]:
    """Return predicted class and positive-class probability for one machine."""
    missing = [name for name in FEATURE_COLUMNS if name not in machine_input]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")
    row = pd.DataFrame([{name: machine_input[name] for name in FEATURE_COLUMNS}])
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            classes = list(getattr(model.named_steps.get("model"), "classes_", []))
        if 1 in classes:
            probability = float(model.predict_proba(row)[0][classes.index(1)])
    return {
        "prediction": prediction,
        "failure_probability": probability,
        "label": "Potential failure detected" if prediction == 1 else "No failure predicted",
    }
