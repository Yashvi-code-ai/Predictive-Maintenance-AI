"""Train and persist the fitted preprocessing + classifier pipeline.

Run from the repository root after placing the UCI CSV at data/ai4i2020.csv.
The notebook remains the source of truth for model comparison. This demo
trainer currently configures Random Forest; align this choice with the notebook's
measured results before final submission.
"""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import (DEFAULT_DATA_PATH, IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS, TARGET_COLUMN, build_preprocessor, clean_dataset, load_dataset)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "predictive_maintenance_pipeline.joblib"

def train_and_save_model(csv_path=DEFAULT_DATA_PATH, model_path=MODEL_PATH):
    raw = load_dataset(csv_path)
    cleaned, _ = clean_dataset(raw)
    X = cleaned.drop(columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS], errors="ignore")
    y = cleaned[TARGET_COLUMN].astype(int)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X, y)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    print(f"Saved fitted pipeline to {destination} (rows={len(X)})")
    return destination

if __name__ == "__main__":
    train_and_save_model()
