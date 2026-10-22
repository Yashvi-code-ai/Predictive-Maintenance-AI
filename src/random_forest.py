"""Random Forest model and permutation-based explainability (Member 2).

Run from repository root:
    python -m src.random_forest

Requires data/ai4i2020.csv. The script writes held-out metrics, a confusion
matrix, and permutation importance for the raw input features. It does not
use the failure subtype flags as predictors.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import load_dataset, prepare_data

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    data = prepare_data(load_dataset())
    model = Pipeline([
        ("preprocessor", data["preprocessor"]),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    proba = model.predict_proba(data["X_test"])[:, 1]
    metrics = {
        "model": "Random Forest",
        "accuracy": float(accuracy_score(data["y_test"], pred)),
        "precision": float(precision_score(data["y_test"], pred, zero_division=0)),
        "recall": float(recall_score(data["y_test"], pred, zero_division=0)),
        "f1": float(f1_score(data["y_test"], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(data["y_test"], proba)),
        "confusion_matrix": confusion_matrix(data["y_test"], pred).tolist(),
        "classification_report": classification_report(
            data["y_test"], pred, zero_division=0, output_dict=True
        ),
    }
    (OUTPUT_DIR / "random_forest_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Permutation importance on raw X_test measures the impact of shuffling
    # each original input column through the complete fitted pipeline.
    importance = permutation_importance(
        model, data["X_test"], data["y_test"],
        scoring="f1", n_repeats=8, random_state=42, n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": data["X_test"].columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "random_forest_permutation_importance.csv", index=False)

    top = importance_df.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Decrease in held-out F1 after permutation")
    ax.set_title("Random Forest permutation importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "random_forest_feature_importance.png", dpi=160)
    plt.close(fig)

    print(
        "Random Forest: "
        f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}"
    )
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:", metrics["confusion_matrix"])
    print(f"Saved model evaluation and explainability outputs under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
