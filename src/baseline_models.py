"""Baseline classifiers owned by Member 1 (Yashvi).

Run from the repository root after placing data/ai4i2020.csv:
    python -m src.baseline_models

Metrics include precision/recall/F1 because machine failures are rare.
All transformations are fitted inside each model Pipeline on training data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import load_dataset, prepare_data

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_model(name, estimator, data):
    pipeline = Pipeline([
        ("preprocessor", data["preprocessor"]),
        ("model", estimator),
    ])
    pipeline.fit(data["X_train"], data["y_train"])
    predictions = pipeline.predict(data["X_test"])
    probabilities = pipeline.predict_proba(data["X_test"])[:, 1]
    return pipeline, {
        "model": name,
        "accuracy": float(accuracy_score(data["y_test"], predictions)),
        "precision": float(precision_score(data["y_test"], predictions, zero_division=0)),
        "recall": float(recall_score(data["y_test"], predictions, zero_division=0)),
        "f1": float(f1_score(data["y_test"], predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(data["y_test"], probabilities)),
        "confusion_matrix": confusion_matrix(data["y_test"], predictions).tolist(),
        "classification_report": classification_report(
            data["y_test"], predictions, zero_division=0, output_dict=True
        ),
    }


def main():
    df = load_dataset()
    data = prepare_data(df)
    models = [
        ("Logistic Regression", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        )),
        ("Decision Tree", DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=10,
            random_state=42
        )),
    ]
    results = []
    for name, estimator in models:
        _, metrics = evaluate_model(name, estimator, data)
        results.append(metrics)
        print(
            f"{name}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}, "
            f"roc_auc={metrics['roc_auc']:.3f}"
        )
        print("Confusion matrix [ [TN, FP], [FN, TP] ]:", metrics["confusion_matrix"])

    summary = pd.DataFrame([
        {key: row[key] for key in ("model", "accuracy", "precision", "recall", "f1", "roc_auc")}
        for row in results
    ]).sort_values("f1", ascending=False)
    summary.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    (OUTPUT_DIR / "baseline_details.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nSaved outputs/baseline_metrics.csv and outputs/baseline_details.json")


if __name__ == "__main__":
    main()
