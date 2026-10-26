"""Compare the six classification models listed in the capstone guidelines.

Run from the repository root:
    python -m src.model_comparison

Uses one stratified train/test split and a preprocessing pipeline fitted on
training data only. The test set is for final comparison, not hyperparameter
tuning. Metrics are written to outputs/model_comparison.csv and
outputs/model_comparison_details.json. No metric values are hard-coded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import load_dataset, prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _to_dense(values):
    """Convert sparse one-hot output to dense for estimators such as GaussianNB."""
    return values.toarray() if hasattr(values, "toarray") else values


def _positive_probabilities(model, X_test):
    """Return positive-class scores where the estimator exposes probabilities."""
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            classes = list(getattr(model.named_steps.get("model"), "classes_", []))
        probabilities = model.predict_proba(X_test)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return scores
    return None


def compare_models(data):
    """Fit/evaluate candidate classifiers; accepts prepare_data() output."""
    candidates = [
        ("Logistic Regression", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42)),
        ("K-Nearest Neighbours", KNeighborsClassifier(n_neighbors=7)),
        ("Decision Tree", DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=10,
            random_state=42)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("Support Vector Machine", SVC(
            class_weight="balanced", probability=True, random_state=42)),
        ("Naive Bayes", GaussianNB()),
    ]
    detailed = []
    for name, estimator in candidates:
        steps = [("preprocessor", clone(data["preprocessor"]))]
        # GaussianNB requires dense input; other estimators accept sparse output.
        if name == "Naive Bayes":
            steps.append(("dense", FunctionTransformer(_to_dense, accept_sparse=True)))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        pipeline.fit(data["X_train"], data["y_train"])
        pred = pipeline.predict(data["X_test"])
        scores = _positive_probabilities(pipeline, data["X_test"])
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(data["y_test"], pred)),
            "precision": float(precision_score(data["y_test"], pred, zero_division=0)),
            "recall": float(recall_score(data["y_test"], pred, zero_division=0)),
            "f1": float(f1_score(data["y_test"], pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(data["y_test"], pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                data["y_test"], pred, labels=[0, 1], zero_division=0, output_dict=True
            ),
        }
        if scores is not None:
            metrics["roc_auc"] = float(roc_auc_score(data["y_test"], scores))
        else:
            metrics["roc_auc"] = None
        detailed.append(metrics)
        print(
            f"{name}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}, "
            f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}"
        )
        print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])

    return detailed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data(load_dataset())
    detailed = compare_models(data)
    summary = pd.DataFrame([
        {key: row[key] for key in (
            "model", "accuracy", "precision", "recall", "f1", "roc_auc"
        )}
        for row in detailed
    ]).sort_values(["f1", "recall"], ascending=False)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "model_comparison_details.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )
    print("\nComparison saved to outputs/model_comparison.csv")
    print("Detailed reports saved to outputs/model_comparison_details.json")
    print("Review precision/recall and false negatives before choosing a final model.")


if __name__ == "__main__":
    main()
