# Applied Machine Learning Capstone Report
## Machine Failure Prediction Using Machine Learning

> **Submission note:** Replace every `[FILL FROM RUN]` field with measured results from your own run. Do not invent metrics or claim that a model is best before comparing the held-out results.

## 1. Problem statement and objectives
Industrial equipment failure can interrupt production and increase maintenance costs. This project investigates whether machine operating conditions can be used to classify whether a failure is likely to occur in the AI4I 2020 dataset.

Objectives:
- Inspect and clean the dataset and document its data quality.
- Build a reproducible preprocessing and classification pipeline.
- Compare at least three classification algorithms using metrics suitable for imbalanced classes.
- Explain important model features and discuss limitations.
- Demonstrate a single-record prediction through a Streamlit interface.

## 2. Dataset description
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Local file:** `data/ai4i2020.csv` (not committed to Git).
- **Target:** `Machine failure` (0 = no failure, 1 = failure).
- **Input features:** Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
- **Rows before cleaning:** [FILL FROM RUN]
- **Rows after cleaning:** [FILL FROM RUN]
- **Class distribution:** [FILL FROM RUN]
- **Missing values:** [FILL FROM RUN]

Describe the units, column meanings, and class imbalance based on the generated EDA outputs.

## 3. Exploratory data analysis
Insert and interpret the generated figures:
- `outputs/target_distribution.png`
- `outputs/numeric_correlation.png`

Discuss target imbalance, sensor-feature distributions/correlations, and any relevant data-quality findings. Avoid treating correlation as causation.

## 4. Data cleaning and preprocessing
Document the actual results from `python -m src.preprocessing` and `python -m src.eda`.
- Exact duplicate rows: [FILL FROM RUN]
- Rows removed for missing target: [FILL FROM RUN]
- Missing feature values: [FILL FROM RUN]
- IQR outliers reported: [FILL FROM RUN]

Exact duplicate rows and rows without a target are removed. Feature imputation and numeric scaling/categorical encoding are fitted inside each pipeline using training data only. Outliers are reported rather than automatically deleted because unusual sensor values can be meaningful.

### Leakage prevention
Exclude `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`, because these failure-mode indicators can disclose the target. Exclude identifiers `UDI` and `Product ID`. Retain `Type` as a categorical predictor.

## 5. Experimental methodology
- Train/test split: stratified 80/20.
- Random seed: 42.
- Preprocessing: median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Models: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Support Vector Machine, and Naive Bayes (or clearly state the subset actually run).
- All transformations are fit on training data only.
- The held-out test set is used for final comparison, not repeated hyperparameter tuning.

## 6. Model evaluation and comparison
Run `python -m src.model_comparison` and transfer the actual values from `outputs/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| K-Nearest Neighbours | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Decision Tree | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Random Forest | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Support Vector Machine | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Naive Bayes | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

Include the confusion matrix for each model from `outputs/model_comparison_details.json`. Explain true positives, true negatives, false positives, and false negatives. Since failures are rare, accuracy alone is insufficient; discuss precision/recall trade-offs and ROC-AUC.

## 7. Final model selection and justification
**Selected model:** [FILL AFTER REVIEWING MEASURED RESULTS]

Justify the selection using the observed metrics and project objective. Explain why the selected precision/recall trade-off is appropriate. If the demo uses Random Forest but another model wins, align the demo trainer with the selected model before submission.

## 8. Explainability
Use `outputs/random_forest_permutation_importance.csv` and `outputs/random_forest_feature_importance.png` if Random Forest remains the chosen model.
- Most influential features observed: [FILL FROM RUN]
- How permutation importance is calculated: shuffle one input feature and measure the resulting decrease in held-out F1.
- Interpretation: importance indicates predictive usefulness for this evaluation, not causality.
- Individual example: [ADD one test/sample prediction and explain its inputs and output without implying causal certainty.]

## 9. Streamlit demonstration
Run `python -m src.train_model`, then `streamlit run app.py`.
Demonstrate at least:
1. A typical input and its predicted class/probability.
2. A second input with different operating conditions.
3. The input summary and interpretation/limitations.
4. A clear explanation that probability may not be calibrated and the app is not a certified safety system.

Add screenshots from your own successful local run.

## 10. Limitations
- AI4I is synthetic and may not generalize to actual factory equipment.
- Failures are imbalanced; the evaluation can vary with split and threshold.
- Model probability is not necessarily calibrated.
- Sensor conditions alone may omit maintenance history, environment, machine-specific differences, and operational costs.
- Feature importance does not prove causation.
- This prototype is not a certified industrial safety system.

## 11. Future scope
- Validate on real industrial sensor and maintenance data.
- Tune thresholds using a validation set and explicit false-negative/false-positive costs.
- Evaluate calibration and temporal/generalization performance.
- Add monitoring for data drift and retraining procedures.
- Add local explanations for individual predictions and human review workflows.

## 12. Team contribution
- **Member 1 — Yashvi:** EDA, baseline classifiers, Streamlit UI, and documentation contributions.
- **Member 2 — Vishesh:** Random Forest/evaluation, model-serving pipeline, testing, and explainability contributions.
- **Shared responsibility:** both members must be able to explain the dataset, preprocessing, leakage prevention, model comparison, metrics, limitations, and live demo.

Adjust this section to match the work each member actually completed.

## 13. Reproducibility
Record Python and package versions, date of execution, dataset source, and commands used. Attach the generated metric files or transcribe their results. Do not commit the raw dataset or generated model artifact.
