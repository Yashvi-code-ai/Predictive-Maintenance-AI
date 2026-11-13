# Applied Machine Learning Capstone Report
## Machine Failure Prediction Using Machine Learning

> **Submission note:** The measured results below were supplied from successful local runs on the UCI AI4I 2020 dataset. Add screenshots from your own Streamlit run and record your actual Python/package versions before final submission.

## 1. Problem statement and objectives
Industrial equipment failure can interrupt production and increase maintenance costs. This project investigates whether machine operating conditions can be used to classify machine failure using the AI4I 2020 dataset.

Objectives:
- Inspect and clean the dataset and document its data quality.
- Build a reproducible preprocessing and classification pipeline.
- Compare six classification algorithms using metrics suitable for imbalanced classes.
- Explain important model features and discuss limitations.
- Demonstrate a single-record prediction through a Streamlit interface.

## 2. Dataset description
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Local file:** `data/ai4i2020.csv` (not committed to Git).
- **Rows before cleaning:** 10,000.
- **Rows after cleaning:** 10,000.
- **Target:** `Machine failure` (0 = no failure, 1 = failure).
- **Class distribution:** 9,661 normal records (96.61%) and 339 failure records (3.39%).
- **Input features (6):** Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min].
- **Missing values:** 0 in every column.

The target is strongly imbalanced: failure records represent only 3.39% of the dataset. Consequently, accuracy alone is not sufficient for model selection.

## 3. Exploratory data analysis
The EDA run retained all 10,000 rows. The target distribution was 9,661 normal cases and 339 machine-failure cases. No missing values were found.

The IQR analysis reported 418 outliers in Rotational speed [rpm] and 69 in Torque [Nm]. No IQR outliers were reported for Air temperature [K], Process temperature [K], or Tool wear [min]. These values were reported, not removed, because unusual operating conditions may contain useful predictive information.

Insert the generated figures from your local `outputs/` directory:
- `outputs/target_distribution.png`
- `outputs/numeric_correlation.png`

Discuss sensor-feature distributions and correlations visible in these figures. Correlation does not establish causation.

## 4. Data cleaning and preprocessing
The preprocessing and EDA runs reported:
- Rows before cleaning: **10,000**.
- Exact duplicate rows removed: **0**.
- Rows removed for missing target: **0**.
- Rows after cleaning: **10,000**.
- Missing values by column: **0**.
- Train/test split: **8,000 / 2,000** (stratified 80/20).
- Training target distribution: **9,661 normal / 339 failure** in the full cleaned dataset; the held-out test set contained 1,932 normal and 68 failure records.
- IQR outliers reported: Rotational speed **418**; Torque **69**; all other numeric input features **0**.

Numeric values are imputed and scaled, and the categorical Type feature is imputed and one-hot encoded inside each model pipeline.

### Leakage prevention
The model uses six input features. Identifiers `UDI` and `Product ID` are excluded. Failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they can disclose information about the failure target.

## 5. Experimental methodology
- Train/test split: stratified 80/20 (8,000 training rows and 2,000 test rows).
- Random seed: 42.
- Models compared: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Support Vector Machine, and Naive Bayes.
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Preprocessing is fitted within the model pipelines.

The results below are the observed held-out test results from the project runs. No threshold tuning or hyperparameter optimization was performed for these reported figures.

## 6. Model evaluation and comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 0.242 | 0.907 |
| K-Nearest Neighbours | 97.25% | 80.95% | 25.00% | 0.382 | 0.856 |
| Decision Tree | 94.70% | 37.66% | 85.29% | 0.523 | 0.915 |
| **Random Forest** | **97.25%** | **57.14%** | **76.47%** | **0.654** | **0.967** |
| Support Vector Machine | 91.75% | 27.85% | **89.71%** | 0.425 | 0.965 |
| Naive Bayes | 95.80% | 25.00% | 11.76% | 0.160 | 0.847 |

### Confusion matrices
Rows represent actual classes [normal, failure] and columns represent predicted classes [normal, failure]. Each matrix is in the order `[[TN, FP], [FN, TP]]`.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 1,593 | 339 | 12 | 56 |
| K-Nearest Neighbours | 1,928 | 4 | 51 | 17 |
| Decision Tree | 1,836 | 96 | 10 | 58 |
| Random Forest | 1,893 | 39 | 16 | 52 |
| Support Vector Machine | 1,774 | 158 | 7 | 61 |
| Naive Bayes | 1,908 | 24 | 60 | 8 |

Random Forest achieved the highest F1-score (0.654) and ROC-AUC (0.967), with 97.25% accuracy. SVM achieved the highest failure recall (89.71%), detecting 61 of 68 failures, but generated 158 false alarms. Random Forest detected 52 of 68 failures and generated 39 false alarms. KNN achieved high precision (80.95%) but detected only 17 of 68 failures.

Because only 68 of the 2,000 test records were failures, accuracy can hide poor failure detection. Precision measures how many predicted failures were correct; recall measures how many actual failures were detected; F1 balances precision and recall; ROC-AUC measures ranking discrimination across thresholds.

## 7. Final model selection and justification
**Selected model: Random Forest Classifier.**

Random Forest was selected as the primary model because it achieved the best F1-score (0.654) and ROC-AUC (0.967) among the six evaluated models, alongside 97.25% accuracy. Its test confusion matrix was:
- True negatives: 1,893
- False positives: 39
- False negatives: 16
- True positives: 52

This is a balanced choice for the project demonstration. SVM may be preferable in a scenario where minimizing missed failures is the overriding priority, but its higher recall came with substantially more false alarms. The selected model is an educational prototype, not an operational maintenance or safety system.

## 8. Explainability
Permutation importance was calculated by shuffling one input feature and measuring the decrease in held-out F1-score. The observed Random Forest results were:

| Rank | Feature | Mean importance | Standard deviation |
|---:|---|---:|---:|
| 1 | Torque [Nm] | 0.3247 | 0.0248 |
| 2 | Rotational speed [rpm] | 0.2927 | 0.0332 |
| 3 | Air temperature [K] | 0.1914 | 0.0317 |
| 4 | Tool wear [min] | 0.1317 | 0.0107 |
| 5 | Type | 0.0398 | 0.0189 |
| 6 | Process temperature [K] | 0.0125 | 0.0149 |

Torque and rotational speed were the two most influential features in this evaluation. Permutation importance indicates predictive usefulness for this dataset and split; it does not show that a feature causes failure. The low measured importance of Process temperature does not by itself justify removing it.

Insert `outputs/random_forest_feature_importance.png` in the final document.

## 9. Streamlit demonstration
Run:
```bash
python train_model.py
streamlit run app.py
```

Add screenshots from your own successful local run. Demonstrate at least:
1. A typical input and its predicted class/probability.
2. A second input with different operating conditions.
3. The input summary and prediction interpretation.
4. A clear explanation that the probability may not be calibrated and the app is not a certified safety system.

## 10. Limitations
- AI4I 2020 is a synthetic dataset and may not generalize to actual factory equipment.
- Failure classes are imbalanced.
- Results depend on the train/test split and model decision threshold.
- Model probabilities are not necessarily calibrated real-world risks.
- Sensor conditions alone may omit maintenance history, environment, machine-specific differences, and operational costs.
- Feature importance does not prove causation.
- This prototype is not a certified industrial safety system.

## 11. Future scope
- Validate on real industrial sensor and maintenance data.
- Evaluate calibration and generalization on additional data.
- Add monitoring for data drift and retraining procedures.
- Add local explanations for individual predictions and human review workflows.

## 12. Team contribution
- **Member 1 — Yashvi:** EDA, baseline classifiers, Streamlit UI, and demo documentation.
- **Member 2 — Vishesh:** Random Forest evaluation, model-serving pipeline, and explainability.
- **Shared responsibility:** preprocessing, model comparison, report, and viva preparation.

Adjust this section if the actual division of work differs.

## 13. Reproducibility
Record the actual Python and package versions and date of execution before submission. The dataset is downloaded separately from the [UCI AI4I 2020 dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and stored locally as `data/ai4i2020.csv`. Generated metrics and figures are written to `outputs/`. Do not commit the raw dataset or generated model artifact.
