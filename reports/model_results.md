# Model Results — AI4I 2020 Predictive Maintenance

These are the measured results supplied from the project's local runs. Dataset: 10,000 rows; stratified 80/20 split; held-out test set: 2,000 rows, including 1,932 normal and 68 failure cases. Failure (class 1) is treated as the positive class.

## Data quality and cleaning

| Item | Result |
|---|---:|
| Rows before cleaning | 10,000 |
| Exact duplicates removed | 0 |
| Rows removed for missing target | 0 |
| Rows after cleaning | 10,000 |
| Train rows | 8,000 |
| Test rows | 2,000 |
| Normal target cases | 9,661 (96.61%) |
| Failure target cases | 339 (3.39%) |
| Missing values | 0 in all columns |
| Rotational speed IQR outliers (reported, not removed) | 418 |
| Torque IQR outliers (reported, not removed) | 69 |
| Other numeric input IQR outliers | 0 |

The six model inputs were Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min]. Identifier columns UDI and Product ID and failure-mode columns TWF, HDF, PWF, OSF, and RNF were excluded.

## Six-model comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 0.242 | 0.907 |
| K-Nearest Neighbours | 97.25% | 80.95% | 25.00% | 0.382 | 0.856 |
| Decision Tree | 94.70% | 37.66% | 85.29% | 0.523 | 0.915 |
| **Random Forest** | **97.25%** | **57.14%** | **76.47%** | **0.654** | **0.967** |
| Support Vector Machine | 91.75% | 27.85% | **89.71%** | 0.425 | 0.965 |
| Naive Bayes | 95.80% | 25.00% | 11.76% | 0.160 | 0.847 |

## Confusion matrices

All matrices use `[[TN, FP], [FN, TP]]`.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 1,593 | 339 | 12 | 56 |
| K-Nearest Neighbours | 1,928 | 4 | 51 | 17 |
| Decision Tree | 1,836 | 96 | 10 | 58 |
| Random Forest | 1,893 | 39 | 16 | 52 |
| Support Vector Machine | 1,774 | 158 | 7 | 61 |
| Naive Bayes | 1,908 | 24 | 60 | 8 |

## Selected model: Random Forest

Random Forest was selected for the project demonstration because it achieved the highest F1-score (0.654) and ROC-AUC (0.967), with 97.25% accuracy. It correctly detected 52 of 68 failures, missed 16, and generated 39 false alarms.

SVM achieved the highest recall (89.71%; 61 of 68 failures detected) but generated 158 false alarms. This illustrates the precision/recall trade-off. The selected Random Forest is a balanced academic prototype, not a certified industrial safety system.

## Random Forest permutation importance

Importance is the mean decrease in held-out F1 after shuffling a feature.

| Rank | Feature | Mean | Standard deviation |
|---:|---|---:|---:|
| 1 | Torque [Nm] | 0.324738 | 0.024758 |
| 2 | Rotational speed [rpm] | 0.292705 | 0.033207 |
| 3 | Air temperature [K] | 0.191436 | 0.031699 |
| 4 | Tool wear [min] | 0.131715 | 0.010667 |
| 5 | Type | 0.039761 | 0.018872 |
| 6 | Process temperature [K] | 0.012487 | 0.014918 |

Torque and rotational speed were the strongest predictors in this evaluation. Permutation importance indicates predictive usefulness, not causation.

## Notes for submission

- Results are from the reported stratified test split; no threshold optimization was performed.
- Add locally generated EDA and feature-importance figures to the final PDF/report.
- Add screenshots from the successful Streamlit demo.
- Record actual Python/package versions and execution date.
- AI4I 2020 is synthetic; do not claim these metrics establish real-world factory performance.
