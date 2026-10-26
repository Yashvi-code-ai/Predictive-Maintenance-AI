# Capstone Completion and Viva Checklist

## Required workflow evidence
- [ ] Problem statement and measurable objectives written.
- [ ] Dataset source, schema, target, and feature meanings documented.
- [ ] EDA class-balance plot and numeric correlation plot generated and discussed.
- [ ] Duplicate, missing-value, target-validity, and outlier checks documented.
- [ ] Identifier and failure-mode leakage columns excluded.
- [ ] Stratified 80/20 split and fixed seed documented.
- [ ] At least three classifiers trained on the same split and preprocessing strategy.
- [ ] Accuracy, precision, recall, F1, ROC-AUC, and confusion matrices reported where supported.
- [ ] Final model justified from measured results, not assumed in advance.
- [ ] Permutation importance interpreted without causal claims.
- [ ] Streamlit app demonstrated with more than one input example.
- [ ] Limitations and future scope discussed.
- [ ] Both members can explain all project stages and answer questions.
- [ ] Report values populated from real outputs; no invented metrics.
- [ ] Screenshots captured from actual local execution.

## Commands from repository root
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.preprocessing
python -m src.eda
python -m src.baseline_models
python -m src.random_forest
python -m src.model_comparison
python -m src.train_model
streamlit run app.py
```

All dataset-dependent commands require `data/ai4i2020.csv`. Download it from the UCI AI4I 2020 Predictive Maintenance Dataset page. Do not commit the raw CSV or generated artifacts.

## Viva questions both members should prepare for
1. Why is this a binary classification task rather than regression?
2. What does each input feature measure?
3. Why is the target class imbalanced, and why can accuracy be misleading?
4. Explain precision, recall, F1-score, ROC-AUC, and each confusion-matrix cell.
5. What is target leakage? Why are TWF, HDF, PWF, OSF, and RNF excluded?
6. Why are UDI and Product ID excluded?
7. Why are preprocessing steps fitted inside a pipeline after splitting?
8. Why report outliers instead of automatically deleting them?
9. How do the candidate classifiers differ?
10. How was the final model selected from the measured results?
11. What does permutation importance mean, and what does it not prove?
12. Why may the displayed failure probability not be calibrated?
13. What are the limitations of using a synthetic dataset?
14. What would be required before using the model in an actual factory?
15. Which files and tasks did each team member personally contribute?

## Demo sequence
1. Explain the problem and dataset.
2. Show the target distribution and describe imbalance.
3. Show the preprocessing and leakage exclusions.
4. Show the model-comparison table and confusion matrices.
5. Show permutation importance and explain the top features.
6. Run Streamlit with two different sets of operating conditions.
7. Finish with limitations and future scope.
