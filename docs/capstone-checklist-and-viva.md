# Capstone checklist and viva preparation

## Before the demonstration

- [ ] Download the UCI AI4I 2020 CSV to `data/ai4i2020.csv`.
- [ ] Install packages from `requirements.txt`.
- [ ] Run preprocessing and EDA.
- [ ] Run baseline models, Random Forest evaluation, and the six-model comparison.
- [ ] Train the saved pipeline using `python train_model.py`.
- [ ] Launch the interface using `streamlit run app.py`.
- [ ] Capture screenshots and record actual metric values for the report.

## Reproducible commands

Run these from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.preprocessing
python -m src.eda
python -m src.baseline_models
python -m src.random_forest
python -m src.model_comparison
python train_model.py
streamlit run app.py
```

The Python scripts also support direct execution, for example `python src/model_comparison.py`. Dataset and generated-output paths are resolved relative to the repository rather than the terminal's current directory.

All dataset-dependent commands require `data/ai4i2020.csv`. Download it from the UCI AI4I 2020 Predictive Maintenance Dataset page. Do not commit the raw CSV or generated artifacts.

## Viva topics

Be prepared to explain the target variable, excluded identifier and leakage columns, train/test split, preprocessing pipeline, class imbalance, precision, recall, F1-score, ROC-AUC, confusion matrix, false negatives, false positives, and why AI4I results do not guarantee real-world industrial performance.
