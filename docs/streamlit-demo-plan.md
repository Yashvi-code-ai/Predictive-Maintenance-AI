# Streamlit Demo — Implementation Notes

## Purpose
The app is an educational prototype that predicts the binary `Machine failure` target from six permitted AI4I features. It is not a certified industrial safety system.

## Runtime contract
- **UI:** `app.py`
- **Inference:** `src/model_service.py`
- **Training:** `src/train_model.py`
- **Saved pipeline:** `outputs/predictive_maintenance_pipeline.joblib` (generated locally; not committed)
- **Input features:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Inference function:** `predict_machine_failure(model, machine_input)`
- **Output keys:** `prediction` (0/1), `failure_probability` (float or null), `label`

The persisted artifact contains the fitted preprocessing pipeline and classifier. Training and inference must use the same feature contract. The input does not include identifiers (`UDI`, `Product ID`) or failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

## Run locally
Run commands from the repository root. Download the UCI AI4I CSV and save it as `data/ai4i2020.csv`.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train_model
streamlit run app.py
```

## Manual acceptance checks
- App shows a useful message when the CSV or model artifact is missing.
- A submitted form returns a class label and probability when the model supports probabilities.
- Inputs shown in the summary match the values submitted.
- Probability is described as a model estimate, not necessarily a calibrated real-world risk.
- The UI says that a negative prediction does not guarantee machine safety.
- Before final submission, align the model trained for the app with the model selected from measured comparison results.

## Team ownership
- **Yashvi:** Streamlit interface, usability, and demo screenshots.
- **Vishesh:** training/inference service, evaluation, and tests.
- **Both:** understand preprocessing, leakage prevention, metrics, and limitations end-to-end.
