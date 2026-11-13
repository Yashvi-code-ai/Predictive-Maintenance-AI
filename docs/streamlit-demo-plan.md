# Streamlit demo plan

## Runtime contract

- **Interface:** `app.py`
- **Inference:** `src/model_service.py`
- **Training entry point:** `train_model.py` (repository root)
- **Saved pipeline:** `outputs/predictive_maintenance_pipeline.joblib` (generated locally; not committed)
- **Dataset:** `data/ai4i2020.csv` (downloaded locally; not committed)

The training script, inference service, and Streamlit app all resolve the model and dataset paths relative to the repository. Start commands from the repository root:

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The dataset must exist before training. If the model artifact is missing, the app displays the exact training command.

## Demo flow

1. Start the Streamlit app.
2. Enter product type and sensor values.
3. Submit the machine operating conditions.
4. Review the predicted class and model probability, if available.
5. Explain false negatives, false positives, class imbalance, and the prototype's limitations.

The prediction probability is a model score and should not be described as a calibrated real-world risk.
