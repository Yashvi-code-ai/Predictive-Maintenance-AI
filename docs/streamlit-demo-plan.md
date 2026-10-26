# Streamlit demo plan

## Goal
A locally runnable educational demo predicting the binary Machine failure target from six permitted AI4I features. It is not an industrial safety system.

## Member 1 — Yashvi (UI)
Branch: feature/streamlit-ui
- Own app.py: page layout, six machine inputs, prediction display, probability display, input summary, and limitation messaging.
- Use src.model_service rather than duplicating preprocessing or training.
- Test the UI after the model-serving branch is merged.

## Member 2 — Vishesh (model serving)
Branch: feature/model-serving
- Add src/train_model.py to fit and persist the full preprocessing + classifier pipeline.
- Add src/model_service.py to load the artifact and validate the six input features.
- Add tests for the prediction interface.
- Exclude UDI, Product ID and failure-mode flags TWF, HDF, PWF, OSF, RNF.

## Shared interface
Artifact: outputs/predictive_maintenance_pipeline.joblib
Training command: python -m src.train_model
Inference function: predict_machine_failure(model, machine_input)
Inputs: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
Output keys: prediction (0/1), failure_probability (float or null), label.

## Integration order
1. Merge model-serving PR first.
2. Merge UI PR second.
3. Download the CSV to data/ai4i2020.csv locally.
4. Install requirements, run unit tests, train model, then run streamlit run app.py.
5. Test predictions, missing-artifact behavior, and edge inputs.
6. Confirm the final model choice matches the notebook's measured comparison before submission.

## Commands
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train_model
streamlit run app.py

## Acceptance checklist
- Do not commit raw CSV or trained binary artifact.
- Persist preprocessing inside the fitted pipeline.
- Exclude target leakage flags and identifiers.
- Clearly state limitations; do not claim certainty.
- Run tests locally before merging.
