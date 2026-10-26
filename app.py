"""Streamlit interface for the Predictive Maintenance AI prototype."""
import streamlit as st
from src.model_service import load_model, predict_machine_failure

st.set_page_config(page_title="Predictive Maintenance AI", page_icon="🛠️", layout="wide")
st.title("Predictive Maintenance AI")
st.caption("Machine failure risk estimation · UCI AI4I 2020 dataset")
st.info("Educational prototype only. Predictions use a synthetic dataset and are not a certified industrial safety system.")

with st.sidebar:
    st.header("About this demo")
    st.write("Enter machine operating conditions to run the fitted classification pipeline.")
    st.caption("Artifact: outputs/predictive_maintenance_pipeline.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The trained model has not been created yet.")
    st.code("python -m src.train_model", language="bash")
    st.write("Download the UCI AI4I CSV to data/ai4i2020.csv first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model artifact: {exc}")
    st.stop()

st.subheader("Machine operating conditions")
with st.form("machine_prediction_form"):
    left, right = st.columns(2)
    with left:
        product_type = st.selectbox("Product type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=298.1, step=0.1)
        process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=308.6, step=0.1)
    with right:
        rpm = st.number_input("Rotational speed (rpm)", min_value=0, max_value=10000, value=1500, step=10)
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=500.0, value=40.0, step=0.5)
        tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=10000, value=100, step=5)
    submitted = st.form_submit_button("Estimate failure risk", type="primary")

if submitted:
    machine = {
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    try:
        result = predict_machine_failure(model, machine)
        st.subheader("Prediction")
        if result["prediction"] == 1:
            st.error("Potential machine failure detected")
        else:
            st.success("No machine failure predicted")
        if result["failure_probability"] is not None:
            probability = result["failure_probability"]
            st.metric("Estimated failure probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption("Model output is not necessarily a calibrated real-world probability.")
        with st.expander("Input summary"):
            st.json(machine)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Interpretation")
st.markdown(
    "- A positive prediction means the classifier predicts the failure class.\n"
    "- A negative prediction does not guarantee that a machine is safe.\n"
    "- Consider false negatives, false positives, and the decision threshold before real-world use."
)
