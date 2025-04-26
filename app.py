import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("best_xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ About this App")
    st.info("""
    This app predicts the risk of heart disease based on patient details using **XGBoost**.

    🔍 **What is XGBoost?**  
    XGBoost (Extreme Gradient Boosting) is a powerful machine learning model that learns from data in stages — like how humans improve by learning from mistakes — to make accurate predictions.
    """)
    st.markdown("---")
    st.caption("🧑‍⚕️ Enter patient values in the form on the right and click 'Predict' to see results.")

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: hotpink;'>💓 Heart Disease Prediction App</h1>", 
    unsafe_allow_html=True
)

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 29, 77, 50)
    sex = st.radio("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina (0)", "Atypical Angina (1)", 
        "Non-anginal Pain (2)", "Asymptomatic (3)"
    ])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
    restecg = st.selectbox("Resting ECG", [
        "Normal (0)", "Abnormality (1)", "Probable or Definite LVH (2)"
    ])

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.radio("Exercise-induced Angina", ["No (0)", "Yes (1)"])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of Peak ST Segment", [
        "Upsloping (0)", "Flat (1)", "Downsloping (2)"
    ])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Type", [
        "Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Other (3)"
    ])

# Convert input fields to numeric
sex = 1 if sex == "Male" else 0
cp = int(cp.split("(")[-1][0])
fbs = int(fbs.split("(")[-1][0])
restecg = int(restecg.split("(")[-1][0])
exang = 1 if exang == "Yes (1)" else 0
slope = int(slope.split("(")[-1][0])
thal = int(thal.split("(")[-1][0])

# Prepare input array
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
features_scaled = scaler.transform(features)

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probability' not in st.session_state:
    st.session_state.probability = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Prediction Section
st.markdown("---")
predict_col1, predict_col2 = st.columns([1, 3])

if predict_col1.button("🔍 Predict"):
    st.session_state.prediction = model.predict(features_scaled)[0]
    st.session_state.probability = model.predict_proba(features_scaled)[0][1]
    
    explainer = shap.Explainer(model)
    st.session_state.shap_values = explainer(features_scaled)

# Show prediction if available
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        predict_col2.error(f"💔 High Risk of Heart Disease Detected! ({st.session_state.probability*100:.1f}%)")
    else:
        predict_col2.success(f"✅ Low Risk – No Heart Disease Detected. ({(1-st.session_state.probability)*100:.1f}%)")

    # SHAP visualization
    if st.checkbox("🔬 Show Model Explanation (SHAP)", value=False):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.plots.waterfall(st.session_state.shap_values[0]))

        # Add a simple explanation for non-tech people
        st.markdown("---")
        st.subheader("🧠 How to Understand the SHAP Chart")
        st.markdown("""
        - Each bar shows how much a specific medical factor influenced the prediction.
        - 🔴 **Red bars** push the prediction toward higher heart disease risk.
        - 🔵 **Blue bars** push the prediction toward lower risk.
        - The longer the bar, the stronger its influence.
        - Top features were the most important in this individual prediction.

        👉 **Example:**  
        - If "Age" has a long red bar, it means the person's age increased their risk.  
        - If "Cholesterol" has a long blue bar, it helped lower the risk prediction.
        """)
