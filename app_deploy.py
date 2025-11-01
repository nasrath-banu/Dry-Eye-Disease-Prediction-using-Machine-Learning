import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===========================
# 🌟 PAGE CONFIG
# ===========================
st.set_page_config(page_title="💧 Dry Eye Predictor (Calibrated Models)", layout="centered")

# Apply custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #c3cfe2 0%, #f5f7fa 100%);
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        height: 3rem;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        transform: scale(1.03);
    }
    h1,h2,h3,h4 {
        text-align: center;
        color: #002B5B;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 📦 LOAD CALIBRATED MODELS + SCALER
# ===========================
@st.cache_resource
def load_artifacts():
    models = {
        "Random Forest": joblib.load("dryeye_rf_model_calibrated.joblib"),
        "SVM": joblib.load("dryeye_svm_model_calibrated.joblib"),
        "LightGBM": joblib.load("dryeye_lgbm_model_calibrated.joblib"),
        "XGBoost": joblib.load("dryeye_xgb_model_calibrated.joblib"),
    }
    scaler = joblib.load("dryeye_scaler.joblib")
    features = joblib.load("dryeye_features.joblib")
    return models, scaler, features

models, scaler, model_features = load_artifacts()

# ===========================
# ⚙️ HELPER FUNCTIONS
# ===========================
def preprocess_input(df):
    for c in model_features:
        if c not in df.columns:
            df[c] = 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def align_and_scale(df):
    aligned = pd.DataFrame(columns=model_features)
    for c in model_features:
        aligned[c] = df[c] if c in df.columns else 0
    aligned = aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaled = scaler.transform(aligned)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, aligned

def predict_all_models(scaled):
    results = {}
    for name, model in models.items():
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0, 1]
        results[name] = (pred, prob)
    return results

# ===========================
# 🧠 APP CONTENT
# ===========================
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("💧 Dry Eye Disease Predictor (Calibrated Models)")
st.markdown("""
This version uses **calibrated machine learning models** — tuned for **more realistic probability outputs**.
Compare how each model interprets your input data.
""")

st.divider()

with st.form("prediction_form"):
    st.subheader("👁️ Enter Your Details")

    col1, col2 = st.columns(2)
    with col1:
        eye_strain = st.selectbox("Discomfort / Eye-strain", [1, 0], format_func=lambda x: "Yes" if x else "No")
        redness = st.selectbox("Redness in eye", [1, 0], format_func=lambda x: "Yes" if x else "No")
        screen_time = st.number_input("Average screen time (hours/day)", 0.0, 24.0, 6.0, 0.5)
        physical_activity = st.number_input("Physical activity (minutes/day)", 0.0, 600.0, 30.0, 5.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", 60.0, 250.0, 120.0, 1.0)
    with col2:
        itchiness = st.selectbox("Itchiness / Irritation", [1, 0], format_func=lambda x: "Yes" if x else "No")
        sleep_duration = st.number_input("Sleep duration (hours)", 0.0, 24.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep quality (1 - Poor → 5 - Excellent)", 1, 5, 3)
        ongoing_med = st.selectbox("Ongoing medication", [1, 0], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    data = {
        'Discomfort Eye-strain': eye_strain,
        'Itchiness/Irritation in eye': itchiness,
        'Redness in eye': redness,
        'Average screen time': screen_time,
        'Sleep duration': sleep_duration,
        'Sleep quality': sleep_quality,
        'Physical activity': physical_activity,
        'Systolic_BP': systolic_bp,
        'Ongoing medication': ongoing_med
    }

    df = pd.DataFrame([data])
    df = preprocess_input(df)
    scaled, aligned = align_and_scale(df)
    results = predict_all_models(scaled)

    st.divider()
    st.subheader("📊 Model Comparison Results (Calibrated)")

    summary = []
    for name, (pred, prob) in results.items():
        status = "⚠️ High Risk" if pred == 1 else "✅ Low Risk"
        summary.append({"Model": name, "Prediction": status, "Probability (%)": round(prob*100, 2)})

    results_df = pd.DataFrame(summary)
    st.dataframe(results_df, hide_index=True)

    # --- Combined Visualization ---
    fig, ax = plt.subplots(figsize=(6,4))
    colors = ['#dc3545' if "High" in p else '#28a745' for p in results_df["Prediction"]]
    ax.bar(results_df["Model"], results_df["Probability (%)"], color=colors)
    ax.set_ylabel("Predicted Probability (%)")
    ax.set_title("Model-wise Predicted Risk Probability (Calibrated)")
    st.pyplot(fig)

    # --- Explanation (same as before) ---
    st.divider()
    st.subheader("🧠 Explanation of Prediction")

    explanation = []

    if eye_strain == 1: 
        explanation.append("• Presence of **eye strain** indicates discomfort in focusing, a key sign of dryness.")
    if redness == 1:
        explanation.append("• **Redness** in eyes shows irritation and inflammation — common in Dry Eye Disease.")
    if itchiness == 1:
        explanation.append("• **Itchiness/Irritation** suggests tear film instability, increasing risk.")

    if screen_time > 6:
        explanation.append(f"• **High screen time ({screen_time} hrs/day)** leads to reduced blinking and poor lubrication.")
    elif screen_time < 3:
        explanation.append(f"• **Low screen time ({screen_time} hrs/day)** helps maintain normal tear moisture.")

    if sleep_duration < 6:
        explanation.append(f"• **Short sleep duration ({sleep_duration} hrs)** affects tear regeneration overnight.")
    elif sleep_duration >= 7:
        explanation.append(f"• **Good sleep duration ({sleep_duration} hrs)** supports healthy eye surface recovery.")

    if sleep_quality <= 2:
        explanation.append("• **Poor sleep quality** affects tear secretion and increases dryness risk.")
    elif sleep_quality >= 4:
        explanation.append("• **Good sleep quality** helps maintain tear stability.")

    if physical_activity < 20:
        explanation.append(f"• **Low physical activity ({physical_activity} mins/day)** may reduce circulation and eye hydration.")
    elif physical_activity >= 40:
        explanation.append(f"• **Active lifestyle ({physical_activity} mins/day)** improves eye health and reduces risk.")

    if systolic_bp > 135:
        explanation.append(f"• **Higher BP ({systolic_bp} mmHg)** may slightly affect eye blood flow, contributing to dryness.")
    elif systolic_bp < 130:
        explanation.append(f"• **Normal BP ({systolic_bp} mmHg)** helps maintain healthy ocular pressure.")

    if ongoing_med == 1:
        explanation.append("• **Ongoing medication** use can include drugs that cause eye dryness as a side effect.")
    else:
        explanation.append("• No ongoing medication — reduces external dryness risk.")

    st.write("### 🔍 Key Factors Considered:")
    for line in explanation:
        st.markdown(line)

    with st.expander("📋 View Processed Input Data"):
        st.dataframe(aligned)

st.markdown("</div>", unsafe_allow_html=True)
