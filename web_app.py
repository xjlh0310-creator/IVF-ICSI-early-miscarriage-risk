import streamlit as st
import pandas as pd
import pickle
import xgboost
import os
import shap
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration (Clean & Professional)
# ==========================================
st.set_page_config(
    page_title="IVF/ICSI Miscarriage Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, professional layout (No Emojis)
st.markdown("""
    <style>
    /* Global font settings */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
    }
    
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Result Card Styling - Clean & Medical */
    .result-box {
        padding: 15px 20px;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        background-color: #f8f9fa;
    }
    
    .result-title {
        color: #333333;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .result-value {
        font-size: 28px;
        font-weight: 700;
        color: #000000;
        margin-bottom: 5px;
    }
    
    .result-desc {
        color: #666666;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Status Indicators (Color bars instead of emojis) */
    .status-high {
        border-left: 5px solid #d32f2f; /* Red */
    }
    .status-low {
        border-left: 5px solid #388e3c; /* Green */
    }
    .status-mod {
        border-left: 5px solid #fbc02d; /* Yellow */
    }

    /* Reduce padding to make it compact */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Metric styling adjustment */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to display SHAP plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150)

st.markdown("### IVF/ICSI Early Miscarriage Risk Prediction")
st.markdown("XGBoost Clinical Decision Support System")
st.markdown("---")

# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model():
    model_filename = 'xgb_model.pkl'
    if not os.path.exists(model_filename):
        alt_path = "2.训练集构建模型/xgb_model.pkl"
        if os.path.exists(alt_path):
            model_filename = alt_path
        else:
            st.error(f"Error: Model file '{model_filename}' not found.")
            st.stop()
        
    try:
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ==========================================
# 3. Sidebar: Inputs
# ==========================================
with st.sidebar:
    st.markdown("#### Patient Clinical Data")
    
    with st.form("input_form"):
        female_age = st.number_input("Female Age (years)", 20.0, 55.0, 32.0, 1.0)
        bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 22.5, 0.1)
        plt_val = st.number_input("Platelet Count (10⁹/L)", 10.0, 600.0, 250.0, 1.0)
        fsh = st.number_input("Basal FSH (IU/L)", 0.0, 100.0, 7.5, 0.1)
        tsh = st.number_input("TSH (mIU/L)", 0.0, 50.0, 2.0, 0.01)
        
        st.markdown("")
        submitted = st.form_submit_button("Calculate Risk")

# ==========================================
# 4. Main Interface
# ==========================================
if submitted:
    input_data = {
        'Female_age': female_age,
        'BMI': bmi,
        'PLT': plt_val,
        'FSH': fsh,
        'TSH': tsh
    }
    df_input = pd.DataFrame([input_data])

    try:
        # 1. Prediction
        prediction_probs = model.predict_proba(df_input)[0]
        risk_prob = float(prediction_probs[1])
        
        # 2. Layout: Split Results and Metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Determine status style
            if risk_prob > 0.5:
                status_class = "status-high"
                status_text = "High Risk"
                advice = "Close monitoring and evaluation recommended."
            elif risk_prob < 0.2:
                status_class = "status-low"
                status_text = "Low Risk"
                advice = "Routine prenatal care suggested."
            else:
                status_class = "status-mod"
                status_text = "Moderate Risk"
                advice = "Clinical judgment required."

            # Render Result Card
            st.markdown(f"""
            <div class="result-box {status_class}">
                <div class="result-title">Prediction Result: {status_text}</div>
                <div class="result-value">{risk_prob*100:.2f}%</div>
                <div class="result-desc">Probability of Early Miscarriage</div>
                <hr style="margin: 10px 0; border-top: 1px solid #eee;">
                <div class="result-desc"><strong>Suggestion:</strong> {advice}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Metrics
            st.markdown("""<div style="padding: 5px;"></div>""", unsafe_allow_html=True) # Spacer
            c1, c2 = st.columns(2)
            c1.metric("Miscarriage Prob.", f"{risk_prob:.2%}")
            c2.metric("Live Birth Prob.", f"{1-risk_prob:.2%}")
            
            # Input summary (Compact)
            with st.expander("Input Data Summary"):
                st.dataframe(df_input, hide_index=True)

        # 3. SHAP Force Plot
        st.markdown("#### Individualized Feature Interpretation (SHAP Force Plot)")
        st.caption("This plot shows how each feature contributes to pushing the risk higher (red) or lower (blue) from the baseline.")
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        # Display Force Plot
        # Note: XGBoost classifiers output log-odds, link="logit" converts it to probability for visualization
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0], df_input.iloc[0], link="logit"), height=120)

    except Exception as e:
        st.error(f"Calculation Error: {str(e)}")
else:
    st.info("Enter clinical data in the sidebar and click 'Calculate Risk'.")

st.markdown("---")
st.caption("Disclaimer: For research use only.")