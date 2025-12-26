import streamlit as st
import pandas as pd
import pickle
import xgboost
import os

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="IVF/ICSI Miscarriage Risk Prediction",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautification
st.markdown("""
    <style>
    /* Main background color */
    .main {
        background-color: #f8f9fa;
    }
    /* High Risk Card (Red) */
    .result-card-high-risk {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
        border-left: 5px solid #FF5252;
    }
    /* Low Risk Card (Green) */
    .result-card-low-risk {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
        border-left: 5px solid #4CAF50;
    }
    /* Moderate Risk Card (Yellow) */
    .result-card-moderate {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
        border-left: 5px solid #FFC107;
    }
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚠️ IVF/ICSI Early Miscarriage Risk Prediction")
st.markdown("### XGBoost-based Clinical Decision Support System")
st.info("ℹ️ **Note:** This tool predicts the **risk of early miscarriage**. A higher probability indicates a higher risk.")
st.markdown("---")

# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model():
    # Priority: Root directory
    model_filename = 'xgb_model.pkl'
    
    # Check if model exists in root, otherwise check subfolder (for compatibility)
    if not os.path.exists(model_filename):
        alt_path = "2.训练集构建模型/xgb_model.pkl"
        if os.path.exists(alt_path):
            model_filename = alt_path
        else:
            st.error(f"❌ Critical Error: Model file `{model_filename}` not found. Please upload it to the GitHub repository.")
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
# 3. Sidebar: Patient Data Entry
# ==========================================
with st.sidebar:
    st.header("📝 Patient Clinical Data")
    st.markdown("Please enter the following indicators:")
    
    with st.form("input_form"):
        # 1. Female_age
        female_age = st.number_input(
            "1. Female Age (years)", 
            min_value=20.0, max_value=55.0, value=32.0, step=1.0
        )
        
        # 2. BMI
        bmi = st.number_input(
            "2. BMI (kg/m²)", 
            min_value=10.0, max_value=50.0, value=22.5, step=0.1,
            help="Body Mass Index"
        )
        
        # 3. PLT
        plt_val = st.number_input(
            "3. Platelet Count (10⁹/L)", 
            min_value=10.0, max_value=600.0, value=250.0, step=1.0,
            help="PLT"
        )
        
        # 4. FSH
        fsh = st.number_input(
            "4. Basal FSH (IU/L)", 
            min_value=0.0, max_value=100.0, value=7.5, step=0.1,
            help="Follicle-Stimulating Hormone"
        )
        
        # 5. TSH
        tsh = st.number_input(
            "5. TSH (mIU/L)", 
            min_value=0.0, max_value=50.0, value=2.0, step=0.01,
            help="Thyroid Stimulating Hormone"
        )
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Calculate Risk", use_container_width=True)

# ==========================================
# 4. Main Interface: Prediction Logic
# ==========================================
if submitted:
    # Construct DataFrame (Columns must match training data exactly)
    input_data = {
        'Female_age': female_age,
        'BMI': bmi,
        'PLT': plt_val,
        'FSH': fsh,
        'TSH': tsh
    }
    df_input = pd.DataFrame([input_data])

    # Display Input Data
    with st.expander("📋 View Input Data", expanded=True):
        st.dataframe(df_input, use_container_width=True)

    try:
        # Prediction
        # predict_proba returns [[prob_class0, prob_class1]]
        # Assuming Class 1 = Early Miscarriage (Risk Event)
        prediction_probs = model.predict_proba(df_input)[0]
        
        # FIX: Explicit float conversion to prevent Streamlit error
        risk_prob = float(prediction_probs[1]) 
        
        # Layout for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Risk Analysis Result")
            
            # Display Progress Bar
            st.write("**Estimated Probability of Early Miscarriage:**")
            st.progress(risk_prob)
            
            # Conditional Logic: High probability = High Risk (Bad)
            if risk_prob > 0.5:
                card_class = "result-card-high-risk"
                icon = "⚠️"
                status = "High Risk"
                advice = "The model predicts a high risk of early miscarriage. Close monitoring and further clinical evaluation are strongly recommended."
                # No balloons for high risk
            elif risk_prob < 0.2:
                card_class = "result-card-low-risk"
                icon = "✅"
                status = "Low Risk"
                advice = "The predicted risk is lower than average. Routine prenatal care is suggested."
                st.balloons() # Balloons for low risk (Good news)
            else:
                card_class = "result-card-moderate"
                icon = "⚖️"
                status = "Moderate Risk"
                advice = "The risk is within an intermediate range. Clinical judgment should be combined with individual patient history."

            # HTML Card for Result
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{icon} Prediction: {status}</h3>
                <h1 style="color: #333; margin: 0;">{risk_prob*100:.2f}%</h1>
                <p style="color: #666; margin-top: 10px;">Probability of Early Miscarriage</p>
                <hr>
                <p><strong>💡 Clinical Suggestion:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Key Metrics")
            st.metric(label="Miscarriage Probability", value=f"{risk_prob:.2%}", delta_color="inverse")
            st.metric(label="Live Birth Probability", value=f"{1-risk_prob:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Possible causes: Input data format issue or model version mismatch.")

else:
    # Empty state
    st.info("👈 Please enter patient data in the sidebar and click 'Calculate Risk' to start.")

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool uses a machine learning model (XGBoost) to estimate the risk of early miscarriage in IVF/ICSI patients. It is for research use only and does not replace professional medical diagnosis.")