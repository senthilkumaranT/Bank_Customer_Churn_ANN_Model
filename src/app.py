import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# 🎨 Set Streamlit theme (for more color and wide layout)
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- PATHS -----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# ----------------- CUSTOM CSS -----------------
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        background-image: linear-gradient(315deg, #ffffff 0%, #d7e1ec 74%);
    }

    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .main-header {
        text-align: center; 
        background: -webkit-linear-gradient(#8e44ad, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* Custom classes */
    .card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #6c5ce7;
    }
    .result-card {
        background: #ffeaa7; 
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px dashed #fdcb6e;
    }
    .safe {
        background: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    .danger {
        background: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- LOAD RESOURCES -----------------
@st.cache_resource
def load_all_models():
    try:
        model_loaded = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model.h5"))
        
        with open(os.path.join(MODELS_DIR, 'label_encoder_gender.pkl'), 'rb') as file:
            le_gender = pickle.load(file)
        
        with open(os.path.join(MODELS_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
            ohe_geo = pickle.load(file)
            
        with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as file:
            s_scaler = pickle.load(file)
            
        return model_loaded, le_gender, ohe_geo, s_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_all_models()

# ----------------- NAVIGATION -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144517.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to:", ["🔮 Predict Churn", "🧩 Model Architecture"])
    
    st.markdown("---")
    st.info("Built with Neural Networks & Streamlit")

# ----------------- PAGE: PREDICTION -----------------
if page == "🔮 Predict Churn":
    st.markdown("<h1 class='main-header'>Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d; font-size:1.2em;'>Enter customer details below to predict the likelihood of churning.</p>", unsafe_allow_html=True)

    if model is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.form("churn_prediction_form"):
            st.markdown("### 📝 Customer Profile")
            
            # Layout: 3 columns for better spacing (CRT column fix)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                geography = st.selectbox('🌎 Geography', onehot_encoder_geo.categories_[0])
                gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
                age = st.slider('🎂 Age', 18, 92, 30)
            
            with col2:
                st.markdown("**Financials**")
                credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, value=650)
                balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0)
                estimated_salary = st.number_input('💰 Salary', min_value=0.0, value=50000.0)
                
            with col3:
                st.markdown("**Account Details**")
                tenure = st.slider('⌛ Tenure (Years)', 0, 10, 3)
                num_of_products = st.slider('🛒 Products', 1, 4, 1)
                has_cr_card = st.radio('💳 Credit Card?', ['Yes', 'No'], horizontal=True)
                is_active_member = st.radio('🟢 Active Member?', ['Yes', 'No'], horizontal=True)

            # Map Yes/No
            has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
            is_active_member_val = 1 if is_active_member == 'Yes' else 0
            
            st.markdown("---")
            submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            # Prepare Input
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card_val],
                'IsActiveMember': [is_active_member_val],
                'EstimatedSalary': [estimated_salary]
            })
            
            # Encode Geo
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale
            input_data_scaled = scaler.transform(input_data)
            
            # Predict
            with st.spinner('Thinking...'):
                prediction = model.predict(input_data_scaled, verbose=0)
                prediction_proba = prediction[0][0]
            
            # Display Results
            st.markdown("### 📊 Prediction Results")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
            
            with col_res2:
                if prediction_proba > 0.5:
                    st.markdown(
                        f"""
                        <div class='result-card danger'>
                            <h2>⚠️ High Risk of Churn</h2>
                            <p>This customer has a <b>{prediction_proba:.2%}</b> probability of leaving.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-card safe'>
                            <h2>✅ Low Risk of Churn</h2>
                            <p>This customer is likely to stay (Probability: <b>{prediction_proba:.2%}</b>).</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

# ----------------- PAGE: DIAGRAM -----------------
elif page == "🧩 Model Architecture":
    st.markdown("<h1 class='main-header'>Model Architecture</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Visual representation of the Neural Network and Data Pipeline.</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Simple View", "Detailed View"])
    
    with tab1:
        img_path = os.path.join(IMAGES_DIR, "pipeline_diagram.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="High-Level Pipeline", use_column_width=True)
        else:
            st.warning("Simple diagram not found.")
            
    with tab2:
        img_path_det = os.path.join(IMAGES_DIR, "pipeline_diagram_detailed.png")
        if os.path.exists(img_path_det):
            st.image(img_path_det, caption="Detailed Architecture", use_column_width=True)
        else:
            st.warning("Detailed diagram not found.")

# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top: 50px; color: #b2bec3;">
        <hr>
        <small>Customer Churn Prediction App | v2.0 | Powered by TensorFlow</small>
    </div>
    """,
    unsafe_allow_html=True
)
