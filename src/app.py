import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# 🎨 Set Streamlit theme
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

# ----------------- CUSTOM CSS (GLASSMORPHISM) -----------------
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background: Dark Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Text Colors */
    h1, h2, h3, p, label, .stMarkdown {
        color: #ffffff !important;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 25px;
        margin-bottom: 25px;
    }

    /* Inputs */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stSlider > div > div > div {
        color: white; 
    }
    
    /* Result Cards */
    .result-safe {
        background: rgba(46, 204, 113, 0.2);
        border: 1px solid #2ecc71;
        color: #2ecc71;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .result-danger {
        background: rgba(231, 76, 60, 0.2);
        border: 1px solid #e74c3c;
        color: #e74c3c;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.2);
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
    st.markdown("<h2 style='text-align: center;'>🔮 Menu</h2>", unsafe_allow_html=True)
    page = st.radio("", ["Predict Churn", "Model Architecture"])
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: rgba(255,255,255,0.7);'>
            <small>Deep Learning Powered</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ----------------- PAGE: PREDICTION -----------------
if page == "Predict Churn":
    st.markdown("<h1 style='text-align: center; margin-bottom: 20px;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

    if model is not None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        with st.form("churn_prediction_form"):
            st.markdown("### 📝 Enter Customer Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Demographics")
                geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
                gender = st.selectbox('Gender', label_encoder_gender.classes_)
                age = st.slider('Age', 18, 92, 30)
            
            with col2:
                st.markdown("#### 💰 Financials")
                credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=650)
                balance = st.number_input('Balance', min_value=0.0, value=10000.0)
                estimated_salary = st.number_input('Salary', min_value=0.0, value=50000.0)
                
            with col3:
                st.markdown("#### 🏦 Account")
                tenure = st.slider('Tenure (Years)', 0, 10, 3)
                num_of_products = st.slider('Products', 1, 4, 1)
                has_cr_card = st.radio('Credit Card?', ['Yes', 'No'], horizontal=True)
                is_active_member = st.radio('Active Member?', ['Yes', 'No'], horizontal=True)

            has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
            is_active_member_val = 1 if is_active_member == 'Yes' else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Analyze Now", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            # Data Processing
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
            
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            input_data_scaled = scaler.transform(input_data)
            
            # Prediction
            with st.spinner('Running Neural Network...'):
                prediction = model.predict(input_data_scaled, verbose=0)
                prediction_proba = prediction[0][0]
            
            # Results
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Analysis Results")
            
            c_res1, c_res2 = st.columns([1, 2])
            
            with c_res1:
                st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
            
            with c_res2:
                if prediction_proba > 0.5:
                    st.markdown(
                        f"""
                        <div class='result-danger'>
                            <h2>⚠️ High Risk</h2>
                            <p>This customer is likely to churn ({prediction_proba:.2%}).</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-safe'>
                            <h2>✅ Low Risk</h2>
                            <p>This customer is likely to stay ({prediction_proba:.2%}).</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE: MODEL ARCHITECTURE -----------------
elif page == "Model Architecture":
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🧠 Model Architecture</h1>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='glass-card'>
            <p>Below is the detailed visual representation of the ANN model architecture.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Simplified view: Single detailed image
    img_path_det = os.path.join(IMAGES_DIR, "pipeline_diagram_detailed.png")
    if os.path.exists(img_path_det):
        st.image(img_path_det, caption="Detailed Neural Network Architecture", use_column_width=True)
    else:
        st.error(f"Image not found at: {img_path_det}")
