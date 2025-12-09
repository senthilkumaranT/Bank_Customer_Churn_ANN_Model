import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# 🎨 Set Streamlit theme (for more color)
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="auto"
)
# Background with gradient using markdown (limited colors in Streamlit natively)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8ffae 0%, #43cea2 100%);
    }
    .highlight {
        padding: 0.5em 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .info-hl { background: #d0f9ff; }
    .warn-hl { background: #ffcccc; }
    .succ-hl { background: #dcffe4; }
    .predict-prob {
        font-size: 2em;
        font-weight: bold;
        color: #fa8231;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "model.h5"))

# Load the encoders and scaler
with open(os.path.join(MODELS_DIR, 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(MODELS_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#3b6978; font-size:2.5em; margin-bottom:0.2em;'>🔮 Customer Churn Prediction 🔮</h1>
        <p style='color:#204051; font-size:1.2em;'>
            Empower your business with colorful, instant predictions!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add a colored horizontal rule for aesthetics
st.markdown("<hr style='border-top: 2px dotted #00b894;'>", unsafe_allow_html=True)

with st.form("churn_prediction_form"):
    st.markdown("<div class='highlight info-hl'>Please fill out the customer details:</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        geography = st.selectbox('🌎 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, 30)
        credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, value=650)
        tenure = st.slider('⌛ Tenure (years)', 0, 10, 3)
    with c2:
        balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0)
        estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0)
        num_of_products = st.slider('🛒 Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'])
        is_active_member = st.selectbox('🟢 Is Active Member', ['No', 'Yes'])

    # Map Yes/No to 1/0
    has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
    is_active_member_val = 1 if is_active_member == 'Yes' else 0
    
    # Rainbow button!
    submitted = st.form_submit_button(
        "🌈 Predict Churn 🌈",
        use_container_width=True
    )

if submitted:
    # Prepare input data
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
    
    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict churn
    with st.spinner('✨ Calculating your colorful prediction...'):
        prediction = model.predict(input_data_scaled, verbose=0)
        prediction_proba = prediction[0][0]
    
    # Show result with custom coloring and emoji
    st.markdown(
        "<div class='highlight succ-hl'>🔥 <b>Prediction Complete!</b> 🔥</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='predict-prob'>Churn Probability: <span style='color:#0984e3'>{prediction_proba:.2%}</span></div>",
        unsafe_allow_html=True
    )
        
    if prediction_proba > 0.5:
        st.markdown(
            "<div class='highlight warn-hl'>"
            "⚠️ <span style='color:#d63031; font-weight:bold;'>The customer is <u>likely</u> to churn.</span> "
            "Take action now! 🚨"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='highlight succ-hl'>"
            "✅ <span style='color:#00b894; font-weight:bold;'>The customer is <u>not likely</u> to churn.</span> "
            "Keep engaging! 🎉"
            "</div>",
            unsafe_allow_html=True
        )
# End with a nice colorful footer
st.markdown(
    """
    <hr style='border-top:2px solid #fdcb6e;'>
    <div style="text-align:center;font-size:1em;color:#636e72;">
        Made with <span style="color:#fd79a8;">&#10084;</span> and <span style="color:#00b894;">Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True
)
