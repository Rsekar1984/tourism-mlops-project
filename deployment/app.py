import os, joblib, requests, numpy as np, pandas as pd, streamlit as st
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Model Hub
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="rknv1984/tourism-project-model",
        filename="best-tourism-model-v1.joblib"
    )
    return joblib.load(path)

model = load_model()

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🏖️")
st.title("🏖️ Tourism Package Purchase Predictor")
st.markdown("Fill in customer details to predict whether they will purchase a travel package.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age               = st.slider("Age", 18, 80, 35)
        monthly_income    = st.number_input("Monthly Income", 5000, 100000, 25000, step=1000)
        duration_pitch    = st.slider("Duration of Pitch (mins)", 1, 60, 15)
        num_trips         = st.slider("Number of Trips", 0, 22, 3)
        num_persons       = st.slider("Persons Visiting", 1, 6, 2)
        num_children      = st.slider("Children Visiting", 0, 5, 0)
        num_followups     = st.slider("Follow-ups", 1, 6, 3)
        pitch_score       = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    with col2:
        city_tier         = st.selectbox("City Tier", [1, 2, 3])
        occupation        = st.selectbox("Occupation", [0, 1, 2, 3])  # encoded
        type_contact      = st.selectbox("Type of Contact", [0, 1])
        gender            = st.selectbox("Gender", [0, 1])
        product_pitched   = st.selectbox("Product Pitched", [0, 1, 2, 3, 4])
        preferred_star    = st.selectbox("Preferred Property Star", [3, 4, 5])
        marital_status    = st.selectbox("Marital Status", [0, 1, 2])
        passport          = st.selectbox("Passport (0=No, 1=Yes)", [0, 1])
        own_car           = st.selectbox("Own Car (0=No, 1=Yes)", [0, 1])
        designation       = st.selectbox("Designation", [0, 1, 2, 3, 4])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age, "TypeofContact": type_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_pitch, "Occupation": occupation,
        "Gender": gender, "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups, "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star, "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children, "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])
    prob  = model.predict_proba(input_data)[0, 1]
    label = "✅ Likely to Purchase" if prob >= 0.45 else "❌ Unlikely to Purchase"
    st.subheader(f"Prediction: {label}")
    st.metric("Purchase Probability", f"{prob:.1%}")
    st.progress(float(prob))