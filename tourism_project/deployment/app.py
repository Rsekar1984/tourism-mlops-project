import os, joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, login

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_REPO = "rknv1984/tourism-project-model"
MODEL_FILE = "best-tourism-model-v1.joblib"

@st.cache_resource
def load_model():
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                                       token=HF_TOKEN or None))

model = load_model()
st.set_page_config(page_title="Tourism Package Predictor", page_icon="🌴")
st.title("🌴 Tourism Package Purchase Predictor")
st.markdown("Fill in the customer details to predict package purchase likelihood.")

col1, col2 = st.columns(2)
with col1:
    age             = st.number_input("Age", 18, 90, 35)
    duration        = st.number_input("Duration of Pitch (mins)", 1, 60, 15)
    num_persons     = st.number_input("Number of Persons Visiting", 1, 10, 2)
    num_children    = st.number_input("Number of Children", 0, 10, 0)
    monthly_income  = st.number_input("Monthly Income (₹)", 5000, 100000, 40000)
    pitch_score     = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car         = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    passport        = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col2:
    gender          = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x else "Female")
    marital_status  = st.selectbox("Marital Status", [0, 1, 2],
                                   format_func=lambda x: ["Unmarried","Married","Divorced"][x])
    education       = st.selectbox("Designation", [0, 1, 2, 3],
                                   format_func=lambda x: ["Executive","Manager","Senior Manager","AVP"][x])
    occupation      = st.selectbox("Occupation", [0, 1, 2, 3],
                                   format_func=lambda x: ["Salaried","Self-Employed","Free Lancer","Small Business"][x])
    product_pitched = st.selectbox("Product Pitched", [0,1,2,3,4],
                                   format_func=lambda x: ["Basic","Deluxe","Super Deluxe","King","Multi"][x])
    num_followups   = st.number_input("Number of Follow-ups", 1, 6, 3)
    hotel_type      = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    num_trips       = st.number_input("Number of Trips (annual)", 0, 22, 2)

if st.button("🔍 Predict Purchase", use_container_width=True):
    features = pd.DataFrame([[
        age, duration, num_persons, num_children, monthly_income,
        pitch_score, own_car, passport, gender, marital_status,
        education, occupation, product_pitched, num_followups,
        hotel_type, num_trips, 1, 2
    ]], columns=[
        "Age","DurationOfPitch","NumberOfPersonVisiting","NumberOfChildrenVisiting",
        "MonthlyIncome","PitchSatisfactionScore","OwnCar","Passport",
        "Gender","MaritalStatus","Designation","Occupation","ProductPitched",
        "NumberOfFollowups","PreferredPropertyStar","NumberOfTrips",
        "TypeofContact","CityTier"
    ])
    proba = model.predict_proba(features)[0][1]
    pred  = int(proba >= 0.45)
    if pred == 1:
        st.success(f"✅ Likely to Purchase — Confidence: {proba*100:.1f}%")
        st.balloons()
    else:
        st.error(f"❌ Unlikely to Purchase — Confidence: {(1-proba)*100:.1f}%")
