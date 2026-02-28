# Good : 0 
# Poor : 1
#Age	Gender	Weight (kg)	Daily Water Intake (liters)	Physical Activity Level	Weather

import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_Gender.pkl")
le_physical_acitivity = joblib.load("label_encoder_Physical Activity Level.pkl")
le_weather = joblib.load("label_encoder_Weather.pkl")
model = joblib.load("best_model.pkl")


st.set_page_config(
    page_title="Insurance Charges Predictor",
    layout="centered"
)

st.title("Health Insurance Charges Prediction App")
st.write("Enter the details below to estimate your insurance charges.")


with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        Weight = st.slider("weight", min_value=10.0, max_value=200.0, value=30.0)
        Water_intake = st.number_input("Water intake in liters", min_value=0, max_value=8, value=0)

    with col2:
        gender = st.selectbox("Gender", options=le_gender.classes_)
        Physical_activity = st.selectbox("Physical Activity level", options=le_physical_acitivity.classes_)
        Weather = st.selectbox("Weather Condition (heat)",options=le_weather.classes_)

    submitted = st.form_submit_button("Predict Hydration Level.")

if submitted:

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Weight (kg)": [Weight],
        "Daily Water Intake (liters)": [Water_intake],
        "Physical Activity Level": [Physical_activity],
        "Weather": [Weather]
    })

    # Encoding
    input_data["Gender"] = le_gender.transform(input_data["Gender"])
    input_data["Physical Activity Level"] = le_physical_acitivity.transform(
        input_data["Physical Activity Level"]
    )
    input_data["Weather"] = le_weather.transform(input_data["Weather"])

   
    num_cols = list(scaler.feature_names_in_)
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    
    input_data = input_data.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("✅ Your Hydration Level is GOOD")
    else:
        st.error("⚠️ Your Hydration Level is POOR")