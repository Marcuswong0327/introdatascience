import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd  # Needed to provide column names for scaler input

# Load the trained model and per-feature scalers
model = joblib.load('stacking_classifier_model.pkl')
scalers = joblib.load('scalers.pkl')  # Dict with 'age', 'bmi', 'avg_glucose_level'

# --- Helper functions ---
def predict_stroke(features):
    features_to_scale = ['bmi', 'age', 'avg_glucose_level']
    scaled_features = []

    # Keep categorical features unscaled
    scaled_features.extend([
        features['gender'],
        features['hypertension'],
        features['heart_disease'],
        features['ever_married'],
        features['work_type'],
        features['Residence_type'],
        features['smoking_status']
    ])

    # Apply feature-specific scaling using DataFrame to avoid warnings
    for key in features_to_scale:
        scaler = scalers[key]
        value_df = pd.DataFrame([[features[key]]], columns=[key])
        scaled_value = scaler.transform(value_df)[0][0]
        scaled_features.append(scaled_value)

    # Make prediction
    X_input = pd.DataFrame([scaled_features])
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    return prediction, probability, X_input

def advice_on_values(age, bmi, glucose):
    advice = []
    if age < 0 or age > 120:
        advice.append("❗ Invalid age input, please enter between 0 and 120.")
    if bmi < 10 or bmi > 60:
        advice.append("❗ BMI input seems off, please double-check.")
    elif bmi < 18.5:
        advice.append("⚠️ BMI is low, consider improving nutrition and gaining healthy weight.")
    if glucose < 30 or glucose > 500:
        advice.append("❗ Blood glucose value abnormal, please verify your input.")
    return advice

# --- Streamlit App ---
def main():
    st.title("🧠 Stroke Risk Prediction & Explanation")
    st.write("Fill in your details below to find out your stroke risk and understand the prediction.")

    # User Inputs
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    ever_married = st_
