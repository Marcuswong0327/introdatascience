import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler dictionary
model = joblib.load('stacking_classifier_model.pkl')
scalers = joblib.load('scalers.pkl')  # Expected to be a dict with keys like 'bmi', 'age', etc.

# Define prediction function
def predict_stroke(features):
    features_to_scale = ['bmi', 'age', 'avg_glucose_level']
    scaled_features = []

    # No scaling for first four features: gender, hypertension, heart_disease, smoking
    scaled_features.extend([features[key] for key in ['gender', 'hypertension','heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']])

    # Apply scaling for needed features
    for key in features_to_scale:
        scaler = scalers[key]
        scaled_value = scaler.transform(np.array([[features[key]]]))[0][0]
        scaled_features.append(scaled_value)

    prediction = model.predict([scaled_features])[0]
    return prediction

# Streamlit UI
def main():
    st.title("🩺 Stroke Prediction App")
    st.write("Fill in your details below to find out your risk of stroke.")

    # Input widgets
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    ever_married = st.selectbox("Ever Married?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    work_type_dict = {
    0: "children",
    1: "Govt_job",
    2: "Private",
    3: "Self-employed",
    4: "Never_worked"
}

    work_type = st.selectbox(
    "Work Type?",
    options=list(work_type_dict.keys()),
    format_func=lambda x: work_type_dict[x]
)

    Residence = st.selectbox("Where you live?", [0, 1], format_func=lambda x: 'Urban' if x == 1 else 'Rural')
    smoking_status = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: 'smokes' if x == 1 else 'never smoked')

    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)

    # Convert gender to numeric
    gender_numeric = 1 if gender == 'Female' else 0

   # Collect features into a dict
    features = {
    'gender': gender_numeric,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'Residence': Residence,
    'smoking_status': smoking_status,
    'bmi': bmi,
    'age': age,
    'avg_glucose_level': glucose
}


    if st.button("Predict"):
        result = predict_stroke(features)
        if result == 1:
            st.error("🔴 You may have Stroke.")
        else:
            st.success("🟢 You are not predicted to have Stroke.")

if __name__ == '__main__':
    main()
