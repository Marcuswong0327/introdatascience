import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load the trained model and per-feature scalers
model = joblib.load("stacking_classifier_model.pkl")
scalers = joblib.load("scalers.pkl")  # Dict with 'age', 'bmi', 'avg_glucose_level'

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

    # Apply feature-specific scaling
    for key in features_to_scale:
        scaler = scalers[key]
        scaled_value = scaler.transform(np.array([[features[key]]]))[0][0]
        scaled_features.append(scaled_value)

    prediction = model.predict([scaled_features])[0]
    probability = model.predict_proba([scaled_features])[0][1]
    return prediction, probability, np.array([scaled_features])

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
    ever_married = st.selectbox("Ever Married?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    work_type_dict = {
        0: "children", 1: "Govt_job", 2: "Private", 3: "Self-employed", 4: "Never_worked"
    }
    work_type = st.selectbox("Work Type", options=list(work_type_dict.keys()), format_func=lambda x: work_type_dict[x])

    residence = st.selectbox("Residence Type", [0, 1], format_func=lambda x: 'Urban' if x == 1 else 'Rural')
    smoking_status = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: 'smokes/formerly smoked' if x == 1 else 'never smoked')

    age = st.number_input("Age", min_value=0, step=1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)

    # Convert gender to numeric
    gender_numeric = 1 if gender == 'Female' else 0

    # Collect features
    features = {
        'gender': gender_numeric,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence,
        'smoking_status': smoking_status,
        'bmi': bmi,
        'age': age,
        'avg_glucose_level': glucose
    }

    if st.button("🔍 Predict Stroke Risk"):
        # Advice section
        for adv in advice_on_values(age, bmi, glucose):
            st.info(adv)

        # Prediction
        prediction, prob, feature_array = predict_stroke(features)

        st.subheader("📊 Stroke Risk Prediction")
        st.metric("🧠 Probability", f"{prob:.2%}")
        if prediction == 1:
            st.error("🔴 You may have a risk of stroke. Please consult a doctor.")
        else:
            st.success("🟢 No predicted stroke risk.")

        # SHAP Explanation
        st.subheader("🔎 SHAP Explanation")
        st.markdown("This explains how each input contributed to the prediction.")

        try:
            explainer = shap.Explainer(model.named_estimators_['final_estimator'], feature_array)
        except:
            explainer = shap.Explainer(model, feature_array)

        shap_values = explainer(feature_array)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')

if __name__ == '__main__':
    main()
