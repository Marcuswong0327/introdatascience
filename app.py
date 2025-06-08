import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler dictionary
try:
    model = joblib.load('stacking_classifier_model.pkl')
    scalers = joblib.load('scalers.pkl')  # Expected to be a dict with keys like 'bmi', 'age', etc.
except Exception as e:
    st.error(f"❌ Error loading model or scalers: {e}")
    st.stop()

# Define prediction function
def predict_stroke(features):
    features_to_scale = ['bmi', 'age', 'avg_glucose_level']
    scaled_features = []

    # No scaling for categorical features
    scaled_features.extend([features[key] for key in [
        'gender', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'smoking_status'
    ]])

    # Apply scaling
    for key in features_to_scale:
        scaler = scalers[key]
        scaled_value = scaler.transform(np.array([[features[key]]]))[0][0]
        scaled_features.append(scaled_value)

    prediction = model.predict([scaled_features])[0]
    probability = model.predict_proba([scaled_features])[0][1]
    return prediction, probability

# Advice logic
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

# Streamlit UI
def main():
    st.title("🩺 Stroke Prediction App")
    st.write("Fill in your details below to find out your risk of stroke.")

    # Input widgets
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    ever_married = st.selectbox("Ever Married?", [0, 1], format_func=lambda x: 'Yes' if x else 'No')

    work_type_dict = {
        0: "children",
        1: "Govt_job",
        2: "Private",
        3: "Self-employed",
        4: "Never_worked"
    }
    work_type = st.selectbox("Work Type?", list(work_type_dict.keys()), format_func=lambda x: work_type_dict[x])

    Residence = st.selectbox("Where you live?", [0, 1], format_func=lambda x: 'Urban' if x == 1 else 'Rural')
    smoking_status = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: 'smokes/formerly smoked' if x else 'never smoked')

    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, step=0.1)
    age = st.number_input("Age", min_value=0, step=1)
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, step=0.1)

    gender_numeric = 1 if gender == 'Female' else 0

    features = {
        'gender': gender_numeric,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence,
        'smoking_status': smoking_status,
        'bmi': bmi,
        'age': age,
        'avg_glucose_level': glucose
    }

    if st.button("Predict"):
        advices = advice_on_values(age, bmi, glucose)
        for adv in advices:
            st.info(adv)

        prediction, prob = predict_stroke(features)

        st.subheader("📊 Stroke Risk Prediction")
        st.metric("🧠 Stroke Probability", f"{prob:.2%}")

        if prediction == 1:
            st.error("🔴 You may have a risk of stroke. Please consult a doctor promptly.")
            if age > 60:
                st.error("⚠️ Older age detected, please monitor cardiovascular health and get regular check-ups.")
            if glucose > 140:
                st.error("⚠️ Elevated blood glucose, watch your diet and monitor for diabetes risk.")
            if bmi > 24.9:
                st.error("⚠️ BMI is high, recommend diet control and increased exercise to reduce obesity risk.")
        else:
            st.success("🟢 No predicted risk of stroke.")
            if age > 60:
                st.warning("⚠️ Older age detected, please monitor cardiovascular health and get regular check-ups.")
            if glucose > 140:
                st.warning("⚠️ Elevated blood glucose, watch your diet and monitor for diabetes risk.")
            if bmi > 24.9:
                st.warning("⚠️ BMI is high, recommend diet control and increased exercise to reduce obesity risk.")
            if age <= 60 and glucose <= 140 and bmi <= 24.9:
                st.success("✅ Great job maintaining a healthy lifestyle!")

# Call main function
if __name__ == '__main__':
    main()
