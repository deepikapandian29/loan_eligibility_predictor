import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model = joblib.load('xgb_loan_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("Loan Eligibility Predictor (XGBoost)")
st.write("Enter applicant details to predict loan eligibility:")

# Create input fields
user_input = {}

for feature in feature_columns:
    if feature == 'Gender':
        user_input[feature] = st.selectbox("Gender", ["Male", "Female"])
        user_input[feature] = 1 if user_input[feature]=="Male" else 0
    elif feature == 'Married':
        user_input[feature] = st.selectbox("Married", ["Yes", "No"])
        user_input[feature] = 1 if user_input[feature]=="Yes" else 0
    elif feature == 'Education':
        user_input[feature] = st.selectbox("Education", ["Graduate", "Not Graduate"])
        user_input[feature] = 1 if user_input[feature]=="Graduate" else 0
    elif feature == 'Self_Employed':
        user_input[feature] = st.selectbox("Self-Employed", ["Yes", "No"])
        user_input[feature] = 1 if user_input[feature]=="Yes" else 0
    elif feature == 'Property_Area':
        user_input[feature] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        user_input[feature] = mapping[user_input[feature]]
    else:
        user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=1.0)

# Predict button
if st.button("Predict Loan Eligibility"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success("Loan Status: " + ("Approved ✅" if prediction==1 else "Rejected ❌"))
