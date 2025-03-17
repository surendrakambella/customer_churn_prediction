import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the saved Random Forest model
with open("random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction using AI & Advanced Ml techniques  ")
st.write("Enter customer details below to predict whether they will churn (1) or stay (0).")

# Input fields
gender = st.radio("Gender", [0, 1])
SeniorCitizen = st.radio("Senior Citizen", [0, 1])
Partner = st.radio("Partner", [0, 1])
Dependents = st.radio("Dependents", [0, 1])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
PhoneService = st.radio("Phone Service", [0, 1])
MultipleLines = st.radio("Multiple Lines", [0, 1])
InternetService = st.radio("Internet Service", [0, 1, 2])
OnlineSecurity = st.radio("Online Security", [0, 1])
OnlineBackup = st.radio("Online Backup", [0, 1])
DeviceProtection = st.radio("Device Protection", [0, 1])
TechSupport = st.radio("Tech Support", [0, 1])
StreamingTV = st.radio("Streaming TV", [0, 1])
StreamingMovies = st.radio("Streaming Movies", [0, 1])
Contract = st.radio("Contract Type", [0, 1, 2])
PaperlessBilling = st.radio("Paperless Billing", [0, 1])
PaymentMethod = st.radio("Payment Method", [0, 1, 2, 3])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01)

# Predict Button
if st.button("Predict"):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([[
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    ]], columns=[
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    # Convert DataFrame to NumPy array
    input_array = input_data.to_numpy()

    # Make prediction
    prediction = rf_model.predict(input_array)

    # Display result
    if prediction[0] == 0:
        st.success("Customer will **NOT** churn (0) 🚀")
    else:
        st.error("Customer will **CHURN** (1) ⚠️")
