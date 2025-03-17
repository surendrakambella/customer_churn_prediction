import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the saved Random Forest model
with open("random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction using AI & Advanced Ml techniques made by Surendra")
st.write("Enter customer details below to predict whether they will churn (1) or stay (0).")

# Input fields with user-friendly text options
gender = st.radio("Gender", ["Male", "Female"])
SeniorCitizen = st.radio("Senior Citizen", ["No", "Yes"])
Partner = st.radio("Partner", ["No", "Yes"])
Dependents = st.radio("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
PhoneService = st.radio("Phone Service", ["No", "Yes"])
MultipleLines = st.radio("Multiple Lines", ["No", "Yes"])
InternetService = st.radio("Internet Service", ["No Internet", "DSL", "Fiber Optic"])
OnlineSecurity = st.radio("Online Security", ["No", "Yes"])
OnlineBackup = st.radio("Online Backup", ["No", "Yes"])
DeviceProtection = st.radio("Device Protection", ["No", "Yes"])
TechSupport = st.radio("Tech Support", ["No", "Yes"])
StreamingTV = st.radio("Streaming TV", ["No", "Yes"])
StreamingMovies = st.radio("Streaming Movies", ["No", "Yes"])
Contract = st.radio("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
PaperlessBilling = st.radio("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.radio("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01)

# Convert text inputs to numeric values
def convert_inputs():
    return [
        1 if gender == "Female" else 0,
        1 if SeniorCitizen == "Yes" else 0,
        1 if Partner == "Yes" else 0,
        1 if Dependents == "Yes" else 0,
        tenure,
        1 if PhoneService == "Yes" else 0,
        1 if MultipleLines == "Yes" else 0,
        ["No Internet", "DSL", "Fiber Optic"].index(InternetService),
        1 if OnlineSecurity == "Yes" else 0,
        1 if OnlineBackup == "Yes" else 0,
        1 if DeviceProtection == "Yes" else 0,
        1 if TechSupport == "Yes" else 0,
        1 if StreamingTV == "Yes" else 0,
        1 if StreamingMovies == "Yes" else 0,
        ["Month-to-Month", "One Year", "Two Year"].index(Contract),
        1 if PaperlessBilling == "Yes" else 0,
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"].index(PaymentMethod),
        MonthlyCharges,
        TotalCharges
    ]

# Predict Button
if st.button("Predict"):
    input_array = np.array([convert_inputs()])
    prediction = rf_model.predict(input_array)
    
    if prediction[0] == 0:
        st.success("Customer will **NOT** churn (0) 🚀")
    else:
        st.error("Customer will **CHURN** (1) ⚠️")
