import streamlit as st
import pandas as pd
import joblib
import json
from tensorflow import keras

# -----------------------------
# Load saved artifacts
# -----------------------------

preprocessor = joblib.load(
    'artifacts/preprocessor.joblib'
)

le = joblib.load(
    'artifacts/label_encoder.joblib'
)

model = keras.models.load_model(
    'artifacts/churn_model.keras'
)

with open('artifacts/metadata.json', 'r') as f:
    metadata = json.load(f)

threshold = metadata['threshold']


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🛒 E-Commerce Customer Churn Prediction")

st.write(
    "Enter customer details to predict the probability of churn."
)


# -----------------------------
# Customer Inputs
# -----------------------------

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=15
)

warehousetohome = st.number_input(
    "Warehouse To Home",
    min_value=0,
    value=29
)

numberofdeviceregistered = st.number_input(
    "Number of Devices Registered",
    min_value=0,
    value=4
)

preferedordercat = st.selectbox(
    "Preferred Order Category",
    [
        "Laptop & Accessory",
        "Mobile",
        "Fashion",
        "Others",
        "Mobile Phone",
        "Grocery"
    ]
)

satisfactionscore = st.slider(
    "Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

maritalstatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

numberofaddress = st.number_input(
    "Number of Address",
    min_value=0,
    value=2
)

complain = st.selectbox(
    "Complain",
    [0, 1]
)

daysincelastorder = st.number_input(
    "Days Since Last Order",
    min_value=0,
    value=7
)

cashbackamount = st.number_input(
    "Cashback Amount",
    min_value=0.0,
    value=143.32
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    new_customer_df = pd.DataFrame({
        'tenure': [tenure],
        'warehousetohome': [warehousetohome],
        'numberofdeviceregistered': [numberofdeviceregistered],
        'preferedordercat': [preferedordercat],
        'satisfactionscore': [satisfactionscore],
        'maritalstatus': [maritalstatus],
        'numberofaddress': [numberofaddress],
        'complain': [complain],
        'daysincelastorder': [daysincelastorder],
        'cashbackamount': [cashbackamount]
    })

    # Preprocess
    X_new = preprocessor.transform(
        new_customer_df
    )

    # Predict probability
    probability = model.predict(
        X_new,
        verbose=0
    ).ravel()[0]

    # Apply threshold
    prediction = int(
        probability >= threshold
    )

    # Display results
    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{probability:.2%}**"
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.write(
        f"Decision Threshold: **{threshold:.2%}**"
    )