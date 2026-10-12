
import streamlit as st
import pandas as pd
import joblib


# Store prediction history for the current browser session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Load the trained model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate their likelihood of leaving."
)

st.divider()

# Create sections for customer details
st.header("1. Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12
    )
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges (£)",
        min_value=0.0,
        max_value=10000.0,
        value=70.0,
        step=1.0
    )
    total_charges = st.number_input(
        "Total Charges (£)",
        min_value=0.0,
        max_value=100000.0,
        value=840.0,
        step=10.0
    )

st.caption(
    "Note: Senior Citizen is encoded as 0 (No) or 1 (Yes). "
    "Charges use the dataset's original currency units."
)

st.divider()

# Build a row with the exact feature names used during training
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# Predict only when the button is clicked
if st.button("Predict Churn", type="primary"):

    if total_charges < monthly_charges * tenure:
        st.warning(
            "Total charges are lower than monthly charges multiplied "
            "by tenure. Please check these values."
        )

    prediction = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]

    churn_index = list(model.classes_).index("Yes")
    churn_probability = probabilities[churn_index]
    
    # Save this prediction
    st.session_state.prediction_history.append({
        "Tenure (months)": tenure,
        "Contract": contract,
        "Monthly Charges": monthly_charges,
        "Predicted Churn": prediction,
        "Churn Probability": round(churn_probability * 100, 2)
    })

    st.header("2. Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == "Yes":
            st.error("Prediction: Customer likely to churn")
        else:
            st.success("Prediction: Customer likely to stay")

    with result_col2:
        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

    st.progress(float(churn_probability))

    if churn_probability >= 0.70:
        st.warning("High estimated churn probability.")
    elif churn_probability >= 0.40:
        st.info("Moderate estimated churn probability.")
    else:
        st.success("Low estimated churn probability.")

    st.caption(
        "This is a model estimate, not a guarantee. "
        "The model's performance depends on the data used to train it."
    )
    
st.divider()
st.header("3. Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No predictions yet. Enter customer details and click Predict Churn.")