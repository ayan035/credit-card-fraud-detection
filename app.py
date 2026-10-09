
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_names = joblib.load("feature_names.pkl")
threshold = joblib.load("best_threshold.pkl")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter transaction details to estimate fraud probability "
    "using a Random Forest model."
)

st.info(f"Decision threshold: {threshold:.0%}")

# Transaction inputs
st.subheader("Transaction Details")

amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
lat = st.number_input("Customer Latitude", value=40.7128)
long = st.number_input("Customer Longitude", value=-74.0060)
city_pop = st.number_input("City Population", min_value=0, value=50000)
merch_lat = st.number_input("Merchant Latitude", value=40.7306)
merch_long = st.number_input("Merchant Longitude", value=-73.9352)

transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
transaction_day = st.slider("Transaction Day", 1, 31, 15)
transaction_month = st.slider("Transaction Month", 1, 12, 6)
transaction_dayofweek = st.selectbox(
    "Day of Week (0 = Monday)", list(range(7)), index=2
)

age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.5)

gender = st.selectbox("Gender", ["M", "F"])

category = st.selectbox(
    "Transaction Category",
    [
        "grocery_pos",
        "grocery_net",
        "gas_transport",
        "shopping_pos",
        "shopping_net",
        "misc_pos",
        "misc_net",
        "entertainment",
        "food_dining",
        "health_fitness",
        "home",
        "kids_pets",
        "personal_care",
        "travel"
    ]
)

state = st.selectbox(
    "State",
    [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC",
        "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
        "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
        "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
        "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
        "WI", "WV", "WY"
    ]
)

if st.button("🔍 Check Transaction", type="primary"):
    sample = pd.DataFrame([{
        "amt": amt,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "transaction_dayofweek": transaction_dayofweek,
        "age": age,
        "distance_km": distance_km,
        "gender": gender,
        "category": category,
        "state": state
    }])

    try:
        processed = preprocessor.transform(sample)
        probability = model.predict_proba(processed)[0, 1]
        prediction = int(probability >= threshold)

        st.subheader("Prediction Result")
        st.metric("Estimated Fraud Probability", f"{probability:.2%}")

        if prediction == 1:
            st.error("⚠️ Model Prediction: FRAUD")
        else:
            st.success("✅ Model Prediction: LEGITIMATE")

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "that a transaction is fraudulent or legitimate."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption("Credit Card Fraud Detection | Random Forest ML Project")
