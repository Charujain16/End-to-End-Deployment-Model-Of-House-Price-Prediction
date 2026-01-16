import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="India House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 India House Price Prediction")
st.markdown("Predict house prices using a trained Machine Learning model")

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_house_price_model.pkl")
    ord_encoder = joblib.load("ordinal_encoder.pkl")
    scaler = joblib.load("robust_scaler.pkl")
    target_mean = joblib.load("target_mean.pkl")
    low_card_cols = joblib.load("low_card_cols.pkl")
    high_card_cols = joblib.load("high_card_cols.pkl")
    numerical_cols = joblib.load("numerical_cols.pkl")
    feature_order = joblib.load("feature_order.pkl")
    mappings = joblib.load("mappings.pkl")
    return (
        model,
        ord_encoder,
        scaler,
        target_mean,
        low_card_cols,
        high_card_cols,
        numerical_cols,
        feature_order,
        mappings
    )

(
    model,
    ord_encoder,
    scaler,
    target_mean,
    low_card_cols,
    high_card_cols,
    numerical_cols,
    feature_order,
    mappings
) = load_artifacts()

# -------------------------------
# Load Dataset (for UI values only)
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("india_housing_prices.csv")

df = load_data()

# -------------------------------
# UI INPUTS
# -------------------------------
st.subheader("📍 Location")

state = st.selectbox("State", sorted(df["State"].unique()))
city = st.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
locality = st.selectbox(
    "Locality",
    sorted(df[(df["State"] == state) & (df["City"] == city)]["Locality"].unique())
)

st.divider()
st.subheader("🏡 Property Details")

property_type = st.selectbox("Property Type", sorted(df["Property_Type"].unique()))
bhk = st.slider("BHK", 1, int(df["BHK"].max()), 2)
size = st.slider("Size (Sq Ft)", 300, 5000, 1000)
floor_no = st.slider("Floor No", 0, int(df["Total_Floors"].max()), 1)
age = st.slider("Age of Property", 0, int(df["Age_of_Property"].max()), 5)

st.divider()
st.subheader("🛋 Amenities & Facilities")

furnishing = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique()))
parking = st.selectbox("Parking Space", sorted(df["Parking_Space"].unique()))
security = st.selectbox("Security", sorted(df["Security"].unique()))
transport = st.selectbox(
    "Public Transport Accessibility",
    sorted(df["Public_Transport_Accessibility"].unique())
)
schools = st.slider("Nearby Schools", 0, 10, 2)
hospitals = st.slider("Nearby Hospitals", 0, 10, 1)

amenities = st.selectbox("Amenities", sorted(df["Amenities"].unique()))
facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
availability = st.selectbox(
    "Availability Status",
    sorted(df["Availability_Status"].unique())
)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔮 Predict House Price"):

    # Step 1: Create empty input row using training schema
    input_df = pd.DataFrame(columns=feature_order)
    input_df.loc[0] = np.nan

    # Step 2: Fill user inputs dynamically
    ui_values = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Floor_No": floor_no,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Furnished_Status": furnishing,
        "Parking_Space": parking,
        "Security": security,
        "Public_Transport_Accessibility": transport,
        "Amenities": amenities,
        "Facing": facing,
        "Availability_Status": availability
    }

    for col, val in ui_values.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    # Step 3: Auto-fill remaining columns
    for col in input_df.columns:
        if pd.isna(input_df.at[0, col]):
            if col in numerical_cols:
                input_df.at[0, col] = df[col].mean()
            else:
                input_df.at[0, col] = df[col].mode()[0]

    # Step 4: Encoding
    if low_card_cols:
        input_df[low_card_cols] = ord_encoder.transform(input_df[low_card_cols])

    for col in high_card_cols:
        input_df[col] = input_df[col].map(mappings[col]).fillna(target_mean)

    # Step 5: Scaling
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Step 6: Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"🏷️ Estimated House Price: ₹ {prediction:,.2f}")
