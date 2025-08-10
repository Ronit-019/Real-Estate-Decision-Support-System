import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title='Real Estate Price Predictor')

st.title("Real Estate Price Prediction")

# Load trained model and scaler
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Locality and Region options
locality_options = ['Ahmedabad', 'Bangalore', 'Chennai', 'Ghaziabad', 'Delhi',
                    'New Delhi', 'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai',
                    'Noida', 'Faridabad', 'Pune']

region_options = ['Gujarat', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh',
                  'Maharashtra', 'Telangana', 'Rajasthan', 'West Bengal']

furnishing_options = ['Semi-Furnished', 'Fully-Furnished', 'Unfurnished']

facing_options = ['East Facing', 'Not Mentioned', 'West Facing', 'North Facing',
                  'North East Facing', 'South East Facing', 'South West Facing',
                  'South Facing', 'North West Facing', 'Northeast Facing']

# Input fields
bhk = st.selectbox("BHK", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 10])
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 10])
builtup_area = st.number_input("Built-up Area (sqft)", min_value=200, max_value=12573, step=50)
carpet_area = st.number_input("Carpet Area (sqft)", min_value=35, max_value=9672, step=10)
floor = st.number_input("Floor", min_value=0, max_value=65, step=1)
total_floors = st.number_input("Total Floors", min_value=1, max_value=98, step=1)

if floor > total_floors:
    st.error("Floor cannot be greater than Total Floors")

luxury_score = st.slider("Luxury Score", min_value=0.0, max_value=51.0, value=5.0, step=1.0)

locality = st.selectbox("Locality", locality_options)
region = st.selectbox("Region", region_options)
furnishing = st.selectbox("Furnishing", furnishing_options)
facing = st.selectbox("Facing Direction", facing_options)

def one_hot_encode(value, options, prefix, drop_first=False):
    encoding = {}
    for opt in options[(1 if drop_first else 0):]:
        key = f"{prefix}_{opt}"
        encoding[key] = int(value == opt)
    return encoding

input_dict = {
    'BHK': bhk,
    'Bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'Built-up Area (sqft)': builtup_area,
    'Carpet Area (sqft)': carpet_area,
    'Floor': floor,
    'Total Floors': total_floors,
    'luxury_score': luxury_score
}

input_dict.update(one_hot_encode(locality, locality_options, "locality", drop_first=True))
input_dict.update(one_hot_encode(region, region_options, "region", drop_first=True))
input_dict.update(one_hot_encode(furnishing, furnishing_options, "Furnishing", drop_first=True))
input_dict.update(one_hot_encode(facing, facing_options, "Facing Direction", drop_first=True))

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_dict])

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    lower_bound = prediction - 12.525
    upper_bound = prediction + 12.525

    def format_price_range(min_value, max_value):
        if max_value < 100:
            return f"₹{min_value:.2f} Lakh – ₹{max_value:.2f} Lakh"
        elif min_value >= 100:
            return f"₹{min_value / 100:.2f} Cr – ₹{max_value / 100:.2f} Cr"
        else:
            return f"₹{min_value:.2f} Lakh – ₹{max_value / 100:.2f} Cr"

    st.info(f"Expected Range: {format_price_range(lower_bound, upper_bound)}")
