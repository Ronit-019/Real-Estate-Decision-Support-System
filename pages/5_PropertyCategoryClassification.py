import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load your data (replace with actual path or source)
df = pd.read_csv("REAL-ESTATE-DATASET.csv")

# 2️⃣ Create Category Labels
def categorize_property(price):
    if price <= 50:
        return 'Affordable'
    elif price <= 150:
        return 'Mid-range'
    else:
        return 'Luxury'


df['price_category'] = df['price'].apply(categorize_property)

# 3️⃣ Encode Locality
le = LabelEncoder()
df['locality_encoded'] = le.fit_transform(df['locality'].astype(str))

# 4️⃣ Prepare Features and Labels
feature_cols = [
    'BHK', 'Built-up Area (sqft)', 'Floor', 'bathrooms',
    'Cctv', 'Community Hall', 'Garden', 'Gym', 'Kids Area',
    'Lift', 'Parking', 'Power Backup', 'Sports Facility', 'Swimming Pool',
    'locality_encoded'
]

X = df[feature_cols]
y = df['price_category']

# 5️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Evaluation Report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# 8️⃣ Streamlit App Interface
st.title("Property Category Classification")

st.subheader("Try Predicting a Property Category")

# Input Form
with st.form("predict_form"):
    bhk = st.number_input("BHK", min_value=1, step=1)
    area = st.number_input("Built-up Area (sqft)", min_value=100)
    floor = st.number_input("Floor Number", min_value=0, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, step=1)

    cctv = st.checkbox("CCTV")
    community_hall = st.checkbox("Community Hall")
    garden = st.checkbox("Garden")
    gym = st.checkbox("Gym")
    kids_area = st.checkbox("Kids Area")
    lift = st.checkbox("Lift")
    parking = st.checkbox("Parking")
    power_backup = st.checkbox("Power Backup")
    sports_facility = st.checkbox("Sports Facility")
    swimming_pool = st.checkbox("Swimming Pool")

    locality_input = st.selectbox("Locality", df['locality'].unique())

    submitted = st.form_submit_button("Predict Category")

    if submitted:
        locality_encoded = le.transform([locality_input])[0]
        input_data = pd.DataFrame([[
            bhk, area, floor, bathrooms,
            int(cctv), int(community_hall), int(garden), int(gym), int(kids_area),
            int(lift), int(parking), int(power_backup), int(sports_facility), int(swimming_pool),
            locality_encoded
        ]], columns=feature_cols)

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Property Category: {prediction}")

st.subheader("Classification Report on Test Set")
st.json(report)
