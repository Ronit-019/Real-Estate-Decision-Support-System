import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Insights Generator", layout="wide")

st.title("Property Price Insights Generator")

# Load Data
df = pd.read_csv('REAL-ESTATE-DATASET.csv')

st.write("### Sample Data")
st.dataframe(df.head())

# Feature Columns
feature_cols = [
    'BHK', 'Bedrooms', 'bathrooms', 'Built-up Area (sqft)', 'Carpet Area (sqft)',
    'Floor', 'Total Floors', 'luxury_score', 'Furnishing', 'Facing Direction',
    'locality', 'region', 'Cctv', 'Community Hall', 'Garden', 'Gym', 'Kids Area',
    'Lift', 'Parking', 'Power Backup', 'Sports Facility', 'Swimming Pool'
]

# Target Column
target_col = 'price'

# Feature Selection via Multiselect
selected_features = st.multiselect("Select Features to Include:", feature_cols, default='BHK')

# Display and Edit Selected Feature Values
sample_row = df[selected_features].iloc[0].copy()

# Preprocessing: Encoding Categorical Features
categorical_cols = ['Furnishing', 'Facing Direction', 'locality', 'region',
                    'Cctv', 'Community Hall', 'Garden', 'Gym', 'Kids Area',
                    'Lift', 'Parking', 'Power Backup', 'Sports Facility', 'Swimming Pool']

used_categorical_cols = [col for col in categorical_cols if col in selected_features]

df_encoded = pd.get_dummies(df[selected_features], columns=used_categorical_cols, drop_first=True)

X = df_encoded
y = df[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature Importance Plot (Random Forest)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

st.write("### Feature Importance")
fig1, ax1 = plt.subplots(figsize=(12,8))
sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax1)
ax1.set_title('Feature Importance')
st.pyplot(fig1)

st.success("Insights generated successfully.")
