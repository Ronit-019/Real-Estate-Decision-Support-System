import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors  # Replace faiss with this

@st.cache_data
def load_data():
    return pd.read_csv("REAL-ESTATE-DATASET.csv")

df = load_data()

st.sidebar.title("🏙️ Property Recommendation Filters")

# Locality selection (Compulsory)
localities = df['locality'].dropna().unique()
user_locality = st.sidebar.selectbox("Select Locality (City)", sorted(localities))

# Optional filters
st.sidebar.subheader("Optional Feature Filters")
feature_values = {}

# Numerical range filters
range_features = ['price', 'Built-up Area (sqft)', 'Carpet Area (sqft)']
for col in range_features:
    if col in df.columns:
        apply_filter = st.sidebar.checkbox(f"Filter by {col}?")
        if apply_filter:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            min_input = st.sidebar.number_input(f"{col} (Min)", min_value=min_val, max_value=max_val, value=min_val,
                                                key=col + "_min")
            max_input = st.sidebar.number_input(f"{col} (Max)", min_value=min_val, max_value=max_val, value=max_val,
                                                key=col + "_max")
            feature_values[col] = (min_input, max_input)

# Fixed-value numerical filters
fixed_num_features = ['BHK', 'bathrooms', 'Bedrooms', 'Floor', 'Total Floors']
for col in fixed_num_features:
    if col in df.columns:
        options = sorted(df[col].dropna().unique())
        options_with_dc = ["Don't care"] + list(map(str, options))
        selected = st.sidebar.selectbox(f"{col}", options_with_dc, key=col)
        if selected != "Don't care":
            feature_values[col] = float(selected)

# Categorical features with "Don't care"
categorical_features = ['Furnishing', 'Facing Direction']
for col in categorical_features:
    if col in df.columns:
        options = df[col].dropna().unique()
        options_with_dc = ["Don't care"] + list(map(str, options))
        selected = st.sidebar.selectbox(f"{col}", options_with_dc, key=col)
        if selected != "Don't care":
            feature_values[col] = selected

# Boolean features
boolean_features = ['Cctv', 'Community Hall', 'Garden', 'Gym', 'Kids Area',
                    'Lift', 'Parking', 'Power Backup', 'Sports Facility', 'Swimming Pool']
for col in boolean_features:
    if col in df.columns:
        selected = st.sidebar.radio(f"{col}", ["Don't care", "Yes", "No"], key=col)
        if selected == "Yes":
            feature_values[col] = 1
        elif selected == "No":
            feature_values[col] = 0

# Number of recommendations
n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

st.title("🏠 Smart Property Recommender")
st.write(f"Showing recommendations based on locality: **{user_locality}**")


def guided_locality_recommender(df, locality, feature_values, n=5):

    df_local = df[df['locality'].str.lower().str.strip() == locality.lower().strip()].reset_index(drop=True)
    if df_local.empty:
        return pd.DataFrame()

    for feat, val in feature_values.items():
        if feat in df_local.columns:
            if isinstance(val, tuple) and len(val) == 2:
                df_local = df_local[(df_local[feat] >= val[0]) & (df_local[feat] <= val[1])]
            else:
                df_local = df_local[df_local[feat] == val]

    df_local = df_local.reset_index(drop=True)
    if df_local.empty or len(df_local) <= 1:
        return pd.DataFrame()

    selected_cols = list(feature_values.keys())
    X = df_local[selected_cols].copy()

    categorical_cols = ['Furnishing', 'Facing Direction']
    X_encoded = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], drop_first=True)

    if X_encoded.empty:
        return pd.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded).astype('float32')

    # Replace faiss index creation and search with sklearn NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='auto', metric='euclidean')
    nbrs.fit(X_scaled)

    reference_idx = 0
    distances, indices = nbrs.kneighbors([X_scaled[reference_idx]])
    recommended_indices = indices[0][1:]

    return df_local.iloc[recommended_indices][['flat_name', 'real_name', 'locality'] + selected_cols]


if st.button("🔍 Show Recommendations", disabled=not feature_values):
    results = guided_locality_recommender(df, user_locality, feature_values, n_recommendations)

    if results.empty:
        st.warning("No properties found with the given filters.")
    else:
        st.success(f"Found {len(results)} recommended properties:")
        st.dataframe(results)
