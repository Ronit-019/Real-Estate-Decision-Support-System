import streamlit as st

st.set_page_config(
    page_title="Real Estate Decision Support System"
)

st.markdown("""
# 🏡 Real Estate Decision Support System

**Make smarter property decisions with the power of Data and Machine Learning.**

This end-to-end platform helps buyers, sellers, investors, and researchers gain data-driven insights into the Indian real estate market through interactive visualizations and machine learning models.

---

## 🔍 Key Features

### 📊 Real Estate Analytics
- 📍 Price Distribution in Selected Locality  
- 🗺️ Average Price Per Region  
- 📈 Price vs Built-up Area (Colored by Bathrooms) 
- 🧱 Room Count Distribution (BHK Types)  
- 🏢 Average Price vs Floor Level  
- 🛠️ Availability of Amenities Across Properties  
- 🧭 Average Price Per Facing Direction  

### 💰 Price Prediction
Enter BHK, area, and floor to get instant price estimates using a trained regression model.

### 🏙️ Flats Recommender
Find similar flats based on price, locality, and amenities using a content-based recommender system.

### 💡 Insights Generator
Discover how features like balconies or extra bathrooms affect property prices using ML-based feature importance.

### 🏷️ Property Category Classification
Classify listings into **Affordable**, **Mid-range**, or **Luxury** based on input features using classification models.

---

## 🛠️ Tools & Technologies
- Python, Pandas, Scikit-learn, Streamlit  
- Plotly, Seaborn, Matplotlib  
- Web scraping from housing website  
- ML Models: Linear Regression, Random Forest, Logistic Regression  

---

## 👨‍💻 Built By
**Ronit Rajput**  
https://www.linkedin.com/in/ronit-rajput/
""")

st.sidebar.success("Select a demo above.")