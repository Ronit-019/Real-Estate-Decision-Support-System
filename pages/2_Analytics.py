import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Welcome to Analysis Page!")

@st.cache_data
def load_data():
    return pd.read_csv("REAL-ESTATE-DATASET.csv")

df = load_data()

df['price_in_rupees'] = df['price'] * 1e5  # Lakhs to Rupees
df['price_per_sqft'] = df['price_in_rupees'] / df['Built-up Area (sqft)']

localities = df['locality'].dropna().unique()
selected_locality = st.selectbox("Choose a locality to visualize:", sorted(localities))

# Filter based on selected locality
filtered_df = df[df['locality'] == selected_locality]

# Show the Mapbox scatter plot
if not filtered_df.empty:
    fig1 = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="price_per_sqft",
        size="Built-up Area (sqft)",
        color_continuous_scale="Viridis",
        mapbox_style="open-street-map",
        zoom=10,
        title=f"Price Distribution in {selected_locality}"
    )
    st.plotly_chart(fig1)
else:
    st.warning("No data available for the selected locality.")

col1,col2 = st.columns(2)

region_price = df.groupby('region')['price'].mean().reset_index().sort_values(by='price')
fig2 = px.bar(region_price, x='region', y='price',text='price')
fig2.update_traces(texttemplate='%{text:,.0f}lakhs', textposition='outside')
fig2.update_layout(xaxis_title='Region Name', yaxis_title='Average Price (₹) in Lakhs',title='Average Price Per Region')
st.plotly_chart(fig2)

fig3 = px.scatter(df, x="Built-up Area (sqft)", y="price", color="bathrooms",title="Price vs. Built-up Area Colored by Number of Bathrooms",
    labels={
        "Built-up Area (sqft)": "Built-up Area (sqft)",
        "price": "Price (₹)",
        "bathrooms": "Number of Bathrooms"
    })
st.plotly_chart(fig3)

room_counts = df['BHK'].value_counts().reset_index()
room_counts.columns = ['BHK', 'Count']

# Plot Pie Chart
fig4 = px.pie(
    room_counts,
    names='BHK',
    values='Count',
    title='Room Count Distribution (BHK Types)',
    color_discrete_sequence=px.colors.qualitative.Set3
)
st.plotly_chart(fig4)

floor_price = df.groupby('Floor')['price'].mean().reset_index()
fig5 = px.line(floor_price, x='Floor', y='price',
              title="Average Price vs. Floor Level")
fig5.update_layout(xaxis_title='Floor', yaxis_title='Average Price (₹) in Lakhs')
st.plotly_chart(fig5)

amenities_cols = ['Cctv', 'Community Hall', 'Garden', 'Gym', 'Kids Area',
                  'Lift', 'Parking', 'Power Backup', 'Sports Facility', 'Swimming Pool']
amenities_count = df[amenities_cols].sum().reset_index()
amenities_count.columns = ['Amenity', 'Count']
fig6 = px.bar(amenities_count, x='Count', y='Amenity', orientation='h')
fig6.update_layout(xaxis_title='Number of Properties', yaxis_title='Amenity Type',title='Availability of Amenities Across Properties')
st.plotly_chart(fig6)

df['Facing Direction'] = df['Facing Direction'].replace('Northeast Facing', 'North East Facing')

region_price = df.groupby('Facing Direction')['price'].mean().reset_index().sort_values(by='price')
fig7 = px.bar(region_price, x='Facing Direction', y='price',text='price')
fig7.update_traces(texttemplate='%{text:,.0f}lakhs', textposition='outside')
fig7.update_layout(xaxis_title='Facing Direction', yaxis_title='Average Price (₹) in Lakhs',title='Average Price Per Facing Direction')
st.plotly_chart(fig7)

st.sidebar.success("Select a demo above.")