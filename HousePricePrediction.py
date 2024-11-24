import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from PIL import Image
import locale

locale.setlocale(locale.LC_ALL, 'en_PK.UTF-8')

# Streamlit app title
st.title("House Price Prediction Model with Inflation Adjustment")

# Load the preprocessed data
df1 = pickle.load(open('df1.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Prepare feature columns
X = df1.drop('price', axis=1)  # Features
y = df1['price']  # Target variable

# Initialize the DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion='friedman_mse', splitter='random', random_state=0)
dtr.fit(X, y)

def predict_price(location, sqft, bedrooms, baths):
  """Predicts property price based on features."""
  loc_index = np.where(X.columns == location)[0][0]
  x = np.zeros(len(X.columns))
  x[0] = baths
  x[1] = sqft
  x[2] = bedrooms
  if loc_index >= 0:
    x[loc_index] = 1
  return dtr.predict([x])[0] / 100000

def predict_price_with_inflation(location, sqft, bedrooms, baths, inflation_rate, year_prediction):
  """Predicts property price with inflation adjustment."""
  base_year = 2019  # Assuming the data is from 2019
  years_passed = year_prediction - base_year

  predicted_price_2019 = predict_price(location, sqft, bedrooms, baths)

  # Adjust price for inflation
  adjusted_price = predicted_price_2019 * (1 + inflation_rate) ** years_passed

  return adjusted_price


image = Image.open('house.jpg')
st.image(image, width=240)

# Streamlit inputs
st.header("Enter Property Details:")
location = st.selectbox("Location", sorted(df['location'].unique()))  # Dropdown menu for locations
sqft = st.number_input("Property Size (sqft)", min_value=100, step=10, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1, value=2)
baths = st.number_input("Number of Bathrooms", min_value=1, step=1, value=1)
inflation_rate = st.slider("Yearly Inflation Rate (%)", min_value=0.0, max_value=20.0, step=0.1, value=7.0) / 100
year_to_predict = st.number_input("Year to Predict", min_value=2020, step=1, value=2024)

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price_with_inflation(
        location, sqft, bedrooms, baths, inflation_rate, year_to_predict
    )
    formatted_price = locale.format_string("%.2f", predicted_price, grouping=True)
    st.success(
        f"The predicted price in {year_to_predict} for a property in {location} is: PKR "
        f"{formatted_price} Lakhs"
    )
