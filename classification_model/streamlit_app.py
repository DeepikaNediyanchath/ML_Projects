# streamlit_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/v1/predict"

st.title("Iris Flower Species Prediction")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 0.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 0.1)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 0.1)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.1)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted species: {result['prediction']}")
    except Exception as e:
        st.error("API not reachable")
