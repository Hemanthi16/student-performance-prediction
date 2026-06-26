import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Predict marks based on study hours using Machine Learning")

hours = st.number_input("Enter Study Hours", min_value=0.0, max_value=24.0)

if st.button("Predict Marks"):
    prediction = model.predict(np.array([[hours]]))
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
