import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🎓 Student Performance Predictor")

# Simple dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Marks": [10,20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = st.number_input("Enter Study Hours")

if st.button("Predict"):
    pred = model.predict(np.array([[hours]]))
    st.success(f"Predicted Marks: {pred[0]:.2f}")
