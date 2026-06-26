import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9],
    "Marks": [10,20,25,40,50,55,70,80,90]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model trained successfully")
