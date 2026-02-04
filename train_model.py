import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("data.csv")

X = data[["hours_studied", "attendance", "internal_marks"]]
y = data["final_score"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
