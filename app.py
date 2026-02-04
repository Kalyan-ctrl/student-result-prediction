from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])
    internal = float(request.form["internal"])

    prediction = model.predict([[hours, attendance, internal]])
    result = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Predicted Final Score: {result}")

if __name__ == "__main__":
    app.run(debug=True)
