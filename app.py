from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    data = pickle.load(file)

# Get actual model from dictionary
model = data["model"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        date = request.form["date"]
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])
        waterfront = float(request.form["waterfront"])
        view = float(request.form["view"])
        condition = float(request.form["condition"])
        sqft_above = float(request.form["sqft_above"])
        sqft_basement = float(request.form["sqft_basement"])
        yr_built = float(request.form["yr_built"])
        yr_renovated = float(request.form["yr_renovated"])
        city = request.form["city"]
        country = request.form["country"]

        input_data = pd.DataFrame({
            "date": [date],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "city": [city],
            "country": [country]
        })

        prediction = model.predict(input_data)[0]

        if prediction < 0:
            prediction = 0

        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:
        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/">Go Back</a>
        """


if __name__ == "__main__":
    app.run(debug=True)


