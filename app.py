from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)


# =========================
# LOAD TRAINED MODEL
# =========================

with open("house_price_model.pkl", "rb") as file:
    data = pickle.load(file)

# If the pickle contains a dictionary
if isinstance(data, dict):
    model = data["model"]
else:
    # If the pickle directly contains the model
    model = data


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        error=None
    )


# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from HTML form

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


        # =========================
        # CONVERT DATE
        # =========================

        date = int(date[:4])


        # =========================
        # CREATE DATAFRAME
        # =========================

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


        # =========================
        # MAKE PREDICTION
        # =========================

        prediction = model.predict(input_data)[0]


        # Prevent negative price

        if prediction < 0:
            prediction = 0


        prediction = round(float(prediction), 2)


        # =========================
        # SHOW RESULT
        # =========================

        return render_template(
            "index.html",
            prediction=prediction,
            error=None
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run()
