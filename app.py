from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    stories = float(request.form["stories"])
    parking = float(request.form["parking"])

    mainroad = float(request.form["mainroad"])
    guestroom = float(request.form["guestroom"])
    basement = float(request.form["basement"])
    hotwaterheating = float(request.form["hotwaterheating"])
    airconditioning = float(request.form["airconditioning"])
    prefarea = float(request.form["prefarea"])
    furnishingstatus = float(request.form["furnishingstatus"])

    # Create 12-feature input
    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        prefarea,
        furnishingstatus
    ]])

    # Prediction
    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)