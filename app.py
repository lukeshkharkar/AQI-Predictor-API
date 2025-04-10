from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load model and pre-processing tools
model = tf.keras.models.load_model("aqi_predictor.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]  # 6x6 values from sensors
    input_arr = np.array(data)
    input_scaled = scaler.transform(input_arr).reshape(1, 6, 6)
    prediction = model.predict(input_scaled)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return jsonify({"prediction": predicted_class})
