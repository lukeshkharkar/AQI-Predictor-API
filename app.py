from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})
