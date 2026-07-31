"""
app.py
------
Task 3: API Development

A Flask REST API that loads the trained heart-disease model and
returns a prediction for a JSON payload of patient details.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and the feature order it expects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


@app.route("/", methods=["GET"])
def home():
    """Simple landing page / health check."""
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient details as JSON to /predict",
        "expected_fields": feature_names
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    Returns:
    { "prediction": "Heart Disease Detected" }
    or
    { "prediction": "No Heart Disease Detected" }
    """
    try:
        data = request.get_json(force=True)

        # Validate that all required fields are present
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {missing}"
            }), 400

        # Build a single-row DataFrame in the exact column order used for training
        input_df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability_of_heart_disease": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
