"""
Diabetes Prediction Backend API
Flask server that serves the trained ML model
Authors: Zaib & Zoha
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call this API

# ──────────────────────────────────────────────
# Load saved model artifacts
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model         = joblib.load(os.path.join(BASE_DIR, "diabetes_prediction_model.pkl"))
    scaler        = joblib.load(os.path.join(BASE_DIR, "diabetes_scaler.pkl"))
    feature_cols  = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    feature_to_idx = {name: i for i, name in enumerate(feature_cols)}
    print("✅  Model, scaler, and feature columns loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️  Could not load model file: {e}")
    print("    Run your Colab notebook first to generate the .pkl files,")
    print("    then copy them into the same folder as app.py.")
    model = scaler = feature_cols = feature_to_idx = None


# ──────────────────────────────────────────────
# Helper: build the feature vector from inputs
# ──────────────────────────────────────────────
def build_input_vector(age, bmi, hba1c, glucose,
                       hypertension=0, heart_disease=0,
                       smoking_history=0, gender="Male",
                       year=2020):
    import pandas as pd
    n = len(feature_cols)
    vec = np.zeros((1, n))

    def set_feat(name, value):
        if name in feature_to_idx:
            vec[0, feature_to_idx[name]] = value

    set_feat("age",                 age)
    set_feat("bmi",                 bmi)
    set_feat("hbA1c_level",        hba1c)
    set_feat("blood_glucose_level", glucose)
    set_feat("year",                year)
    set_feat("hypertension",        hypertension)
    set_feat("heart_disease",       heart_disease)
    set_feat("smoking_history",     smoking_history)

    if gender == "Male":
        set_feat("gender_Male", 1)
    else:
        set_feat("gender_Male", 0)

    for col in feature_cols:
        if col.startswith("location_"):
            set_feat(col, 1 if col == "location_Alabama" else 0)
        if col.startswith("race_"):
            set_feat(col, 1 if col == "race_Caucasian" else 0)

    # Scaling — bilkul training jaisa
    vec_df = pd.DataFrame(vec, columns=feature_cols)
    numeric_cols = [c for c in feature_cols if not c.startswith(('gender_', 'location_', 'race_'))]
    vec_df[numeric_cols] = scaler.transform(vec_df[numeric_cols])
    vec = vec_df.values

    return vec


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Diabetes Prediction API is running.",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "age": 45,
        "bmi": 28.5,
        "hba1c": 6.5,
        "glucose": 140,
        "hypertension": 0,      ← optional (0 or 1)
        "heart_disease": 0,     ← optional (0 or 1)
        "smoking_history": 0,   ← optional (0 or 1)
        "gender": "Male"        ← optional ("Male" or "Female")
    }

    Returns:
    {
        "prediction": "Diabetic" | "Non-Diabetic",
        "risk_percent": 73.4,
        "risk_label": "High" | "Moderate" | "Low",
        "confidence": 73.4
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Place .pkl files next to app.py and restart."}), 503

    data = request.get_json(force=True)

    # ── Validate required fields ──
    required = ["age", "bmi", "hba1c", "glucose"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        age       = float(data["age"])
        bmi       = float(data["bmi"])
        hba1c     = float(data["hba1c"])
        glucose   = float(data["glucose"])
        hypert    = int(data.get("hypertension",   0))
        heart_d   = int(data.get("heart_disease",  0))
        smoking   = int(data.get("smoking_history",0))
        gender    = str(data.get("gender", "Male"))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # ── Range validation ──
    if not (0 < age < 120):
        return jsonify({"error": "Age must be between 1 and 120."}), 400
    if not (10 < bmi < 70):
        return jsonify({"error": "BMI must be between 10 and 70."}), 400
    if not (3.5 < hba1c < 15):
        return jsonify({"error": "HbA1c must be between 3.5 and 15."}), 400
    if not (50 < glucose < 500):
        return jsonify({"error": "Blood glucose must be between 50 and 500 mg/dL."}), 400

    # ── Build vector & predict ──
    try:
        vec        = build_input_vector(age, bmi, hba1c, glucose, hypert, heart_d, smoking, gender)
        proba      = model.predict_proba(vec)[0][1]
        pred      = 1 if proba >= 0.35 else 0
        risk_pct   = round(float(proba) * 100, 1)
        label      = "Diabetic" if pred == 1 else "Non-Diabetic"

        if risk_pct >= 60:
            risk_label = "High"
        elif risk_pct >= 35:
            risk_label = "Moderate"
        else:
            risk_label = "Low"

        return jsonify({
            "prediction":   label,
            "risk_percent": risk_pct,
            "risk_label":   risk_label,
            "confidence":   risk_pct if label == "Diabetic" else round(100 - risk_pct, 1)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict for multiple patients at once.
    Expects: { "patients": [ {age, bmi, hba1c, glucose, ...}, ... ] }
    """
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    data     = request.get_json(force=True)
    patients = data.get("patients", [])

    if not patients:
        return jsonify({"error": "No patients provided."}), 400

    results = []
    for i, p in enumerate(patients):
        try:
            vec      = build_input_vector(
                float(p["age"]), float(p["bmi"]),
                float(p["hba1c"]), float(p["glucose"]),
                int(p.get("hypertension", 0)),
                int(p.get("heart_disease", 0)),
                int(p.get("smoking_history", 0)),
                str(p.get("gender", "Male"))
            )
            proba      = model.predict_proba(vec)[0][1]
            pred       = 1 if proba >= 0.35 else 0
            risk_pct = round(float(proba) * 100, 1)
            results.append({
                "index":      i,
                "prediction": "Diabetic" if pred == 1 else "Non-Diabetic",
                "risk_percent": risk_pct
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"results": results})


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
