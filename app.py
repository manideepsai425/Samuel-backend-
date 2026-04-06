from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Safe path handling
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'career_model.pkl')
columns_path = os.path.join(BASE_DIR, 'columns.pkl')

# Load model and columns
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'rb') as f:
    columns = pickle.load(f)

# -----------------------------
# Data mappings
# -----------------------------
career_skill_map = {
    "Data Scientist": ["Python", "SQL"],
    "Software Engineer": ["Python", "SQL", "Java"],
    "Embedded Engineer": ["Java"]
}

skill_recommendations = {
    "Python": ["Learn Python Basics", "Do ML Projects"],
    "SQL": ["Practice SQL Queries", "Work with Databases"],
    "Java": ["Learn OOP in Java", "Build Backend Projects"]
}

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return jsonify({"message": "Zionyx Backend is running 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        age = data.get("age", 18)
        math_score = data.get("math_score", 50)

        # Prepare input
        input_data = np.zeros(len(columns))

        for i, col in enumerate(columns):
            if col == "Age":
                input_data[i] = age
            elif col == "Mathematics_Score":
                input_data[i] = math_score

        # Prediction
        prediction = model.predict([input_data])[0]

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([input_data])[0]
            confidence = float(max(probs) * 100)
        else:
            confidence = None

        # Skills & recommendations
        skills = career_skill_map.get(prediction, [])
        recommendations = []

        for skill in skills:
            recommendations.extend(skill_recommendations.get(skill, []))

        return jsonify({
            "career": prediction,
            "confidence": confidence,
            "skills": skills,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run server (Render compatible)
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
