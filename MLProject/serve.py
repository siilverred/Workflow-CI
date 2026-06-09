import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model dari file lokals
MODEL_PATH = "model_output/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[+] Model loaded successfully!")
except Exception as e:
    print(f"[Warning] Model not found: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'attrition_probability': float(probability)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)