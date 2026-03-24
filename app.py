from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.train_model import FEATURE_COLUMNS, train_and_save_model

app = Flask(__name__)

MODEL_PATH = Path('models/uhi_model.pkl')
DATASET_PATH = Path('data/uhi_synthetic_dataset.csv')

LABEL_MAP = {
    0: 'Normal Temperature Zone',
    1: 'Moderate Heat Zone',
    2: 'High Heat Zone',
}


def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save_model(MODEL_PATH, DATASET_PATH, n_rows=1000)


MODEL = load_or_train_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}

    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    input_df = pd.DataFrame([{col: float(payload[col]) for col in FEATURE_COLUMNS}])
    prediction = int(MODEL.predict(input_df)[0])

    probabilities = MODEL.predict_proba(input_df)[0]
    confidence = float(probabilities[prediction])

    return jsonify(
        {
            'prediction': prediction,
            'label': LABEL_MAP[prediction],
            'confidence': round(confidence, 4),
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
