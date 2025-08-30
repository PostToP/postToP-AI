import pickle
import dill
import numpy as np
from flask import Flask, request, jsonify
from data.text_cleaning import TextPreprocessor
from model.ModelWrapper import ModelWrapper

model = ModelWrapper.deserialize('model/v1.pkl')
model.load_model()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        title = data['title']
        description = data['description']
        categories = data['categories']
        duration = [data['duration']]

        prediction = model.predict(
            title, description, categories, duration)

        return jsonify({
            'prediction': prediction.tolist(),
            'version': 'v1.0.0'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
        }), 400


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
