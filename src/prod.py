import pickle
import dill
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from text_cleaning import TextPreprocessor

with open('model.pkl', 'rb') as f:
    model = dill.load(f)

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json(force=True)
            title = data['title']
            description = data['description']
            categories = data['categories']
            duration = data['duration']
            duration = TextPreprocessor.convert_duration(duration)

            prediction = model.predict(
                title, description, categories, duration)

            print(prediction)

            return jsonify({
                'prediction': prediction.tolist(),
                'status': 'success'
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 400

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
