from flask import Flask, request, jsonify
from model.ModelWrapper import ModelWrapper
import logging

logger = logging.getLogger("prod")
logger.setLevel(logging.DEBUG)

model = ModelWrapper.deserialize('model/v1.pkl')
model.load_model()
logger.info("Model loaded")
model.warmup()
logger.info("Model warmed up")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        title = data['title']
        description = data['description']
        categories = data['categories']
        duration = [data['duration']]

        logger.info(f"Processing request with title: {title[:30]}...")

        prediction = model.predict(
            title, description, categories, duration)
        logger.info(f"Prediction: {prediction}")

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
    logger.info("Server started on port 5000")
    serve(app, host="0.0.0.0", port=5000)
