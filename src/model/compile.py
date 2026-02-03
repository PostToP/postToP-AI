import pandas as pd
import tensorflow as tf
import joblib

from model.model_wrapper import ModelWrapper


def create_model_wrapper():
    model = tf.keras.models.load_model("model/final_model.keras")
    model.trainable = False
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    pipelines = {
        "title": joblib.load("model/pipelines/Title_pipeline.joblib"),
        "description": joblib.load("model/pipelines/Description_pipeline.joblib"),
        "category": joblib.load("model/pipelines/Categories_pipeline.joblib"),
        "duration": joblib.load("model/pipelines/Duration_pipeline.joblib"),
    }

    model_wrapper = ModelWrapper(
        title_pipeline=pipelines["title"],
        description_pipeline=pipelines["description"],
        category_pipeline=pipelines["category"],
        duration_pipeline=pipelines["duration"],
        interpreter=tflite_model,
    )
    model_wrapper.serialize("model/model_wrapper.tar.gz")
