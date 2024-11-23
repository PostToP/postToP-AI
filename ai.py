import spacy
import tensorflow as tf
import numpy as np
from preprocess import detect_and_translate, preprocess

interpreter = tf.lite.Interpreter(model_path="./ai/model.tflite")
interpreter.allocate_tensors()

def predict_tflite(title_vector, description_vector, category_vector):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    title_vector = title_vector.astype(np.float32)
    description_vector = description_vector.astype(np.float32)
    category_vector = category_vector.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], category_vector)
    interpreter.set_tensor(input_details[1]['index'], title_vector)
    interpreter.set_tensor(input_details[2]['index'], description_vector)


    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][0]

import dill
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

with open("ai/category_vec.pkl", "rb") as f:
    temp = dill.load(f)
    category_tokenizer = temp[0]
    category_vectorizer = temp[1]

with open("ai/title_vec.pkl", "rb") as f:
    temp = dill.load(f)
    title_tokenizer = temp[0]
    title_vectorizer = temp[1]

with open("ai/description_vec.pkl", "rb") as f:
    temp = dill.load(f)
    description_tokenizer = temp[0]
    description_vectorizer = temp[1]