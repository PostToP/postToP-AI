import spacy
import os
from dotenv import load_dotenv

load_dotenv()
import tensorflow as tf
import numpy as np
from preprocess import detect_and_translate, preprocess
import dill
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from ai import predict_tflite, category_tokenizer, category_vectorizer, title_tokenizer, title_vectorizer, description_tokenizer, description_vectorizer


def convert_to_vector(text, tokenizer, vectorizer):
    text = preprocess(text)
    text = tokenizer.encode(text)
    text = vectorizer.encode(text)
    return text

import requests
API_KEY = os.getenv("API_KEY")

def call_yt(id):
  url = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CtopicDetails&hl=en&id={id}&key={API_KEY}'
  headers = {
        'Accept': 'application/json'
  }
  response = requests.get(url, headers=headers)
  if response.status_code == 200:
      video_data = response.json()
      title = video_data["items"][0]["snippet"]["localized"]["title"];
      description = video_data["items"][0]["snippet"]["localized"]["description"];
      if "topicDetails" not in video_data["items"][0]:
        categories = []
      else:
        categories = video_data["items"][0]["topicDetails"]["topicCategories"];
        categories = [x.replace("https://en.wikipedia.org/wiki/","").replace("_"," ") for x in categories]
      return title,description,categories
  else:
      print(f"Failed to fetch data: {response.status_code}")

def predict_is_music(title,description,categories):
  title = detect_and_translate(title)
  description = detect_and_translate(description)
  title_cv = convert_to_vector(title, title_tokenizer, title_vectorizer)
  description_cv = convert_to_vector(description, description_tokenizer, description_vectorizer)
  category_tok = category_tokenizer.encode(categories)
  categories_vectorized = category_vectorizer.encode(category_tok)

  title_cv = title_cv.reshape(1, title_cv.shape[0])
  description_cv = description_cv.reshape(1, description_cv.shape[0])
  categories_vectorized = categories_vectorized.reshape(1, categories_vectorized.shape[0])

  prediction = [predict_tflite(title_cv,description_cv,categories_vectorized)]

  return prediction



from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Disable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/predict")
async def predict(ytid: str):
    if not ytid:
        raise HTTPException(status_code=400, detail="ytid is required")

    try:
        title, description, categories = call_yt(ytid)
        prediction_result = predict_is_music(title, description, categories)[0]
        return {"confidence": float(prediction_result*100), "is_music": bool(prediction_result > 0.75)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)