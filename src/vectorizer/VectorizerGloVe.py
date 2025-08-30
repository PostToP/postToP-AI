from sentence_transformers import SentenceTransformer
import numpy as np
from vectorizer.IVectorizer import IVectorizer

class VectorizerGloVe(IVectorizer):
    def __init__(self):
        self.vectorizer = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d')

    def train(self, dataset):
        pass

    def encode(self, text):
        if len(text) <= 1:
            return np.zeros(300)
        text = " ".join(text)
        text = [text]
        return self.vectorizer.encode(text)[0]

    def encode_batch(self, texts):
        return np.array([self.encode(text) for text in texts])

    def getDict(self):
        return {}
    
    def __repr__(self):
        return "VectorizerGloVe"