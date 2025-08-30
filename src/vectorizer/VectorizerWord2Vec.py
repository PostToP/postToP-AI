from gensim.models import Word2Vec
import numpy as np
from vectorizer.IVectorizer import IVectorizer
import multiprocessing

class VectorizerWord2Vec(IVectorizer):
    def __init__(self, max_features=300):
        self.max_features = max_features

    def train(self, dataset):
        max_workers = multiprocessing.cpu_count()
        self.vectorizer = Word2Vec(
            sentences=dataset, vector_size=self.max_features, window=3, min_count=1, workers=max_workers)

    def encode(self, text):
        doc_vec = np.zeros(self.vectorizer.vector_size)
        word_count = 0
        for word in text:
            if word in self.vectorizer.wv:
                doc_vec += self.vectorizer.wv[word]
                word_count += 1
        if word_count > 0:
            doc_vec /= word_count
        return doc_vec

    def encode_batch(self, texts):
        return np.array([self.encode(text) for text in texts])

    def getDict(self):
        return self.vectorizer.wv.key_to_index
    
    def __repr__(self):
        return f"VectorizerWord2Vec_{self.max_features}"