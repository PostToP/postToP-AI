import numpy as np
from sentence_transformers import SentenceTransformer

from vectorizer.vectorizer import IVectorizer


class VectorizerGloVe(IVectorizer):
    def __init__(self) -> None:
        self.vectorizer = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d")
        self.output_dim = 300

    def train(self, dataset: list[list[str]]) -> None:
        pass

    def encode(self, text: list[str]) -> np.ndarray:
        if len(text) <= 1:
            return np.zeros(300)
        text = " ".join(text)
        text = [text]
        return self.vectorizer.encode(text)[0]

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        return np.array([self.encode(text) for text in texts])

    def get_dict(self) -> dict[str, int]:
        pass

    def __repr__(self) -> str:
        return "VectorizerGloVe"
