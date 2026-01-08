import multiprocessing

import numpy as np
from gensim.models import Word2Vec

from vectorizer.vectorizer import IVectorizer


class VectorizerWord2Vec(IVectorizer):
    def __init__(self, max_features: int = 300) -> None:
        self.output_dim = max_features

    def train(self, dataset: list[list[str]]) -> None:
        max_workers = multiprocessing.cpu_count()
        self.vectorizer = Word2Vec(
            sentences=dataset,
            vector_size=self.output_dim,
            window=3,
            min_count=1,
            workers=max_workers,
        )

    def encode(self, text: list[str]) -> np.ndarray:
        doc_vec = np.zeros(self.vectorizer.vector_size)
        word_count = 0
        for word in text:
            if word in self.vectorizer.wv:
                doc_vec += self.vectorizer.wv[word]
                word_count += 1
        if word_count > 0:
            doc_vec /= word_count
        return doc_vec

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        return np.array([self.encode(text) for text in texts])

    def get_dict(self) -> dict[str, int]:
        return self.vectorizer.wv.key_to_index

    def __repr__(self) -> str:
        return f"VectorizerWord2Vec_{self.max_features}"

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "vectorizer")
