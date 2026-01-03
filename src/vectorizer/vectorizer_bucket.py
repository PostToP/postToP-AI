import numpy as np
from sklearn.preprocessing import OneHotEncoder

from vectorizer.vectorizer import IVectorizer


class VectorizerBucket(IVectorizer):
    def __init__(self, edges: list[int]) -> None:
        self.edges = edges
        self.vectorizer = OneHotEncoder(sparse_output=False)

    def train(self, dataset: list[int]) -> None:
        data = np.array(dataset).reshape(-1, 1)
        buckets = np.digitize(data, bins=self.edges, right=False)
        self.vectorizer.fit(buckets)

    def encode(self, number: int) -> np.ndarray:
        text = np.array([[number]])
        buckets = np.digitize(text, bins=self.edges, right=False)
        return self.vectorizer.transform(buckets)

    def encode_batch(self, numbers: list[int]) -> np.ndarray:
        data = np.array(numbers).reshape(-1, 1)
        buckets = np.digitize(data, bins=self.edges, right=False)
        return self.vectorizer.transform(buckets)

    def get_dict(self) -> dict[str, int]:
        voc = self.vectorizer.vocabulary_
        voc = dict(sorted(voc.items(), key=lambda item: item[1]))
        return voc

    def __repr__(self) -> str:
        return f"VectorizerCount_{len(self.vectorizer.vocabulary_)}"
