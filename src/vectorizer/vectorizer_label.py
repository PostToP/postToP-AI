import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from vectorizer.vectorizer import IVectorizer


class VectorizerLabel(IVectorizer):
    def __init__(self) -> None:
        self.vectorizer = MultiLabelBinarizer()

    def train(self, dataset: list[list[str]]) -> None:
        self.vectorizer.fit(dataset)

    def encode(self, text: list[str]) -> np.ndarray:
        text = [text]
        return self.vectorizer.transform(text)[0]

    def encode_batch(self, texts: list[list[str]]) -> list[list[int]]:
        return self.vectorizer.transform(texts)

    def get_dict(self) -> dict[str, int]:
        voc = self.vectorizer.classes_
        voc = dict(zip(voc, range(len(voc))))
        return voc

    def __repr__(self) -> str:
        return "VectorizerLabel"
