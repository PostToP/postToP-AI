import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from vectorizer.vectorizer import IVectorizer


class VectorizerTFIDF(IVectorizer):
    def __init__(self, max_features: int = 5000) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            lowercase=False,
            max_features=max_features,
        )
        self.output_dim = max_features

    def train(self, dataset: list[list[str]]) -> None:
        self.vectorizer.fit(dataset)

    def encode(self, text: list[str]) -> np.ndarray:
        text = [text]
        return self.vectorizer.transform(text).toarray()[0]

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def get_dict(self) -> dict[str, int]:
        voc = self.vectorizer.vocabulary_
        voc = dict(sorted(voc.items(), key=lambda item: item[1]))
        return voc

    def __repr__(self) -> str:
        return f"VectorizerTFIDF_{len(self.vectorizer.vocabulary_)}"

    def __sklearn_is_fitted__(self) -> bool:
        return len(self.vectorizer.vocabulary_) > 0
