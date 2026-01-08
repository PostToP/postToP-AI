import numpy as np
from sentence_transformers import SentenceTransformer

from vectorizer.vectorizer import IVectorizer


class VectorizerBert(IVectorizer):
    def __init__(self) -> None:
        self.vectorizer = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        self.output_dim = 384

    def train(self, dataset: list[list[str]]) -> None:
        # BERT models are pre-trained and don't need further training
        pass

    def encode(self, text: list[str]) -> np.ndarray:
        if not text or len(text) <= 1:
            return np.zeros(384)
        # The encode method already handles single texts, no need to create a list
        return self.vectorizer.encode(text)

    def encode_batch(self, texts: list[list[str]]) -> np.ndarray:
        # Filter out empty texts
        valid_texts = [text for text in texts if text and len(text) > 1]

        if not valid_texts:
            # Return empty array with correct shape if no valid texts
            return np.zeros((len(texts), 384))

        # Use the built-in batch encoding capability
        embeddings = self.vectorizer.encode(valid_texts)

        # Create result array with zeros for invalid texts
        result = np.zeros((len(texts), 384))
        valid_idx = 0

        for i, text in enumerate(texts):
            if text and len(text) > 1:
                result[i] = embeddings[valid_idx]
                valid_idx += 1

        return result

    def get_dict(self) -> dict[str, int]:
        voc = self.vectorizer.tokenizer.get_vocab()
        return dict(sorted(voc.items(), key=lambda item: item[1]))

    def __repr__(self) -> str:
        return "VectorizerBert"

    def __sklearn_is_fitted__(self) -> bool:
        return True
