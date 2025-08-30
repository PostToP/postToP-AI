from sentence_transformers import SentenceTransformer
import numpy as np
from vectorizer.IVectorizer import IVectorizer

class VectorizerBert(IVectorizer):
    def __init__(self):
        self.vectorizer = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

    def train(self, dataset):
        # BERT models are pre-trained and don't need further training
        pass

    def encode(self, text):
        if not text or len(text) <= 1:
            return np.zeros(384)
        # The encode method already handles single texts, no need to create a list
        return self.vectorizer.encode(text)

    def encode_batch(self, texts):
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

    def getDict(self):
        voc = self.vectorizer.tokenizer.get_vocab()
        return dict(sorted(voc.items(), key=lambda item: item[1]))
    
    def __repr__(self):
        return "VectorizerBert"