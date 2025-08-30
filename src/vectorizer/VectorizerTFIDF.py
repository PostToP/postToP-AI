from sklearn.feature_extraction.text import TfidfVectorizer
from vectorizer.IVectorizer import IVectorizer

class VectorizerTFIDF(IVectorizer):
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda doc: doc, preprocessor=lambda doc: doc, lowercase=False, max_features=max_features)

    def train(self, dataset):
        self.vectorizer.fit(dataset)

    def encode(self, text):
        text = [text]
        return self.vectorizer.transform(text).toarray()[0]

    def encode_batch(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def getDict(self):
        voc = self.vectorizer.vocabulary_
        voc = dict(sorted(voc.items(), key=lambda item: item[1]))
        return voc
    
    def __repr__(self):
        return f"VectorizerTFIDF_{len(self.vectorizer.vocabulary_)}"