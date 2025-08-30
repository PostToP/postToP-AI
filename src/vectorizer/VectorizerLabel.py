from sklearn.preprocessing import MultiLabelBinarizer
from vectorizer.IVectorizer import IVectorizer

class VectorizerLabel(IVectorizer):
    def __init__(self):
        self.vectorizer = MultiLabelBinarizer()

    def train(self, dataset):
        self.vectorizer.fit(dataset)

    def encode(self, text):
        text = [text]
        return self.vectorizer.transform(text)[0]

    def encode_batch(self, texts):
        return self.vectorizer.transform(texts)

    def getDict(self):
        voc = self.vectorizer.classes_
        voc = dict(zip(voc, range(len(voc))))
        return voc
    
    def __repr__(self):
        return f"VectorizerLabel"