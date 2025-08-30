from sentence_transformers import SentenceTransformer
import multiprocessing
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

class IVectorizer:
    def train(self, text):
        pass

    def encode(self, text):
        pass

    def encode_batch(self, texts):
        pass

    def getDict(self):
        pass

    def __repr__(self):
        pass


class VectorizerSequential(IVectorizer):
    def __init__(self, vocab_size=5000, max_sequence_length=100):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.vectorizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    def train(self, dataset):
        trainer = WordLevelTrainer(
            vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]"], show_progress=False)
        self.vectorizer.train_from_iterator(dataset, trainer)

    def encode(self, text):
        sequence = self.vectorizer.encode(text, is_pretokenized=True)
        padded_ids = sequence.ids[:self.max_sequence_length] + \
            [0] * (self.max_sequence_length - len(sequence.ids))
        return padded_ids[:self.max_sequence_length]

    def encode_batch(self, texts):
        sequence = self.vectorizer.encode_batch(texts, is_pretokenized=True)
        padded_ids = [s.ids[:self.max_sequence_length] + [0] * (self.max_sequence_length - len(s.ids)) for s in sequence]
        return np.array([ids[:self.max_sequence_length] for ids in padded_ids])

    def getDict(self):
        return self.vectorizer.__dict__()
    
    def __repr__(self):
        return f"VectorizerSequential_{self.vocab_size}_{self.max_sequence_length}"


class VectorizerCount(IVectorizer):
    def __init__(self, max_features=5000):
        self.vectorizer = CountVectorizer(
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
        return f"VectorizerCount_{len(self.vectorizer.vocabulary_)}"


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


class VectorizerGloVe(IVectorizer):
    def __init__(self):
        self.vectorizer = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d')

    def train(self, dataset):
        pass

    def encode(self, text):
        if len(text) <= 1:
            return np.zeros(300)
        text = " ".join(text)
        text = [text]
        return self.vectorizer.encode(text)[0]

    def encode_batch(self, texts):
        return np.array([self.encode(text) for text in texts])

    def getDict(self):
        return {}
    
    def __repr__(self):
        return "VectorizerGloVe"
