from tokenizers import Tokenizer
from vectorizer.IVectorizer import IVectorizer
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer


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