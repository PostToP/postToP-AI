from tokenizers import Tokenizer
from vectorizer.IVectorizer import IVectorizer
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer


class VectorizerSequential(IVectorizer):
    def __init__(self, vocab_size=5000, max_sequence_length=100):
        self.vocab_size = vocab_size
        self.vectorizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.output_dim = max_sequence_length

    def train(self, dataset):
        trainer = WordLevelTrainer(
            vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]"], show_progress=False)
        self.vectorizer.train_from_iterator(dataset, trainer)

    def encode(self, text):
        sequence = self.vectorizer.encode(text, is_pretokenized=True)
        padded_ids = sequence.ids[:self.output_dim] + \
            [0] * (self.output_dim - len(sequence.ids))
        return padded_ids[:self.output_dim]

    def encode_batch(self, texts):
        sequence = self.vectorizer.encode_batch(texts, is_pretokenized=True)
        padded_ids = [s.ids[:self.output_dim] + [0] * (self.output_dim - len(s.ids)) for s in sequence]
        return np.array([ids[:self.output_dim] for ids in padded_ids])

    def getDict(self):
        return self.vectorizer.__dict__()
    
    def __repr__(self):
        return f"VectorizerSequential_{self.vocab_size}_{self.output_dim}"