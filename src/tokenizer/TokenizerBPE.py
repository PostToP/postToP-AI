from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizer.ITokenizer import ITokenizer


class TokenizerBPE(ITokenizer):
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, text):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=[
                             "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(text, trainer=trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).tokens

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
    
    def __repr__(self):
        return f"TokenizerBPE_{self.vocab_size}"