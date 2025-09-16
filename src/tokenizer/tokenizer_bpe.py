from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from tokenizer.tokenizer import ITokenizer


class TokenizerBPE(ITokenizer):
    def __init__(self, vocab_size: int = 10000) -> None:
        self.vocab_size: int = vocab_size
        self.tokenizer: Tokenizer | None = None

    def train(self, text: list[str]) -> None:
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(text, trainer=trainer)

    def encode(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.encode(text) for text in texts]

    def __repr__(self) -> str:
        return f"TokenizerBPE_{self.vocab_size}"
