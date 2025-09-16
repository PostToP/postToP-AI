from nltk import download as nltk_download
from nltk.tokenize import word_tokenize

from tokenizer.tokenizer import ITokenizer

nltk_download("punkt_tab")


class TokenizerWord(ITokenizer):
    def encode(self, text: str) -> list[str]:
        return word_tokenize(text)

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.encode(text) for text in texts]

    def __repr__(self) -> str:
        return "TokenizerWord"
