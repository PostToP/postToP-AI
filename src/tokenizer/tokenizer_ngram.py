from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from tokenizer.tokenizer import ITokenizer


class TokenizerNgram(ITokenizer):
    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.ngram_range: tuple[int, int] = ngram_range

    def ngram_tokenize_text_range(self, text: str) -> list[str]:
        tokens: list[str] = word_tokenize(text)
        ngrams_list: list[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams_list.extend([" ".join(gram) for gram in ngrams(tokens, n)])
        return ngrams_list

    def encode(self, text: str) -> list[str]:
        return self.ngram_tokenize_text_range(text)

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.encode(text) for text in texts]

    def __repr__(self) -> str:
        return f"TokenizerNgram_{self.ngram_range[0]}-{self.ngram_range[1]}"

    def __sklearn_is_fitted__(self) -> bool:
        return True
