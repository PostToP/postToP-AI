from tokenizer.tokenizer import ITokenizer


class TokenizerNone(ITokenizer):
    def encode(self, text: str) -> str:
        return text

    def encode_batch(self, texts: list[str]) -> list[str]:
        return texts

    def __repr__(self) -> str:
        return "TokenizerNone"

    def __sklearn_is_fitted__(self) -> bool:
        return True
