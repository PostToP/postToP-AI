from tokenizer.tokenizer import ITokenizer


class TokenizerWhitespace(ITokenizer):
    def encode(self, text: str) -> list[str]:
        return text.split()

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        return [text.split() for text in texts]

    def __repr__(self) -> str:
        return "TokenizerWhitespace"

    def __sklearn_is_fitted__(self) -> bool:
        return True
