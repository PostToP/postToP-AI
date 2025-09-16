class ITokenizer:
    def train(self, text: list[str]) -> None:
        pass

    def encode(self, text: str) -> list[str]:
        pass

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        pass

    def __repr__(self) -> str:
        pass
