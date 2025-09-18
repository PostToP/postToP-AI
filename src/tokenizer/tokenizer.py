class ITokenizer:
    def train(self, text: list[str]) -> None:
        pass

    def encode(self, text: str) -> list[str]:
        pass

    def encode_batch(self, texts: list[str]) -> list[list[str]]:
        pass

    def fit(self, X: list[str], y: list[str]) -> "ITokenizer":
        self.train(X)
        return self

    def transform(self, X: list[str]) -> list[list[str]]:
        return self.encode_batch(X)

    def __repr__(self) -> str:
        pass
