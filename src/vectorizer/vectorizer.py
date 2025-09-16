class IVectorizer:
    output_dim: int = 0

    def train(self, text: list[str]) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        pass

    def get_dict(self) -> dict[str, int]:
        pass

    def __repr__(self) -> str:
        pass
