class Pipeline:
    def __init__(self):
        self.tokenizer = None
        self.vectorizer = None

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def train(self, data):
        if self.tokenizer:
            self.tokenizer.train(data)
            data = self.tokenizer.encode_batch(data)
        if self.vectorizer:
            self.vectorizer.train(data)

    def process(self, data):
        if self.tokenizer:
            data = self.tokenizer.encode(data)
        if self.vectorizer:
            data = self.vectorizer.encode(data)
        return data

    def process_batch(self, data):
        if self.tokenizer:
            data = self.tokenizer.encode_batch(data)
        if self.vectorizer:
            data = self.vectorizer.encode_batch(data)
        return data

    def train_and_process(self, train, val=None, test=None):
        self.train(train)
        train_processed = self.process_batch(train)
        val_processed = self.process_batch(val) if val is not None else None
        test_processed = self.process_batch(test) if test is not None else None
        return train_processed, val_processed, test_processed

    def __repr__(self):
        return f"Pipeline(tokenizer={self.tokenizer}, vectorizer={self.vectorizer})"
