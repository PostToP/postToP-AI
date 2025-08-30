from tokenizer.ITokenizer import ITokenizer

class TokenizerWhitespace(ITokenizer):
    def encode(self, text):
        return text.split()

    def encode_batch(self, texts):
        return [text.split() for text in texts]
    
    def __repr__(self):
        return "TokenizerWhitespace"