from tokenizer.ITokenizer import ITokenizer

class TokenizerNone(ITokenizer):
    def encode(self, text):
        return text

    def encode_batch(self, texts):
        return texts
    
    def __repr__(self):
        return "TokenizerNone"