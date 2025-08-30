from nltk.tokenize import word_tokenize
from tokenizer.ITokenizer import ITokenizer
from nltk import download as nltk_download
nltk_download('punkt_tab')

class TokenizerWord(ITokenizer):
    def encode(self, text):
        return word_tokenize(text)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
    
    def __repr__(self):
        return "TokenizerWord"