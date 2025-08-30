from tokenizer.ITokenizer import ITokenizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

class TokenizerNgram(ITokenizer):
    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range

    def ngram_tokenize_text_range(self, text):
        tokens = word_tokenize(text)
        ngrams_list = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams_list.extend([' '.join(gram) for gram in ngrams(tokens, n)])
        return ngrams_list

    def encode(self, text):
        return self.ngram_tokenize_text_range(text)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
    
    def __repr__(self):
        return f"TokenizerNgram_{self.ngram_range[0]}-{self.ngram_range[1]}"