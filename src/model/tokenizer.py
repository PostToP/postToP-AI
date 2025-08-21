from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk import download as nltk_download
nltk_download('punkt_tab')


class ITokenizer:
    name = "Tokenizer"

    def train(self, text):
        pass

    def encode(self, text):
        pass

    def encode_batch(self, texts):
        pass


class TokenizerNone(ITokenizer):
    def encode(self, text):
        return text

    def encode_batch(self, texts):
        return texts


class TokenizerWord(ITokenizer):
    name = "TokenizerWord"

    def encode(self, text):
        return word_tokenize(text)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]


class TokenizerNgram(ITokenizer):
    name = "TokenizerNgram"

    def __init__(self, ngram_range=(1, 2)):
        self.name = f"TokenizerNgram_{ngram_range[0]}_{ngram_range[1]}"
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


class TokenizerBPE(ITokenizer):
    name = "TokenizerBPE"

    def __init__(self, vocab_size=10000):
        self.name = f"TokenizerBPE_{vocab_size}"
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, text):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=[
                             "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train_from_iterator(text, trainer=trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).tokens

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
