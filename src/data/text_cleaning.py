import multiprocessing
import random
import re
from functools import partial
from typing import Callable
from unicodedata import combining, normalize

import contractions
import pandas as pd
from ftfy import fix_text
from nltk import download as nltk_download
from nltk.corpus import stopwords, wordnet, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from tldextract import extract
from tqdm import tqdm
from unidecode import unidecode

from config.config import (
    custom_contractions,
    custom_keepwords,
    custom_stopwords,
)

# Download required NLTK resources
nltk_download("words", quiet=True)
nltk_download("stopwords", quiet=True)
nltk_download("punkt", quiet=True)
nltk_download("wordnet", quiet=True)

# Define custom contractions
for key, value in custom_contractions.items():
    contractions.add(key, value)

# Define stop words
stop_words = set(stopwords.words("english"))
stop_words.update(custom_stopwords)


class TextPreprocessor:
    """Text preprocessing utilities.

    This class contains static methods for cleaning and processing text data.
    """

    @staticmethod
    def clean_categories(categories: list[str]) -> list[str]:
        """Extract category names from Wikipedia URLs.

        Args:
            categories (list[str]): List of Wikipedia URLs

        Returns:
            list[str]: List of category names

        """
        return [x.replace("https://en.wikipedia.org/wiki/", "").replace("_", " ") for x in categories]

    @staticmethod
    def convert_duration(duration: str) -> int:
        """Convert ISO 8601 duration to seconds.

        Args:
            duration (str): ISO 8601 duration string

        Returns:
            int: Duration in seconds

        """
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return 0
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def remove_escaped_characters(text: str) -> str:
        """Remove escaped characters from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with escaped characters removed

        """
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r"\r", " ", text)
        text = re.sub(r"\"", " ", text)
        return text

    @staticmethod
    def normalize_text_to_ascii(text: str) -> str:
        """Convert fancy unicode characters from text to ASCII.

        Args:
            text (str): Input text

        Returns:
            str: Text with fancy unicode characters changed to ASCII

        """
        text = fix_text(text)
        normalized_text = normalize("NFKD", text)
        ascii_text = unidecode(
            "".join([c for c in normalized_text if not combining(c)]),
        )
        return ascii_text

    @staticmethod
    def __get_second_level_domain(url: str) -> str:
        """Extract the second-level domain from a URL.

        Args:
            url (str): Input URL

        Returns:
            str: Second-level domain

        """
        extracted = extract(url)
        return extracted.domain

    @staticmethod
    def replace_links_with_sld(text: str) -> str:
        """Replace links in text with their second-level domain.

        Args:
            text (str): Input text

        Returns:
            str: Text with links replaced by second-level domains

        """
        urls = re.findall(
            r"(https?:[ |\n]?\/\/[ |\n]?[^\s]+)",
            text,
            flags=re.IGNORECASE,
        )
        urls_starting_with_www = re.findall(r"www\.[^\s]+", text, flags=re.IGNORECASE)
        urls = urls + urls_starting_with_www
        urls = [(x, "".join(x.split(" "))) for x in urls]
        for intext, url in urls:
            sld = TextPreprocessor.__get_second_level_domain(url)
            text = text.replace(intext, sld)
        return " " + text + " "

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with email addresses removed

        """
        EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.sub(EMAIL_REGEX, " ", text)

    @staticmethod
    def remove_ats(text: str) -> str:
        """Remove mentions from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with mentions removed

        """
        return re.sub(r"@\S+", " ", text)

    @staticmethod
    def lemmatize_text(text: str) -> str:
        """Lemmatize text using NLTK.

        Args:
            text (str): Input text

        Returns:
            str: Lemmatized text

        """
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        lemmatized_sentence = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_sentence)

    @staticmethod
    def batch_lemmatize(texts: list[str]) -> list[str]:
        """Lemmatize a list of texts.

        Args:
            texts (list[str]): List of input texts

        Returns:
            list[str]: List of lemmatized texts

        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_texts = []

        for text in texts:
            tokens = word_tokenize(text)
            lemmatized_texts.append(
                " ".join([lemmatizer.lemmatize(token) for token in tokens]),
            )

        return lemmatized_texts

    @staticmethod
    def get_artist_names(descriptions: list[str], channel_names: list[str]) -> set[str]:
        """#TODO."""
        yt_generated_regex = r"Provided to YouTube by (?P<company>.+)\n\n(?P<title>[^·]+) · (?P<authors>.+)\n\n(?P<album>.*)|Producer:\s(?P<producer>.+)|Composer:\s(?P<composer>.+)"
        artists = set()

        for x in descriptions:
            if "Auto-generated by YouTube.".lower() not in x.lower():
                continue
            m = re.findall(yt_generated_regex, x)
            if m:
                for x in m:
                    if x[0]:  # company
                        artists.add(x[0])
                    if x[2]:  # authors
                        authors = x[2].split("·")
                        authors = [author.strip() for author in authors]
                        for author in authors:
                            artists.add(author.lower())
                    if x[4]:  # producers
                        artists.add(x[4])
                    if x[5]:  # composers
                        artists.add(x[5])

        for channel in channel_names:
            artists.add(channel)

        english_words = set(words.words())

        def is_english_word(word: str) -> bool:
            if word.lower() in english_words:
                return True
            lemma = TextPreprocessor.lemmatize_text(word)
            return lemma.lower() in english_words

        cleaned_artists = set()
        for x in list(artists):
            text = re.sub(
                r"[^a-zA-Z0-9\s\u4e00-\u9fff\uac00-\ud7af\u0900-\u097f\u0a00-\u0a7f\u0b00-\u0b7f\u0c00-\u0c7f\u0d00-\u0d7f\u0e00-\u0e7f]",
                " ",
                x,
            )
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r" - Topic", " ", text)
            text = text.lower()
            text = text.strip()
            cleaned_artists.add(text)

        filtered_artist_names = set()
        for x in cleaned_artists:
            for w in x.split():
                if not is_english_word(w):
                    filtered_artist_names.add(w)

        filtered_artist_names = filtered_artist_names - set(custom_keepwords)
        filtered_artist_names.update(artists)
        filtered_artist_names.discard("")
        return filtered_artist_names

    @staticmethod
    def remove_wordpairs(text: str, filtered_artist_names: list[str]) -> str:
        pattern = r"\b(?:" + r"|".join(map(re.escape, filtered_artist_names)) + r")\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    @staticmethod
    def to_lower(text: str) -> str:
        """Convert text to lowercase.

        Args:
            text (str): Input text

        Returns:
            str: Text converted to lowercase

        """
        return text.lower()

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand contractions in text.

        Args:
            text (str): Input text

        Returns:
            str: Text with contractions expanded

        """
        result = contractions.fix(text)
        return str(result) if result is not None else ""

    @staticmethod
    def remove_hyphen(text: str) -> str:
        """Remove hyphens between words.

        Args:
            text (str): Input text

        Returns:
            str: Text with hyphens removed

        """
        return re.sub(r"(?<=\w)-(?=\w)", "", text)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Remove non-ASCII characters and numbers from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with special characters removed

        """
        # Keep alphanumeric, spaces, and Japanese characters
        return re.sub(r"[^a-zA-Z\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", " ", text)

    @staticmethod
    def collapse_whitespaces(text: str) -> str:
        """Collapse consecutive whitespaces into a single space.

        Args:
            text (str): Input text

        Returns:
            str: Text with consecutive whitespaces collapsed

        """
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def augment_text_with_synonyms(text: str, percentage: float = 0.1) -> str:
        words = text.split()
        num_words_to_change = int(len(words) * percentage)
        randomly_selected_words = random.sample(words, num_words_to_change)
        augmented_words = []

        for _, word in enumerate(words):
            if word in randomly_selected_words:
                synonyms = []
                for syn in wordnet.synsets(word):
                    if syn is not None:
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace("_", " ").lower()
                            if synonym.isalpha():
                                synonyms.append(synonym)
                new_word = random.choice(synonyms) if synonyms else word
                augmented_words.append(new_word)
                continue
            augmented_words.append(word)

        return " ".join(augmented_words).lower()

    @staticmethod
    def remove_stopwords(text: str) -> str:
        """Remove stopwords from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed

        """
        word_tokens = text.split(" ")
        filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
        return " ".join(filtered_sentence)

    @staticmethod
    def remove_consecutive_duplicates(text: str) -> str:
        """Remove consecutive duplicate words from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with consecutive duplicate words removed

        """
        words = text.split()
        new_words = []
        for i in range(len(words)):
            if i == 0 or words[i] != words[i - 1]:
                new_words.append(words[i])
        return " ".join(new_words)


class DatasetPreprocessor:
    @staticmethod
    def remove_similar_rows(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """Remove rows with similar content based on TF-IDF similarity."""
        vectorizer = TfidfVectorizer()
        title_description = df["Title"] + " " + df["Description"]
        tfidf_matrix = vectorizer.fit_transform(title_description)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        rows_to_remove = set()

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] >= threshold:
                    # Skip if any of the values to compare is None
                    if df["Is Music"].iloc[i] is None or df["Is Music"].iloc[j] is None:
                        continue
                    if df["Is Music"].iloc[i] != df["Is Music"].iloc[j]:
                        continue
                    # Use actual index value instead of position
                    rows_to_remove.add(df.index[j])

        new_df = df.drop(index=list(rows_to_remove)).reset_index(drop=True)
        return new_df


def generate_new_pipeline() -> Pipeline:
    def wrapper(func: Callable) -> Callable:
        return FunctionTransformer(lambda x: x.apply(func), validate=False)

    pipe = Pipeline(
        [
            ("remove_escaped_characters", wrapper(TextPreprocessor.remove_escaped_characters)),
            ("normalize_text_to_ascii", wrapper(TextPreprocessor.normalize_text_to_ascii)),
            ("replace_links_with_sld", wrapper(TextPreprocessor.replace_links_with_sld)),
            ("remove_emails", wrapper(TextPreprocessor.remove_emails)),
            ("remove_ats", wrapper(TextPreprocessor.remove_ats)),
            # todo: remove artist names, maybe
            ("to_lower", wrapper(TextPreprocessor.to_lower)),
            ("expand_contractions", wrapper(TextPreprocessor.expand_contractions)),
            ("remove_hyphen", wrapper(TextPreprocessor.remove_hyphen)),
            ("remove_special_characters", wrapper(TextPreprocessor.remove_special_characters)),
            ("collapse_whitespaces", wrapper(TextPreprocessor.collapse_whitespaces)),
            ("lemmatize_text", wrapper(TextPreprocessor.lemmatize_text)),
            ("remove_stopwords", wrapper(TextPreprocessor.remove_stopwords)),
            ("remove_consecutive_duplicates", wrapper(TextPreprocessor.remove_consecutive_duplicates)),
            ("to_lower2", wrapper(TextPreprocessor.to_lower)),
            ("convert_to_dataframe", FunctionTransformer(lambda x: pd.DataFrame(x))),
        ],
    )
    return pipe


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = "Is Music"
    if column not in dataset.columns:
        column = dataset.columns[-1]
    # Create balanced dataset
    positive_samples = dataset[dataset[column] == 1]
    negative_samples = dataset[dataset[column] == 0]

    min_samples = min(len(positive_samples), len(negative_samples))
    balanced_dataset = pd.concat(
        [positive_samples.head(min_samples), negative_samples.head(min_samples)],
    )

    train_df, test_df = train_test_split(
        balanced_dataset,
        test_size=test_size,
        random_state=42,
    )

    # Add overflow samples to test set
    overflow_positive = positive_samples[min_samples:]
    overflow_negative = negative_samples[min_samples:]
    test_df = pd.concat([test_df, overflow_positive, overflow_negative])

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df
