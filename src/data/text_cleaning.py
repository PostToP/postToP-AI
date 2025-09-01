import multiprocessing
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
import random
from nltk.corpus import words, wordnet, stopwords
from nltk import download as nltk_download
from tldextract import extract
from unidecode import unidecode
from ftfy import fix_text
from unicodedata import normalize, combining
from functools import partial
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk_download('words', quiet=True)
nltk_download('stopwords', quiet=True)
nltk_download('punkt', quiet=True)
nltk_download('wordnet', quiet=True)

# Define custom contractions
contractions.add("feat", "featuring")
contractions.add("ft", "featuring")
contractions.add("mv", "music video")
contractions.add("ost", "original soundtrack")
contractions.add("yt", "youtube")
contractions.add("fb", "facebook")
contractions.add("hq", "high quality")
contractions.add("q", "quality")
contractions.add("hd", "high definition")
contractions.add("bpm", "beats per minute")
contractions.add("p", "producer")
contractions.add("prod", "producer")
contractions.add("vo", "vocalist")

# Define stop words
stop_words = set(stopwords.words('english'))
additional_stopwords = [
    "youtube", "video", "discord", "link", "about", "follow", "subscribe",
    "twitter", "like", "channel", "twitch", "instagram", "tiktok", "social"
]
stop_words.update(additional_stopwords)

# Define useful words for artist name filtering
useful_words = ["music", "production", "nightcore", "records", "band", "dj", "feat", "p", "utau", "asmr", "vevo", "english", "lofi", "meme", "vlog", "vod", "vods", "feat", "studio", "asmr", "english",
                "online", "playlist", "google", "pc", "software", "programming", "mp3", "cd", "dj", "lofi", "linux", "internet", "genshin", "vods", "vod", "3d", "p", "vtuber", "uploading", "demo", "ch", "lil", "tv"]

artists = ["ado", "eve", "utsu"]


class TextPreprocessor:
    """Text preprocessing utilities

    This class contains static methods for cleaning and processing text data.
    """

    @staticmethod
    def clean_categories(categories: list[str]) -> list[str]:
        """Extract category names from Wikipedia URLs

        Args:
            categories (list[str]): List of Wikipedia URLs

        Returns:
            list[str]: List of category names
        """
        return [x.replace("https://en.wikipedia.org/wiki/", "").replace("_", " ") for x in categories]

    @staticmethod
    def convert_duration(duration: str) -> int:
        """Convert ISO 8601 duration to seconds

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
        """Remove escaped characters from text

        Args:
            text (str): Input text

        Returns:
            str: Text with escaped characters removed
        """
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\r', ' ', text)
        text = re.sub(r'\"', ' ', text)
        return text

    @staticmethod
    def normalize_text_to_ascii(text: str) -> str:
        """Convert fancy unicode characters from text to ASCII

        Args:
            text (str): Input text

        Returns:
            str: Text with fancy unicode characters changed to ASCII"""
        text = fix_text(text)
        normalized_text = normalize("NFKD", text)
        ascii_text = unidecode(
            "".join([c for c in normalized_text if not combining(c)]))
        return ascii_text

    @staticmethod
    def __get_second_level_domain(url: str) -> str:
        """Extract the second-level domain from a URL

        Args:
            url (str): Input URL

        Returns:
            str: Second-level domain
        """
        extracted = extract(url)
        return extracted.domain

    @staticmethod
    def replace_links_with_sld(text: str) -> str:
        """Replace links in text with their second-level domain

        Args:
            text (str): Input text

        Returns:
            str: Text with links replaced by second-level domains
        """
        urls = re.findall(
            r'(https?:[ |\n]?\/\/[ |\n]?[^\s]+)', text, flags=re.IGNORECASE)
        urls_starting_with_www = re.findall(
            r'www\.[^\s]+', text, flags=re.IGNORECASE)
        urls = urls + urls_starting_with_www
        urls = [(x, "".join(x.split(" "))) for x in urls]
        for intext, url in urls:
            sld = TextPreprocessor.__get_second_level_domain(url)
            text = text.replace(intext, sld)
        return " " + text + " "

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text

        Args:
            text (str): Input text

        Returns:
            str: Text with email addresses removed
        """
        EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.sub(EMAIL_REGEX, " ", text)

    @staticmethod
    def remove_ats(text: str) -> str:
        """Remove mentions from text

        Args:
            text (str): Input text

        Returns:
            str: Text with mentions removed
        """
        return re.sub(r'@\S+', ' ', text)

    @staticmethod
    def lemmatize_text(text: str) -> str:
        """Lemmatize text using NLTK

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
        """Lemmatize a list of texts

        Args:
            texts (list[str]): List of input texts

        Returns:
            list[str]: List of lemmatized texts
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_texts = []
        
        for text in texts:
            tokens = word_tokenize(text)
            lemmatized_texts.append(" ".join([lemmatizer.lemmatize(token) for token in tokens]))
        
        return lemmatized_texts

    @staticmethod
    def get_artist_names(descriptions, channel_names):
        """#TODO"""
        yt_generated_regex = r"Provided to YouTube by (?P<company>.+)\n\n(?P<title>[^·]+) · (?P<authors>.+)\n\n(?P<album>.*)|Producer:\s(?P<producer>.+)|Composer:\s(?P<composer>.+)"
        artists = set()

        for x in descriptions:
            if 'Auto-generated by YouTube.'.lower() not in x.lower():
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

        def is_english_word(word):
            if word.lower() in english_words:
                return True
            lemma = TextPreprocessor.lemmatize_text(word)
            if lemma.lower() in english_words:
                return True
            return False

        cleaned_artists = set()
        for x in list(artists):
            x = re.sub(
                r'[^a-zA-Z0-9\s\u4e00-\u9fff\uac00-\ud7af\u0900-\u097f\u0a00-\u0a7f\u0b00-\u0b7f\u0c00-\u0c7f\u0d00-\u0d7f\u0e00-\u0e7f]', ' ', x)
            x = re.sub(r'\s+', ' ', x)
            x = re.sub(r' - Topic', ' ', x)
            x = x.lower()
            x = x.strip()
            cleaned_artists.add(x)

        filtered_artist_names = set()
        for x in cleaned_artists:
            for w in x.split():
                if not is_english_word(w):
                    filtered_artist_names.add(w)

        filtered_artist_names = filtered_artist_names - set(useful_words)
        filtered_artist_names.update(artists)
        filtered_artist_names.discard("")
        return filtered_artist_names

    @staticmethod
    def remove_wordpairs(text, filtered_artist_names):
        pattern = r'\b(?:' + r'|'.join(map(re.escape,
                                           filtered_artist_names)) + r')\b'
        return re.sub(pattern, '', text, flags=re.IGNORECASE)

    @staticmethod
    def to_lower(text: str) -> str:
        """Convert text to lowercase

        Args:
            text (str): Input text

        Returns:
            str: Text converted to lowercase
        """
        return text.lower()

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand contractions in text

        Args:
            text (str): Input text

        Returns:
            str: Text with contractions expanded
        """
        return contractions.fix(text)

    @staticmethod
    def remove_hyphen(text: str) -> str:
        """Remove hyphens between words

        Args:
            text (str): Input text

        Returns:
            str: Text with hyphens removed
        """
        return re.sub(r'(?<=\w)-(?=\w)', '', text)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Remove non-ASCII characters and numbers from text

        Args:
            text (str): Input text

        Returns:
            str: Text with special characters removed
        """
        # Keep alphanumeric, spaces, and Japanese characters
        return re.sub(r"[^a-zA-Z\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", " ", text)

    @staticmethod
    def collapse_whitespaces(text: str) -> str:
        """Collapse consecutive whitespaces into a single space

        Args:
            text (str): Input text

        Returns:
            str: Text with consecutive whitespaces collapsed
        """
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def augment_text_with_synonyms(text, percentage):
        words = text.split()
        num_words_to_change = int(len(words) * percentage)
        randomly_selected_words = random.sample(words, num_words_to_change)
        augmented_words = []

        for i, word in enumerate(words):
            if word in randomly_selected_words:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for l in syn.lemmas():
                        synonym = l.name().replace("_", " ").lower()
                        if synonym.isalpha():
                            synonyms.append(synonym)
                if synonyms:
                    new_word = random.choice(synonyms)
                else:
                    new_word = word
                augmented_words.append(new_word)
                continue
            augmented_words.append(word)

        return " ".join(augmented_words).lower()

    @staticmethod
    def remove_stopwords(text: str) -> str:
        """Remove stopwords from text

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed
        """
        word_tokens = text.split(" ")
        filtered_sentence = [
            w for w in word_tokens if not w.lower() in stop_words]
        return " ".join(filtered_sentence)

    @staticmethod
    def remove_consecutive_duplicates(text: str) -> str:
        """Remove consecutive duplicate words from text

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
        return ' '.join(new_words)


class DatasetPreprocessor:
    def __init__(self):
        self.preprocessing_steps = []

    def add_step(self, func):
        """Add a preprocessing step to the pipeline"""
        self.preprocessing_steps.append(func)
        return self

    def process_text_columns(self, df, columns, verbose=False):
        """Apply all preprocessing steps to specified text columns with progress tracking"""
        df_copy = df.copy()

        for i, func in enumerate(self.preprocessing_steps):
            step_name = getattr(func, "__name__", f"Step {i+1}")

            for col in columns:
                if verbose:
                    df_copy[col] = [func(text) for text in tqdm(
                        df_copy[col], desc=f"{step_name} - {col}")]
                else:
                    df_copy[col] = df_copy[col].apply(func)

        return df_copy

    def process_text_columns_multiprocessing(self, df, columns, n_jobs=-1):
        """Apply all preprocessing steps to specified text columns using multiprocessing

        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): List of column names to process
            n_jobs (int): Number of processes to use, -1 for all available

        Returns:
            pd.DataFrame: Processed dataframe
        """

        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()

        df_copy = df.copy()

        for col in columns:
            n_rows = len(df_copy)
            chunk_size = n_rows // n_jobs
            chunks = []

            for i in range(n_jobs):
                start_idx = i * chunk_size
                end_idx = n_rows if i == n_jobs - 1 else (i + 1) * chunk_size
                chunks.append(df_copy[col].iloc[start_idx:end_idx].tolist())

            # Process chunks in parallel
            with multiprocessing.Pool(processes=n_jobs) as pool:
                results = pool.starmap(self._process_chunk, [
                                       (chunk,) for chunk in chunks])

            # Combine results
            processed_col = []
            for chunk_result in results:
                processed_col.extend(chunk_result)

            df_copy[col] = processed_col

        return df_copy

    def _process_chunk(self, chunk):
        """Process a chunk of texts"""
        return [self.process_text(text) for text in chunk]

    def process_text(self, text):
        """Apply all preprocessing steps to a single text"""
        for func in self.preprocessing_steps:
            text = func(text)
        return text

    @staticmethod
    def remove_similar_rows(df, threshold=0.9):
        """Remove rows with similar content based on TF-IDF similarity"""
        vectorizer = TfidfVectorizer()
        title_description = df['Title'] + " " + df['Description']
        tfidf_matrix = vectorizer.fit_transform(title_description)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        rows_to_remove = set()

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] >= threshold:
                    # Skip if any of the values to compare is None
                    if df['Is Music'].iloc[i] is None or df['Is Music'].iloc[j] is None:
                        continue
                    if df['Is Music'].iloc[i] != df['Is Music'].iloc[j]:
                        continue
                    # Use actual index value instead of position
                    rows_to_remove.add(df.index[j])

        new_df = df.drop(index=list(rows_to_remove)).reset_index(drop=True)
        return new_df


def generate_train_preprocess_pipeline(train_df: pd.DataFrame) -> DatasetPreprocessor:
    """Generate a preprocessing pipeline for the training set

        Args:
            train_df (pd.DataFrame): Training dataset

        Returns:
            DatasetPreprocessor: Preprocessing pipeline for the training set
    """
    filtered_artist_names = TextPreprocessor.get_artist_names(
        train_df['Description'],
        train_df['Channel Name']
    )
    preprocessor = DatasetPreprocessor()

    # Add preprocessing steps in sequence
    preprocessor.add_step(TextPreprocessor.remove_escaped_characters)
    preprocessor.add_step(TextPreprocessor.normalize_text_to_ascii)
    preprocessor.add_step(TextPreprocessor.replace_links_with_sld)
    preprocessor.add_step(TextPreprocessor.remove_emails)
    preprocessor.add_step(TextPreprocessor.remove_ats)
    preprocessor.add_step(partial(
        TextPreprocessor.remove_wordpairs, filtered_artist_names=filtered_artist_names))
    preprocessor.add_step(TextPreprocessor.to_lower)
    preprocessor.add_step(TextPreprocessor.expand_contractions)
    preprocessor.add_step(TextPreprocessor.remove_hyphen)
    preprocessor.add_step(TextPreprocessor.remove_special_characters)
    preprocessor.add_step(TextPreprocessor.collapse_whitespaces)
    preprocessor.add_step(TextPreprocessor.lemmatize_text)
    preprocessor.add_step(TextPreprocessor.remove_stopwords)
    preprocessor.add_step(TextPreprocessor.remove_consecutive_duplicates)
    preprocessor.add_step(TextPreprocessor.to_lower)

    return preprocessor


def generate_test_preprocess_pipeline() -> DatasetPreprocessor:
    """Generate a preprocessing pipeline for the test set

        Returns:
            DatasetPreprocessor: Preprocessing pipeline for the test set
    """
    preprocessor = DatasetPreprocessor()

    # Add preprocessing steps in sequence
    preprocessor.add_step(TextPreprocessor.remove_escaped_characters)
    preprocessor.add_step(TextPreprocessor.normalize_text_to_ascii)
    preprocessor.add_step(TextPreprocessor.replace_links_with_sld)
    preprocessor.add_step(TextPreprocessor.remove_emails)
    preprocessor.add_step(TextPreprocessor.remove_ats)
    preprocessor.add_step(TextPreprocessor.to_lower)
    preprocessor.add_step(TextPreprocessor.expand_contractions)
    preprocessor.add_step(TextPreprocessor.remove_hyphen)
    preprocessor.add_step(TextPreprocessor.remove_special_characters)
    preprocessor.add_step(TextPreprocessor.collapse_whitespaces)
    preprocessor.add_step(TextPreprocessor.lemmatize_text)
    preprocessor.add_step(TextPreprocessor.remove_stopwords)
    preprocessor.add_step(TextPreprocessor.remove_consecutive_duplicates)
    preprocessor.add_step(TextPreprocessor.to_lower)

    return preprocessor


def split_dataset(dataset, test_size=0.1):
    column = 'Is Music'
    if column not in dataset.columns:
        column = dataset.columns[-1]
    # Create balanced dataset
    positive_samples = dataset[dataset[column] == 1]
    negative_samples = dataset[dataset[column] == 0]

    min_samples = min(len(positive_samples), len(negative_samples))
    balanced_dataset = pd.concat([
        positive_samples.head(min_samples),
        negative_samples.head(min_samples)
    ])

    # Train-test split
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        balanced_dataset, test_size=test_size, random_state=42)

    # Add overflow samples to test set
    overflow_positive = positive_samples[min_samples:]
    overflow_negative = negative_samples[min_samples:]
    test_df = pd.concat([test_df, overflow_positive, overflow_negative])

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df