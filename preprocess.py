import spacy
from re import sub,findall
from unicodedata import normalize,combining
from ftfy import fix_text
from unidecode import unidecode
from tldextract import extract
import contractions
from bs4 import BeautifulSoup
import tensorflow as tf
import numpy as np
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

def detect_and_translate(text):
    def convert_to_chunks(text,size):
      size = size-1
      chunks = []
      while len(text) > size:
          split_index = text[:size].rfind(' ')
          if split_index == -1:
              split_index = size
          chunks.append(text[:split_index])
          text = text[split_index:].strip()
      chunks.append(text)
      return chunks
    if len(text) >= 5000:
        chunks = convert_to_chunks(text,5000)
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        translated = " ".join(translated_chunks)
        return translated
    translated = translator.translate(text)
    return translated


def pp_remove_escaped_characters(text):
  text = sub(r'\n', ' ', text)
  text = sub(r'\t', ' ', text)
  text = sub(r'\r', ' ', text)
  text = sub(r'\"', ' ', text)
  return text


def pp_remove_html(text):
  soup = BeautifulSoup(text, "html.parser")
  return soup.get_text()


def pp_remove_fancy_text(text):
    text = fix_text(text)
    normalized_text = normalize("NFKD", text)
    ascii_text = unidecode("".join([c for c in normalized_text if not combining(c)]))
    return ascii_text


def get_second_level_domain(url):
    extracted = extract(url)
    return extracted.domain


def pp_replace_links_with_sld(text):
    urls = findall(r'(https?://[^\s]+)', text)
    urls_starting_with_www = findall(r'www\.[^\s]+', text)
    urls = urls + urls_starting_with_www
    for url in urls:
      sld = get_second_level_domain(url)
      text = text.replace(url, sld)
    return " "+text+" "

def pp_to_lower(text):
  return text.lower()




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

def pp_expand_contractions(text):
  return contractions.fix(text)


def pp_remove_hyphen(text):
  return sub(r'(?<=\w)-(?=\w)', '', text)


def pp_collapse_whitespaces(text):
  return sub(r'\s+', ' ', text).strip()


def pp_lemmatize_text(text):
    doc = nlp(text)
    lemmatized_sentence = [token.lemma_ for token in doc]
    return " ".join(lemmatized_sentence)


def pp_remove_consecutive_duplicates(text):
  words = text.split()
  new_words = []
  for i in range(len(words)):
    if i == 0 or words[i] != words[i - 1]:
      new_words.append(words[i])

  return ' '.join(new_words)

def preprocess(text):
    text = pp_remove_escaped_characters(text)
    text = pp_remove_html(text)
    text = pp_remove_fancy_text(text)
    text = pp_replace_links_with_sld(text)
    text = pp_to_lower(text)
    text = pp_expand_contractions(text)
    text = pp_remove_hyphen(text)
    text = pp_collapse_whitespaces(text)
    text = pp_lemmatize_text(text)
    text = pp_remove_consecutive_duplicates(text)
    return text