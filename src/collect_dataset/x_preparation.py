import platform

import pandas as pd
import string
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from markdown import markdown
import numpy as np
from pathlib import Path
from textblob import TextBlob
import spacy
import os

def get_default_paths():
    system_name = platform.system()
    if system_name == "Linux":
        input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
    elif system_name == "Darwin":  # macOS
        input_directory = "/path/to/mac/input"
        output_directory = "/path/to/mac/output"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return input_directory, output_directory


## For clean text which is not in markdown format
def clean_md_to_text(text: str):
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")
    # Remove comment tags
    for comment in soup.findAll(name="comment"):
        comment.decompose()
    text = soup.get_text()
    return text

## Using to clean template sentense in markdown format
def delete_sentences(text, sentences):
    # Remove all occurrences of the sentences from the text
    for sentence in sentences:
        text = re.sub(sentence, "", text)
    return text

## Using to clean template sentense in markdown format
def get_and_clean_data(data, sentenses, save_path):
    # Drop rows with missing values in 'title_n_body'
    description = data['title_n_body'].dropna()

    # Pre-compile regex patterns for efficiency
    html_comment_regex = re.compile(r"(<!--.*?-->)", flags=re.DOTALL)
    link_regex = re.compile(r"(https?://\S+)")
    ellipsis_regex = re.compile(r"(.*?)… …(.*)")
    single_ellipsis_regex = re.compile(r"(…)")

    # Define transformations
    def clean_text(s):
        # Convert Markdown to plain text
        s = clean_md_to_text(str(s))
        # Delete long sentences
        s = delete_sentences(s, sentenses)
        # Remove HTML-style comments
        s = html_comment_regex.sub(" ", s)
        # Convert to lowercase
        s = s.lower()
        # Merge title and body content, remove ellipses
        s = ellipsis_regex.sub(r"\1\2", s)
        s = single_ellipsis_regex.sub(" ", s)
        # Remove links
        s = link_regex.sub(" ", s)
        # Remove punctuation
        s = s.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
        # Remove excess whitespace
        s = s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
        s = ' '.join(w.strip() for w in s.split())
        return s

    # Apply cleaning transformations
    cleaned_description = description.apply(clean_text)
    cleaned_description = pd.Series(cleaned_description, name='cleaned_title_n_body')
    cleaned_description = pd.concat([data, cleaned_description], axis=1)

    # Save cleaned data to a pickle file
    output_path = Path(save_path + "/flink_clean_description.pkl")
    cleaned_description['pull_number'] = cleaned_description['url'].apply(lambda url: url.rstrip('/').split('/')[-1])
    cleaned_description.to_pickle(output_path)

    return cleaned_description


## Main
if __name__ == "__main__":
    # Load the data
    try:
        input_directory, output_directory = get_default_paths()
        input_path = Path(input_directory + "/flink_use_for_run_pre_process.pkl")
        data = pd.read_pickle(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # Define sentences to remove
    try:
        sentences = ["What changes were proposed in this pull request\?",
                     "Why are the changes needed\?",
                     "Does this PR introduce any user-facing change\?",
                     "How was this patch tested\?"]
    except Exception as e:
        print(f"Error defining sentences: {e}")
        raise

    # Clean the data
    try:
        cleaned_description = get_and_clean_data(data, sentences, output_directory)
    except Exception as e:
        print(f"Error cleaning data: {e}")
        raise
