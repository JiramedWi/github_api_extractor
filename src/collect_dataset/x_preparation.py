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
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
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
    for pattern in sentences:
        # Use re.MULTILINE to match lines correctly
        text = re.sub(pattern, "", text, flags=re.MULTILINE).strip()
    return text


## Using to clean template sentense in markdown format
def get_and_clean_data(data, sentences, save_path):
    description = data['title_n_body'].dropna()

    def clean_text(s):
        s = delete_sentences(s, sentences)  # Remove templates first
        s = clean_md_to_text(s)  # Convert Markdown after templates are gone
        s = re.sub(r"(<!--.*?-->)", " ", s, flags=re.DOTALL)  # Remove HTML comments
        s = re.sub(r"(https?://\S+)", " ", s)  # Remove URLs
        s = re.sub(r"(.*?)… …(.*)", r"\1\2", s)  # Remove ellipses
        s = re.sub(r"(…)", " ", s)
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
        s = ' '.join(w.strip() for w in s.split())
        return s

    cleaned_description = description.apply(clean_text)
    cleaned_description = pd.Series(cleaned_description, name='cleaned_title_n_body')
    cleaned_description = pd.concat([data, cleaned_description], axis=1)

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
        sentences_hive = [
            r"What changes were proposed in this pull request[\?\:]?",
            r"Why are the changes needed[\?\:]?",
            r"Does this PR introduce any user-facing change[\?\:]?",
            r"How was this patch tested[\?\:]?"
        ]

        sentences_for_flink = [
            r"What is the purpose of the change[\?\:]?",
            r"Brief change log[\?\:]?",
            r"Verifying this change[\?\:]?",
            r"Does this pull request potentially affect one of the following parts[\?\:]?",
            r"Dependencies \(does it add or upgrade a dependency\):?",
            r"The public API, i\.e\., is any changed class annotated with `@Public\(Evolving\)`[\?\:]?",
            r"The serializers[\?\:]?",
            r"The runtime per-record code paths \(performance sensitive\)[\?\:]?",
            r"Anything that affects deployment or recovery: JobManager \(and its components\), Checkpointing, Kubernetes/Yarn, ZooKeeper[\?\:]?",
            r"The S3 file system connector[\?\:]?",
            r"Documentation[\?\:]?",
            r"Does this pull request introduce a new feature[\?\:]?",
            r"If yes, how is the feature documented[\?\:]?",
            r"Make sure that the pull request corresponds to a \[JIRA issue\]",
            r"Name the pull request in the form \"\[FLINK-XXXX\] \[component\] Title of the pull request\"",
            r"Fill out the template below to describe the changes contributed by the pull request",
            r"Make sure that the change passes the automated tests, i\.e\., `mvn clean verify` passes",
            r"Each pull request should address only one issue, not mix up code from multiple issues",
            r"Each commit in the pull request has a meaningful commit message \(including the JIRA id\)",
            r"Once all items of the checklist are addressed, remove the above text and this checklist, leaving only the filled out template below",
            r"The sections below can be removed for hotfixes of typos",
            r"This change is already covered by existing tests, such as \*\(please describe tests\)\*",
        ]

    except Exception as e:
        print(f"Error defining sentences: {e}")
        raise

    # Clean the data
    try:
        cleaned_description = get_and_clean_data(data, sentences_for_flink, output_directory)
    except Exception as e:
        print(f"Error cleaning data: {e}")
        raise
