import os
import platform
import pandas as pd
import string
import re
from bs4 import BeautifulSoup
from markdown import markdown
from pathlib import Path


def get_paths():
    """Get input and output directories from environment variables or default to system-specific paths."""
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR")

    if not input_directory or not output_directory:
        system_name = platform.system()
        print(f"Detected OS: {system_name}")

        if system_name == "Linux":
            input_directory = "/app/resources/tsdetect/test_smell_flink"
            output_directory = "/app/resources/tsdetect/test_smell_flink"
        elif system_name == "Darwin":  # macOS
            input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
            output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        else:
            raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


def clean_md_to_text(text: str):
    """Convert markdown to plain text."""
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.findAll(name="comment"):
        comment.decompose()
    return soup.get_text()


def delete_sentences(text, sentences):
    for pattern in sentences:
        text = re.sub(pattern, "", text, flags=re.MULTILINE).strip()
    return text


def get_and_clean_data(data, sentences, save_path):
    description = data['title_n_body'].dropna()

    def clean_text(s):
        s = delete_sentences(s, sentences)
        s = clean_md_to_text(s)
        s = re.sub(r"(<!--.*?-->)", " ", s, flags=re.DOTALL)
        s = re.sub(r"(https?://\S+)", " ", s)
        s = re.sub(r"(.*?)… …(.*)", r"\1\2", s)
        s = re.sub(r"(…)", " ", s)
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
        return ' '.join(w.strip() for w in s.split())

    cleaned_description = description.apply(clean_text)
    cleaned_description = pd.Series(cleaned_description, name='cleaned_title_n_body')
    cleaned_description = pd.concat([data, cleaned_description], axis=1)

    output_path = save_path / "flink_clean_description.pkl"
    cleaned_description['pull_number'] = cleaned_description['url'].apply(lambda url: url.rstrip('/').split('/')[-1])
    cleaned_description.to_pickle(output_path)

    return cleaned_description


if __name__ == "__main__":
    input_directory, output_directory = get_paths()
    input_path = input_directory / "flink_use_for_run_pre_process.pkl"

    try:
        data = pd.read_pickle(input_path)
    except Exception as e:
        print(f"Error loading data from {input_path}: {e}")
        raise

    sentences_for_flink = [
        r"What is the purpose of the change[\?\:]?",
        r"Brief change log[\?\:]?",
        r"Verifying this change[\?\:]?",
        r"Does this pull request potentially affect one of the following parts[\?\:]?",
        r"Documentation[\?\:]?",
        r"Does this pull request introduce a new feature[\?\:]?",
        r"If yes, how is the feature documented[\?\:]?",
        r"Make sure that the pull request corresponds to a \[JIRA issue\]",
        r"Name the pull request in the form \"\[FLINK-XXXX\] \[component\] Title of the pull request\"",
    ]

    try:
        cleaned_description = get_and_clean_data(data, sentences_for_flink, output_directory)
        print(f"Processing complete. Output saved to {output_directory}/flink_clean_description.pkl")
    except Exception as e:
        print(f"Error cleaning data: {e}")
        raise
