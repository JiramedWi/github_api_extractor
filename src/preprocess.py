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


def clean_md_to_text(text):
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")
    # Remove comment tags
    for comment in soup.findAll(name="comment"):
        comment.decompose()
    text = soup.get_text()
    return text


def delete_sentences(text, sentences):
    # Remove all occurrences of the sentences from the text
    for sentence in sentences:
        text = re.sub(sentence, "", text)
    return text


sentences = ["What changes were proposed in this pull request\?",
             "Why are the changes needed\?",
             "Does this PR introduce any user-facing change\?",
             "How was this patch tested\?"]

sample_text_md = "JIRA link : https://issues.apache.org/jira/browse/HIVE-27090\r\n\r\nThis test was newly added in https://issues.apache.org/jira/browse/HIVE-20651 but the test has continuously failed on branch-3. Needs a fix. No other change had been committed to this test after this ticket. Since the ticket was merged in branch-3 without validating the test, fixing it now.\r\n\r\n<!--\r\nThanks for sending a pull request!  Here are some tips for you:\r\n  1. If this is your first time, please read our contributor guidelines: https://cwiki.apache.org/confluence/display/Hive/HowToContribute\r\n  2. Ensure that you have created an issue on the Hive project JIRA: https://issues.apache.org/jira/projects/HIVE/summary\r\n  3. Ensure you have added or run the appropriate tests for your PR: \r\n  4. If the PR is unfinished, add '[WIP]' in your PR title, e.g., '[WIP]HIVE-XXXXX:  Your PR title ...'.\r\n  5. Be sure to keep the PR description updated to reflect all changes.\r\n  6. Please write your PR title to summarize what this PR proposes.\r\n  7. If possible, provide a concise example to reproduce the issue for a faster review.\r\n\r\n-->\r\n\r\n### What changes were proposed in this pull request?\r\n<!--\r\nPlease clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue. \r\nIf possible, please consider writing useful notes for better and faster reviews in your PR. See the examples below.\r\n  1. If you refactor some codes with changing classes, showing the class hierarchy will help reviewers.\r\n  2. If you fix some SQL features, you can provide some references of other DBMSes.\r\n  3. If there is design documentation, please add the link.\r\n  4. If there is a discussion in the mailing list, please add the link.\r\n-->\r\n\r\n\r\n### Why are the changes needed?\r\n<!--\r\nPlease clarify why the changes are needed. For instance,\r\n  1. If you propose a new API, clarify the use case for a new API.\r\n  2. If you fix a bug, you can clarify why it is a bug.\r\n-->\r\n\r\n\r\n### Does this PR introduce _any_ user-facing change?\r\n<!--\r\nNote that it means *any* user-facing change including all aspects such as the documentation fix.\r\nIf yes, please clarify the previous behavior and the change this PR proposes - provide the console output, description, screenshot and/or a reproducable example to show the behavior difference if possible.\r\nIf possible, please also clarify if this is a user-facing change compared to the released Hive versions or within the unreleased branches such as master.\r\nIf no, write 'No'.\r\n-->\r\n\r\n\r\n### How was this patch tested?\r\n<!--\r\nIf tests were added, say they were added here. Please make sure to add some test cases that check the changes thoroughly including negative and positive cases if possible.\r\nIf it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.\r\nIf tests were not added, please describe why they were not added and/or why it was difficult to add.\r\n-->\r\n"
sample_text_sentences = "### What changes were proposed in this pull request?\r\nfix data correctness bug when query " \
                        "on map in orc table in vectorization mode.\r\n\r\n### Why are the changes needed?\r\nto fix " \
                        "data correctness bug\r\n\r\n### Does this PR introduce _any_ user-facing " \
                        "change?\r\nNo\r\n\r\n### How was this patch tested?\r\nTested with qtest and locally"

no_md = clean_md_to_text(sample_text_sentences)
output_no_sentences = delete_sentences(no_md, sentences)

import numpy as np
from pathlib import Path
import os


# cleaned_description = description.apply(
#         lambda s: s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' ' * 4,
#                                                                                                         ' ').replace
#         (' ' * 3, ' ').replace(' ' * 2, ' ').strip())

def get_and_clean_data(data):
    # description = data['body'].dropna()
    description = data['title_n_body'].dropna()
    # Convert .md to text only
    description = description.apply(lambda s: clean_md_to_text(s))
    # Delete long sentences in text
    description = description.apply(lambda s: delete_sentences(s, sentences))
    # Covert to lower
    cleaned_description = description.str.lower()
    # Connect title and body together
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r"(.*?)… …(.*)", r"\1\2", s))
    # Clean links
    link_regex = r"(https?://\S+)"
    cleaned_description = cleaned_description.apply(lambda s: re.sub(link_regex, " ", s))
    # Delete the number
    # cleaned_description = cleaned_description.apply(lambda s: ' '.join(re.split('(-?\d+\.?\d*)', s)))
    # cleaned_description = cleaned_description.apply(lambda s: ''.join(i for i in s if not i.isdigit()))
    # Remove punctuation
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    # Only keep English characters
    # cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^\w\s]+', '', s))
    # Delete white space and join white space
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.apply(lambda s: ' '.join(w.strip() for w in s.split()))
    cleaned_description.to_pickle(Path(os.path.abspath('../resources/clean_demo.pkl')))
    return cleaned_description


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def preProcess_porterstemmer(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def preProcess_lemmetizer(s):
    s = word_tokenize(s)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    tags = nltk.pos_tag(s)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tags))
    s = [lemmatizer.lemmatize(word) if tag == 'n' or tag == 'v' else None for word, tag in wordnet_tagged]
    s = list(filter(None, s))
    s = [w for w in s if w not in stop_dict]
    s = ' '.join(s)
    return s


def preProcess_lemmetizer_without_cleantags(s):
    s = word_tokenize(s)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    s = [lemmatizer.lemmatize(word) for word in s]
    s = [w for w in s if w not in stop_dict]
    s = ' '.join(s)
    return s


sample_text = "Jiramed is a totally new kind of learning experience kites dogs babies feet"
sample_text1 = preProcess_lemmetizer(sample_text)

hive_request = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
df = get_and_clean_data(hive_request)
df = pd.concat([hive_request, df], axis=1)
