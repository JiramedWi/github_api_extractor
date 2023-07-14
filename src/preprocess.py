import pandas as pd
import string
import requests
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from pathlib import Path
import os
from get_github_api import github_api


def get_and_clean_data():
    data = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))
    description = data['body'].dropna()
    cleaned_description = description.str.lower()
    cleaned_description = cleaned_description.apply(
            lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.apply(lambda s: ' '.join(re.split('(-?\d+\.?\d*)', s)))
    cleaned_description = cleaned_description.apply(lambda s: ''.join( i for i in s if not i.isdigit()))
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    # cleaned_description = description.apply(
    #         lambda s: s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' ' * 4,
    #                                                                                                         ' ').replace
    #         (' ' * 3, ' ').replace(' ' * 2, ' ').strip())
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.apply(lambda s: ' '.join(w.strip() for w in s.split()))
    cleaned_description.to_pickle(Path(os.path.abspath('../resources/clean_demo.pkl')))
    return cleaned_description


def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


result = get_and_clean_data()
