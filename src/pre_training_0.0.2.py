import nltk
import os
import numpy as np
import pandas as pd
import requests
import spacy
import joblib
import inspect
import time

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def pos_tagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def pos_tagger_for_spacy(tag):
    # Mapping NLTK POS tags to spaCy POS tags
    tag_dict = {'N': 'NOUN', 'V': 'VERB', 'R': 'ADV', 'J': 'ADJ'}
    return tag_dict.get(tag, 'n')


nlp = spacy.load("en_core_web_sm")


def pre_process_spacy(s):
    doc = nlp(s)
    s = " ".join([token.lemma_ if token.pos_ in ['NOUN', 'VERB'] else token.text for token in doc if
                  token.pos_ in ['NOUN', 'VERB']])
    return s


def pre_process_textblob(s):
    blob = TextBlob(s)
    # Remove stopwords
    s = [word for word in blob.words if word not in nltk.corpus.stopwords.words('english')]
    s = " ".join(s)
    return s


def pre_process_porterstemmer(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def pre_process_lemmatizer(s):
    s = word_tokenize(s)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    tags = nltk.pos_tag(s)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tags))
    s = [lemmatizer.lemmatize(word, tag) if tag == 'n' or tag == 'v' else None for word, tag in wordnet_tagged]
    s = list(filter(None, s))
    s = [w for w in s if w not in stop_dict]
    s = ' '.join(s)
    return s


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


def scale_sparse_matrix(matrix):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(matrix.toarray())
    return csr_matrix(x_scaled)


def log_transform_tfidf(matrix):
    return np.log1p(matrix)


def set_smote(x_y_fit_blind_transform):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print("start to set smote at: " + start_time_gmt)
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        x_smote, y_smote = smote.fit_resample(x_y_fit_blind_transform_dict['x_fit'],
                                              x_y_fit_blind_transform_dict['y_fit'])
        # Check value is balanced or not
        if x_smote.shape[0] > x_y_fit_blind_transform_dict['x_fit'].shape[0] and y_smote.shape[0] > \
                x_y_fit_blind_transform_dict['y_fit'].shape[0]:
            print("set Balanced: x value old =" + str(x_y_fit_blind_transform_dict['x_fit'].shape[0]) + "x value new = "
                  + str(x_smote.shape[0]) + "y value old" + str(x_y_fit_blind_transform_dict['x_fit'].shape[0]) +
                  " y value new = " + str(y_smote.shape[0]))
            raise Exception("Not balanced")
        x_y_fit_blind_transform_dict['x_fit'] = x_smote
        x_y_fit_blind_transform_dict['y_fit'] = y_smote
    joblib.dump(x_y_fit_blind_transform, f'../resources/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return x_y_fit_blind_transform


class MachineLearningScript:
    def __init__(self, source_x: str, source_y: str, term_represented: list, pre_process_steps: list,
                 n_gram_range: list):
        self.source_x = joblib.load(source_x)
        self.source_y = joblib.load(source_y)
        self.term_represented = term_represented
        self.pre_process_steps = pre_process_steps
        self.n_gram_range = n_gram_range
        self.scaler = MinMaxScaler()
        self.smote = SMOTE()
        self.gbc = GradientBoostingClassifier()
        self.line_url = 'https://notify-api.line.me/api/notify'
        self.headers = {'content-type': 'application/x-www-form-urlencoded',
                        'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}

    def indexing_x(self):
        temp_x = []
        for term_represented in self.term_represented:
            for pre_process_step in self.pre_process_steps:
                for n_gram_range in self.n_gram_range:
                    if term_represented == CountVectorizer:
                        vectorizer = CountVectorizer(preprocessor=pre_process_step, ngram_range=n_gram_range, )
                    else:
                        vectorizer = TfidfVectorizer(preprocessor=pre_process_step, ngram_range=n_gram_range,
                                                     use_idf=True)
                    name = f"{get_var_name(term_represented)}_{get_var_name(pre_process_step)}_n_grams_{n_gram_range[0]}_{n_gram_range[1]}"
                    data_combined = {
                        name: vectorizer
                    }
                    temp_x.append(data_combined)
                    process_noti = f"Total process: {len(temp_x)} at {name}"
                    r = requests.post(self.line_url, headers=self.headers, data={'message': process_noti})
                    print(r.text)
                    joblib.dump(temp_x, f'../resources/indexing_0.0.2.pkl')
        return temp_x

    def data_fit_transform(self, terms_x):
        temp_x = []
        x_cleaned = self.source_x['title_n_body_clean']
        y_list = self.source_y
        # start time for fit transform
        start_time_at_first = time.time()
        start_time_gmt_at_first = time.gmtime(start_time_at_first)
        start_time_gmt_at_first = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt_at_first)
        start_noti_fit = (f"Total of Terms_x: {len(terms_x)}" + f"Total of y_list: {len(y_list)}"
                          + f"start to fit data at: {start_time_gmt_at_first}")
        r = requests.post(self.line_url, headers=self.headers, data={'message': start_noti_fit})
        print(r.text)
        for y_dict in y_list:
            for y_name, y_value in y_dict.items():
                print(f"start at y_dict name: {y_name}")
                x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(x_cleaned, y_value, test_size=0.2,
                                                                            stratify=y_value)
                for term_x in terms_x:
                    for term_x_name, term_x_value in term_x.items():
                        # start time for fit transform
                        start_time = time.time()
                        start_time_gmt = time.gmtime(start_time)
                        start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
                        start_noti = f"start to fit transform at: {start_time_gmt}"
                        r = requests.post(self.line_url, headers=self.headers, data={'message': start_noti})
                        print(r.text)
                        term_x_value.fit(x_cleaned)
                        term_x_train = term_x_value.transform(x_fit)
                        term_x_test = term_x_value.transform(x_blind_test)
                        data_combination = {
                            "combination": term_x_name,
                            "x_fit": term_x_train,
                            "x_blind_test": term_x_test,
                            "y_name": y_name,
                            "y_fit": y_fit,
                            "y_blind_test": y_blind_test,
                        }
                        temp_x.append(data_combination)
                        count = len(temp_x)
                        # end time for fit transform
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_time_gmt = time.gmtime(result_time)
                        result_time = time.strftime("%H:%M:%S", result_time_gmt)
                        end_term_noti = f"Total time of fit transform: {result_time} at {term_x_name} in process at :{count} with y at {y_name}"
                        r = requests.post(self.line_url, headers=self.headers, data={'message': end_term_noti})
                        print(r.text)
        joblib.dump(temp_x, f'../resources/x_y_fit_blind_transform_0_0_2.pkl')
        end_time_at_last = time.time()
        result_time_last = end_time_at_last - start_time_at_first
        result_time_gmt = time.gmtime(result_time_last)
        result_time_last = time.strftime("%H:%M:%S", result_time_gmt)
        result_noti = f"Total time: {result_time_last}" + f"While start time at first: {start_time_gmt_at_first}"
        r = requests.post(self.line_url, headers=self.headers, data={'message': result_noti})
        print(r.text)
        return temp_x


x = '../resources/x_0_0_2.pkl'
y_source = '../resources/y_0_0_2.pkl'
term_representations = [CountVectorizer, TfidfVectorizer]
pre_process_steps = [pre_process_porterstemmer, pre_process_lemmatizer, pre_process_textblob, pre_process_spacy]
n_grams_ranges = [(1, 1), (1, 2), (1, 3)]

run = MachineLearningScript(x, y_source, term_representations, pre_process_steps, n_grams_ranges)
indexer = run.indexing_x()
x = run.data_fit_transform(indexer)
