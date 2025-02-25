import time

import nltk
import os
import numpy as np
import pandas as pd
import requests
import spacy
import joblib
import inspect

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
    count = 0
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        # count loop
        count += 1
        # Set SMOTE oversampling
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        x_smote, y_smote = smote.fit_resample(x_y_fit_blind_transform_dict['x_fit'],
                                              x_y_fit_blind_transform_dict['y_fit'])
        # Check value is smoted or not
        if x_smote.shape[0] > x_y_fit_blind_transform_dict['x_fit'].shape[0] and y_smote.shape[0] > \
                x_y_fit_blind_transform_dict['y_fit'].shape[0]:
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape[0]) + ", x value new = "
                  + str(x_smote.shape[0]) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape[0]) +
                  ", y value new = " + str(y_smote.shape[0]))
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape) + ", x value new = "
                  + str(x_smote.shape) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape) +
                  ", y value new = " + str(y_smote.shape))
            pass
        else:
            raise Exception("It not set smote")
        # Check ratio of Y_smote
        class_distribution_train_smote = pd.Series(y_smote).value_counts()
        print(f"count_y_smote {class_distribution_train_smote}")
        ratio_class_1_train_smote = class_distribution_train_smote[1] / len(y_smote)
        ratio_class_0_train_smote = class_distribution_train_smote[0] / len(y_smote)
        print(f"\nRatio of class '1' in the training smote set: {ratio_class_1_train_smote:.2%}")
        print(f"\nRatio of class '0' in the training smote set: {ratio_class_0_train_smote:.2%}")
        x_y_fit_blind_transform_dict['x_fit'] = x_smote
        x_y_fit_blind_transform_dict['y_fit'] = y_smote
        x_y_fit_blind_transform_dict['y_smote_1_ratio'] = f"{ratio_class_1_train_smote:.2%}"
        x_y_fit_blind_transform_dict['y_smote_0_ratio'] = f"{ratio_class_0_train_smote:.2%}"
        print(f"Total process: {count}")
    joblib.dump(x_y_fit_blind_transform, f'/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_blind_SMOTE_transform_optuna.pkl')
    return x_y_fit_blind_transform


def normalize_x(x_y_fit_blind_transform, normalize_method):
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        if normalize_method == 'min_max':
            x_y_fit_blind_transform_dict['x_fit'] = scale_sparse_matrix(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = scale_sparse_matrix(
                x_y_fit_blind_transform_dict['x_blind_test'])
        elif normalize_method == 'log':
            x_y_fit_blind_transform_dict['x_fit'] = log_transform_tfidf(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = log_transform_tfidf(
                x_y_fit_blind_transform_dict['x_blind_test'])
    joblib.dump(x_y_fit_blind_transform,
                f'/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/normalize_{get_var_name(x_y_fit_blind_transform)}_{normalize_method}_transform.pkl')

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
                    joblib.dump(temp_x, f'/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/indexing.pkl')
        return temp_x

    def data_fit_transform(self, terms_x):
        temp_x = []
        x_cleaned = self.source_x['cleaned_title_n_body']
        y_list = self.source_y
        for y_dict in y_list:
            for y_name, y_value in y_dict.items():
                print(f"start at y_dict name: {y_name}")
                x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(x_cleaned, y_value, test_size=0.2,
                                                                            stratify=y_value)
                # Reset index
                x_fit = x_fit.reset_index(drop=True)
                x_blind_test = x_blind_test.reset_index(drop=True)
                y_fit = y_fit.reset_index(drop=True)
                y_blind_test = y_blind_test.reset_index(drop=True)
                # Check ratio of Y
                class_distribution_train = pd.Series(y_fit).value_counts()
                print(f"count_y {class_distribution_train}")
                class_distribution_test = pd.Series(y_blind_test).value_counts()
                print(f"count_y_test {class_distribution_test}")
                ratio_class_train = len(y_fit) / len(y_value)
                ratio_class_test = len(y_blind_test) / len(y_value)
                ratio_class_1_train = class_distribution_train[1] / len(y_fit)
                ratio_class_1_test = class_distribution_test[1] / len(y_blind_test)
                ratio_class_0_train = class_distribution_train[0] / len(y_fit)
                ratio_class_0_test = class_distribution_test[0] / len(y_blind_test)
                print(f"\nRatio of class in the training set: {ratio_class_train:.2%}")
                print(f"\nRatio of class in the test set: {ratio_class_test:.2%}")
                print(f"\nRatio of class '1' in the training set: {ratio_class_1_train:.2%}")
                print(f"\nRatio of class '0' in the training set: {ratio_class_0_train:.2%}")
                print(f"\nRatio of class '1' in the test set: {ratio_class_1_test:.2%}")
                print(f"\nRatio of class '0' in the test set: {ratio_class_0_test:.2%}")
                for term_x in terms_x:
                    for term_x_name, term_x_value in term_x.items():
                        term_x_value.fit(x_cleaned)
                        term_x_train = term_x_value.transform(x_fit)
                        term_x_test = term_x_value.transform(x_blind_test)
                        data_combination = {
                            "combination": term_x_name,
                            "x_fit": term_x_train,
                            "x_blind_test": term_x_test,
                            "y_name": y_name,
                            "y_fit": y_fit,
                            "y_amount": len(y_fit),
                            "y_blind_test": y_blind_test,
                            "y_fit_ratio": f"{ratio_class_train:.2%}",
                            "y_blind_test_ratio": f"{ratio_class_test:.2%}",
                            "y_fit_1_ratio": f"{ratio_class_1_train:.2%}",
                            "y_fit_0_ratio": f"{ratio_class_0_train:.2%}",
                            "y_blind_test_1_ratio": f"{ratio_class_1_test:.2%}",
                            "y_blind_test_0_ratio": f"{ratio_class_0_test:.2%}",
                            "y_fit_ratio_0_1": f"{class_distribution_train[0]}:{class_distribution_train[1]}",
                            "y_blind_test_ratio_0_1": f"{class_distribution_test[0]}:{class_distribution_test[1]}"
                        }
                        temp_x.append(data_combination)
                        count = len(temp_x)

        joblib.dump(temp_x, f'/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_blind_transform_optuna.pkl')
        return temp_x


def main():
    x = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_for_pre_training.pkl'
    y_source = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/y_for_pre_training.pkl'
    term_representations = [CountVectorizer, TfidfVectorizer]
    pre_process_steps = [pre_process_porterstemmer, pre_process_lemmatizer, pre_process_textblob, pre_process_spacy]
    n_grams_ranges = [(1, 1), (1, 2)]

    # To run datafit
    run = MachineLearningScript(x, y_source, term_representations, pre_process_steps, n_grams_ranges)
    indexer = run.indexing_x()
    indexer = run.data_fit_transform(indexer)

    # To run smote
    time.sleep(10)
    smote = joblib.load('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_blind_transform_optuna.pkl')
    smote = set_smote(smote)

    # To run normalize
    time.sleep(10)
    normalize = joblib.load('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_blind_SMOTE_transform_optuna.pkl')
    mix_max = normalize_x(normalize, 'min_max')
    log = normalize_x(normalize, 'log')

    print("Done")

if __name__ == '__main__':
    main()

