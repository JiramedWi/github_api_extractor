import string
import requests
import re
import nltk
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from markdown import markdown
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.preprocess import pre_process_porterstemmer, pre_process_lemmetizer
import smote_variants as sv
from sklearn.ensemble import GradientBoostingClassifier

# TODO: Prepare X, Y as Same length
filepath = Path(os.path.abspath('../resources/clean_demo.pkl'))
x = pd.read_pickle(filepath)
word_counts = x.str.count(' ') + 1

# Check X
data = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
data = data[data['title_n_body'].notnull()]
data.rename(columns={'title_n_body': 'title_n_body_not_clean'}, inplace=True)
data = pd.concat([data, x.dropna()], axis=1)

y1 = pd.read_csv(Path(os.path.abspath('../resources/tsdetect/all_test_smell/df_test_semantic_smell.csv')))
y2 = pd.read_csv(Path(os.path.abspath('../resources/tsdetect/all_test_smell/df_issue_in_test_step.csv')))
y3 = pd.read_csv(Path(os.path.abspath('../resources/tsdetect/all_test_smell/df_code_related.csv')))
y4 = pd.read_csv(Path(os.path.abspath('../resources/tsdetect/all_test_smell/df_dependencies.csv')))
y5 = pd.read_csv(Path(os.path.abspath('../resources/tsdetect/all_test_smell/df_test_execution.csv')))


def compare_y_to_x(dfx, dfy):
    return dfy.loc[dfy['url'].isin(dfx['url'])]


y1_to_x = compare_y_to_x(data, y1)

# TODO: Prepare X: TF-IDF, Ngram 1, Normalization MinMax(0,1)
# vectorizer_porter_pre = TfidfVectorizer(use_idf=True, preprocessor=pre_process_porterstemmer)
tfidf_vectorizer_lemm_pre = TfidfVectorizer(use_idf=True, preprocessor=pre_process_lemmetizer, ngram_range=(1, 1))

# TODO: Prepare X,Y: Split80:20, Set SMOTE
x_fit, x_test, y_train, y_test = model_selection.train_test_split(x, y1, test_size=0.2)
# we need random_state?
X_tfidf_train = tfidf_vectorizer_lemm_pre.fit_transform(x_fit)
X_tfidf_test = tfidf_vectorizer_lemm_pre.fit_transform(x_test)

# TODO: ML Model: GBM
gbm_model = GradientBoostingClassifier(n_estimators=5000,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       max_features='log2',
                                       verbose=1)
# what we should set on setting?
model = gbm_model.fit(X_tfidf_train, y_train)

# TODO: ML Model: Cross_validation, Metric
precision = model_selection.cross_val_score(gbm_model, X_tfidf_train, y_train, cv=5,
n_jobs=-2, scoring='precision_macro').mean()
recall = model_selection.cross_val_score(gbm_model, X_tfidf_train, y_train, cv=5,
n_jobs=-2, scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_train, y_train, cv=5,
n_jobs=-2, scoring='f1_macro').mean()