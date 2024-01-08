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

line_url = 'https://notify-api.line.me/api/notify'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


gbm_model = GradientBoostingClassifier(n_estimators=5000,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       max_features='log2',
                                       verbose=1)


def train_cv(datasets):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    start_noti = f"start to train cv at: {start_time_gmt}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_noti)
    # result_cv = []
    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']
        y_name = dataset['y_name']
        # start_dataset_noti = f"start to train cv at: {term_x_name}"
        # r = requests.post(line_url, headers=headers, data={'message': start_dataset_noti})
        print(r.text)
        # Train and evaluate the model
        scoring_metrics = ['precision_macro', 'recall_macro', 'f1_macro']
        cv_results = model_selection.cross_validate(gbm_model, x_fit, y_fit, cv=5, n_jobs=-2,
                                                    scoring=scoring_metrics)
        data_combination_cv_score = {
            "combination": term_x_name,
            "y_name": y_name,
        }
        for metric in scoring_metrics:
            score = {
                metric: cv_results[f'test_{metric}'].mean()
            }
            data_combination_cv_score.update(score)
        # Add the results to the list of datasets
        result_cv = pd.DataFrame(data_combination_cv_score, index=[0])
        dataset['cv_results'] = result_cv
        # result_cv.append(data_combination_cv_score)
    # Save the model and results path
    results_cv_path = f"../resources/cv_score_{get_var_name(datasets)}.pkl"
    # result_cv = pd.DataFrame(result_cv)
    joblib.dump(datasets, results_cv_path)
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    end_time_noti = f"Total time end CV: {result_time}"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)
    return datasets


def train_ml(datasets):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    start_noti = f"start to train ml at: {start_time_gmt}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_noti)
    # results_predict = []
    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        x_blind_test = dataset['x_blind_test']
        y_fit = dataset['y_fit']
        y_blind_test = dataset['y_blind_test']
        y_name = dataset['y_name']
        # start_dataset_noti = f"start to train ml at: {term_x_name}"
        # r = requests.post(line_url, headers=headers, data={'message': start_dataset_noti})
        print(r.text)
        # Train and evaluate the model
        gbm_model.fit(x_fit, y_fit)
        predict = gbm_model.predict(x_blind_test)

        # Calculate metrics
        precision_test_score = precision_score(y_blind_test, predict)
        recall_test_score = recall_score(y_blind_test, predict)
        f1_test_score = f1_score(y_blind_test, predict)

        predicted_probabilities = gbm_model.predict_proba(x_blind_test)[:, 1]
        auc_test_score = roc_auc_score(y_blind_test, predicted_probabilities)

        data_combination_result_score = {
            "combination": term_x_name,
            "y_name": y_name,
            "precision": precision_test_score,
            "recall": recall_test_score,
            "f1": f1_test_score,
            "auc": auc_test_score
        }
        # Add the results to the list of datasets
        results_predict = pd.DataFrame(data_combination_result_score, index=[0])
        dataset['predict_results'] = results_predict
        # results_predict.append(data_combination_result_score)

    results_predict_path = f"../resources/predict_{get_var_name(datasets)}.pkl"
    # results_predict = pd.DataFrame(results_predict)
    joblib.dump(datasets, results_predict_path)
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    end_time_noti = f"Total time end ML: {result_time}"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)
    return datasets


normal_result_for_train = joblib.load('../resources/result_0.0.2/x_y_fit_blind_transform_0_0_2.pkl')
normal_result_for_train_normalize_min_max = joblib.load(
    '../resources/normalize_x_y_fit_blind_transform_0_0_2_min_max_transform_0.0.2.pkl')
normal_result_for_train_normalize_log = joblib.load(
    '../resources/normalize_x_y_fit_blind_transform_0_0_2_log_transform_0.0.2.pkl')

SMOTE_result_for_train = joblib.load('../resources/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')
SMOTE_result_for_train_normalize_min_max = joblib.load(
    '../resources/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_min_max_transform_0.0.2.pkl')
SMOTE_result_for_train_normalize_log = joblib.load(
    '../resources/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_log_transform_0.0.2.pkl')

# normal result
result_normal_cv = train_cv(normal_result_for_train)
result_normal_predict = train_ml(normal_result_for_train)

result_normal_normalize_min_max_cv = train_cv(normal_result_for_train_normalize_min_max)
result_normal_normalize_min_max_predict = train_ml(normal_result_for_train_normalize_min_max)

result_normal_normalize_log_cv = train_cv(normal_result_for_train_normalize_log)
result_normal_normalize_log_predict = train_ml(normal_result_for_train_normalize_log)


# SMOTE result
result_SMOTE_cv = train_cv(SMOTE_result_for_train)
result_SMOTE_predict = train_ml(SMOTE_result_for_train)

result_SMOTE_normalize_min_max_cv = train_cv(SMOTE_result_for_train_normalize_min_max)
result_SMOTE_normalize_min_max_predict = train_ml(SMOTE_result_for_train_normalize_min_max)

result_SMOTE_normalize_log_cv = train_cv(SMOTE_result_for_train_normalize_log)
result_SMOTE_normalize_log_predict = train_ml(SMOTE_result_for_train_normalize_log)

