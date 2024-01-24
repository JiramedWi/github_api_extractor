import nltk
import os
import numpy as np
import pandas as pd
import requests
import spacy
import joblib
import inspect
import time

from datetime import datetime, timedelta, timezone
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
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, confusion_matrix, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc

line_url = 'https://notify-api.line.me/api/notify'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}
tz = timezone(timedelta(hours=7))


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


def calculate_precision(y_true, y_pred):
    true_positive = confusion_matrix(y_true, y_pred)[1, 1]
    false_positive = confusion_matrix(y_true, y_pred)[0, 1]

    if true_positive + false_positive == 0:
        return 0
    result = true_positive / (true_positive + false_positive)
    return result


def y_ratio_train_0(y_true, y_pred):
    zero_count = np.mean(y_true == 0)
    # find ratio of 0 in y_true
    return zero_count


def y_ratio_pred_0(y_true, y_pred):
    zero_count = np.mean(y_pred == 0)
    # find ratio of 0 in y_true
    return zero_count


def calculate_recall(y_true, y_pred):
    true_positive = confusion_matrix(y_true, y_pred)[1, 1]
    false_negative = confusion_matrix(y_true, y_pred)[1, 0]

    if true_positive + false_negative == 0:
        return 0
    result = true_positive / (true_positive + false_negative)
    return result


def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision + recall == 0:
        raise

    result = 2 * (precision * recall) / (precision + recall)
    return result


def custom_precision_recall_curve(y_true, probas_pred):
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    area_under_curve = auc(recall, precision)
    return area_under_curve


def train_cv(datasets):
    # Start the clock, train and evaluate the model, stop the clock with gmt+7 time
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train cv at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_time_announce)
    custom_precision_scorer = make_scorer(calculate_precision, greater_is_better=True)
    custom_recall_scorer = make_scorer(calculate_recall, greater_is_better=True)
    custom_f1_score_scorer = make_scorer(calculate_f1_score, greater_is_better=True)
    custom_pr_auc_scorer = make_scorer(custom_precision_recall_curve, greater_is_better=True, needs_proba=True)
    # matthews_corrcoef = make_scorer(matthews_corrcoef, greater_is_better=True)
    # auc = make_scorer(auc, greater_is_better=True)
    # roc_auc_score = make_scorer(roc_auc_score, greater_is_better=True)
    # precision_recall_curve = make_scorer(precision_recall_curve, greater_is_better=True)

    scoring_metrics = {
        'y_ratio_train_0': make_scorer(y_ratio_train_0, greater_is_better=True),
        'y_ratio_pred_0': make_scorer(y_ratio_pred_0, greater_is_better=True),
        'precision_score': custom_precision_scorer,
        'recall_score': custom_recall_scorer,
        'f1_score': custom_f1_score_scorer,
        # 'auc_score': make_scorer(auc, greater_is_better=True),
        'roc_auc_score': make_scorer(roc_auc_score, greater_is_better=True),
        'precision_recall_curve': custom_pr_auc_scorer,
        'matthews_corrcoef': make_scorer(matthews_corrcoef, greater_is_better=True),
    }

    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']
        y_name = dataset['y_name']
        # start_dataset_noti = f"start to train cv at: {term_x_name}"
        # r = requests.post(line_url, headers=headers, data={'message': start_dataset_noti})
        print(r.text)
        # Train and evaluate the model
        # to get the train or test value using => cv_results['indices']['train'][0].tolist()
        data_combination_cv_score = {}
        cv_results = None
        try:
            cv_results = model_selection.cross_validate(gbm_model, x_fit, y_fit, cv=5, n_jobs=-2,
                                                        scoring=scoring_metrics, return_indices=True,
                                                        error_score='raise')
            err_text = ''
            for metric in scoring_metrics:
                try:
                    score = {
                        f"{metric}_fold_1": cv_results[f'test_{metric}'][0],
                        f"{metric}_fold_2": cv_results[f'test_{metric}'][1],
                        f"{metric}_fold_3": cv_results[f'test_{metric}'][2],
                        f"{metric}_fold_4": cv_results[f'test_{metric}'][3],
                        f"{metric}_fold_5": cv_results[f'test_{metric}'][4],
                        metric: cv_results[f'test_{metric}'].mean(),
                    }
                    data_combination_cv_score.update(score)
                except Exception as e:
                    print(f"Error during model evaluation: {e}")
                    err_text = err_text + 'error in ' + metric + ' ' + 'error type ' + type(e).__name__ + '\n'
                    score = {
                        'error': err_text
                    }
                    data_combination_cv_score.update(score)
                    pass
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            score = {
                'error': type(e).__name__,
            }
            data_combination_cv_score.update(score)
            pass
        # Add the results to the list of datasets
        dataset.update(data_combination_cv_score)
        del data_combination_cv_score
    # Save the model and results path
    results_cv_path = f"../resources/result_0.0.2/cv_score_{get_var_name(datasets)}.pkl"
    # result_cv = pd.DataFrame(result_cv)
    joblib.dump(datasets, results_cv_path)
    end_time = datetime.now(tz)
    result_time = end_time - start_time
    result_time_in_sec = result_time.total_seconds()
    # Make it short to 2 decimal
    in_minute = result_time_in_sec / 60
    in_minute = "{:.2f}".format(in_minute)
    # Make it short to 5 decimal
    in_hour = result_time_in_sec / 3600
    in_hour = "{:.5f}".format(round(in_hour, 2))
    end_time_noti = f"Total time of CV train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)
    return datasets


def train_ml(datasets):
    # Start the clock, train and evaluate the model, stop the clock with gmt+7 time
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train ML at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_time_announce)
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
        mcc = matthews_corrcoef(y_blind_test, predict)

        predicted_probabilities = gbm_model.predict_proba(x_blind_test)[:, 1]
        # auc_score = auc(recall_test_score, precision_test_score)
        roc_auc_test_score = roc_auc_score(y_blind_test, predicted_probabilities)
        precision_recall_curve_test_score = precision_recall_curve(y_blind_test, predicted_probabilities)

        data_combination_result_score = {
            "precision": precision_test_score,
            "recall": recall_test_score,
            "f1": f1_test_score,
            "mcc": mcc,
            # "auc": auc_score,
            "roc_auc": roc_auc_test_score,
            "precision_recall_curve": precision_recall_curve_test_score,
        }
        # Add the results to the list of datasets
        dataset.update(data_combination_result_score)
        # results_predict = pd.DataFrame(data_combination_result_score, index=[0])
        # dataset['predict_results'] = results_predict
        # results_predict.append(data_combination_result_score)

    results_predict_path = f"../resources/result_0.0.2/predict_{get_var_name(datasets)}.pkl"
    # results_predict = pd.DataFrame(results_predict)
    joblib.dump(datasets, results_predict_path)
    end_time = datetime.now(tz)
    result_time = end_time - start_time
    result_time_in_sec = result_time.total_seconds()
    # Make it short to 2 decimal
    in_minute = result_time_in_sec / 60
    in_minute = "{:.2f}".format(in_minute)
    # Make it short to 5 decimal
    in_hour = result_time_in_sec / 3600
    in_hour = "{:.5f}".format(round(in_hour, 2))
    end_time_noti = f"Total time of ML train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)
    return datasets


start_time = datetime.now(tz)
start_time_announce = start_time.strftime("%c")
start_noti = f"start to train CV and ML at: {start_time_announce}"
r = requests.post(line_url, headers=headers, data={'message': start_noti})
print(r.text, start_time_announce)

normal_result_for_train = joblib.load('../resources/result_0.0.2/x_y_fit_blind_transform_0_0_2.pkl')
normal_result_for_train_normalize_min_max = joblib.load(
    '../resources/result_0.0.2/normalize_x_y_fit_blind_transform_0_0_2_min_max_transform_0.0.2.pkl')
normal_result_for_train_normalize_log = joblib.load(
    '../resources/result_0.0.2/normalize_x_y_fit_blind_transform_0_0_2_log_transform_0.0.2.pkl')

SMOTE_result_for_train = joblib.load('../resources/result_0.0.2/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')
SMOTE_result_for_train_normalize_min_max = joblib.load(
    '../resources/result_0.0.2/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_min_max_transform_0.0.2.pkl')
SMOTE_result_for_train_normalize_log = joblib.load(
    '../resources/result_0.0.2/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_log_transform_0.0.2.pkl')

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

end_time = datetime.now(tz)
result_time = end_time - start_time
result_time_in_sec = result_time.total_seconds()
# Make it short to 2 decimal
in_minute = result_time_in_sec / 60
in_minute = "{:.2f}".format(in_minute)
# Make it short to 5 decimal
in_hour = result_time_in_sec / 3600
in_hour = "{:.5f}".format(round(in_hour, 2))
end_time_noti = f"Total time of CV and ML train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
print(r.text, end_time_noti)
