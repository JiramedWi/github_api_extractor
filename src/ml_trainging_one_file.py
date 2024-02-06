import nltk
import os
import numpy as np
import pandas as pd
import requests
import spacy
import joblib
import inspect
import time
import logging

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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, confusion_matrix, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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


# TODO: 1. Tune dataset
gbm_model_1st_try = GradientBoostingClassifier(n_estimators=5000,
                                               learning_rate=0.05,
                                               max_depth=3,
                                               subsample=0.5,
                                               validation_fraction=0.1,
                                               n_iter_no_change=20,
                                               max_features='log2',
                                               verbose=1)

gbm_model_2nd = GradientBoostingClassifier(n_estimators=10000,
                                           learning_rate=0.1,
                                           max_depth=5,
                                           min_samples_split=5,
                                           min_samples_leaf=2,
                                           subsample=0.9,
                                           validation_fraction=0.1,
                                           n_iter_no_change=20,
                                           max_features='log2',
                                           verbose=1)

gbm_model_3rd = GradientBoostingClassifier(n_estimators=10000,
                                           learning_rate=0.1,
                                           max_depth=7,
                                           min_samples_split=10,
                                           min_samples_leaf=5,
                                           subsample=1.0,
                                           validation_fraction=0.1,
                                           n_iter_no_change=20,
                                           max_features=None,
                                           verbose=1)


def calculate_precision(y_true, y_pred):
    true_positive = confusion_matrix(y_true, y_pred)[1, 1]
    false_positive = confusion_matrix(y_true, y_pred)[0, 1]

    if true_positive + false_positive == 0:
        return 0
    result = true_positive / (true_positive + false_positive)
    return result


def y_ratio_train_1(y_true, y_pred):
    zero_count = np.mean(y_true == 1)
    # find ratio of 0 in y_true
    return zero_count


def y_ratio_pred_1(y_true, y_pred):
    zero_count = np.mean(y_pred == 1)
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
        raise ValueError("Precision and Recall both are zero, F1 score is undefined.")

    result = 2 * (precision * recall) / (precision + recall)
    return result


def custom_precision_recall_curve(y_true, probas_pred):
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    area_under_curve = auc(recall, precision)
    return area_under_curve


def train_cv(datasets, model):
    # Start the clock, train and evaluate the model, stop the clock with gmt+7 time
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train cv at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}" + ' ' + f"model: {get_var_name(model)}"
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

    scoring_metrics_dict = {
        'y_ratio_train_1': y_ratio_train_1,
        # 'y_ratio_train_0': make_scorer(y_ratio_train_0, greater_is_better=True),
        'y_ratio_pred_1': y_ratio_pred_1,
        # 'y_ratio_pred_0': make_scorer(y_ratio_pred_0, greater_is_better=True),
        'precision_score': calculate_precision,
        'recall_score': calculate_recall,
        'f1_score': calculate_f1_score,
        # 'auc_score': make_scorer(auc, greater_is_better=True),
        'roc_auc_score': roc_auc_score,
        # 'roc_auc_score': make_scorer(roc_auc_score, greater_is_better=True),
        'precision_recall_curve': custom_precision_recall_curve,
        'matthews_corrcoef': matthews_corrcoef,
        # 'matthews_corrcoef': make_scorer(matthews_corrcoef, greater_is_better=True),
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
            # Assuming gbm_model is your GradientBoostingClassifier
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Initialize an empty list to store the trained models
            trained_models = []
            for fold, (train_index, test_index) in enumerate(skf.split(x_fit, y_fit), 1):
                X_train, X_test = x_fit[train_index], x_fit[test_index]
                Y_train, Y_test = y_fit[train_index], y_fit[test_index]
                # Train your model
                model.fit(X_train, Y_train)
                # Store the trained model
                trained_models.append(model)
                # Make predictions on the test set
                predictions = model.predict(X_test)
                # # Fold index
                fold_result = {}
                err_text = ''
                for metric_name, scorer in scoring_metrics_dict.items():
                    try:
                        # Try to calculate the score
                        scorer_result = scorer(Y_test, predictions)
                        score = {
                            f"{metric_name}_fold_{fold}": scorer_result
                        }
                        fold_result.update(score)
                    except Exception as e:
                        logger.error(f"Error during model evaluation: {e} inside")
                        err_text = f"{e} error in {metric_name} error type {type(e).__name__}\n"
                        score = {'error': err_text}
                        fold_result.update(score)
                data_combination_cv_score.update(fold_result)
                del fold_result
        except Exception as e:
            logger.error(f"Error during model evaluation: {e} outside")
            score = {'error': f"{e} error in {term_x_name} Y = {y_name} error type {type(e).__name__}\n"}
            data_combination_cv_score.update(score)
            pass
        # Add the results to the list of datasets
        dataset.update(data_combination_cv_score)
        del data_combination_cv_score
    # Save the model and results path
    results_cv_path = f"../resources/result_0.0.2/cv_score_{get_var_name(datasets)}_{get_var_name(model)}.pkl"
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


def train_ml(datasets, model):
    # Start the clock, train and evaluate the model, stop the clock with gmt+7 time
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train ML at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}" + ' ' + f"model: {get_var_name(model)}"
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
        model.fit(x_fit, y_fit)
        predict = model.predict(x_blind_test)

        # Calculate metrics
        precision_test_score = precision_score(y_blind_test, predict)
        recall_test_score = recall_score(y_blind_test, predict)
        f1_test_score = f1_score(y_blind_test, predict)
        mcc = matthews_corrcoef(y_blind_test, predict)

        predicted_probabilities = model.predict_proba(x_blind_test)[:, 1]
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

    results_predict_path = f"../resources/result_0.0.2/predict_{get_var_name(datasets)}_{get_var_name(model)}.pkl"
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
# result_normal_cv = train_cv(normal_result_for_train)
# result_normal_predict = train_ml(normal_result_for_train)
#
# result_normal_normalize_min_max_cv = train_cv(normal_result_for_train_normalize_min_max)
# result_normal_normalize_min_max_predict = train_ml(normal_result_for_train_normalize_min_max)
#
# result_normal_normalize_log_cv = train_cv(normal_result_for_train_normalize_log)
# result_normal_normalize_log_predict = train_ml(normal_result_for_train_normalize_log)

# SMOTE result
result_SMOTE_cv_1st_model = train_cv(SMOTE_result_for_train, gbm_model_1st_try)
result_SMOTE_predict_1st_model = train_ml(SMOTE_result_for_train, gbm_model_1st_try)

result_SMOTE_cv_2nd_model = train_cv(SMOTE_result_for_train, gbm_model_2nd)
result_SMOTE_predict_2nd_model = train_ml(SMOTE_result_for_train, gbm_model_2nd)

result_SMOTE_cv_3rd_model = train_cv(SMOTE_result_for_train, gbm_model_3rd)
result_SMOTE_predict_3rd_model = train_ml(SMOTE_result_for_train, gbm_model_3rd)

# result_SMOTE_normalize_min_max_cv = train_cv(SMOTE_result_for_train_normalize_min_max)
# result_SMOTE_normalize_min_max_predict = train_ml(SMOTE_result_for_train_normalize_min_max)
#
# result_SMOTE_normalize_log_cv = train_cv(SMOTE_result_for_train_normalize_log)
# result_SMOTE_normalize_log_predict = train_ml(SMOTE_result_for_train_normalize_log)

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
