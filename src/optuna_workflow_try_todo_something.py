import threading

import nltk
import os
import numpy as np
import optuna
import pandas as pd
import requests
import spacy
import joblib
import inspect
import time
import logging

from datetime import datetime, timedelta, timezone

from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

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


class TimeoutCallback:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds

    def on_trial_begin(self, trial, state):
        timer = threading.Timer(self.timeout_seconds, lambda: trial.should_stop.set(True))
        timer.start()


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 500, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 128, 512)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 64, 256)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)

    gbm = GradientBoostingClassifier(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     subsample=subsample,
                                     random_state=42)

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=3, scoring='roc_auc')
    print(result)
    auc_scores = result['test_score']
    return np.mean(auc_scores)


# callback = TimeoutCallback(timeout_seconds=600)


def save_plot_optuna(study, path, name):
    # Plot optimization history
    fig1 = optuna.visualization.plot_optimization_history(study)

    # Plot parallel coordinates
    fig2 = optuna.visualization.plot_parallel_coordinate(study)

    # Plot slice for specified parameters
    target_params = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample']
    fig3 = optuna.visualization.plot_slice(study, params=target_params)

    # Plot parameter importances
    fig4 = optuna.visualization.plot_param_importances(study)

    # Combine the subplots into a single figure
    combined_fig = make_subplots(rows=2, cols=2)

    # Add traces from each subplot to the combined figure
    for trace in fig1.data:
        combined_fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        combined_fig.add_trace(trace, row=1, col=2)

    for trace in fig3.data:
        combined_fig.add_trace(trace, row=2, col=1)

    for trace in fig4.data:
        combined_fig.add_trace(trace, row=2, col=2)

    # Save the figure
    combined_fig.update_layout(title_text=f"{name} Figure")
    combined_fig.show()
    combined_fig.write_image(f"{path}.png")


def find_best_parameter(datasets: list):
    # Set up time and line notification
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    start_noti = f"start to find best parameter cv at: {start_time_gmt}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_noti)

    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']
        y_name = dataset['y_name']
        if term_x_name == 'TfidfVectorizer_pre_process_porterstemmer_n_grams_1_1' and y_name == 'issue_in_test_step':
            # Find best parameter
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, x_fit, y_fit),
                n_trials=5,
                timeout=600,
            )
            trial = study.best_trial
            result = trial.value
            best_params = trial.params
            dataset['best_params'] = best_params
            dataset['result'] = result
            noti_study = f"WE get the results at index {datasets.index(dataset)} with {result}"
            r = requests.post(line_url, headers=headers, data={'message': noti_study})
            print(r.text, noti_study)
        # save_plot_path = f"../resources/optuna_plot/plot_optuna_{get_var_name(datasets)}_{term_x_name}"
        # save_plot_optuna(study, save_plot_path, get_var_name(datasets))

    # Save the model and results path
    results_cv_path = f"../resources/optuna_result/cv_score_{get_var_name(datasets)}.pkl"
    # result_cv = pd.DataFrame(result_cv)
    joblib.dump(datasets, results_cv_path)
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    end_time_noti = f"Total time end finding best parameter CV: {result_time}"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)
    return datasets


normal_result = joblib.load('../resources/result_0_0_2/x_y_fit_blind_transform_optuna.pkl')
# smote_result = joblib.load('../resources/result_0_0_2/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')

parameter_result_normal = find_best_parameter(normal_result)
# parameter_result_smote = find_best_parameter(smote_result)
