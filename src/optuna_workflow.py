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
    return result


def save_plot_optuna(study, path):
    # Create a single figure to accommodate all plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot optimization history
    optuna.visualization.plot_optimization_history(study, target_names=['objective'], ax=axes[0, 0])
    axes[0, 0].set_title("Optimization History")

    # Plot parallel coordinates
    optuna.visualization.plot_parallel_coordinate(study, ax=axes[0, 1])
    axes[0, 1].set_title("Parallel Coordinates")

    # Plot slice for specified parameters
    target_params = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample']
    optuna.visualization.plot_slice(study, params=target_params, ax=axes[1, 0])
    axes[1, 0].set_title("Slice Plot")

    # Plot parameter importances
    optuna.visualization.plot_param_importances(study, ax=axes[1, 1])
    axes[1, 1].set_title("Parameter Importance")

    # Adjust spacing and layout as needed
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{path}.png")


def find_best_parameter(datasets):
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
        # start_dataset_noti = f"start to train cv at: {term_x_name}"
        # r = requests.post(line_url, headers=headers, data={'message': start_dataset_noti})
        print(r.text)
        # Find best parameter
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, x_fit, y_fit),
                n_trials=15,
                callbacks=[TimeoutCallback(1800)]
            )
            trial = study.best_trial
            result = trial.value
            best_params = trial.params
            dataset['best_params'] = best_params
            dataset['result'] = result
            save_plot_path = f"../resources/optuna_plot/plot_optuna_{get_var_name(datasets)}_{term_x_name}"
            save_plot_optuna(study, save_plot_path)
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            err_text = f"{e}" + 'error type ' + type(e).__name__ + '\n'
            dataset['best_params'] = err_text
            dataset['result'] = err_text
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

