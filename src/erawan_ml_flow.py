import nltk
import os
import numpy as np
import pandas as pd
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import csr_matrix
from textblob import TextBlob

dirname = os.path.expanduser('~')
data_set_path = os.path.join(dirname, 'data_set')
if not os.path.isdir(data_set_path):
    os.makedirs(data_set_path)
result_path = os.path.join(dirname, 'result')
if not os.path.isdir(result_path):
    os.makedirs(result_path)


def read_pickle_files(directory):
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    data = {}
    for file_name in pickle_files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'rb') as file:
            data[file_name] = joblib.load(file)

    return data


data = read_pickle_files(data_set_path)


# Function to train and evaluate GradientBoostingClassifier
def train_gbm_evaluate_model(idx, X_train, Y_train, X_test, Y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)

    # Predictions on the test set
    Y_pred = model.predict(X_test)

    # Calculate metrics
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

    # Return metrics and the trained model
    return {
        "combination": idx,
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# Function to load and train models for all dataset combinations
def train_models_gbm(data_combinations, output_dir):
    results = []
    for idx, data_combination in data_combinations.items():
        X_train = data_combination["X_train"]
        Y_train = data_combination["Y_train"]
        X_test = data_combination["X_test"]
        Y_test = data_combination["Y_test"]

        # Train and evaluate the model
        result = train_gbm_evaluate_model(X_train, Y_train, X_test, Y_test)
        results.append(result)

        # Save the model and results
        model_file = f"{output_dir}/model_{idx}.pkl"
        result_file = f"{output_dir}/result_{idx}.pkl"

        joblib.dump(result["model"], model_file)
        joblib.dump(result, result_file)

    return results


def train_models_gbm_one_time(data_combinations, output_dir):
    results = []
    for idx, data_combination in data_combinations.items():
        X_train = data_combination["X_train"]
        Y_train = data_combination["Y_train"]
        X_test = data_combination["X_test"]
        Y_test = data_combination["Y_test"]

        # Train and evaluate the model
        result = train_gbm_evaluate_model(idx, X_train, Y_train, X_test, Y_test)
        results.append(result)

        # Save the model and results
        model_file = f"{output_dir}/model_{idx}.pkl"
        result_file = f"{output_dir}/result_{idx}.pkl"

        joblib.dump(result["model"], model_file)
        joblib.dump(result, result_file)
        if len(results) > 2:
            return results


result = train_models_gbm_one_time(data, result_path)
df_result = pd.DataFrame(result)