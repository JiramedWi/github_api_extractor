import joblib
import pandas as pd
import logging

import string
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from markdown import markdown
import numpy as np
from pathlib import Path
import os

import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_combination(combination_text):
    # Split the text using underscores
    words = combination_text.split('_')

    # Extract specific indices
    count_vectorizer = words[0]
    pre_process = words[3]
    n_gram_first = int(words[-2])
    n_gram_second = int(words[-1])

    # Return the components
    return count_vectorizer, pre_process, n_gram_first, n_gram_second


def loop_dict_normal_list_to_df(dict_list):
    temp = []
    for a_dict in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(a_dict['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}"
        }
        new_dict.update(a_dict)
        # Calculate the average for each metric
        metrics = ['precision_score', 'recall_score', 'f1_score', 'roc_auc_score', 'precision_recall_curve',
                   'matthews_corrcoef']
        for metric in metrics:
            try:
                fold_values = [a_dict[f'{metric}_fold_{i}'] for i in range(1, 6)]
                new_dict[f'{metric}_avg'] = np.mean(fold_values)
            except Exception as e:
                logger.error(f"Error during model evaluation: {e}")
                err_text = f"{e}" + 'error in ' + metric + ' ' + 'error type ' + type(e).__name__ + '\n'
                new_dict[f'{metric}_avg'] = err_text

        for col in ['x_fit', 'x_blind_test', 'y_fit', 'y_blind_test']:
            print(a_dict['combination'])
            print(a_dict['x_fit'].shape)
            print(a_dict['y_name'])
            print(a_dict['x_blind_test'].shape)
            print('-----------------------------')
            new_dict.pop(col)
        new_df = pd.DataFrame([new_dict])
        temp.append(new_df)
        del new_dict
    df = pd.concat(temp)
    return df


def print_x_features_from_dict_list(dict_list):
    for a_dict in dict_list:
        print(a_dict['combination'])
        print(a_dict['x_fit'].shape)
        # print(a_dict['x_blind_test'].shape)
        print('-----------------------------')


cv_score_normal_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_normal_result_for_train_gbm_model_3rd.pkl')))
cv_score_normal_result_for_train = loop_dict_normal_list_to_df(cv_score_normal_result_for_train)
# cv_score_normal_result_for_train = loop_dict_normal_list_to_df(cv_score_normal_result_for_train)
predict_normal_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_normal_result_for_train_gbm_model_3rd.pkl')))

SMOTE_result_for_train_cv_3rd = joblib.load('../resources/result_0.0.2/cv_score_SMOTE_result_for_train_gbm_model_3rd.pkl')
SMOTE_result_for_train_cv_3rd = loop_dict_normal_list_to_df(SMOTE_result_for_train_cv_3rd)
# SMOTE_result_for_cv_result_3rd = loop_dict_normal_list_to_df(SMOTE_result_for_train_cv_3rd)

SMOTE_result_for_train_result_predict_3rd = joblib.load('../resources/result_0.0.2/predict_SMOTE_result_for_train_gbm_model_3rd.pkl')
# df_SMOTE_result_for_train_predict_result_3rd = loop_dict_normal_list_to_df(SMOTE_result_for_train_result_predict_3rd)

# print_x_features_from_dict_list(cv_score_normal_result_for_train.to_dict('records'))

# # Extract relevant columns for metrics comparison
# metrics_columns = [col for col in cv_score_normal_result_for_train.columns if 'fold' in col]
#
# # Melt the DataFrame to long format for better visualization
# melted_df = pd.melt(cv_score_normal_result_for_train, id_vars=['combination'], value_vars=metrics_columns,
#                     var_name='metric', value_name='value')
#
# # Plot the boxplot
# plt.figure(figsize=(20, 8))
# sns.boxplot(x='metric', y='value', hue='combination', data=melted_df)
# plt.xticks(rotation=45, ha='right')
# plt.title('Comparison of Metrics Across Folds')
# plt.subplots_adjust(right=0.7)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# plt.show()
