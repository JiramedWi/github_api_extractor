import joblib
import pandas as pd
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

cv_score_SMOTE_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_smote_result_for_train.pkl')))
cv_score_SMOTE_result_for_train_normalize_min_max = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_SMOTE_result_for_train_normalize_min_max.pkl')))
cv_score_SMOTE_result_for_train_normalize_log = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_SMOTE_result_for_train_normalize_log.pkl')))

predict_SMOTE_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_smote_result_for_train.pkl')))
predict_SMOTE_result_for_train_normalize_min_max = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_SMOTE_result_for_train_normalize_min_max.pkl')))
predict_SMOTE_result_for_train_normalize_log = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_SMOTE_result_for_train_normalize_log.pkl')))

cv_score_normal_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_normal_result_for_train.pkl')))
cv_score_normal_result_for_train_normalize_log = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_normal_result_for_train_normalize_log.pkl')))
cv_score_normal_result_for_train_normalize_min_max = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/cv_score_normal_result_for_train_normalize_min_max.pkl')))

predict_normal_result_for_train = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_normal_result_for_train.pkl')))
predict_normal_result_for_train_normalize_log = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_normal_result_for_train_normalize_log.pkl')))
predict_normal_result_for_train_normalize_min_max = joblib.load(
    Path(os.path.abspath('../resources/result_0.0.2/predict_normal_result_for_train_normalize_min_max.pkl')))


def loop_cv_dict_normal_list_to_df(dict_list):
    temp = []
    new_dict = {}
    for i in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(i['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}",
            'Y_name': i['y_name'],
            'y_fit_ratio': i['y_fit_ratio'],
            'y_blind_test_ratio': i['y_blind_test_ratio'],
            # 'y_fit_ratio_0_1': i['y_fit_ratio_0_1'],
            # 'y_blind_test_ratio_0_1': i['y_blind_test_ratio_0_1'],
            'y_fit_1_ratio': i['y_fit_1_ratio'],
            'y_fit_0_ratio': i['y_fit_0_ratio'],
            'y_blind_test_1_ratio': i['y_blind_test_1_ratio'],
            'y_blind_test_0_ratio': i['y_blind_test_0_ratio'],
            'cv_precision': i['cv_results']['precision_macro'],
            'cv_recall': i['cv_results']['recall_macro'],
            'cv_f1': i['cv_results']['f1_macro'],
        }
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
    df = pd.concat(temp)
    return df


def loop_predict_dict_normal_list_to_df(dict_list):
    temp = []
    new_dict = {}
    for i in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(i['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}",
            'Y_name': i['y_name'],
            'y_fit_ratio': i['y_fit_ratio'],
            'y_blind_test_ratio': i['y_blind_test_ratio'],
            # 'y_fit_ratio_0_1': i['y_fit_ratio_0_1'],
            # 'y_blind_test_ratio_0_1': i['y_blind_test_ratio_0_1'],
            'y_fit_1_ratio': i['y_fit_1_ratio'],
            'y_fit_0_ratio': i['y_fit_0_ratio'],
            'y_blind_test_1_ratio': i['y_blind_test_1_ratio'],
            'y_blind_test_0_ratio': i['y_blind_test_0_ratio'],
            'predicts_precision': i['predict_results']['precision'],
            'predicts_recall': i['predict_results']['recall'],
            'predicts_f1': i['predict_results']['f1'],
            'predicts_auc': i['predict_results']['auc'],
        }
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
    df = pd.concat(temp)
    return df


def loop_cv_dict_SMOTE_list_to_df(dict_list):
    temp = []
    new_dict = {}
    for i in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(i['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}",
            'Y_name': i['y_name'],
            'y_fit_ratio': i['y_fit_ratio'],
            'y_blind_test_ratio': i['y_blind_test_ratio'],
            # 'y_fit_ratio_0_1': i['y_fit_ratio_0_1'],
            # 'y_blind_test_ratio_0_1': i['y_blind_test_ratio_0_1'],
            'y_fit_1_ratio': i['y_fit_1_ratio'],
            'y_fit_0_ratio': i['y_fit_0_ratio'],
            'y_blind_test_1_ratio': i['y_blind_test_1_ratio'],
            'y_blind_test_0_ratio': i['y_blind_test_0_ratio'],
            'y_smote_1_ratio': i['y_smote_1_ratio'],
            'y_smote_0_ratio': i['y_smote_0_ratio'],
            'cv_precision': i['cv_results']['precision_macro'],
            'cv_recall': i['cv_results']['recall_macro'],
            'cv_f1': i['cv_results']['f1_macro']
        }
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
    df = pd.concat(temp)
    return df


def loop_predict_dict_SMOTE_list_to_df(dict_list):
    temp = []
    new_dict = {}
    for i in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(i['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}",
            'Y_name': i['y_name'],
            'y_fit_ratio': i['y_fit_ratio'],
            'y_blind_test_ratio': i['y_blind_test_ratio'],
            # 'y_fit_ratio_0_1': i['y_fit_ratio_0_1'],
            # 'y_blind_test_ratio_0_1': i['y_blind_test_ratio_0_1'],
            'y_fit_1_ratio': i['y_fit_1_ratio'],
            'y_fit_0_ratio': i['y_fit_0_ratio'],
            'y_blind_test_1_ratio': i['y_blind_test_1_ratio'],
            'y_blind_test_0_ratio': i['y_blind_test_0_ratio'],
            'y_smote_1_ratio': i['y_smote_1_ratio'],
            'y_smote_0_ratio': i['y_smote_0_ratio'],
            'predicts_precision': i['predict_results']['precision'],
            'predicts_recall': i['predict_results']['recall'],
            'predicts_f1': i['predict_results']['f1'],
            'predicts_auc': i['predict_results']['auc'],
        }
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
    df = pd.concat(temp)
    return df


def loop_normal_list_to_df(dict_list):
    temp = []
    new_dict = {}
    for i in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(i['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}",
            'Y_name': i['y_name'],
            'y_fit_ratio': i['y_fit_ratio'],
            'y_blind_test_ratio': i['y_blind_test_ratio'],
            'y_fit_ratio_0_1': i['y_fit_ratio_0_1'],
            'y_blind_test_ratio_0_1': i['y_blind_test_ratio_0_1'],
            'y_fit_1_ratio': i['y_fit_1_ratio'],
            'y_fit_0_ratio': i['y_fit_0_ratio'],
            'y_blind_test_1_ratio': i['y_blind_test_1_ratio'],
            'y_blind_test_0_ratio': i['y_blind_test_0_ratio'],
        }
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
    df = pd.concat(temp)
    return df


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


# Dataframe dataset
normal_result_for_train = joblib.load('../resources/x_y_fit_blind_transform_0_0_2.pkl')
normal_result_for_train_normalize_min_max = joblib.load(
    '../resources/normalize_x_y_fit_blind_transform_0_0_2_min_max_transform_0.0.2.pkl')
normal_result_for_train_normalize_log = joblib.load(
    '../resources/normalize_x_y_fit_blind_transform_0_0_2_log_transform_0.0.2.pkl')

SMOTE_result_for_train = joblib.load('../resources/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')
SMOTE_result_for_train_normalize_min_max = joblib.load(
    '../resources/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_min_max_transform_0.0.2.pkl')
SMOTE_result_for_train_normalize_log = joblib.load(
    '../resources/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_log_transform_0.0.2.pkl')

normal_result_for_train = loop_normal_list_to_df(normal_result_for_train)
normal_result_for_train_normalize_min_max = loop_normal_list_to_df(normal_result_for_train_normalize_min_max)
normal_result_for_train_normalize_log = loop_normal_list_to_df(normal_result_for_train_normalize_log)

SMOTE_result_for_train = loop_normal_list_to_df(SMOTE_result_for_train)
SMOTE_result_for_train_normalize_min_max = loop_normal_list_to_df(SMOTE_result_for_train_normalize_min_max)
SMOTE_result_for_train_normalize_log = loop_normal_list_to_df(SMOTE_result_for_train_normalize_log)

# Dataframe for the results
cv_score_normal_result_for_train = loop_cv_dict_normal_list_to_df(cv_score_normal_result_for_train)
cv_score_normal_result_for_train_normalize_min_max = loop_cv_dict_normal_list_to_df(
    cv_score_normal_result_for_train_normalize_min_max)
cv_score_normal_result_for_train_normalize_log = loop_cv_dict_normal_list_to_df(
    cv_score_normal_result_for_train_normalize_log)

predict_normal_result_for_train = loop_predict_dict_normal_list_to_df(predict_normal_result_for_train)
predict_normal_result_for_train_normalize_min_max = loop_predict_dict_normal_list_to_df(
    predict_normal_result_for_train_normalize_min_max)
predict_normal_result_for_train_normalize_log = loop_predict_dict_normal_list_to_df(
    predict_normal_result_for_train_normalize_log)

cv_score_smote_result_for_train = loop_cv_dict_SMOTE_list_to_df(cv_score_SMOTE_result_for_train)
cv_score_smote_result_for_train_normalize_min_max = loop_cv_dict_SMOTE_list_to_df(
    cv_score_SMOTE_result_for_train_normalize_min_max)
cv_score_smote_result_for_train_normalize_log = loop_cv_dict_SMOTE_list_to_df(
    cv_score_SMOTE_result_for_train_normalize_log)

predict_smote_result_for_train = loop_predict_dict_SMOTE_list_to_df(predict_SMOTE_result_for_train)
predict_SMOTE_result_for_train_normalize_min_max = loop_predict_dict_SMOTE_list_to_df(
    predict_SMOTE_result_for_train_normalize_min_max)
predict_SMOTE_result_for_train_normalize_log = loop_predict_dict_SMOTE_list_to_df(
    predict_SMOTE_result_for_train_normalize_log)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(x="n_gram", y="cv_precision", data=cv_score_smote_result_for_train)
plt.title("Precision for Different Combinations")
plt.xticks(rotation=45)
plt.show()

# Create a pivot table for better heatmap visualization
for vectorizer in cv_score_smote_result_for_train['count_vectorizer'].unique():
    vectorizer_data = cv_score_smote_result_for_train[cv_score_smote_result_for_train['count_vectorizer'] == vectorizer]
    pivot_df = vectorizer_data.pivot_table(values='cv_f1', index='pre_process', columns='n_gram', aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f", linewidths=.5)
    plt.title(f'{vectorizer} - F1 Macro Scores Heatmap')
    plt.show()

# Create separate boxplots for each count_vectorizer
for vectorizer in cv_score_smote_result_for_train['count_vectorizer'].unique():
    vectorizer_data = cv_score_smote_result_for_train[cv_score_smote_result_for_train['count_vectorizer'] == vectorizer]

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='n_gram', hue='pre_process', y='cv_f1', data=vectorizer_data, width=0.6, palette='viridis')
    plt.title(f'{vectorizer} - F1 Macro Scores Boxplot')
    # set hue legend outside the plot
    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


def boxplot_cv(df, metric):
    for vectorizer in df['count_vectorizer'].unique():
        vectorizer_data = df[df['count_vectorizer'] == vectorizer]

        # Create the boxplot
        plt.figure(figsize=(12, 6))
        plot = sns.boxplot(x='n_gram', hue='pre_process', y=metric, data=vectorizer_data, width=0.6, palette='viridis')
        plt.title(f'{vectorizer} - {metric} Macro Scores Boxplot')
        # set hue legend outside the plot
        plt.subplots_adjust(right=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        fig = plot.get_figure()
        path = Path(
            os.path.abspath(f'../resources/all_graph_0_0_2/{get_var_name(df)}_{vectorizer}_{metric}_cv_boxplot.png'))
        fig.savefig(path)


def boxplot_predict(df, metric):
    for vectorizer in df['count_vectorizer'].unique():
        vectorizer_data = df[df['count_vectorizer'] == vectorizer]

        # Create the boxplot
        plt.figure(figsize=(12, 6))
        plot = sns.boxplot(x='n_gram', hue='pre_process', y=metric, data=vectorizer_data, width=0.6, palette='viridis')
        plt.title(f'{vectorizer} - {metric}  Scores Boxplot')
        # set hue legend outside the plot
        plt.subplots_adjust(right=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        fig = plot.get_figure()
        path = Path(os.path.abspath(
            f'../resources/all_graph_0_0_2/{get_var_name(df)}_{vectorizer}_{metric}_predict_boxplot.png'))
        fig.savefig(path)

# Call the function to save graph for each metric
# boxplot_cv(cv_score_normal_result_for_train, 'cv_f1')
# boxplot_cv(cv_score_normal_result_for_train, 'cv_precision')
# boxplot_cv(cv_score_normal_result_for_train, 'cv_recall')
# boxplot_cv(cv_score_normal_result_for_train_normalize_min_max, 'cv_f1')
# boxplot_cv(cv_score_normal_result_for_train_normalize_min_max, 'cv_precision')
# boxplot_cv(cv_score_normal_result_for_train_normalize_min_max, 'cv_recall')
# boxplot_cv(cv_score_normal_result_for_train_normalize_log, 'cv_f1')
# boxplot_cv(cv_score_normal_result_for_train_normalize_log, 'cv_precision')
# boxplot_cv(cv_score_normal_result_for_train_normalize_log, 'cv_recall')
#
# boxplot_cv(cv_score_smote_result_for_train, 'cv_f1')
# boxplot_cv(cv_score_smote_result_for_train, 'cv_precision')
# boxplot_cv(cv_score_smote_result_for_train, 'cv_recall')
# boxplot_cv(cv_score_smote_result_for_train_normalize_min_max, 'cv_f1')
# boxplot_cv(cv_score_smote_result_for_train_normalize_min_max, 'cv_precision')
# boxplot_cv(cv_score_smote_result_for_train_normalize_min_max, 'cv_recall')
# boxplot_cv(cv_score_smote_result_for_train_normalize_log, 'cv_f1')
# boxplot_cv(cv_score_smote_result_for_train_normalize_log, 'cv_precision')
# boxplot_cv(cv_score_smote_result_for_train_normalize_log, 'cv_recall')
#
# boxplot_predict(predict_normal_result_for_train, 'predicts_f1')
# boxplot_predict(predict_normal_result_for_train, 'predicts_precision')
# boxplot_predict(predict_normal_result_for_train, 'predicts_recall')
# boxplot_predict(predict_normal_result_for_train, 'predicts_auc')
# boxplot_predict(predict_normal_result_for_train_normalize_min_max, 'predicts_f1')
# boxplot_predict(predict_normal_result_for_train_normalize_min_max, 'predicts_precision')
# boxplot_predict(predict_normal_result_for_train_normalize_min_max, 'predicts_recall')
# boxplot_predict(predict_normal_result_for_train_normalize_min_max, 'predicts_auc')
# boxplot_predict(predict_normal_result_for_train_normalize_log, 'predicts_f1')
# boxplot_predict(predict_normal_result_for_train_normalize_log, 'predicts_precision')
# boxplot_predict(predict_normal_result_for_train_normalize_log, 'predicts_recall')
# boxplot_predict(predict_normal_result_for_train_normalize_log, 'predicts_auc')
#
# boxplot_predict(predict_smote_result_for_train, 'predicts_f1')
# boxplot_predict(predict_smote_result_for_train, 'predicts_precision')
# boxplot_predict(predict_smote_result_for_train, 'predicts_recall')
# boxplot_predict(predict_smote_result_for_train, 'predicts_auc')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_min_max, 'predicts_f1')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_min_max, 'predicts_precision')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_min_max, 'predicts_recall')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_min_max, 'predicts_auc')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_log, 'predicts_f1')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_log, 'predicts_precision')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_log, 'predicts_recall')
# boxplot_predict(predict_SMOTE_result_for_train_normalize_log, 'predicts_auc')
