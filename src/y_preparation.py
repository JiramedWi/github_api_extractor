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

y = pd.read_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/merge_data.csv')

sha_opened_columns = ["url", "sha_opened", "Test_Smell_Classification_Type", "Total_Of_Test_Smell"]
sha_closed_columns = ["url", "sha_closed", "Test_Smell_Classification_Type", "Total_Of_Test_Smell"]

sha_opened = pd.concat([y[column] for column in sha_opened_columns], axis=1)
sha_opened.rename(columns={'Total_Of_Test_Smell': 'opened_total_test_smell'}, inplace=True)
sha_opened = sha_opened.dropna().reset_index(drop=True)
sha_closed = pd.concat([y[column] for column in sha_closed_columns], axis=1)
sha_closed.rename(columns={'Total_Of_Test_Smell': 'closed_total_test_smell'}, inplace=True)
sha_closed = sha_closed.dropna().reset_index(drop=True)

y_compare = pd.concat([sha_opened, sha_closed['sha_closed'], sha_closed['closed_total_test_smell']], axis=1)
url_value = y_compare['url'].nunique()
y_compare['y'] = pd.DataFrame(
    np.where(y_compare['closed_total_test_smell'] > y_compare['opened_total_test_smell'], 1, 0))
y_compare = y_compare.drop_duplicates(subset=['url', 'Test_Smell_Classification_Type'], keep='last').reset_index(
    drop=True)
y_compare.to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/y_result.csv')


def seperate_test_smell_type(df, classification_type):
    return df[df['Test_Smell_Classification_Type'] == classification_type]


df_test_semantic_smell = seperate_test_smell_type(y_compare, "test_semantic_smell")
df_test_semantic_smell.reset_index(
    drop=True).to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/df_test_semantic_smell.csv')
df_issue_in_test_step = seperate_test_smell_type(y_compare, "issue_in_test_step")
df_issue_in_test_step.reset_index(
    drop=True).to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/df_issue_in_test_step.csv')
df_code_related = seperate_test_smell_type(y_compare, "code_related")
df_code_related.reset_index(
    drop=True).to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/df_code_related.csv')
df_dependencies = seperate_test_smell_type(y_compare, "dependencies")
df_dependencies.reset_index(
    drop=True).to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/df_dependencies.csv')
df_test_execution = seperate_test_smell_type(y_compare, "test_execution")
df_test_execution.reset_index(
    drop=True).to_csv('/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/all_test_smell/df_test_execution.csv')

value = df_test_semantic_smell['y'].value_counts()