from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os

from src.preprocess import pre_process_porterstemmer, pre_process_lemmetizer
import nltk

filepath = Path(os.path.abspath('../resources/clean_demo.pkl'))
df = pd.read_pickle(filepath)
word_counts = df.str.count(' ') + 1

# Check
data = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
data = data[data['title_n_body'].notnull()]
data = pd.concat([data, df.dropna()], axis=1)


vectorizer_porter_pre = TfidfVectorizer(use_idf=True, preprocessor=pre_process_porterstemmer)
vectorizer_lemm_pre = TfidfVectorizer(use_idf=True, preprocessor=pre_process_lemmetizer)
# vectorizer_lemm_alltags_pre = TfidfVectorizer(use_idf=True, preprocessor=preProcess_lemmetizer_without_cleantags)

x1 = vectorizer_porter_pre.fit_transform(df)
result_porter_pre = pd.DataFrame(x1.toarray(), columns=vectorizer_porter_pre.get_feature_names_out(), index=data['url'])
x2 = vectorizer_lemm_pre.fit_transform(df)
result_lem_pre = pd.DataFrame(x2.toarray(), columns=vectorizer_lemm_pre.get_feature_names_out(), index=data['url'])
# x3 = vectorizer_lemm_alltags_pre.fit_transform(df)
# result_lemm_alltags_pre = pd.DataFrame(x3.toarray(), columns=vectorizer_lemm_alltags_pre.get_feature_names_out(), index=data['url'])


