from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os

filepath = Path(os.path.abspath('../resources/clean_demo.pkl'))
df = pd.read_pickle(filepath)
word_counts = df.str.count(' ') + 1

data = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))
data = data[data['body'].notnull()]
# data = data['body'].dropna()


vectorizer = TfidfVectorizer(use_idf=True, token_pattern=r'(?u)\b\w+\b')
x = vectorizer.fit_transform(df)
result = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out(), index=data['url'])
print(vectorizer.get_feature_names_out())

