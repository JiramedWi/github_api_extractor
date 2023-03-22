import pandas as pd
from pathlib import Path
import os
from get_github_api import github_api


df = pd.read_pickle(Path(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl')))
df = df.reset_index()
# df = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl')))

url = github_api.extract_url('https://github.com/apache/hive')
result = github_api.get_starred_at(url[0], url[1])
df1 = pd.json_normalize(result)

s = github_api.get_stargazers(url[0], url[1], 9)
s = pd.json_normalize(s)
