import pandas as pd
from pathlib import Path
import os

hive_test = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_testing_merged_pulls.pkl')))
hive_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))

url = hive_test['pulls_url'].drop_duplicates()
hive_issue = hive_issue[hive_issue['url'].isin(url)]
column = ['url', 'base.sha', 'merge_commit_sha']
hive_issue_filtered = hive_issue.filter(column)
hive_issue_filtered = hive_issue_filtered.rename(columns={'base.sha': 'closed', 'merge_commit_sha': 'open'})
hive_issue_filtered = hive_issue_filtered.reset_index(drop=True)
hive_issue_filtered.to_pickle(Path(os.path.abspath('../resources/hive_use_for_run_java.pkl')))
