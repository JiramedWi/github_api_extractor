import pandas as pd
from pathlib import Path
import os

hive_test = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_testing_merged_pulls.pkl')))
hive_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))

# To extract the SHA from pull request
url = hive_test['pulls_url'].drop_duplicates()
hive_issue = hive_issue[hive_issue['url'].isin(url)]
column = ['url', 'base.sha', 'merge_commit_sha']
hive_issue_filtered = hive_issue.filter(column)
hive_issue_filtered = hive_issue_filtered.rename(columns={'base.sha': 'open', 'merge_commit_sha': 'closed'})
hive_issue_filtered = hive_issue_filtered.reset_index(drop=True)
hive_issue_filtered.to_pickle(Path(os.path.abspath('../resources/hive_use_for_run_java.pkl')))

# To extract issue description
hive_issue_description = hive_issue[hive_issue['url'].isin(url)]
hive_issue_description['title_n_body'] = hive_issue_description['title'] + " " + hive_issue_description['body']
column = ['url', 'body', 'base.sha', 'merge_commit_sha', 'title_n_body']
hive_issue_filtered = hive_issue_description.filter(column)
hive_issue_filtered = hive_issue_filtered.rename(
    columns={'body': 'body', 'base.sha': 'open', 'merge_commit_sha': 'closed'})
hive_issue_filtered = hive_issue_filtered.reset_index(drop=True)
hive_issue_filtered.to_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
