import pandas as pd
from pathlib import Path
import os
from get_github_api import github_api


hive_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))
hive_test = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_testing_merged_pulls.pkl')))
flink_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl')))
json_iteretor = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl')))
json_iteretor_testpull = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_testing_pulls.pkl')))
json_iteretor_merge_testpull = pd.read_pickle(Path(os.path.abspath('../resources/json'
                                                                   '-iterator_java_testing_merged_pulls.pkl')))

# test_pull = test_pull.reset_index(drop=True)
hive_test = hive_test.set_index('pulls_url')
hive_test.to_pickle(Path(os.path.abspath('../resources/apache_hive_test_fixed_Merged_requests.pkl')))



flink_issue = flink_issue.reset_index(drop=True)
flink_issue_merge = flink_issue[flink_issue['merged_at'].notna()]
hive_issue_merge = hive_issue[hive_issue['merged_at'].notna()]
hive_issue_merge.to_pickle(Path(os.path.abspath('../resources/apache_hive_all_Merged_requests.pkl')))
print('done')
# df = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl')))


url = github_api.extract_url('https://github.com/apache/hive')
result = github_api.get_starred_at(url[0], url[1])
df1 = pd.json_normalize(result)

s = github_api.check_rate_limit('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')
s = pd.json_normalize(s)
