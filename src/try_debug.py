import pickle

import pandas as pd

# check the data in logger
# last_path = '/home/pee/repo/github_api_extractor/resources/Logger/apache_flink_testing_pulls_at_16925_at4290.pkl'
# df_last = pd.read_pickle(last_path)
#
# first_path ='/home/pee/repo/github_api_extractor/resources/Logger/apache_flink_testing_pulls_at_25067_at30.pkl'
# df_first = pd.read_pickle(first_path)
# unique_pull_url_first = df_first['pulls_url'].unique()
# unique_pull_url_last = df_last['pulls_url'].unique()

flink_pull = '/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_pulls.pkl'
df_flink_pull = pd.read_pickle(flink_pull)
df_flink_pull_tail = df_flink_pull.tail(5)

casandra_pull = '/home/pee/repo/github_api_extractor/resources/pull_request_projects/cassandra_pulls.pkl'
df_cassandra_pull = pd.read_pickle(casandra_pull)
df_cassandra_pull_tail = df_cassandra_pull.tail(5)

testing_pulls_flink = '/home/pee/repo/github_api_extractor/resources/Logger/flink_testing_pulls.pkl'
df_testing_pulls_flink = pd.read_pickle(testing_pulls_flink)
unique_pull_url_flink = df_testing_pulls_flink['pulls_url'].unique()

testing_pulls_casssandra = '/home/pee/repo/github_api_extractor/resources/Logger/cassandra_testing_pulls.pkl'
df_testing_pulls_cassandra = pd.read_pickle(testing_pulls_casssandra)
unique_pull_url_cassandra = df_testing_pulls_cassandra['pulls_url'].unique()

# open hive_use_for_run_pre_process.pkl
hive_use_for_run = pd.read_pickle('/home/pee/repo/github_api_extractor/resources/hive_use_for_run_pre_process.pkl')
