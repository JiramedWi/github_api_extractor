import pandas as pd

file = '../../../home/jiramed_withun/resource_pull_request/apache_flink_all_closed_pull_requests.pkl'

df = pd.read_pickle(file)

value = 'https://api.github.com/repos/apache/flink/pulls/17731'
index_of_value = (df['url'] == value).idxmax()
df = df.loc[index_of_value:]
df.to_pickle('../../../home/jiramed_withun/resource_pull_request/apache_flink_all_closed_pull_requests_start_at_17731.pkl')