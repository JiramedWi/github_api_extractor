import pandas as pd
import os
from pathlib import Path

flink_pull_request = "../resources/pull_request_projects/flink_pulls.pkl"
flink_df = pd.read_pickle(flink_pull_request)

cassandra_pull_request = "../resources/pull_request_projects/cassandra_pulls.pkl"
cassandra_df = pd.read_pickle(cassandra_pull_request)


def remove_invalid_rows(df, columns, invalid_values=[None, '-', '']):
    filtered_df = df[
        ~df[columns].apply(lambda x: x.isin(invalid_values).any(), axis=1)
    ]
    return filtered_df[columns]


filtered_df_flink = remove_invalid_rows(flink_df, ['url', 'title', 'body', 'base.sha', 'merge_commit_sha'])
