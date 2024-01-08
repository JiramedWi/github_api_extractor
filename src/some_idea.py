from itertools import combinations

import joblib
import pandas as pd
from pathlib import Path
import os
from get_github_api import github_api

path = os.path.dirname(__file__)
# path = os.getcwd()
# hive_request = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
filepath = Path(os.path.abspath(os.path.join(path, '../resources/clean_demo.pkl')))


# x = pd.read_pickle(filepath)
# data = hive_request[hive_request['title_n_body'].notnull()]
# data.rename(columns={'title_n_body': 'title_n_body_not_clean'}, inplace=True)
# data = pd.concat([data, x.dropna()], axis=1)
# data.rename(columns={0: 'title_n_body_clean'}, inplace=True)
# data.to_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))

# fruits = ["apple", "banana", "cherry"]
# for i, x in enumerate(fruits):
#     for item in combinations(fruits, i + 1):
#         print(item)
def check_unique_values(pandas_objects):
    for idx, obj in enumerate(pandas_objects):
        if isinstance(obj, pd.DataFrame):
            unique_values = obj.apply(lambda col: col.nunique())
        elif isinstance(obj, pd.Series):
            unique_values = obj.unique()
        else:
            unique_values = None

        print(f"Unique values in object {idx + 1}:\n{unique_values}\n")


def check_same_unique_values(dataframes):
    reference_unique_values = dataframes[0].apply(lambda col: col.unique()).to_dict()

    for df in dataframes[1:]:
        current_unique_values = df.apply(lambda col: col.unique()).to_dict()

        if current_unique_values != reference_unique_values:
            return False

    return True


result_list = joblib.load(Path(os.path.abspath('../resources/result_0.0.2/x_y_fit_blind_transform_0_0_2.pkl')))
df_temp = []
for result in result_list:
    result_x = result['x_fit']
    result_combi = result['combination']
    df = pd.DataFrame(result_x)
    df_temp.append(df)
df = pd.concat(df_temp)

fn = os.path.join(path, '../resource')
# fn = Path(__file__).parent/'..'
# fn =  os.path.join(os.path.dirname(__file__), '..','resources')
print(path)
print(os.path.abspath(fn))

# hive_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_all_closed_requests.pkl')))
# hive_test = pd.read_pickle(Path(os.path.abspath('../resources/apache_hive_testing_merged_pulls.pkl')))
# flink_issue = pd.read_pickle(Path(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl')))
# json_iteretor = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl')))
# json_iteretor_testpull = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_testing_pulls.pkl')))
# json_iteretor_merge_testpull = pd.read_pickle(Path(os.path.abspath('../resources/json'
#                                                                    '-iterator_java_testing_merged_pulls.pkl')))
#
# # test_pull = test_pull.reset_index(drop=True)
# hive_test = hive_test.set_index('pulls_url')
# hive_test.to_pickle(Path(os.path.abspath('../resources/apache_hive_test_fixed_Merged_requests.pkl')))
#
#
#
# flink_issue = flink_issue.reset_index(drop=True)
# flink_issue_merge = flink_issue[flink_issue['merged_at'].notna()]
# hive_issue_merge = hive_issue[hive_issue['merged_at'].notna()]
# hive_issue_merge.to_pickle(Path(os.path.abspath('../resources/apache_hive_all_Merged_requests.pkl')))
# print('done')
# df = pd.read_pickle(Path(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl')))


# url = github_api.extract_url('https://github.com/apache/hive')
# # result = github_api.get_starred_at(url[0], url[1])
# # df1 = pd.json_normalize(result)

# s = github_api.check_rate_limit('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')
# s = pd.json_normalize(s)
