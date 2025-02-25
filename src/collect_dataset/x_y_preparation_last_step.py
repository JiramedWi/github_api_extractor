import pandas as pd
from joblib import dump, load
# get x and y data
x = pd.read_pickle('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/flink_clean_description.pkl')
y = pd.read_pickle('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/y_labeled_test_smells.pkl')

# match row with pull_number column x and y
x = x[x['pull_number'].isin(y['pull_number'])]

# make sure x and y sort in the same order with 'pull_number'
x = x.sort_values(by='pull_number')
y = y.sort_values(by='pull_number')

# reset index
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

# if x['pull_number'].equals(y['pull_number']):
#     print("DataFrames aligned successfully!")
# else:
#     print("Mismatch in pull_number columns after sorting.")

# create a list of dictionaries
list_of_dicts = [
    {"test_semantic_smell": y["label_test_semantic_smell"]},
    {"issue_in_test_step": y["label_issue_in_test_step"]},
    {"code_related": y["label_code_related"]},
    {"dependencies": y["label_dependencies"]},
    {"test_execution": y["label_test_execution"]}
]

# pickle data set
dump(x, '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_for_pre_training.pkl')
dump(list_of_dicts, '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/y_for_pre_training.pkl')
