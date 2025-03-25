

# test run subprocess using java
import subprocess
import os
import pandas as pd


# get the current working directory

# subprocess.run(['java', '-jar', '/home/pee/repo/tmp_flink/tsdetect/TestSmellDetector.jar',
#                          '/home/pee/repo/tmp_flink/csv/closed_flink_file_0_82b628d4730eef32b2f7a022e3b73cb18f950e6e.csv',
#                          'closed', '82b628d4730eef32b2f7a022e3b73cb18f950e6e', '25132'],)

# result = subprocess.run(['java', '-jar', '/home/pee/repo/tmp_flink/tsdetect/TestSmellDetector.jar',
#                          '/home/pee/repo/tmp_flink/csv/closed_flink_file_0_82b628d4730eef32b2f7a022e3b73cb18f950e6e.csv',
#                          'closed', '82b628d4730eef32b2f7a022e3b73cb18f950e6e', '25132'],
#                capture_output=True, text=True, check=True)

# try:
#     result = subprocess.run(['java', '-jar', '/home/pee/repo/tmp_flink/tsdetect/TestSmellDetector.jar',
#                          '/home/pee/repo/tmp_flink/csv/closed_flink_file_0_82b628d4730eef32b2f7a022e3b73cb18f950e6e.csv',
#                          'closed', '82b628d4730eef32b2f7a022e3b73cb18f950e6e', '25132']
#                             , capture_output=True, text=True, check=True)
#     print("Output:", result.stdout)
# except subprocess.CalledProcessError as e:
#     print("Error Output:", e.stderr)

# read file csv with pandas
file_path = '/home/pee/repo/tmp_flink/csv/closed_flink_file_0_82b628d4730eef32b2f7a022e3b73cb18f950e6e.csv'
df = pd.read_csv(file_path, header=None, names=['app','url'])
index_result = df.index[df['url'] == "/home/pee/repo/tmp_flink/clone_repo_0_flink/flink-table/flink-table-code-splitter/src/test/resources/declaration/expected/TestNotRewriteLocalVariableInFunctionWithReturnValue.java"]

test_csv = pd.read_csv('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/test_smell.csv', header=0)
