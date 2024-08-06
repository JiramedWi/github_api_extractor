import os
import joblib

import pandas as pd


def compare_y_to_x(dfx, dfy):
    return dfy.loc[dfy['url'].isin(dfx['url'])]


class DataPreparation:
    def __init__(self, x: str, y: list, ):
        self.x_path = x
        self.y_paths = y

    def read_data(self):
        x = pd.read_pickle(self.x_path)
        y_output = []
        for y_path in self.y_paths:
            y_temp = pd.read_csv(y_path)
            y_to_x = compare_y_to_x(x, y_temp)
            file_path, extension = os.path.splitext(y_path)
            parts = file_path.split('/')
            filename = parts[-1]
            filename = filename.replace('df_', '')
            data_combined = {
                filename: y_to_x['y']
            }
            y_output.append(data_combined)
        return x, y_output


x = '../resources/hive_use_for_run_pre_process.pkl'
y_source = ['../resources/tsdetect/all_test_smell/df_test_semantic_smell.csv',
            '../resources/tsdetect/all_test_smell/df_issue_in_test_step.csv',
            '../resources//tsdetect/all_test_smell/df_code_related.csv',
            '../resources/tsdetect/all_test_smell/df_dependencies.csv',
            '../resources/tsdetect/all_test_smell/df_test_execution.csv']

run = DataPreparation(x, y_source)
x_result, y_result = run.read_data()
joblib.dump(x_result, '../resources/result_0_0_2/x_0_0_2.pkl')
joblib.dump(y_result, '../resources/result_0_0_2/y_0_0_2.pkl')
