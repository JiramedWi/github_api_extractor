import os
from pathlib import Path

import joblib
import pandas as pd

result_list = joblib.load(Path(os.path.abspath('../resources/x_y_fit_blind_transform_0_0_2.pkl')))
df_temp_x = []
df_temp_x_test = []
df_temp_y = []
df_temp_y_test = []
for result in result_list:
    result_x = result['x_fit']
    result_x_test = result['x_blind_test']
    result_y = result['y_fit']
    result_y_test = result['y_blind_test']
    result_combi = result['combination']
    df_x = pd.DataFrame(result_x)
    df_x_test = pd.DataFrame(result_x_test)
    df_y = pd.DataFrame(result_y)
    count_y = df_y.value_counts()
    print(f"count_y {count_y}")
    df_y_test = pd.DataFrame(result_y_test)
    count_y_test = df_y_test.value_counts()
    print(f"count_y_test {count_y_test}")
    ratio_class_1_train = count_y[1] / len(df_y)
    ratio_class_1_test = count_y_test[1] / len(df_y_test)
    ratio_class_0_train = count_y[0] / len(df_y)
    ratio_class_0_test = count_y_test[0] / len(df_y_test)
    print(f"\nRatio of class '1' in the training set: {ratio_class_1_train:.2%}")
    print(f"\nRatio of class '0' in the training set: {ratio_class_0_train:.2%}")

    print(f"\nRatio of class '1' in the test set: {ratio_class_1_test:.2%}")
    print(f"\nRatio of class '0' in the test set: {ratio_class_0_test:.2%}")
#     df_temp_x.append(df_x)
#     df_temp_y.append(df_y)
#
# df_x_result = pd.concat(df_temp_x)
# df_y_result = pd.concat(df_temp_y)
