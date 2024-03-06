import logging
from datetime import datetime, timezone, timedelta

import joblib
import pandas as pd
import requests

from src.static_method_class import loop_dict_normal_list_to_df

# read file from results optuna
cv_score_datasets_normal = joblib.load('../resources/result_optuna_parameter_tuning/cv_score_datasets_normal.pkl')
cv_score_datasets_smote = joblib.load('../resources/result_optuna_parameter_tuning/cv_score_datasets_smote.pkl')
predict_score_datasets_normal = joblib.load(
    '../resources/result_optuna_parameter_tuning/predict_score_datasets_normal.pkl')
predict_score_datasets_smote = joblib.load(
    '../resources/result_optuna_parameter_tuning/predict_score_datasets_smote.pkl')

# read file best param
best_param_normal = joblib.load('../resources/optuna_result/best_param_of_normal.pkl')
best_param_smote = joblib.load('../resources/optuna_result/best_param_of_smote.pkl')

df_cv_score_datasets_normal = loop_dict_normal_list_to_df(cv_score_datasets_normal,
                                                          ['x_fit', 'x_blind_test', 'y_fit', 'y_blind_test'])
df_cv_score_datasets_smote = loop_dict_normal_list_to_df(cv_score_datasets_smote,
                                                         ['x_fit', 'x_blind_test', 'y_fit', 'y_blind_test'])
df_predict_score_datasets_normal = loop_dict_normal_list_to_df(predict_score_datasets_normal,
                                                               ['x_fit', 'x_blind_test', 'y_fit', 'y_blind_test'])
df_predict_score_datasets_smote = loop_dict_normal_list_to_df(predict_score_datasets_smote,
                                                              ['x_fit', 'x_blind_test', 'y_fit', 'y_blind_test'])


# concat the best parameter to the df
def concat_best_param(df_datasets, df_parameters):
    df = df_datasets
    df['combination'] = df['combination'].str.lower()
    df['y_name'] = df['y_name'].str.lower()
    df_parameters['combination'] = df_parameters['combination'].str.lower()
    df_parameters['y_name'] = df_parameters['y_name'].str.lower()
    df = pd.merge(df, df_parameters, on=['combination', 'y_name'])
    return df


df_cv_score_datasets_normal = concat_best_param(df_cv_score_datasets_normal, best_param_normal)
df_cv_score_datasets_smote = concat_best_param(df_cv_score_datasets_smote, best_param_smote)
df_predict_score_datasets_normal = concat_best_param(df_predict_score_datasets_normal, best_param_normal)
df_predict_score_datasets_smote = concat_best_param(df_predict_score_datasets_smote, best_param_smote)

print(df_cv_score_datasets_normal.groupby('y_name')['roc_auc'].describe().to_markdown())
print(df_cv_score_datasets_smote.groupby('y_name')['roc_auc'].describe().to_markdown())
print(df_predict_score_datasets_smote.groupby('y_name')['roc_auc_test_score'].describe().to_markdown())
print(df_predict_score_datasets_normal.groupby('y_name')['roc_auc_test_score'].describe().to_markdown())
