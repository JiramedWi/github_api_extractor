import os.path

import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of file names
file_names = [
    "cv_score_normal_dataset_lda_lsa_df.pkl",
    "cv_score_normal_dataset_lda_lsa_normalized_df.pkl",
    "cv_score_normal_dataset_normalized_df.pkl",
    "cv_score_normal_df.pkl",
    "cv_score_smote_dataset_lda_lsa_normalized_polynom_df.pkl",
    "cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df.pkl",
    "cv_score_smote_dataset_lda_lsa_polynom_df.pkl",
    "cv_score_smote_dataset_lda_lsa_prowsyn_df.pkl",
    "cv_score_smote_dataset_normalized_polynom_df.pkl",
    "cv_score_smote_dataset_normalized_prowsyn_df.pkl",
    "cv_score_smote_polynom_fit_df.pkl",
    "cv_score_smote_prowsyn_fit_df.pkl",
    "predict_score_normal_dataset_lda_lsa_df.pkl",
    "predict_score_normal_dataset_lda_lsa_normalized_df.pkl",
    "predict_score_normal_dataset_normalized_df.pkl",
    "predict_score_normal_df.pkl",
    "predict_score_smote_dataset_lda_lsa_normalized_polynom_df.pkl",
    "predict_score_smote_dataset_lda_lsa_normalized_prowsyn_df.pkl",
    "predict_score_smote_dataset_lda_lsa_polynom_df.pkl",
    "predict_score_smote_dataset_lda_lsa_prowsyn_df.pkl",
    "predict_score_smote_dataset_normalized_polynom_df.pkl",
    "predict_score_smote_dataset_normalized_prowsyn_df.pkl",
    "predict_score_smote_polynom_fit_df.pkl",
    "predict_score_smote_prowsyn_fit_df.pkl"
]

# Dictionary to store DataFrames
dataframes = {}
file_path = '../resources/result_optuna_parameter_tuning_round_2/result_as_df/'
# Loop through the file names and read each into a DataFrame
for file_name in file_names:
    # Create a key based on the file name without the extension
    key = file_name.replace(".pkl", "")

    # Read the pickle file into a DataFrame
    dataframes[key] = joblib.load(os.path.join(file_path, file_name))

# create all dataset of df to see CV score and predict result
all_df_cv_score = pd.concat([dataframes[key] for key in dataframes if "cv_score" in key])
all_df_predict_result = pd.concat([dataframes[key] for key in dataframes if "predict_score" in key])

# Over-all agreement of the best combination in CV and predict
over_all_agreement_cv_result = all_df_cv_score[["count_vectorizer", "pre_process", "n_gram", "term",
                                                "y_name", "smote", "normalization",
                                                "precision_macro", "recall_macro", "f1_macro", "roc_auc"]]
over_all_agreement_cv_result['normalization'].fillna('no', inplace=True)
over_all_agreement_cv_result['term'].fillna('Not use', inplace=True)
over_all_agreement_cv_result.to_csv(file_path + "/over_all.csv")
over_all_agreement_predict_result = all_df_predict_result[["count_vectorizer", "pre_process", "n_gram",
                                                           "term", "y_name", "smote",
                                                           "precision_test_score", "recall_test_score",
                                                           "f1_test_score", "roc_auc_test_score"]]


def rank_y_names_by_scores_ajan_kong(data, y_name):
    data = data.copy()
    # Group the data by y_name and calculate the mean of the scores
    data_at_y = data[data['y_name'] == y_name]
    data_at_y.set_index(
        ['count_vectorizer', 'pre_process', 'n_gram', 'term', 'y_name', 'smote', 'normalization'],
        inplace=True, append=True, drop=False)
    data_at_y_ranked = data_at_y.rank(axis=0).mean(axis=1)
    return data_at_y_ranked


def rank_y_name_by_me(df, y_name_value):
    df = df.copy()

    # Filter the DataFrame by the specified y_name
    df = df[df['y_name'] == y_name_value]
    # identify which columns are metrics
    metric_columns = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    # Rank the techniques for each metric
    for metric in metric_columns:
        df[metric + '_rank'] = df[metric].rank(ascending=False, method='min')
    # and mean the ranks to get the overall rank
    df['overall_rank'] = df[[metric + '_rank' for metric in metric_columns]].mean(axis=1)
    # Convert overall_rank to an integer rank
    df['overall_rank'] = df['overall_rank'].rank(ascending=True, method='min').astype(int)
    # Sort the DataFrame by the overall rank
    df = df.sort_values(by='overall_rank').reset_index(drop=True)

    # Swap columns to make it look good, put technique columns first
    technique_columns = ['count_vectorizer', 'pre_process', 'n_gram', 'term', 'smote', 'normalization']
    # technique_columns = df.columns.difference(
    #     metric_columns + ['overall_rank'] + [metric + '_rank' for metric in metric_columns])
    df = df[list(technique_columns) + ['overall_rank'] + [metric + '_rank' for metric in metric_columns]]

    return df


file_path_to_save_over_all_rank = '../resources/result_optuna_parameter_tuning_round_2/result_as_overall_rank'

ranked_df_code_related = rank_y_name_by_me(over_all_agreement_cv_result, 'code_related')
# ranked_df_code_related.to_pickle(file_path_to_save_over_all_rank + "/ranked_df_code_related.pkl")
ranked_df_test_semantic_smell = rank_y_name_by_me(over_all_agreement_cv_result, 'test_semantic_smell')
# ranked_df_test_semantic_smell.to_pickle(file_path_to_save_over_all_rank + "/ranked_df_test_semantic_smell.pkl")
ranked_df_issue_in_test_step = rank_y_name_by_me(over_all_agreement_cv_result, 'issue_in_test_step')
# ranked_df_issue_in_test_step.to_pickle(file_path_to_save_over_all_rank + "/ranked_df_issue_in_test_step.pkl")
ranked_df_dependency = rank_y_name_by_me(over_all_agreement_cv_result, 'dependencies')
ranked_df_test_execution = rank_y_name_by_me(over_all_agreement_cv_result, 'test_execution')

print('-------------------code related-------------------')
print(ranked_df_code_related.head(5).to_markdown())

print('-------------------test semantic smell-------------------')
print(ranked_df_test_semantic_smell.head(5).to_markdown())

print('-------------------issue in test step-------------------')
print(ranked_df_issue_in_test_step.head(5).to_markdown())

# print('-------------------dependencies-------------------')
# print(ranked_df_dependency.head(5).to_markdown())
#
# print('-------------------test execution-------------------')
# print(ranked_df_test_execution.head(5).to_markdown())

# check combination rank 1 in each y_name
# data = over_all_agreement_cv_result.copy()
# data_at_y = data[data['y_name'] == 'code_related']
# data_at_y.set_index(['count_vectorizer', 'pre_process', 'n_gram', 'term', 'y_name', 'smote'],
#                     inplace=True, append=True, drop=False)
# data_at_y_ranked = data_at_y['precision_macro'].rank(ascending=False, method='min').sort_values(ascending=True)
# print(data_at_y_ranked.head(5).to_markdown())
