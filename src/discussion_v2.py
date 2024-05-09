import os.path

import joblib
import pandas as pd

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
file_path = '/home/pee/repo/github_api_extractor/resources/result_optuna_parameter_tuning_round_2/result_as_df/'
# Loop through the file names and read each into a DataFrame
for file_name in file_names:
    # Create a key based on the file name without the extension
    key = file_name.replace(".pkl", "")

    # Read the pickle file into a DataFrame
    dataframes[key] = joblib.load(os.path.join(file_path, file_name))

# find best scores in each models and print them
best_scores_cv = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
# best_scores_cv = ['precision_macro']
best_scores_predict = ['precision_test_score', 'recall_test_score', 'f1_test_score', 'roc_auc_test_score']
# best_scores_predict = ['precision_test_score', 'recall_test_score', 'f1_test_score', 'roc_auc_test_score', 'mcc']


# find best score function
def find_best_score(df, score):
    grouped = df.groupby("y_name")[score].mean()
    ranked = grouped.mean(axis=1).sort_values(ascending=False)
    # print("Ranking of y_name by average scores:")
    # print(ranked)

    # Get the best combination for the top-ranked y_name
    top_ranked_name = ranked.idxmax()
    best_combination = df[df["y_name"] == top_ranked_name]
    # print(f"\nBest combinations for {top_ranked_name}:")
    # print(best_combination)
    return top_ranked_name, best_combination, grouped, ranked


# find best score loop in each model
result_best_scores = {}
for key in dataframes:
    # seperate model key as cv and predict
    if "cv" in key:
        top_ranked_name, best_combination, grouped, ranked = find_best_score(dataframes[key], best_scores_cv)
        result_best_scores[key] = {
            'top_ranked_y_name': top_ranked_name,
            'all_scores': grouped,
            'best_combination': best_combination
        }
    elif "predict" in key:
        print(f"at {key}")
        top_ranked_name, best_combination, grouped, ranked = find_best_score(dataframes[key], best_scores_predict)
        result_best_scores[key] = {
            'top_ranked_y_name': top_ranked_name,
            'all_scores': grouped,
            'best_combination': best_combination
        }
    else:
        print("error!! somethings wrong")


def compare_cv_predict_print(result_best_scores_dataframes, cv_key, predict_key):
    print(f"\nComparison of cv and predict results for each model {cv_key} and {predict_key}:")
    # seperate count_vectorizer and tfidf_vectorizer
    result_cv = result_best_scores_dataframes[cv_key]
    result_predict = result_best_scores_dataframes[predict_key]
    # print result
    print(f" 1st rank y average score of {cv_key} = \n{result_cv['top_ranked_y_name']}")  # 1st rank of cv
    print(f" 1st rank y average score of {predict_key} = \n{result_predict['top_ranked_y_name']}")  # 1st rank of predict
    print(
        f" all rank score of {cv_key} = \n{result_cv['all_scores'].sort_values(by=['y_name'], ascending=False).to_markdown()}")  # all scores of cv
    print(
        f" all rank score of {predict_key} = \n{result_predict['all_scores'].sort_values(by=['y_name'], ascending=False).to_markdown()}")  # all scores of predict
    # print(result_cv['best_combination'], result_predict['best_combination'])


# Compare cv and predict results for each model lda_lsa
cv_lda_lsa = [key for key in result_best_scores.keys() if "lda_lsa" in key and "cv" in key]
predict_lda_lsa = [key for key in result_best_scores.keys() if "lda_lsa" in key and "predict" in key]

for cv_key, predict_key in zip(cv_lda_lsa, predict_lda_lsa):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')
# Compare cv and predict results for each model normalized
cv_normalized = [key for key in result_best_scores.keys() if "normalized" in key and "cv" in key]
predict_normalized = [key for key in result_best_scores.keys() if "normalized" in key and "predict" in key]

for cv_key, predict_key in zip(cv_normalized, predict_normalized):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')

# Compare cv and predict results for each model smote
cv_smote_poly = [key for key in result_best_scores.keys() if "polynom" in key and "cv" in key]
cv_smote_prowsyn = [key for key in result_best_scores.keys() if "prowsyn" in key and "cv" in key]
predict_smote_poly = [key for key in result_best_scores.keys() if "polynom" in key and "predict" in key]
predict_smote_prowsyn = [key for key in result_best_scores.keys() if "prowsyn" in key and "predict" in key]

# compare in polynom
for cv_key, predict_key in zip(cv_smote_poly, predict_smote_poly):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')

# compare in prowsyn
for cv_key, predict_key in zip(cv_smote_prowsyn, predict_smote_prowsyn):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')

# normal
normal_cv = [key for key in result_best_scores.keys() if "normal_df" in key and "cv" in key]
normal_predict = [key for key in result_best_scores.keys() if "normal_df" in key and "predict" in key]

for cv_key, predict_key in zip(normal_cv, normal_predict):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')

# normal normalized
normal_normalized_cv = [key for key in result_best_scores.keys() if "normalized_df" in key and "cv" in key]
normal_normalized_predict = [key for key in result_best_scores.keys() if "normalized_df" in key and "predict" in key]

for cv_key, predict_key in zip(normal_normalized_cv, normal_normalized_predict):
    # seperate count_vectorizer and tfidf_vectorizer
    compare_cv_predict_print(result_best_scores, cv_key, predict_key)
print('---------------------------------------------------------')

