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
    print(
        f" 1st rank y average score of {predict_key} = \n{result_predict['top_ranked_y_name']}")  # 1st rank of predict
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

# create all dataset of df to see CV score and predict result
all_df_cv_score = pd.concat([dataframes[key] for key in dataframes if "cv_score" in key])
all_df_predict_result = pd.concat([dataframes[key] for key in dataframes if "predict_score" in key])

# Over-all agreement of the best combination in CV and predict
over_all_agreement_cv_result = all_df_cv_score[["count_vectorizer", "pre_process", "n_gram", "term",
                                                "y_name", "smote",
                                                "precision_macro", "recall_macro", "f1_macro", "roc_auc"]]
over_all_agreement_predict_result = all_df_predict_result[["count_vectorizer", "pre_process", "n_gram",
                                                           "term", "y_name", "smote", "precision_test_score",
                                                           "recall_test_score", "f1_test_score", "roc_auc_test_score"]]


def rank_y_names_by_scores(data):
    # Group the data by y_name and calculate the mean of the scores
    data['overall_agreement_score_rank'] = data[['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']].rank(
        axis=0).mean(axis=1)
    print('')
    # grouped = data.groupby("y_name")[["overall_agreement_score"]].mean()
    #
    # # Rank the y_name based on the overall agreement scores
    # ranked = grouped.mean(axis=1).sort_values(ascending=False).reset_index()
    # ranked.columns = ['y_name', 'average_score']
    #
    # # Add a rank column
    # ranked['rank'] = ranked['average_score'].rank(ascending=False)

    # Add a rank column by overall agreement score
    # data['overall_agreement_score_rank'] = data['overall_agreement_score'].rank(ascending=False)

    # Merge ranked data back with original dataframe to get full information
    # merged_df = pd.merge(data, ranked, on='y_name', how='inner')

    # Melt the DataFrame for easier plotting with seaborn
    melted_df = pd.melt(
        data,
        id_vars=[
            "y_name", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
            "overall_agreement_score_rank"],
        value_vars=['count_vectorizer', 'pre_process', 'n_gram', 'term'],
        var_name='technique',
        value_name='technique_value'
    )

    # Generate a boxplot for each technique
    # This plot shows the distribution of overall agreement scores for each technique is not impact about the rank
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='technique_value', y='overall_agreement_score_rank', hue='technique', data=melted_df)
    plt.title('Boxplot of Overall Agreement Scores by Technique')
    plt.xlabel('Technique')
    plt.ylabel('Overall Agreement rank')
    plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../resources/optuna_plot/overall_agreement_scores_by_technique.png')
    plt.show()

    return data


rank_y_names_by_scores(over_all_agreement_cv_result)


def find_top_n_ranks(data, N, y_name):
    df = data.copy()
    # Change into float
    for column in ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Filter metrics we used
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    top_n_list = []
    # Recieve only y_name that we want
    df_filtered = df[df['y_name'] == y_name]

    # Find the top N ranks for each metric
    for metric in metrics:
        top_n_df = df_filtered.nlargest(N, metric).copy()
        top_n_df['metric'] = metric
        top_n_df['rank_at_metric'] = top_n_df[metric].rank(ascending=False)
        top_n_list.append(top_n_df)

    # Combine the top N DataFrames into a single DataFrame
    top_n_combined_df = pd.concat(top_n_list, ignore_index=True)
    top_n_combined_df = top_n_combined_df[
        ['rank_at_metric', 'metric', 'count_vectorizer', 'pre_process', 'n_gram', 'term',
         'y_name', 'smote','precision_macro', 'recall_macro', 'f1_macro', 'roc_auc',
         'overall_agreement_score_rank']]

    return top_n_combined_df


top_5_cv_rank_code_relate = find_top_n_ranks(over_all_agreement_cv_result, 5, 'code_related')
top_10_cv_rank_code_relate = find_top_n_ranks(over_all_agreement_cv_result, 10, 'code_related')
top_20_cv_rank_code_relate = find_top_n_ranks(over_all_agreement_cv_result, 20, 'code_related')

top_5_cv_rank_test_sementic = find_top_n_ranks(over_all_agreement_cv_result, 5, 'test_sementic')
top_10_cv_rank_test_sementic = find_top_n_ranks(over_all_agreement_cv_result, 10, 'test_sementic')
top_20_cv_rank_test_sementic = find_top_n_ranks(over_all_agreement_cv_result, 20, 'test_sementic')

top_5_cv_rank_issue_in_test_step = find_top_n_ranks(over_all_agreement_cv_result, 5, 'issue_in_test_step')
top_10_cv_rank_issue_in_test_step = find_top_n_ranks(over_all_agreement_cv_result, 10, 'issue_in_test_step')
top_20_cv_rank_issue_in_test_step = find_top_n_ranks(over_all_agreement_cv_result, 20, 'issue_in_test_step')
