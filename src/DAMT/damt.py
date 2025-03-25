import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

ranked_df_code_related = pd.read_pickle(
    "/Users/Jumma/git_repo/github_api_extractor/src/DAMT/ranked_df_code_related.pkl")
ranked_df_issue_in_test_step = pd.read_pickle(
    "/Users/Jumma/git_repo/github_api_extractor/src/DAMT/ranked_df_issue_in_test_step.pkl")
ranked_df_test_semantic_smell = pd.read_pickle(
    "/Users/Jumma/git_repo/github_api_extractor/src/DAMT/ranked_df_test_semantic_smell.pkl")


# Define a function to map the old values to the new values
def map_values(df, mapping_dict):
    df = df.copy()
    for col in df.columns:
        if col in mapping_dict:
            df[col] = df[col].map(mapping_dict[col]).fillna(df[col])
    return df


def generate_box_plots(df, save_path):
    plt.figure(figsize=(15, 15))

    def print_boxplot_stats(column, group_by):
        grouped = df.groupby(group_by)[column]
        stats = grouped.describe(percentiles=[.25, .5, .75])
        print(f"\nBoxplot stats for {column} grouped by {group_by}:\n")
        print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
        for name, group in grouped:
            q1 = group.quantile(0.25)
            median = group.median()
            q3 = group.quantile(0.75)
            print(f"{name}: Q1={q1}, Median={median}, Q3={q3}")

    # Specify the desired order for each categorical variable
    pre_process_order = sorted(df['pre_process'].unique())
    smote_order = sorted(df['smote'].unique())
    n_gram_order = sorted(df['n_gram'].unique())
    count_vectorizer_order = sorted(df['count_vectorizer'].unique())

    # Box plot of pre processing for overall ranks
    plt.subplot(2, 2, 1)
    sns.boxplot(x="pre_process", y="overall_rank", data=df, order=pre_process_order)
    plt.title("Overall agreement rank by stemming and lemmatization techniques")
    plt.xlabel("Stemming and lemmatization techniques")
    plt.ylabel("Overall rank")
    print_boxplot_stats('overall_rank', 'pre_process')

    # Box plot of smote for overall ranks
    plt.subplot(2, 2, 2)
    sns.boxplot(x="smote", y="overall_rank", data=df, order=smote_order)
    plt.title("Overall Rank by imbalanced class handling technique")
    plt.xlabel("Imbalanced class handling techniques")
    plt.ylabel("Overall rank")
    print_boxplot_stats('overall_rank', 'smote')

    # Box plot of n_gram for overall ranks
    plt.subplot(2, 2, 3)
    sns.boxplot(x="n_gram", y="overall_rank", data=df, order=n_gram_order)
    plt.title("Overall agreement rank by N-Gram")
    plt.xlabel("N-gram")
    plt.ylabel("Overall rank")
    print_boxplot_stats('overall_rank', 'n_gram')

    # Box plot of vectorizer technique for overall ranks
    plt.subplot(2, 2, 4)
    sns.boxplot(x="count_vectorizer", y="overall_rank", data=df, order=count_vectorizer_order)
    plt.title("Overall agreement rank by Textual feature techniques")
    plt.xlabel("Textual feature techniques")
    plt.ylabel("Overall rank")
    print_boxplot_stats('overall_rank', 'count_vectorizer')

    # Adjust layout to avoid overlapping
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    plt.savefig(save_path)
    plt.show()


# Example mapping dictionary
mapping_result_for_paper = {
    'count_vectorizer': {
        'TfidfVectorizer': 'TF-IDF',
        'CountVectorizer': 'TF'
    },
    'pre_process': {
        'lemmatizer': 'wordnet',
        'porterstemmer': 'porterstemmer',  # No change
        'spacy': 'spacy',  # No change
        'textblob': 'textblob',  # No change
    }
}
file_path_over_all_rank = "/Users/Jumma/git_repo/github_api_extractor/src/DAMT/"
ranked_df_code_related = map_values(ranked_df_code_related, mapping_result_for_paper)
ranked_df_issue_in_test_step = map_values(ranked_df_issue_in_test_step, mapping_result_for_paper)
ranked_df_test_semantic_smell = map_values(ranked_df_test_semantic_smell, mapping_result_for_paper)

generate_box_plots(ranked_df_code_related, file_path_over_all_rank + '/code_related.png')
print('=========================')
generate_box_plots(ranked_df_issue_in_test_step, file_path_over_all_rank + '/issue_in_test_step.png')
print('=========================')
generate_box_plots(ranked_df_test_semantic_smell, file_path_over_all_rank + '/test_semantic_smell.png')

# def plot_cross_combination_boxplots(df, title):
#     techniques = ["count_vectorizer", "pre_process", "n_gram", "smote", "term", "normalization"]
#     combs = list(combinations(techniques, 2))
#
#     # Create a subplot grid with sufficient size
#     num_combinations = len(combs)
#     cols = 3  # Number of columns for the subplot grid
#     rows = (num_combinations // cols) + (num_combinations % cols > 0)
#
#     fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
#     fig.suptitle('Overall Rank Boxplots by Technique Combinations ' + title, fontsize=20, y=1.02)  # Add a main title
#
#     for i, (tech1, tech2) in enumerate(combs):
#         row = i // cols
#         col = i % cols
#
#         sns.boxplot(x=tech1, y="overall_rank", hue=tech2, data=df, ax=axes[row, col])
#         axes[row, col].set_title(f"Overall Rank by {tech1} and {tech2}")
#         axes[row, col].set_xlabel(tech1)
#         axes[row, col].set_ylabel("Overall Rank")
#         axes[row, col].legend(title=tech2, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # Remove any empty subplots
#     for i in range(num_combinations, rows * cols):
#         fig.delaxes(axes.flatten()[i])
#
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)  # Adjust the top to make space for the main title
#     plt.show()
#
#
# plot_cross_combination_boxplots(ranked_df_code_related, 'code related')
# plot_cross_combination_boxplots(ranked_df_issue_in_test_step, 'Test smell in testing process step')
# plot_cross_combination_boxplots(ranked_df_test_semantic_smell, 'test semantic smell')
