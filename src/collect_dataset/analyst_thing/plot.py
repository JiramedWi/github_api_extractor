import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




def generate_box_plots_ranks(
    df,
    dataset_key,
    save_dir,
    rank_metrics=None,
    mapping_dict=None,
    dataset_title_map=None
):
    """
    Plots a 2x3 grid of boxplots for each specified rank metric, with custom big titles and output filenames
    based on dataset_key and dataset_title_map.

    Parameters:
        df (pd.DataFrame): DataFrame with experiment results.
        dataset_key (str): Key for the dataset (e.g. "code_related", "dependencies", etc.).
        save_dir (str): Directory to save the PNGs.
        rank_metrics (list of str): List of rank columns to plot. If None, uses common defaults.
        mapping_dict (dict): Optional mapping for categorical column values (for pretty labels).
        dataset_title_map (dict): Mapping of dataset_key to human-readable title for plot.
    """

    if rank_metrics is None:
        rank_metrics = [
            "overall_rank",
            "cv_precision_macro_rank",
            "cv_recall_macro_rank",
            "cv_f1_macro_rank",
            "cv_roc_auc_rank"
        ]

    group_columns = [
        "Textual feature",    # Formerly count_vectorizer
        "Stem lemma",         # Formerly pre_process
        "N-gram",             # Formerly n_gram
        "Topic modeling",     # New!
        "Imba handling"       # Formerly smote
    ]

    # Optional: Map for display names in plot
    dataset_title = dataset_title_map.get(dataset_key, dataset_key) if dataset_title_map else dataset_key

    # Optional: Mapping for categorical value display
    if mapping_dict:
        df = df.copy()
        for col, mapvals in mapping_dict.items():
            if col in df.columns:
                df[col] = df[col].map(mapvals).fillna(df[col])

    for metric in rank_metrics:
        plt.figure(figsize=(18, 10))
        for i, group_col in enumerate(group_columns):
            plt.subplot(2, 3, i+1)
            order = sorted(df[group_col].unique())
            sns.boxplot(x=group_col, y=metric, data=df, order=order)
            plt.title(f"overall agreement rank by {group_col} technique")
            plt.xlabel(group_col)
            plt.ylabel("Rank (lower is better)")
            print_boxplot_stats(df, metric, group_col)
        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.text(0.5, 0.5, "Empty / Summary Cell", ha='center', va='center', fontsize=14, alpha=0.4)
        # Main big title:
        plt.suptitle(
            f'Boxplot of the overall ranking analysis of each technique used in combination to predict "{dataset_title}"',
            fontsize=18
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save with descriptive filename
        filename = f"{save_dir}/boxplot_{metric}_{dataset_key}.png"
        plt.savefig(filename)
        plt.show()


def print_boxplot_stats(df, column, group_by):
    """
    Prints Q1, Median, Q3 and summary stats for a rank column, grouped by a categorical variable.
    """
    grouped = df.groupby(group_by)[column]
    stats = grouped.describe(percentiles=[.25, .5, .75])
    print(f"\nBoxplot stats for {column} grouped by {group_by}:\n")
    print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
    for name, group in grouped:
        q1 = group.quantile(0.25)
        median = group.median()
        q3 = group.quantile(0.75)
        print(f"{name}: Q1={q1}, Median={median}, Q3={q3}")

dataset_title_map = {
    "code_related": "Code-related smell",
    "dependencies": "Dependencies smell",
    "issue_in_test_step": "Issue in test step smell",
    "test_execution": "Test execution smell",
    "test_semantic_smell": "Test semantic smell"
}

# Example mapping dictionary (update as needed for your value mappings)
mapping_result_for_paper = {
    "Textual feature": {
        "TF-IDF": "TF-IDF",
        "TF": "TF"
    },
    "Stem lemma": {
        "lemmatizer": "wordnet",
        "porterstemmer": "porterstemmer",
        "spacy": "spacy",
        "textblob": "textblob"
    }
}

# - Replace these with your actual DataFrame variables
file_path_prefix = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/tables"  # Example, change as needed

# Apply mapping if you want pretty labels
# ranked_df_code_related = map_values(ranked_df_code_related, mapping_result_for_paper)
# If you have no mapping function, just pass mapping_result_for_paper to the plotting function

dataset_dict = {
    "code_related": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_code_related.csv", na_filter=False),
    "dependencies": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_dependencies.csv", na_filter=False),
    "issue_in_test_step": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_issue_in_test_step.csv", na_filter=False),
    "test_execution": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_execution.csv", na_filter=False),
    "test_semantic_smell": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_semantic_smell.csv", na_filter=False)
}

for dataset_key, df in dataset_dict.items():
    generate_box_plots_ranks(
        df,
        dataset_key=dataset_key,
        save_dir=file_path_prefix,
        mapping_dict=mapping_result_for_paper,
        dataset_title_map=dataset_title_map
    )