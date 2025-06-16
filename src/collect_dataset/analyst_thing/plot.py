import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_box_plots_ranks(
        df,
        save_path,
        rank_metrics=None,
        mapping_dict=None
):
    """
    Plots a 2x3 grid of boxplots for each specified rank metric.
    Each grid: 5 boxplots for the columns
      ['Textual feature', 'Stem lemma', 'N-gram', 'Topic modeling', 'Imba handling'].

    Args:
        df (pd.DataFrame): DataFrame containing your results.
        save_path (str): Prefix for the PNG files (metric will be added to the filename).
        rank_metrics (list): List of rank metric columns to plot.
        mapping_dict (dict): Optional. Dict for renaming categorical values.
    """

    # Default metrics to plot if not provided
    if rank_metrics is None:
        rank_metrics = [
            "cv_precision_macro_rank",
            "cv_recall_macro_rank",
            "cv_f1_macro_rank",
            "cv_roc_auc_rank"
        ]

    # The five grouping columns (for your new data structure)
    group_columns = [
        "Textual feature",  # Formerly count_vectorizer
        "Stem lemma",  # Formerly pre_process
        "N-gram",  # Formerly n_gram
        "Topic modeling",  # New column
        "Imba handling"  # Formerly smote
    ]

    # Apply mapping if provided (for pretty plot labels)
    if mapping_dict:
        df = df.copy()
        for col, mapvals in mapping_dict.items():
            if col in df.columns:
                df[col] = df[col].map(mapvals).fillna(df[col])

    for metric in rank_metrics:
        plt.figure(figsize=(18, 10))
        for i, group_col in enumerate(group_columns):
            plt.subplot(2, 3, i + 1)
            # Get sorted unique values for plot order
            order = sorted(df[group_col].unique())
            sns.boxplot(x=group_col, y=metric, data=df, order=order)
            plt.title(f"{metric.replace('_', ' ').capitalize()} by {group_col}")
            plt.xlabel(group_col)
            plt.ylabel("Rank (lower is better)")
            # Print stats to console
            print_boxplot_stats(df, metric, group_col)
        # Optionally: leave the 6th cell empty or add a note
        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.text(0.5, 0.5, "Empty / Summary Cell", ha='center', va='center', fontsize=14, alpha=0.4)
        plt.suptitle(f"{metric.replace('_', ' ').capitalize()} Grid", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_path}_{metric}.png")
        plt.show()


def print_boxplot_stats(df, column, group_by):
    """
    Prints Q1, Median, Q3 and full describe stats for a rank column, grouped by a categorical variable.
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


# ===============================
# Example usage for your workflow:
# ===============================

# Example mapping (update as needed)
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
    # Add more mappings as you like.
}

# For each of your Y datasets:
# - Replace these with your actual DataFrame variables
file_path_prefix = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/tables"  # Example, change as needed

# Apply mapping if you want pretty labels
# ranked_df_code_related = map_values(ranked_df_code_related, mapping_result_for_paper)
# If you have no mapping function, just pass mapping_result_for_paper to the plotting function

ranked_df_code_related = pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_code_related.csv")

generate_box_plots_ranks(
    ranked_df_code_related,
    file_path_prefix,
    mapping_dict=mapping_result_for_paper
)

# Repeat for other Y datasets as needed