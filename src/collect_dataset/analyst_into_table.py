import pandas as pd

# Read the CSV (adjust path if needed)
# df = pd.read_csv('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/final_training_16_05/final_training/summary_all_results.csv')
df = pd.read_csv('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/training_result_27_5/final_training/summary_all_results.csv')

# Rename columns to match the ranking function’s expectations
df = df.rename(columns={
    'label': 'y_name',
    'test_precision': 'precision_macro',
    'test_recall': 'recall_macro',
    'test_f1': 'f1_macro',
    'test_roc_auc': 'roc_auc'
})

# Split the 'combination' column into separate technique columns
def split_combination(comb_str):
    """
    Expected format:
      Term_pre_process_<pre_process>_n_grams_<min>_<max>
    e.g., 'CountVectorizer_pre_process_spacy_n_grams_1_2'
    """
    term_part, rest = comb_str.split('_pre_process_')
    pre_process, gram_part = rest.split('_n_grams_')
    n_gram = gram_part  # e.g., '1_2'
    return pd.Series({
        'term': term_part,
        'pre_process': pre_process,
        'n_gram': n_gram
    })

# Apply splitting
techniques = df['combination'].apply(split_combination)
df = pd.concat([df, techniques], axis=1)

# Define the ranking function, dropping any rows with missing metrics
def rank_y_name_by_me(df, y_name_value):
    df_sub = df[df['y_name'] == y_name_value].dropna(subset=[
        'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc'])
    df_sub = df_sub.copy()
    metric_columns = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    for metric in metric_columns:
        df_sub[f'{metric}_rank'] = df_sub[metric].rank(ascending=False, method='min')
    df_sub['overall_rank'] = df_sub[[f'{metric}_rank' for metric in metric_columns]].mean(axis=1)
    df_sub['overall_rank'] = df_sub['overall_rank'].rank(ascending=True, method='min').astype(int)
    df_sub = df_sub.sort_values(by='overall_rank').reset_index(drop=True)
    return df_sub

# Apply the ranking function for each y_name and collect results
ranked_list = []
for y_val in df['y_name'].unique():
    ranked_df = rank_y_name_by_me(df, y_val)
    ranked_df['y_name'] = y_val
    ranked_list.append(ranked_df)

ranked_all = pd.concat(ranked_list, ignore_index=True)

# For each unique combination of techniques, check if it achieved top rank (1) for any y_name
best_rank_per_combo = (
    ranked_all
    .groupby(['term', 'pre_process', 'n_gram'])['overall_rank']
    .min()
    .reset_index()
    .rename(columns={'overall_rank': 'best_overall_rank'})
)
best_rank_per_combo['achieve_top_rank'] = best_rank_per_combo['best_overall_rank'] == 1

# Display the summary
print(best_rank_per_combo)

# Optionally, save the summary to CSV
best_rank_per_combo.to_csv('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/combination_top_rank_summary.csv', index=False)
