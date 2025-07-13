import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

def pairwise_wilcoxon(
    df,
    metric_column="overall_rank",
    id_column="combination_id",
    alpha=0.05,
    group_column=None,
    group_values=None
):
    df = df.copy()

    # Step 1: Filter by group if specified
    if group_column and group_values is not None:
        df = df[df[group_column].isin(group_values)]
        if df.empty:
            raise ValueError("No data left after filtering by group_column and group_values.")

    # Step 2: Create combo ID
    if isinstance(id_column, list):
        df["__combo_id__"] = df[id_column].astype(str).agg('_'.join, axis=1)
        id_col = "__combo_id__"
    else:
        id_col = id_column

    configs = df[id_col].unique()
    results = []
    summary = {cfg: {"win": 0, "tie": 0, "loss": 0} for cfg in configs}

    # Step 3: Loop through all config pairs
    for cfg1, cfg2 in combinations(configs, 2):
        val1 = df[df[id_col] == cfg1][metric_column].values
        val2 = df[df[id_col] == cfg2][metric_column].values

        if len(val1) == len(val2) and len(val1) > 0:
            try:
                stat, p = wilcoxon(val1, val2)
                if p < alpha:
                    if np.mean(val1) > np.mean(val2):  # Lower = better, Changing this logic if needed
                        summary[cfg1]["win"] += 1
                        summary[cfg2]["loss"] += 1
                        outcome = "win"
                    else:
                        summary[cfg1]["loss"] += 1
                        summary[cfg2]["win"] += 1
                        outcome = "loss"
                else:
                    summary[cfg1]["tie"] += 1
                    summary[cfg2]["tie"] += 1
                    outcome = "tie"
            except:
                stat, p = np.nan, np.nan
                outcome = "error"

            results.append({
                "cfg1": cfg1,
                "cfg2": cfg2,
                "statistic": stat,
                "p-value": p,
                "outcome": outcome
            })

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df["win-loss"] = summary_df["win"] - summary_df["loss"]
    summary_df = summary_df.sort_values(by="win-loss", ascending=False)

    return summary_df, pd.DataFrame(results)


file_path_prefix = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/tables"
dataset_dict = {
    "code_related": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_code_related.csv", na_filter=False),
    "dependencies": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_dependencies.csv", na_filter=False),
    "issue_in_test_step": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_issue_in_test_step.csv",
                                      na_filter=False),
    "test_execution": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_execution.csv", na_filter=False),
    "test_semantic_smell": pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_test_semantic_smell.csv",
                                       na_filter=False)
}

columns_to_identify = ["Textual feature"]
code_related_result = pd.read_csv(f"{file_path_prefix}/summary_cv_full_label_code_related.csv", na_filter=False)
summary, pairwise_results = pairwise_wilcoxon(
    code_related_result,
    metric_column="overall_rank",
    id_column= columns_to_identify)
# summary.to_csv("win_tie_loss_summary.csv")
# all_results.to_csv("pairwise_test_results.csv", index=False)
