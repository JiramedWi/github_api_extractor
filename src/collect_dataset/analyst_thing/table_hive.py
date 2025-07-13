import os
import pandas as pd

def rank_table_summary(
    df, metrics_prefix, y_name, top_n=5, aggregation_method="average_all", use_both=False
):
    df_sub = df[df['y_name'] == y_name].copy()
    if use_both:
        metric_columns = [
            "cv_precision_macro", "cv_recall_macro", "cv_f1_macro", "cv_roc_auc",
            "test_precision", "test_recall", "test_f1", "test_roc_auc"
        ]
    elif metrics_prefix == "cv":
        metric_columns = [
            "cv_precision_macro", "cv_recall_macro", "cv_f1_macro", "cv_roc_auc"
        ]
    elif metrics_prefix == "test":
        metric_columns = [
            "test_precision", "test_recall", "test_f1", "test_roc_auc"
        ]
    else:
        raise ValueError("Unknown metrics_prefix for summary table.")
    df_sub = df_sub.dropna(subset=metric_columns)
    for metric in metric_columns:
        df_sub[f"{metric}_rank"] = df_sub[metric].rank(ascending=False, method='min')
    if use_both:
        if aggregation_method == "average_all":
            df_sub["overall_rank"] = df_sub[[f"{m}_rank" for m in metric_columns]].mean(axis=1)
        elif aggregation_method == "average_rank":
            cv_metrics = ["cv_precision_macro_rank", "cv_recall_macro_rank", "cv_f1_macro_rank", "cv_roc_auc_rank"]
            test_metrics = ["test_precision_rank", "test_recall_rank", "test_f1_rank", "test_roc_auc_rank"]
            df_sub["cv_overall"] = df_sub[cv_metrics].mean(axis=1)
            df_sub["test_overall"] = df_sub[test_metrics].mean(axis=1)
            df_sub["overall_rank"] = df_sub[["cv_overall", "test_overall"]].mean(axis=1)
    else:
        df_sub["overall_rank"] = df_sub[[f"{m}_rank" for m in metric_columns]].mean(axis=1)
    df_sub["overall_rank"] = df_sub["overall_rank"].rank(ascending=True, method='min').astype(int)
    for metric in metric_columns:
        rank_col = f"{metric}_rank"
        df_sub[rank_col] = df_sub[rank_col].rank(ascending=True, method='min').astype(int)
    df_sub = df_sub.sort_values("overall_rank").reset_index(drop=True)
    if use_both:
        actual_results = ["cv_f1_macro", "cv_roc_auc", "test_f1", "test_roc_auc"]
    elif metrics_prefix == "cv":
        actual_results = ["cv_f1_macro", "cv_roc_auc"]
    else:
        actual_results = ["test_f1", "test_roc_auc"]
    output_cols = [
        "Textual feature", "Stem lemma", "N-gram", "Topic modeling", "Imba handling", "overall_rank",
        f"{metrics_prefix}_precision_macro_rank",
        f"{metrics_prefix}_recall_macro_rank",
        f"{metrics_prefix}_f1_macro_rank",
        f"{metrics_prefix}_roc_auc_rank",
    ] + actual_results
    if "result" in df_sub.columns:
        output_cols.append("result")
    if "source_name" in df_sub.columns:
        output_cols.append("source_name")
    output_cols = [c for c in output_cols if c in df_sub.columns]
    full_table = df_sub[output_cols]
    top_n_table = df_sub[output_cols].head(top_n)
    return full_table, top_n_table

def make_summary_table_cv(csv_path, top_n=5, output_prefix="summary_cv", save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path, na_filter=False)
    results = {}
    for y_name in df["y_name"].unique():
        full_table, top_n_table = rank_table_summary(df, "cv", y_name, top_n=top_n, use_both=False)
        full_path = os.path.join(save_dir, f"{output_prefix}_full_{y_name}.csv")
        topn_path = os.path.join(save_dir, f"{output_prefix}_top{top_n}_{y_name}.csv")
        full_table.to_csv(full_path, index=False, na_rep="None")
        top_n_table.to_csv(topn_path, index=False, na_rep="None")
        results[y_name] = {"full": full_table, "top": top_n_table}
    print(f"Saved CV summary tables for all test smell categories in {save_dir}.")
    return results

def make_summary_table_predict(csv_path, top_n=5, output_prefix="summary_predict", save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    results = {}
    for y_name in df["y_name"].unique():
        full_table, top_n_table = rank_table_summary(df, "test", y_name, top_n=top_n, use_both=False)
        full_path = os.path.join(save_dir, f"{output_prefix}_full_{y_name}.csv")
        topn_path = os.path.join(save_dir, f"{output_prefix}_top{top_n}_{y_name}.csv")
        full_table.to_csv(full_path, index=False, na_rep="None")
        top_n_table.to_csv(topn_path, index=False, na_rep="None")
        results[y_name] = {"full": full_table, "top": top_n_table}
    print(f"Saved Predict summary tables for all test smell categories in {save_dir}.")
    return results

def make_summary_table_both(csv_path, top_n=5, aggregation_method="average_all", output_prefix="summary_both", save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path, na_filter=False)
    results = {}
    for y_name in df["y_name"].unique():
        full_table, top_n_table = rank_table_summary(
            df, "cv", y_name, top_n=top_n, use_both=True, aggregation_method=aggregation_method
        )
        full_path = os.path.join(save_dir, f"{output_prefix}_full_{aggregation_method}_{y_name}.csv")
        topn_path = os.path.join(save_dir, f"{output_prefix}_top{top_n}_{aggregation_method}_{y_name}.csv")
        full_table.to_csv(full_path, index=False, na_rep="None")
        top_n_table.to_csv(topn_path, index=False, na_rep="None")
        results[y_name] = {"full": full_table, "top": top_n_table}
    print(f"Saved BOTH summary tables (aggregation_method={aggregation_method}) for all test smell categories in {save_dir}.")
    return results


csv_path = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/merged_summary.csv"
save_dir = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/tables"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Generate summary tables for CV, Predict, and Both methods
make_summary_table_cv(csv_path, top_n=10, save_dir=save_dir)
make_summary_table_predict(csv_path, top_n=10, save_dir=save_dir)
make_summary_table_both(csv_path, top_n=10, aggregation_method="average_all", save_dir=save_dir)
make_summary_table_both(csv_path, top_n=10, aggregation_method="average_rank", save_dir=save_dir)
