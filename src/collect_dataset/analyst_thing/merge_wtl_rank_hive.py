import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", None)
import os
from pathlib import Path
import platform
import logging


# ==========================
# Path Setup
# ==========================
def get_paths():
    system_name = platform.system()
    logging.info(f"Detected OS: {system_name}")

    if system_name == "Linux":
        base_dir = Path("/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result")
    elif system_name == "Darwin":
        base_dir = Path("/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result")
    elif system_name == "Windows":
        base_dir = Path("C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result")
    else:
        raise EnvironmentError("Unsupported platform")

    return base_dir, base_dir / "wtl_20_summary", base_dir / "tables"


# ==========================
# Load & Merge WTL + Rank
# ==========================
def merge_wtl_with_rank(wtl_path: Path, rank_path: Path, y_name: str):
    wtl_df = pd.read_csv(wtl_path / f"train_test_wtl_{y_name}_f1_macro.csv")
    rank_df = pd.read_csv(rank_path / f"summary_cv_full_{y_name}.csv")

    merge_cols = ["Textual feature", "Stem lemma", "N-gram", "Topic modeling", "Imba handling"]
    merged_df = pd.merge(wtl_df, rank_df[merge_cols + ["overall_rank", "cv_f1_macro", "cv_roc_auc"]],
                         on=merge_cols, how="left")

    if merged_df["overall_rank"].isna().any():
        logging.warning("Some combinations have missing overall_rank after merge!")

    # Compute win-loss ranking
    merged_df["win_loss_rank"] = merged_df["win-loss"].rank(method="min", ascending=False).astype(int)

    return merged_df


# ==========================
# Grouping Summary
# ==========================
def summarize_by_each_technique(merged_df: pd.DataFrame):
    group_cols = ["Textual feature", "Stem lemma", "N-gram", "Topic modeling", "Imba handling"]

    # Ensure 'None' strings are filled
    for col in group_cols:
        merged_df[col] = merged_df[col].fillna("None")

    for col in group_cols:
        print(f"\n📊 Grouped by: {col}")
        summary = merged_df.groupby(col)[
            ["overall_rank", "win_loss_rank", "win", "tie", "loss"]
        ].agg({
            "overall_rank": ["mean", "std", "count"],
            "win_loss_rank": ["mean", "std"],
            "win": "mean",
            "tie": "mean",
            "loss": "mean"
        }).sort_values(("overall_rank", "mean"))
        print(summary)


# ==========================
# Comparison Function
# ==========================
def compare_group_vs_others(merged_df: pd.DataFrame, focus_conditions: dict, groupby_cols: list, metric: str = "overall_rank"):
    for col in groupby_cols:
        merged_df[col] = merged_df[col].fillna("None")

    focus_mask = pd.Series(True, index=merged_df.index)
    for key, value in focus_conditions.items():
        focus_mask &= (merged_df[key] == value)

    focus_group = merged_df[focus_mask].copy()
    other_group = merged_df[~focus_mask].copy()

    print("\n📌 Focus Group Summary:")
    print(focus_group[[metric, "win", "tie", "loss"]].describe())

    print("\n📌 Other Group Summary:")
    print(other_group[[metric, "win", "tie", "loss"]].describe())

    return pd.concat([focus_group.assign(group="Focus Group"), other_group.assign(group="Others")], ignore_index=True)


# ==========================
# Flexible Summary Generator by Imba x Stem
# ==========================
def generate_pair_summary(
    merged_df: pd.DataFrame,
    imba_list: list = None,
    stem_list: list = None,
    sort_by: str = "overall_rank"
):
    imba_list = imba_list or ["ProWSyn", "Polynomial Fit", "None"]
    stem_list = stem_list or ["textblob", "spacy", "lemmatizer", "porterstemmer"]

    subset_df = merged_df[
        merged_df["Imba handling"].isin(imba_list)
        & merged_df["Stem lemma"].isin(stem_list)
    ].copy()

    if subset_df.empty:
        logging.warning("⚠️ No data matched the specified Imba × Stem filter.")
        return

    subset_df["group"] = subset_df["Imba handling"] + " + " + subset_df["Stem lemma"]

    summary_df = subset_df.groupby("group")[
        ["overall_rank", "win", "tie", "loss"]
    ].agg(["mean", "std", "count"])

    summary_df = summary_df.sort_values((sort_by, "mean"))

    print("\n📊 Summary Table (Custom Imba x Stem Combination):")
    print(summary_df)

    return summary_df


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    log_file = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/merge_wtl_rank_hive.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    base_path, wtl_path, rank_path = get_paths()
    y_name = "code_related"  # change to any label like 'test_execution', etc.

    merged = merge_wtl_with_rank(wtl_path, rank_path, y_name)
    summarize_by_each_technique(merged)

    # Optional: compare ProWSyn + spacy group
    focus_conditions = {"Imba handling": "ProWSyn", "Stem lemma": "spacy"}
    groupby_cols = ["Imba handling", "Stem lemma"]
    comparison_df = compare_group_vs_others(merged, focus_conditions, groupby_cols)

    # Flexible pair summary: try any set!
    generate_pair_summary(
        merged,
        imba_list=["ProWSyn", "Polynomial Fit", "None"],
        stem_list=["textblob", "spacy", "lemmatizer", "porterstemmer"]
    )

    output_file = base_path / f"merged_wtl_rank_{y_name}.csv"
    merged.to_csv(output_file, index=False)
    logging.info(f"✅ Saved merged file with ranks: {output_file}")
