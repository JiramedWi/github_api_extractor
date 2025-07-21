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


def summarize_by_each_technique_save_csv(merged_df: pd.DataFrame, y_name: str, output_path: Path):
    group_cols = ["Textual feature", "Stem lemma", "N-gram", "Topic modeling", "Imba handling"]

    for col in group_cols:
        merged_df[col] = merged_df[col].fillna("None")

    for col in group_cols:
        summary = merged_df.groupby(col)[
            ["overall_rank", "win_loss_rank", "win", "tie", "loss"]
        ].agg({
            "overall_rank": ["mean", "std", "count"],
            "win_loss_rank": ["mean", "std", "count"],
            "win": ["mean", "std", "count"],
            "tie": ["mean", "std", "count"],
            "loss": ["mean", "std", "count"]
        }).sort_values(("overall_rank", "mean"))

        # Console view
        print(f"\n📊 Grouped by: {col}")
        print(summary)

        # Save each summary to CSV
        filename = f"summary_groupby_{col.replace(' ', '_')}_{y_name}.csv"
        output_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path / filename)
        logging.info(f"💾 Saved group summary: {filename}")


# ==========================
# Pair Combination Summary
# ==========================
def generate_pair_summary(merged_df: pd.DataFrame):
    merged_df["Stem lemma"] = merged_df["Stem lemma"].fillna("None")
    merged_df["Imba handling"] = merged_df["Imba handling"].fillna("None")
    merged_df["group"] = merged_df["Imba handling"] + " + " + merged_df["Stem lemma"]

    summary = merged_df.groupby("group")[
        ["overall_rank", "win", "tie", "loss", "win_loss_rank"]
    ].agg(["mean", "std", "count"])

    print("\n📊 Summary Table (Custom Imba x Stem Combination):")
    print(summary)


def generate_pair_summary_save_csv(merged_df: pd.DataFrame, y_name: str, output_path: Path):
    focus_pairs = [
        ("Polynomial Fit", "lemmatizer"),
        ("Polynomial Fit", "porterstemmer"),
        ("Polynomial Fit", "textblob"),
        ("Polynomial Fit", "spacy"),
        ("ProWSyn", "porterstemmer"),
        ("ProWSyn", "textblob"),
        ("ProWSyn", "lemmatizer"),
        ("ProWSyn", "spacy"),
        ("None", "spacy"),
        ("None", "textblob"),
        ("None", "lemmatizer"),
        ("None", "porterstemmer"),
    ]

    merged_df["group"] = merged_df["Imba handling"] + " + " + merged_df["Stem lemma"]

    summary_rows = []

    for imba, stem in focus_pairs:
        label = f"{imba} + {stem}"
        subset = merged_df[(merged_df["Imba handling"] == imba) & (merged_df["Stem lemma"] == stem)].copy()

        if subset.empty:
            logging.warning(f"🚫 No data found for group: {label}")
            continue

        stats = {
            "group": label,
            "overall_rank_mean": subset["overall_rank"].mean(),
            "overall_rank_std": subset["overall_rank"].std(),
            "overall_rank_count": subset["overall_rank"].count(),
            "win_mean": subset["win"].mean(),
            "win_std": subset["win"].std(),
            "win_count": subset["win"].count(),
            "tie_mean": subset["tie"].mean(),
            "tie_std": subset["tie"].std(),
            "tie_count": subset["tie"].count(),
            "loss_mean": subset["loss"].mean(),
            "loss_std": subset["loss"].std(),
            "loss_count": subset["loss"].count(),
            "win_loss_rank_mean": subset["win_loss_rank"].mean(),
            "win_loss_rank_std": subset["win_loss_rank"].std(),
            "win_loss_rank_count": subset["win_loss_rank"].count(),
        }

        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    print("\n📊 Summary Table (Custom Imba x Stem Combination):")
    print(summary_df.set_index("group"))

    # Save as CSV
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"summary_combine_technique_hive_{y_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    logging.info(f"📄 Saved pair summary to {csv_path.name}")


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
    # y_name = "code_related"  # change to any label like 'test_execution', etc.
    # y_name = "test_semantic_smell"
    # y_name = "dependencies"
    # y_name = 'test_execution'
    y_name = 'issue_in_test_step'

    save_csv_path = base_path / "discussion_tables"
    merged = merge_wtl_with_rank(wtl_path, rank_path, y_name)
    summarize_by_each_technique(merged)
    summarize_by_each_technique_save_csv(merged, y_name, save_csv_path)
    generate_pair_summary(merged)
    generate_pair_summary_save_csv(merged, y_name, save_csv_path)

    output_file = base_path / f"merged_wtl_rank_{y_name}.csv"
    merged.to_csv(output_file, index=False)
    logging.info(f"✅ Saved merged file with ranks: {output_file}")
