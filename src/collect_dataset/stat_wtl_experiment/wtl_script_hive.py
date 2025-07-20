import os
import platform

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon
from collections import defaultdict



def get_paths():
    input_directory = os.getenv("INPUT_DIR_TRAINING")
    output_directory = os.getenv("OUTPUT_DIR_TRAINING")

    if input_directory and output_directory:
        logging.info("Using environment variables for paths.")
        return Path(input_directory), Path(output_directory)

    system_name = platform.system()
    logging.info(f"Detected OS: {system_name}")

    if system_name == "Linux":
        input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result_10_6"
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/wtl_20_summary"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/predict_20_runs"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/wtl_20_summary"
    elif system_name == "Windows":
        input_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result_04_6"
        output_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_hive/new_training_result_09_6"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


# Logging setup
log_file = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/train_test_20.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# =====================
# Parsing Functions
# =====================
def parse_combination(combination_str):
    if not isinstance(combination_str, str) or not combination_str:
        return pd.Series({"Textual feature": None, "Stem lemma": None, "N-gram": None})
    if "_pre_process_" in combination_str and "_n_grams_" in combination_str:
        before_pre, after_pre = combination_str.split("_pre_process_")
        stem_and_ngram = after_pre.split("_n_grams_")
        stem_lemma = stem_and_ngram[0]
        n_gram_rest = stem_and_ngram[1]
        if "countvectorizer" in before_pre.lower():
            textual_feature = "TF"
        elif "tfidfvectorizer" in before_pre.lower():
            textual_feature = "TF-IDF"
        if n_gram_rest in ["11", "1_1"]:
            n_gram = "1"
        elif n_gram_rest in ["12", "1_2"]:
            n_gram = "2"
    else:
        textual_feature = combination_str
        stem_lemma = None
        n_gram = None
    return pd.Series({"Textual feature": textual_feature, "Stem lemma": stem_lemma, "N-gram": n_gram})


def infer_imba_handling(source_name):
    s = source_name.lower()
    if "poly" in s:
        return "Polynomial Fit"
    elif "prowsyn" in s:
        return "ProWSyn"
    else:
        return "None"


def infer_topic_model(source_name, vectorizer_name):
    s = source_name.lower()
    v = vectorizer_name
    if "normal" in s:
        return "None"
    elif "topic" in s and v == "TF":
        return "LDA"
    elif "topic" in s and v == "TF-IDF":
        return "LSA"
    else:
        return "None"


# =====================
# WTL Functions
# =====================
def calculate_wtl_and_merge(dataset_file_list, y_name, metric_name="f1_macro", alpha=0.05):
    all_rows = []
    all_records = []
    for source_name, file_path in dataset_file_list:
        records = joblib.load(file_path)
        for record in records:
            if record.get("y_name") == y_name:
                record["source_name"] = source_name
                all_records.append(record)
    logging.info(f"✅ Loaded {len(all_records)} records for y_name = {y_name}")
    score_dict = {
        f"{r['source_name']}__{r['combination']}": [entry[metric_name] for entry in r["cv_multi_run_scores"]]
        for r in all_records
    }
    combinations_list = list(score_dict.keys())

    summary = {cfg: {"win": 0, "tie": 0, "loss": 0} for cfg in combinations_list}
    fail_count = 0
    pairwise_count = 0

    for cfg1, cfg2 in combinations(combinations_list, 2):
        vals1 = score_dict[cfg1]
        vals2 = score_dict[cfg2]
        pairwise_count += 1

        try:
            stat, p = wilcoxon(vals1, vals2)
            if p < alpha:
                if np.mean(vals1) > np.mean(vals2):
                    summary[cfg1]["win"] += 1
                    summary[cfg2]["loss"] += 1
                else:
                    summary[cfg1]["loss"] += 1
                    summary[cfg2]["win"] += 1
            else:
                summary[cfg1]["tie"] += 1
                summary[cfg2]["tie"] += 1
        except Exception as e:
            fail_count += 1
            logging.warning(f"[Skip] Wilcoxon failed: {cfg1} vs {cfg2}: {e}")

    # === Post-check diagnostic for each combination ===
    expected = len(combinations_list) - 1
    invalid_rows = 0
    for comb, result in summary.items():
        total = result["win"] + result["tie"] + result["loss"]
        if total != expected:
            logging.warning(f"[⚠️] {comb}: W+T+L={total} (Expected: {expected})")
            invalid_rows += 1

    logging.info(f"✅ Total valid combinations: {len(combinations_list)}")
    logging.info(f"🔁 Pairwise comparisons attempted: {pairwise_count}")
    logging.info(f"❌ Wilcoxon failures skipped: {fail_count}")
    logging.info(f"⚠️ Combinations with W+T+L ≠ {expected}: {invalid_rows}")
    # Merge WTL + Technique Info
    for record in all_records:
        comb_row = parse_combination(record["combination"])
        row = {
            "combination": record["combination"],
            "y_name": record["y_name"],
            "source_name": record["source_name"],
            "Imba handling": infer_imba_handling(record["source_name"]),
            "Topic modeling": infer_topic_model(record["source_name"], comb_row["Textual feature"]),
            "Textual feature": comb_row["Textual feature"],
            "Stem lemma": comb_row["Stem lemma"],
            "N-gram": comb_row["N-gram"],
            "win": summary[record['combination']]["win"],
            "tie": summary[record['combination']]["tie"],
            "loss": summary[record['combination']]["loss"],
            "win-loss": summary[record['combination']]["win"] - summary[record['combination']]["loss"]
        }
        all_rows.append(row)
    final_df = pd.DataFrame(all_rows)
    return final_df


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Ensured parent directory exists for: {path}")


# =====================
# Main Runner
# =====================
if __name__ == "__main__":
    logging.info("Starting WTL analyst on datasets...")
    input_path, output_path = get_paths()
    ensure_parent_dir(output_path)
    dataset_files = [
        ("normal", input_path / "predict_20_loop_result_normal.pkl"),
        ("topic", input_path / "predict_20_loop_result_topic_model.pkl"),
        ("smote_poly_normal", input_path / "predict_20_loop_result_smote_poly_normal.pkl"),
        ("smote_prowsyn_normal", input_path / "predict_20_loop_result_smote_prowsyn_normal.pkl"),
        ("smote_poly_topic", input_path / "predict_20_loop_result_smote_poly_topic.pkl"),
        ("smote_prowsyn_topic", input_path / "predict_20_loop_result_smote_prowsyn_topic.pkl")
    ]

    # y_name = "test_semantic_smell"
    # y_name = "dependencies"
    y_name = 'test_execution'
    # y_name = 'issue_in_test_step'
    # y_name = 'code_related'
    metric_name = "f1_macro"

    save_path_file = output_path / f"train_test_wtl_{y_name}_{metric_name}.csv"
    df = calculate_wtl_and_merge(dataset_files, y_name, metric_name)
    df.to_csv(save_path_file, index=False)
    logging.info("✅ Final WTL summary table saved!")
