import os
import platform

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon


def get_paths():
    input_directory = os.getenv("INPUT_DIR_TRAINING")
    output_directory = os.getenv("OUTPUT_DIR_TRAINING")

    if input_directory and output_directory:
        logging.info("Using environment variables for paths.")
        return Path(input_directory), Path(output_directory)

    system_name = platform.system()
    logging.info(f"Detected OS: {system_name}")

    if system_name == "Linux":
        input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_10_6"
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_30_loop"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/wtl_30_summary"
    elif system_name == "Windows":
        input_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_04_6"
        output_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/new_training_result_09_6"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


# Logging setup
log_file = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/cv_predict_train_30_loop_v2.log"
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
    all_records = []

    # === Load and collect records ===
    for source_name, file_path in dataset_file_list:
        records = joblib.load(file_path)
        for record in records:
            if record.get("y_name") == y_name:
                record["source_name"] = source_name
                all_records.append(record)

    logging.info(f"✅ Loaded {len(all_records)} records for y_name = {y_name}")

    # === Build score_dict with full unique keys ===
    score_dict = {}
    skipped_empty = 0
    skipped_missing_metric = 0
    skipped_bad_shape = 0

    for r in all_records:
        full_id = f"{r['source_name']}__{r['combination']}"
        run_scores = r.get("cv_multi_run_scores")

        if not run_scores:
            skipped_empty += 1
            continue

        metric_scores = [entry.get(metric_name) for entry in run_scores]
        if None in metric_scores:
            skipped_missing_metric += 1
            continue

        if len(metric_scores) != 20:
            skipped_bad_shape += 1
            continue

        score_dict[full_id] = metric_scores

    logging.info(f"✅ Valid combinations with 20 runs of '{metric_name}': {len(score_dict)}")
    logging.info(f"⛔ Skipped (empty 'cv_multi_run_scores'): {skipped_empty}")
    logging.info(f"⛔ Skipped (missing '{metric_name}' in some runs): {skipped_missing_metric}")
    logging.info(f"⛔ Skipped (not exactly 20 scores): {skipped_bad_shape}")

    # === WTL Comparison ===
    summary = {cfg: {"win": 0, "tie": 0, "loss": 0} for cfg in score_dict.keys()}
    for cfg1, cfg2 in combinations(score_dict.keys(), 2):
        vals1 = score_dict[cfg1]
        vals2 = score_dict[cfg2]
        # Skip comparison if all elements are exactly equal
        if vals1 == vals2:
            summary[cfg1]["tie"] += 1
            summary[cfg2]["tie"] += 1
            continue
        try:
            stat, p = wilcoxon(vals1, vals2, zero_method="zsplit")
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
            logging.exception(f"❌ Error comparing {cfg1} vs {cfg2}: {str(e)}")

    # === Merge WTL Results + Technique Info ===
    all_rows = []
    for r in all_records:
        full_id = f"{r['source_name']}__{r['combination']}"
        if full_id not in summary:
            continue

        comb_row = parse_combination(r["combination"])
        row = {
            "combination": r["combination"],
            "source_name": r["source_name"],
            "y_name": r["y_name"],
            "Textual feature": comb_row["Textual feature"],
            "Stem lemma": comb_row["Stem lemma"],
            "N-gram": comb_row["N-gram"],
            "Imba handling": infer_imba_handling(r["source_name"]),
            "Topic modeling": infer_topic_model(r["source_name"], comb_row["Textual feature"]),
            "win": summary[full_id]["win"],
            "tie": summary[full_id]["tie"],
            "loss": summary[full_id]["loss"],
            "win-loss": summary[full_id]["win"] - summary[full_id]["loss"]
        }
        all_rows.append(row)

    return pd.DataFrame(all_rows)


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

    # y_name = "label_test_semantic_smell"
    # y_name = "label_dependencies"
    # y_name = 'label_test_execution'
    # y_name = 'label_issue_in_test_step'
    y_name = 'label_code_related'
    metric_name = "f1_macro"

    save_path_file = output_path / f"train_test_wtl_{y_name}_{metric_name}.csv"
    df = calculate_wtl_and_merge(dataset_files, y_name, metric_name)
    df.to_csv(save_path_file, index=False)
    logging.info("✅ Final WTL summary table saved!")
