import math
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
# Main WTL Function
# =====================
def calculate_wtl_and_merge(dataset_file_list, y_name, metric_name="f1_macro", alpha=0.05):
    all_records = []
    comparison_logs = []

    for source_name, file_path in dataset_file_list:
        records = joblib.load(file_path)
        for record in records:
            if record.get("y_name") == y_name:
                record["source_name"] = source_name
                all_records.append(record)

    logging.info(f"✅ Loaded {len(all_records)} records for y_name = {y_name}")

    score_dict = {}
    parsed_metadata = {}
    for r in all_records:
        full_id = f"{r['source_name']}__{r['combination']}"
        run_scores = r.get("cv_multi_run_scores")
        if not run_scores or len(run_scores) != 20:
            continue
        metric_scores = [entry.get(metric_name) for entry in run_scores]
        if None in metric_scores:
            continue
        score_dict[full_id] = metric_scores
        comb_row = parse_combination(r["combination"])
        parsed_metadata[full_id] = {
            "Textual feature": comb_row["Textual feature"],
            "Stem lemma": comb_row["Stem lemma"],
            "N-gram": comb_row["N-gram"],
            "Imba handling": infer_imba_handling(r["source_name"]),
            "Topic modeling": infer_topic_model(r["source_name"], comb_row["Textual feature"]),
        }

    logging.info(f"Starting WTL comparisons for {math.comb(len(score_dict), 2)} configurations...")

    summary = {cfg: {"win": 0, "tie": 0, "loss": 0} for cfg in score_dict.keys()}

    for cfg1, cfg2 in combinations(score_dict.keys(), 2):
        vals1 = score_dict[cfg1]
        vals2 = score_dict[cfg2]

        meta1 = parsed_metadata[cfg1]
        meta2 = parsed_metadata[cfg2]

        if vals1 == vals2:
            summary[cfg1]["tie"] += 1
            summary[cfg2]["tie"] += 1
            result = "tie"
            p = 1.0
        else:
            try:
                stat, p = wilcoxon(vals1, vals2, zero_method="zsplit")
                if p < alpha:
                    if np.mean(vals1) > np.mean(vals2):
                        summary[cfg1]["win"] += 1
                        summary[cfg2]["loss"] += 1
                        result = "win"
                    else:
                        summary[cfg1]["loss"] += 1
                        summary[cfg2]["win"] += 1
                        result = "loss"
                else:
                    summary[cfg1]["tie"] += 1
                    summary[cfg2]["tie"] += 1
                    result = "tie"
            except Exception as e:
                logging.exception(f"Error comparing {cfg1} vs {cfg2}")
                continue

        comparison_logs.append({
            "cfg1": cfg1,
            "cfg2": cfg2,
            "cfg1_mean": np.mean(vals1),
            "cfg2_mean": np.mean(vals2),
            "p_value": p,
            "result": result,
            "cfg1_textual": meta1["Textual feature"],
            "cfg1_stem": meta1["Stem lemma"],
            "cfg1_ngram": meta1["N-gram"],
            "cfg1_imba": meta1["Imba handling"],
            "cfg1_topic": meta1["Topic modeling"],
            "cfg2_textual": meta2["Textual feature"],
            "cfg2_stem": meta2["Stem lemma"],
            "cfg2_ngram": meta2["N-gram"],
            "cfg2_imba": meta2["Imba handling"],
            "cfg2_topic": meta2["Topic modeling"]
        })

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

    return pd.DataFrame(all_rows), pd.DataFrame(comparison_logs)


# =====================
# Main Runner
# =====================
if __name__ == "__main__":
    logging.info("Starting WTL analyst on datasets...")
    input_path, output_path = get_paths()
    dataset_files = [
        ("normal", input_path / "predict_20_loop_result_normal.pkl"),
        ("topic", input_path / "predict_20_loop_result_topic_model.pkl"),
        ("smote_poly_normal", input_path / "predict_20_loop_result_smote_poly_normal.pkl"),
        ("smote_prowsyn_normal", input_path / "predict_20_loop_result_smote_prowsyn_normal.pkl"),
        ("smote_poly_topic", input_path / "predict_20_loop_result_smote_poly_topic.pkl"),
        ("smote_prowsyn_topic", input_path / "predict_20_loop_result_smote_prowsyn_topic.pkl")
    ]

    y_name = "code_related"
    metric_name = "f1_macro"
    #TODO change metric_name to others for analysis

    summary_df, comparison_log_df = calculate_wtl_and_merge(dataset_files, y_name, metric_name)

    summary_df.to_csv(output_path / f"new_way_prove_train_test_wtl_{y_name}_{metric_name}.csv", index=False)
    comparison_log_df.to_csv(output_path / f"new_way_train_test_wtl_pairwise_comparison_{y_name}_{metric_name}.csv", index=False)
    logging.info("✅ Final WTL summary and pairwise comparison tables saved.")