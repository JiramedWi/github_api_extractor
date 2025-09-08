import gc
import os
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import logging

import psutil
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from lightgbm import LGBMClassifier


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
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_cv_loop"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_30_loop"
    elif system_name == "Windows":
        input_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_flink"
        output_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_30_loop"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)

# Logging setup
log_file = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/cv_predict_train_loop_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
SCORING = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc', 'accuracy']

def train_cv_loop_runs(dataset_name: str, dataset_path: Path, output_path: Path, n_runs: int = 20):
    logging.info(f"📂 Loading dataset: {dataset_path}")

    output_dir = output_path / "cv_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"cv_loop_result_{dataset_name}.pkl"

    # === Load input dataset ===
    try:
        datasets = joblib.load(dataset_path)
    except Exception as e:
        logging.exception(f"[💥 ERROR] Failed to load dataset from: {dataset_path}")
        return

    # If output file exists, load it and update datasets with completed results
    if output_file.exists():
        try:
            completed_datasets = joblib.load(output_file)
            completed_count = sum(1 for d in completed_datasets if "cv_multi_run_scores" in d)
            total = len(datasets)
            logging.info(f"[RESUME] {completed_count} out of {total} datasets already processed in {output_file}. Resuming remaining...")
            # Update datasets with completed results
            for i, d in enumerate(completed_datasets):
                if "cv_multi_run_scores" in d:
                    datasets[i] = d
            if completed_count == total:
                logging.info(f"[DONE] All datasets already processed. Nothing left to do.")
                return
        except Exception as e:
            logging.warning(f"[WARN] Could not load or update from output file: {e}")

    for idx, data in enumerate(datasets):
        if "cv_multi_run_scores" in data:
            logging.info(f"[{dataset_name}|{idx}] Skipping: already has cv_multi_run_scores.")
            continue

        x_fit = data["x_fit"]
        y_fit = data["y_fit"]

        # Validate and convert as needed
        arr_type = type(x_fit).__name__
        arr_dtype = getattr(x_fit, "dtype", None)
        is_sparse = "csr" in arr_type.lower()
        log_pre = f"[{dataset_name}|{idx}|x_fit]"
        if is_sparse:
            if arr_dtype not in ("float32", "float64"):
                logging.info(f"{log_pre} Converting CSR dtype {arr_dtype} to float32")
                x_fit = x_fit.astype("float32")
            else:
                logging.info(f"{log_pre} CSR type, dtype OK: {arr_dtype}")
        else:
            if arr_dtype not in ("float32", "float64"):
                logging.info(f"{log_pre} Converting ndarray dtype {arr_dtype} to float32")
                x_fit = x_fit.astype("float32")
            else:
                logging.info(f"{log_pre} ndarray, dtype OK: {arr_dtype}")
        arr_shape = x_fit.shape
        arr_nnz = x_fit.nnz if is_sparse else "N/A"
        logging.info(f"{log_pre} shape: {arr_shape}, nnz: {arr_nnz}")

        logging.info(f"[{dataset_name}|{idx}] RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

        params = data["best_params"]
        cv_metrics = []

        logging.info(f"🔁 [{dataset_name}] 20x CV | Index {idx} | {data['combination']}")

        try:
            for run_idx in range(n_runs):
                kf = KFold(n_splits=5, shuffle=True)  # random split every run
                clf = LGBMClassifier(**params, n_jobs=-1)
                cv_result = cross_validate(clf, x_fit, y_fit, cv=kf, scoring=SCORING, n_jobs=1)

                cv_metrics.append({
                    'f1_macro': np.mean(cv_result['test_f1_macro']),
                    'precision_macro': np.mean(cv_result['test_precision_macro']),
                    'recall_macro': np.mean(cv_result['test_recall_macro']),
                    'roc_auc': np.mean(cv_result['test_roc_auc']),
                    'accuracy': np.mean(cv_result['test_accuracy']),
                })

            data["cv_multi_run_scores"] = cv_metrics

        except Exception as e:
            logging.exception(f"❌ Error in CV loop at index {idx}: {str(e)}")
            data["cv_multi_run_scores"] = None
            data["cv_error"] = str(e)

        # Save and clean
        for k in ["x_fit", "x_blind_test", "y_fit", "y_blind_test"]:
            if k in data:
                del data[k]

        joblib.dump(datasets, output_file)
        logging.info(f"💾 Checkpoint saved after index {idx}")

        gc.collect()

    logging.info(f"✅ Finished: cv_30_loop_result_{dataset_name}.pkl")
    gc.collect()

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    logging.info("Starting 20 random trainings on datasets...")
    input_path, output_path = get_paths()

    dataset_files = [
        ("normal", input_path / "optuna_result_normal.pkl"),
        ("topic_model", input_path / "optuna_result_topic_model.pkl"),
        ("smote_poly_normal", input_path / "optuna_result_smote_poly_normal.pkl"),
        ("smote_prowsyn_normal", input_path / "optuna_result_smote_prowsyn_normal.pkl"),
        ("smote_poly_topic", input_path / "optuna_result_smote_poly_topic_model.pkl"),
        ("smote_prowsyn_topic", input_path / "optuna_result_smote_prowsyn_topic_model.pkl")
    ]

    for dataset_name, file_path in dataset_files:
        # log information about the dataset being processed
        logging.info(f"Processing dataset: {dataset_name} from {file_path}")
        train_cv_loop_runs(
            dataset_name=dataset_name,
            dataset_path=file_path,
            output_path=output_path,
            n_runs=20,
        )