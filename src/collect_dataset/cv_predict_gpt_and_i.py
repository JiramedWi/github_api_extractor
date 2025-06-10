import gc
import os
import logging
import platform
import psutil

import joblib
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
from sklearn.exceptions import UndefinedMetricWarning

# Function to determine paths dynamically

def get_paths():
    input_directory = os.getenv("INPUT_DIR_TRAINING")
    output_directory = os.getenv("OUTPUT_DIR_TRAINING")

    if input_directory and output_directory:
        logging.info("Using environment variables for paths.")
        return Path(input_directory), Path(output_directory)

    system_name = platform.system()
    logging.info(f"Detected OS: {system_name}")

    if system_name == "Linux":
        input_directory = "/app/resources/tsdetect/test_smell_flink"
        output_directory = "/app/resources/tsdetect/test_smell_flink/optuna_result"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result"
    elif system_name == "Windows":
        input_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_04_6"
        output_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/new_training_result_09_6"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)

# Logging setup
log_file = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/log_result_04_6/training_result.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Suppress warnings during cross-validation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# Training and evaluation metrics
SCORING = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']


def train_cv_and_predict(dataset_name: str, dataset_path: Path):
    logging.info(f"📂 Loading dataset: {dataset_path}")

    # === NEW: Load checkpoint if it exists (resuming capability) ===
    output_dir = output_path / "final_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"predict_score_{dataset_name}.pkl"

    if output_file.exists():
        logging.info(f"🔄 Resuming from existing checkpoint: {output_file}")
        datasets = joblib.load(output_file)
    else:
        datasets = joblib.load(dataset_path)

    # Determine which indices are already processed (checkpointing)
    processed_indices = set()
    for idx, data in enumerate(datasets):
        if "test_f1" in data:  # Already processed if test_f1 exists
            processed_indices.add(idx)

    for idx, data in enumerate(datasets):
        if idx in processed_indices:
            logging.info(f"[{dataset_name}|{idx}] Skipping: already processed in checkpoint.")
            continue

        x_fit = data["x_fit"]
        y_fit = data["y_fit"]
        x_blind_test = data["x_blind_test"]
        y_blind_test = data["y_blind_test"]

        # Validate and convert as needed, log shape and nnz
        for arr_name, arr in [("x_fit", x_fit), ("x_blind_test", x_blind_test)]:
            arr_type = type(arr).__name__
            arr_dtype = getattr(arr, "dtype", None)
            is_sparse = "csr" in arr_type.lower()
            log_pre = f"[{dataset_name}|{idx}|{arr_name}]"
            if is_sparse:
                if arr_dtype not in ("float32", "float64"):
                    logging.info(f"{log_pre} Converting CSR dtype {arr_dtype} to float32 (no .toarray())")
                    arr = arr.astype("float32")
                else:
                    logging.info(f"{log_pre} CSR type, dtype OK: {arr_dtype}")
            else:
                if arr_dtype not in ("float32", "float64"):
                    logging.info(f"{log_pre} Converting ndarray dtype {arr_dtype} to float32")
                    arr = arr.astype("float32")
                else:
                    logging.info(f"{log_pre} ndarray, dtype OK: {arr_dtype}")
            # === Log shape and sparsity ===
            arr_shape = arr.shape
            arr_nnz = arr.nnz if is_sparse else "N/A"
            logging.info(f"{log_pre} shape: {arr_shape}, nnz: {arr_nnz}")
            # Write back to correct variable
            if arr_name == "x_fit":
                x_fit = arr
            else:
                x_blind_test = arr

        # === NEW: Log RAM usage here ===
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logging.info(f"[{dataset_name}|{idx}] Current RAM usage: {mem_mb:.2f} MB")

        params = data["best_params"]

        logging.info(f"🔁 [{dataset_name}] CV + Predict | Index {idx} | {data['combination']}")

        try:
            clf = LGBMClassifier(**params, n_jobs=-1)
            cv_results = cross_validate(clf, x_fit, y_fit, cv=5, scoring=SCORING, n_jobs=-1)
            data.update({
                'cv_precision_macro': cv_results['test_precision_macro'].mean(),
                'cv_recall_macro': cv_results['test_recall_macro'].mean(),
                'cv_f1_macro': cv_results['test_f1_macro'].mean(),
                'cv_roc_auc': cv_results['test_roc_auc'].mean()
            })

            clf.fit(x_fit, y_fit)
            y_pred = clf.predict(x_blind_test)
            y_prob = clf.predict_proba(x_blind_test)
            data.update({
                'test_precision': precision_score(y_blind_test, y_pred),
                'test_recall': recall_score(y_blind_test, y_pred),
                'test_f1': f1_score(y_blind_test, y_pred),
                'test_mcc': matthews_corrcoef(y_blind_test, y_pred),
                'test_roc_auc': roc_auc_score(y_blind_test, y_prob[:, 1]),
            })

        except Exception as e:
            logging.error(f"❌ Error on index {idx}: {e}")
            data['error'] = str(e)

        # === NEW: Save checkpoint after this index ===
        joblib.dump(datasets, output_file)
        logging.info(f"💾 Checkpoint saved after index {idx}")

        # === NEW: Explicit memory cleanup ===
        del x_fit, x_blind_test, y_fit, y_blind_test
        gc.collect()

    logging.info(f"✅ Finished: cv_and_predict_score_{dataset_name}.pkl")

if __name__ == "__main__":

    input_path, output_path = get_paths()

    # List your tuned optuna results here (dataset_name, file_path)
    dataset_files = [
        ("normal", input_path / "optuna_result_normal.pkl"),
        ("topic_model", input_path / "optuna_result_topic_model.pkl"),
        ("smote_poly_normal", input_path / "optuna_result_smote_poly_normal.pkl"),
        ("smote_prowsyn_normal", input_path / "optuna_result_smote_prowsyn_normal.pkl"),
        ("smote_poly_topic", input_path / "optuna_result_smote_poly_topic_model.pkl"),
        ("smote_prowsyn_topic", input_path / "optuna_result_smote_prowsyn_topic_model.pkl")
    ]

    for dataset_name, file_path in dataset_files:
        train_cv_and_predict(dataset_name, file_path)


