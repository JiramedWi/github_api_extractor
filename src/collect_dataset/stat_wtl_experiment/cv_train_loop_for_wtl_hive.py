import os
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
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
        input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result"
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/train_30_loop"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result"
    elif system_name == "Windows":
        input_directory = "C:/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_flink"
        output_directory = "C:/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_30_loop"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)

# Logging setup
log_file = "C:/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/cv_predict_train_30_loop_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
# ==========================
# Type-Safe Helper Function
# ==========================
def ensure_float32_and_log(arr, name, dataset_name, combo_idx):
    arr_type = type(arr).__name__
    arr_dtype = getattr(arr, "dtype", None)
    is_sparse = "csr" in arr_type.lower()
    log_pre = f"[{dataset_name}|{combo_idx}|{name}]"

    if is_sparse:
        if arr_dtype not in ("float32", "float64"):
            logging.info(f"{log_pre} Converting CSR dtype {arr_dtype} to float32")
            arr = arr.astype("float32")
        else:
            logging.info(f"{log_pre} CSR type, dtype OK: {arr_dtype}")
    else:
        if arr_dtype not in ("float32", "float64"):
            logging.info(f"{log_pre} Converting ndarray dtype {arr_dtype} to float32")
            arr = arr.astype("float32")
        else:
            logging.info(f"{log_pre} ndarray, dtype OK: {arr_dtype}")

    arr_shape = arr.shape
    arr_nnz = arr.nnz if is_sparse else "N/A"
    logging.info(f"{log_pre} shape: {arr_shape}, nnz: {arr_nnz}")

    return arr
# ==========================
# Main Function
# ==========================
def run_30_random_trainings_on_dataset(
    dataset_name,
    input_path,
    output_folder,
    n_runs=30,
    test_size=0.2,
    verbose=True
):
    output_path = Path(output_folder) / f"cv_30_loop_result_{dataset_name}.pkl"

    # ✅ Skip if already done
    if output_path.exists():
        logging.info(f"[⏩] Skipping {dataset_name} — already exists at {output_path}")
        return

    data = joblib.load(input_path)
    updated_data = []

    for idx, record in enumerate(data):
        x = record["x_fit"]
        y = record["y_fit"]
        params = record.get("best_params", {})

        # ✅ Make sure x is float32 and log info
        x = ensure_float32_and_log(x, "x_fit", dataset_name, idx)

        run_metrics = []

        for _ in range(n_runs):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

            model = LGBMClassifier(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1] if len(np.unique(y)) == 2 else None

            result = {
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision_macro": precision_score(y_test, y_pred, average="macro"),
                "recall_macro": recall_score(y_test, y_pred, average="macro"),
                "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            }

            run_metrics.append(result)

        record["multi_run_scores"] = run_metrics

        for field in ["x_fit", "x_blind_test", "y_fit", "y_blind_test"]:
            if field in record:
                del record[field]

        updated_data.append(record)

        if verbose:
            logging.info(f"[{idx + 1}/{len(data)}] Finished 30 runs for combination: {record['combination']}")

    joblib.dump(updated_data, output_path)
    logging.info(f"[✅] Saved updated dataset to: {output_path}")

# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    logging.info("Starting 30 random trainings on datasets...")
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
        run_30_random_trainings_on_dataset(
            dataset_name=dataset_name,
            input_path=file_path,
            output_folder=output_path
        )