import os
import gc
import joblib
import logging
import platform
import psutil
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
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
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/train_test_20_loop"
    elif system_name == "Darwin":  # macOS
        input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive"
        output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result"
    elif system_name == "Windows":
        input_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result"
        output_directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/train_test_20_loop"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


# Logging setup
log_file = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/latest_result/train_test_predict_train_20_loop.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ==========================
# Main Function
# ==========================
def train_predict_20_runs(dataset_name: str, dataset_path: Path, output_path: Path, n_runs: int = 20):
    logging.info(f"📂 Loading dataset: {dataset_path}")

    output_dir = output_path / "predict_20_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"predict_20_loop_result_{dataset_name}.pkl"

    try:
        datasets = joblib.load(dataset_path)
    except Exception as e:
        logging.exception(f"[💥 ERROR] Failed to load dataset: {dataset_path}")
        return

    if output_file.exists():
        try:
            completed_datasets = joblib.load(output_file)
            for i, d in enumerate(completed_datasets):
                if "cv_multi_run_scores" in d:
                    datasets[i] = d
            logging.info(f"[RESUME] Resuming from existing results in {output_file}")
        except Exception as e:
            logging.warning(f"[WARN] Could not resume from output file: {e}")

    for idx, data in enumerate(datasets):
        if "cv_multi_run_scores" in data:
            logging.info(f"[{dataset_name}|{idx}] Skipping: already processed.")
            continue

        x_fit = data["x_fit"]
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

        x_test = data["x_blind_test"]
        arr_type = type(x_test).__name__
        arr_dtype = getattr(x_test, "dtype", None)
        is_sparse = "csr" in arr_type.lower()
        log_pre = f"[{dataset_name}|{idx}|x_test]"
        if is_sparse:
            if arr_dtype not in ("float32", "float64"):
                logging.info(f"{log_pre} Converting CSR dtype {arr_dtype} to float32")
                x_test = x_test.astype("float32")
            else:
                logging.info(f"{log_pre} CSR type, dtype OK: {arr_dtype}")
        else:
            if arr_dtype not in ("float32", "float64"):
                logging.info(f"{log_pre} Converting ndarray dtype {arr_dtype} to float32")
                x_test = x_test.astype("float32")
            else:
                logging.info(f"{log_pre} ndarray, dtype OK: {arr_dtype}")
        arr_shape = x_test.shape
        arr_nnz = x_test.nnz if is_sparse else "N/A"
        logging.info(f"{log_pre} shape: {arr_shape}, nnz: {arr_nnz}")

        y_fit = data["y_fit"]
        y_test = data["y_blind_test"]

        logging.info(
            f"[{dataset_name}|{idx}] RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

        params = data["best_params"]
        predict_metrics = []

        logging.info(f"🔁 [{dataset_name}] 20x Predict | Index {idx} | {data['combination']}")

        try:
            for run_idx in range(n_runs):
                params_with_random = params.copy()
                params_with_random['random_state'] = np.random.randint(0, 1000000)
                clf = LGBMClassifier(**params_with_random, n_jobs=-1)
                param = clf.get_params()
                logging.info(f"{dataset_name} | {idx} | Run {run_idx + 1}/{n_runs} | Params: {param}")
                clf.fit(x_fit, y_fit)
                y_pred = clf.predict(x_test)
                y_proba = clf.predict_proba(x_test)[:, 1] if len(set(y_test)) == 2 else None

                result = {
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                    "precision_macro": precision_score(y_test, y_pred, average="macro"),
                    "recall_macro": recall_score(y_test, y_pred, average="macro"),
                    "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                    "accuracy": accuracy_score(y_test, y_pred),
                }
                predict_metrics.append(result)
                # Log the result in each run
                logging.info(f"[{dataset_name}|{idx}|Run {run_idx + 1}] Metrics: {result}")

            data["cv_multi_run_scores"] = predict_metrics

        except Exception as e:
            logging.exception(f"❌ Error during prediction at index {idx}: {str(e)}")
            data["cv_multi_run_scores"] = None
            data["cv_error"] = str(e)

        for key in ["x_fit", "x_blind_test", "y_fit", "y_blind_test"]:
            if key in data:
                del data[key]

        joblib.dump(datasets, output_file)
        logging.info(f"💾 Checkpoint saved after index {idx}")
        gc.collect()

    logging.info(f"✅ Finished: predict_20_loop_result_{dataset_name}.pkl")


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    input_path, output_path = get_paths()
    dataset_files = [
        ("normal", input_path / "optuna_result_normal.pkl"),
        ("topic_model", input_path / "optuna_result_topic_model.pkl"),
        ("smote_poly_normal", input_path / "optuna_result_smote_poly_normal.pkl"),
        ("smote_prowsyn_normal", input_path / "optuna_result_smote_prowsyn_normal.pkl"),
        ("smote_poly_topic", input_path / "optuna_result_smote_poly_topic_model.pkl"),
        ("smote_prowsyn_topic", input_path / "optuna_result_smote_prowsyn_topic_model.pkl"),
    ]
    for dataset_name, file_path in dataset_files:
        logging.info(f"Processing dataset: {dataset_name}")
        train_predict_20_runs(dataset_name, file_path, output_path)
