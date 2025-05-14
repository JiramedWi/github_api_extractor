import os
import logging
from sys import platform

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
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR_OPTUNA")

    if not input_directory or not output_directory:
        system_name = platform.system()
        logging.info(f"Detected OS: {system_name}")

        if system_name == "Linux":
            input_directory = "/app/resources/tsdetect/test_smell_flink"
            output_directory = "/app/resources/tsdetect/test_smell_flink/optuna_result"
        elif system_name == "Darwin":  # macOS
            input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
            output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        else:
            raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Suppress warnings during cross-validation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# Training and evaluation metrics
SCORING = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']


def train_cv_and_predict(dataset_name: str, dataset_path: Path):
    logging.info(f"📂 Loading dataset: {dataset_path}")
    datasets = joblib.load(dataset_path)

    for idx, data in enumerate(datasets):
        x_fit = data["x_fit"]
        y_fit = data["y_fit"]
        x_blind_test = data["x_blind_test"]
        y_blind_test = data["y_blind_test"]

        # 🔧 Ensure data is in correct format
        if hasattr(x_fit, "toarray"):
            x_fit = x_fit.toarray()
        if hasattr(x_blind_test, "toarray"):
            x_blind_test = x_blind_test.toarray()

        x_fit = x_fit.astype("float32")
        x_blind_test = x_blind_test.astype("float32")
        params = data["best_params"]

        logging.info(f"🔁 [{dataset_name}] CV + Predict | Index {idx} | {data['combination']}")

        try:
            # LightGBM model from tuned params
            clf = LGBMClassifier(**params, n_jobs=-1)

            # === Cross-validation ===
            cv_results = cross_validate(clf, x_fit, y_fit, cv=5, scoring=SCORING, n_jobs=-1)
            data.update({
                'cv_precision_macro': cv_results['test_precision_macro'].mean(),
                'cv_recall_macro': cv_results['test_recall_macro'].mean(),
                'cv_f1_macro': cv_results['test_f1_macro'].mean(),
                'cv_roc_auc': cv_results['test_roc_auc'].mean()
            })

            # === Train Final Model + Predict ===
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

    # Save results
    output_dir = output_path / "final_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save updated dataset with scores
    joblib.dump(datasets, output_dir / f"cv_score_{dataset_name}.pkl")
    joblib.dump(datasets, output_dir / f"predict_score_{dataset_name}.pkl")

    logging.info(f"✅ Saved: cv_score_{dataset_name}.pkl & predict_score_{dataset_name}.pkl")

    return datasets


def merge_all_results(output_dir: Path, summary_file: str = "summary_all_results.csv"):
    logging.info(f"🧾 Merging all result summaries into: {summary_file}")
    all_rows = []

    for file in output_dir.glob("predict_score_*.pkl"):
        dataset_name = file.stem.replace("predict_score_", "")
        datasets = joblib.load(file)

        for idx, data in enumerate(datasets):
            row = {
                "dataset": dataset_name,
                "index": idx,
                "combination": data.get("combination", ""),
                "label": data.get("y_name", ""),
                "cv_roc_auc": data.get("cv_roc_auc"),
                "test_roc_auc": data.get("test_roc_auc"),
                "test_f1": data.get("test_f1"),
                "test_precision": data.get("test_precision"),
                "test_recall": data.get("test_recall"),
                "test_mcc": data.get("test_mcc"),
                "params": data.get("best_params")
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    summary_path = output_dir / summary_file
    df.to_csv(summary_path, index=False)
    logging.info(f"📄 Merged summary saved to: {summary_path}")


# ========== ENTRY POINT ==========
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

    # Optional summary table for advisor
    merge_all_results(output_path / "final_training")
