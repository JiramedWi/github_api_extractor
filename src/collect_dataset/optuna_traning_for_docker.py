import logging
import os
import platform
import joblib
import numpy as np
import optuna
import json
import gc
from pathlib import Path
from sklearn import model_selection
from lightgbm import LGBMClassifier
from optuna.exceptions import TrialPruned

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/resources/tsdetect/test_smell_flink/optuna_result/optuna_tuning.log"),
        logging.StreamHandler()
    ]
)

# Checkpoint file path
CHECKPOINT_FILE = "/app/resources/tsdetect/test_smell_flink/optuna_result/optuna_checkpoint.json"

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

# Load checkpoint file if exists
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save checkpoint file
def save_checkpoint(done_datasets):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(done_datasets, f)

# Early stopping callback
def early_stopping_callback(early_stopping_rounds):
    def callback(study, trial):
        current_trial = trial.number

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) == 0:
            return

        best_trial_number = study.best_trial.number

        if current_trial < early_stopping_rounds:
            return

        trials_without_improvement = current_trial - best_trial_number
        if trials_without_improvement >= early_stopping_rounds:
            logging.info(f"\U0001F515 Early stopping: {early_stopping_rounds} trials without improvement.")
            logging.info(f"\u2705 Best ROC AUC so far: {study.best_value:.4f}, from trial {best_trial_number}")
            study.stop()
    return callback

# Objective function with optional pruning
def objective(trial, x, y):
    # Ensure correct dtype
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": 8
    }

    clf = LGBMClassifier(**params)

    result = model_selection.cross_validate(clf, x, y, cv=5, n_jobs=3, scoring='roc_auc')
    auc_scores = result['test_score']
    score = np.mean(auc_scores)

    if score < 0.55:
        raise TrialPruned()

    return score

# Start optuna
def find_best_parameter(datasets: list, dataset_name: str, done_datasets: dict):
    if dataset_name not in done_datasets:
        done_datasets[dataset_name] = []

    finished_indexes = set(done_datasets[dataset_name])
    results_cv_path = output_path / f"optuna_result_{dataset_name}.pkl"
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, dataset in enumerate(datasets):
        if idx in finished_indexes:
            logging.info(f"\u23E9 Skipping {dataset_name} index {idx}, already completed.")
            continue

        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']

        logging.info(f"\n\U0001F680 Starting Optuna tuning for {dataset_name} index {idx}")

        study = optuna.create_study(direction='maximize', study_name=f"{dataset_name}_{idx}", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=1000,
            timeout=600,
            callbacks=[early_stopping_callback(early_stopping_rounds=30)]
        )

        trial = study.best_trial
        logging.info(f"\U0001F3AF Best Trial for {dataset_name} index {idx}: Score = {trial.value:.4f} | Params = {trial.params}")

        dataset['best_params'] = trial.params
        dataset['result'] = trial.value

        # Save checkpoint and partial result after every index
        done_datasets[dataset_name].append(idx)
        save_checkpoint(done_datasets)
        joblib.dump(datasets, results_cv_path)

        del study, x_fit, y_fit
        gc.collect()

    logging.info(f"\U0001F4E6 Finished processing {dataset_name}, final result at: {results_cv_path}")
    gc.collect()

# Main execution
if __name__ == '__main__':
    input_path, output_path = get_paths()
    done_datasets = load_checkpoint()

    dataset_files = {
        "normal": input_path / "x_y_fit_optuna.pkl",
        # "topic_model": input_path / "x_y_fit_topic_model_with_LDA_LSA.pkl",
        # "smote_poly_normal": input_path / "x_y_SMOTE_normal_fit_polynom_transform.pkl",
        # "smote_prowsyn_normal": input_path / "x_y_SMOTE_normal_fit_prowsyn_transform.pkl",
        # "smote_poly_topic_model": input_path / "x_y_SMOTE_topic_model_polynom_transform.pkl",
        # "smote_prowsyn_topic_model": input_path / "x_y_SMOTE_topic_model_prowsyn_transform.pkl"
    }

    for dataset_name, file_path in dataset_files.items():
        logging.info(f"\U0001F4C2 Loading dataset from: {file_path}")
        datasets = joblib.load(file_path)
        find_best_parameter(datasets, dataset_name, done_datasets)
        del datasets
        gc.collect()

    logging.info("\u2705 All datasets processed successfully.")
