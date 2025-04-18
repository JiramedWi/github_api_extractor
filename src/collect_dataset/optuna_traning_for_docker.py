import logging
import os
import platform
import joblib
import numpy as np
import optuna
from pathlib import Path
from sklearn import model_selection
from lightgbm import LGBMClassifier
from optuna.exceptions import TrialPruned

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optuna_tuning.log"),
        logging.StreamHandler()
    ]
)

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


# Get input and output paths
input_path, output_path = get_paths()


# Early stopping callback
def early_stopping_callback(early_stopping_rounds):
    def callback(study, trial):
        current_trial = trial.number
        best_trial_number = study.best_trial.number

        if current_trial < early_stopping_rounds:
            return

        trials_without_improvement = current_trial - best_trial_number
        if trials_without_improvement >= early_stopping_rounds:
            logging.info(f"🔕 Early stopping: {early_stopping_rounds} trials without improvement.")
            logging.info(f"✅ Best ROC AUC so far: {study.best_value:.4f}, from trial {best_trial_number}")
            study.stop()
    return callback


# Objective function with optional pruning
def objective(trial, x, y):
    # Suggest hyperparameters for LightGBM
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

    logging.info(f"Trial {trial.number}: ROC AUC = {score:.4f} | Params = {trial.params}")

    if score < 0.55:
        raise TrialPruned()

    return score


# Start optuna
def find_best_parameter(datasets: list, dataset_name: str):
    for dataset in datasets:
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']

        logging.info(f"\n🚀 Starting Optuna tuning for dataset: {dataset_name}")
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=1000,
            timeout=600,
            callbacks=[early_stopping_callback(early_stopping_rounds=30)]
        )

        trial = study.best_trial
        logging.info(f"🎯 Best Trial for {dataset_name}: Score = {trial.value:.4f} | Params = {trial.params}")

        dataset['best_params'] = trial.params
        dataset['result'] = trial.value

    # Save results in the output directory
    results_cv_path = output_path / f"optuna_result_{dataset_name}.pkl"
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(datasets, results_cv_path)

    logging.info(f"📦 Saved Optuna results to: {results_cv_path}")
    return datasets


# Main execution
if __name__ == '__main__':
    data_file = input_path / "x_y_fit_blind_SMOTE_transform_optuna.pkl"
    logging.info(f"📂 Loading dataset from: {data_file}")
    datasets = joblib.load(data_file)

    find_best_parameter(datasets, 'flink_smote')
