import os
import platform
import joblib
import numpy as np
import optuna
from pathlib import Path
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

# Function to determine paths dynamically
def get_paths():
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR_OPTUNA")

    if not input_directory or not output_directory:
        system_name = platform.system()
        print(f"Detected OS: {system_name}")

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

# Set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 500, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 128, 512)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 64, 256)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)

    gbm = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=42
    )

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=3, scoring='roc_auc')
    print(result)
    auc_scores = result['test_score']
    score = np.mean(auc_scores)
    return score

# Start optuna
def find_best_parameter(datasets: list, dataset_name: str):
    for dataset in datasets:
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, x_fit, y_fit), n_trials=1000, timeout=600)

        trial = study.best_trial
        dataset['best_params'] = trial.params
        dataset['result'] = trial.value

    # Save results in the output directory
    results_cv_path = output_path / f"optuna_result_{dataset_name}.pkl"
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure directories exist
    joblib.dump(datasets, results_cv_path)
    
    return datasets

# Main execution
if __name__ == '__main__':
    # Load the dataset from input directory
    data_file = input_path / "x_y_fit_blind_SMOTE_transform_optuna.pkl"
    datasets = joblib.load(data_file)
    
    # Find best parameter
    find_best_parameter(datasets, 'flink_smote')
