import joblib
import numpy as np
import optuna
import requests
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier


# set up objective for using optuna
def objective(trial, x, y):
    n_estimators = trial.suggest_int('n_estimators', 500, 5000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 128, 512)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 64, 256)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)

    gbm = GradientBoostingClassifier(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     subsample=subsample,
                                     random_state=42)

    result = model_selection.cross_validate(gbm, x, y, cv=5, n_jobs=3, scoring='roc_auc')
    print(result)
    auc_scores = result['test_score']
    score = np.mean(auc_scores)
    return score


# start optuna
#TODO: Do early stopping for optuna
def find_best_parameter(datasets: list, dataset_name: str):
    # Set up time and line notification
    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']
        y_name = dataset['y_name']
        # Find best parameter
        # try:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, x_fit, y_fit),
            n_trials=1000,
            timeout=600,
        )
        trial = study.best_trial
        result = trial.value
        best_params = trial.params
        dataset['best_params'] = best_params
        dataset['result'] = result
    # Save the model and results path
    results_cv_path = f"/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result/cv_score_result_{dataset_name}.pkl"
    joblib.dump(datasets, results_cv_path)
    return datasets

# main execution
if __name__ == '__main__':
    #TODO:Check the dataset to see if it is correct
    # Load the data
    datasets = joblib.load('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_blind_SMOTE_transform_optuna.pkl')
    # Find best parameter
    find_best_parameter(datasets, 'flink_smote')
