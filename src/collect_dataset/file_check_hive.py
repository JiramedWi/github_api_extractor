import os

import joblib

directory_dataset = "/home/pee/repo/github_api_extractor/resources/result_0_0_3/x_y_fit_normal_0_0_3.pkl"
directory_path_hive_optuna = "/home/pee/repo/github_api_extractor/resources/optuna_result_round_2"
directory_path_hive_optuna_not_round2 = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive/optuna_result'

datasets_normal = joblib.load(directory_dataset)
optuna_result_not_round2_hive_normal = joblib.load(os.path.join(directory_path_hive_optuna_not_round2, "cv_score_normal_dataset.pkl"))
optuna_result_hive_normal = joblib.load(os.path.join(directory_path_hive_optuna, "best_param_of_normal.pkl"))
