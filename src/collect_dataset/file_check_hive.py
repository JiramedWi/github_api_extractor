import os

import joblib


dataset_path = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/result_0_0_3"
result_path = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/result_as_df"

x_2 = joblib.load("/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/result_0_0_2/x_0_0_2.pkl")
y_2 = joblib.load("/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_hive/result_0_0_2/y_0_0_2.pkl")
datasets_normal = joblib.load(os.path.join(dataset_path, "x_y_fit_normal_0_0_3.pkl"))
dataset_normal_prowsyn = joblib.load(os.path.join(dataset_path, "x_y_normal_smote_prowsyn.pkl"))
dataset_normal_polynom_fit = joblib.load(os.path.join(dataset_path, "x_y_normal_smote_polynom_fit.pkl"))

result_normal = joblib.load(os.path.join(result_path, "cv_score_normal_df.pkl"))