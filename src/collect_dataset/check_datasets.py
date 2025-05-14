import joblib


data_path_file_normal = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_optuna.pkl'
datasets = joblib.load(data_path_file_normal)

data_path_file_smote = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_SMOTE_normal_fit_polynom_transform.pkl'
datasets_smote = joblib.load(data_path_file_smote)

data_path_topic = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_topic_model_with_LDA_LSA.pkl'
datasets_topic = joblib.load(data_path_topic)


data_path_optuna_normal = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result/optuna_result_normal.pkl'
datasets_optuna_normal = joblib.load(data_path_optuna_normal)

data_path_optuna_smote = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result/optuna_result_smote_poly_normal.pkl'
datasets_optuna_smote = joblib.load(data_path_optuna_smote)
