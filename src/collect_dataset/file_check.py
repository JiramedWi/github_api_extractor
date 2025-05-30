import os
import joblib


# directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/final_training_16_05/final_training/"
directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/training_result_27_5/final_training/"

cv_score_normal           = joblib.load(os.path.join(directory, "cv_score_normal.pkl"))
cv_score_topic            = joblib.load(os.path.join(directory, "cv_score_topic_model.pkl"))
cv_score_polynorm_normal  = joblib.load(os.path.join(directory, "cv_score_smote_poly_normal.pkl"))
cv_score_polynorm_topic   = joblib.load(os.path.join(directory, "cv_score_smote_poly_topic.pkl"))
cv_score_prowsyn_normal   = joblib.load(os.path.join(directory, "cv_score_smote_prowsyn_normal.pkl"))
cv_score_prowsyn_topic    = joblib.load(os.path.join(directory, "cv_score_smote_prowsyn_topic.pkl"))

predict_score_normal           = joblib.load(os.path.join(directory, "predict_score_normal.pkl"))
predict_score_polynorm_normal  = joblib.load(os.path.join(directory, "predict_score_smote_poly_normal.pkl"))
predict_score_polynorm_topic   = joblib.load(os.path.join(directory, "predict_score_smote_poly_topic.pkl"))
predict_score_prowsyn_normal   = joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_normal.pkl"))
predict_score_prowsyn_topic    = joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_topic.pkl"))
predict_score_topic            = joblib.load(os.path.join(directory, "predict_score_topic_model.pkl"))

dataset_path = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
datasets_normal = joblib.load(os.path.join(dataset_path, "x_y_fit_optuna.pkl"))
datasets_topic = joblib.load(os.path.join(dataset_path, "x_y_fit_topic_model_with_LDA_LSA.pkl"))
datasets_smote_poly_normal = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_normal_fit_polynom_transform.pkl"))
datasets_smote_prowsyn_normal = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_normal_fit_prowsyn_transform.pkl"))
datasets_smote_poly_topic = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_topic_model_polynom_transform.pkl"))
datasets_smote_prowsyn_topic = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_topic_model_prowsyn_transform.pkl"))

optuna_path = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_15_5"
optuna_normal = joblib.load(os.path.join(optuna_path, "optuna_result_normal.pkl"))
optuna_topic = joblib.load(os.path.join(optuna_path, "optuna_result_topic_model.pkl"))
optuna_smote_poly_normal = joblib.load(os.path.join(optuna_path, "optuna_result_smote_poly_normal.pkl"))
optuna_smote_prowsyn_normal = joblib.load(os.path.join(optuna_path, "optuna_result_smote_prowsyn_normal.pkl"))
optuna_smote_poly_topic = joblib.load(os.path.join(optuna_path, "optuna_result_smote_poly_topic_model.pkl"))
optuna_smote_prowsyn_topic = joblib.load(os.path.join(optuna_path, "optuna_result_smote_prowsyn_topic_model.pkl"))

