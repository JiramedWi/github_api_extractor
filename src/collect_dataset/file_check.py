import os
import joblib

# โฟลเดอร์ที่เก็บไฟล์ .pkl ของคุณ
# directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/"
# directory = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/training_result/final_training/"
directory_optuna = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_04_6/"

# ——— Load each Optuna result into a clear variable ———
optuna_result_normal               = joblib.load(os.path.join(directory_optuna, "optuna_result_normal.pkl"))
optuna_result_smote_poly_normal    = joblib.load(os.path.join(directory_optuna, "optuna_result_smote_poly_normal.pkl"))
optuna_result_smote_poly_topic     = joblib.load(os.path.join(directory_optuna, "optuna_result_smote_poly_topic_model.pkl"))
optuna_result_smote_prowsyn_normal = joblib.load(os.path.join(directory_optuna, "optuna_result_smote_prowsyn_normal.pkl"))
optuna_result_smote_prowsyn_topic  = joblib.load(os.path.join(directory_optuna, "optuna_result_smote_prowsyn_topic_model.pkl"))
optuna_result_topic_model          = joblib.load(os.path.join(directory_optuna, "optuna_result_topic_model.pkl"))

#
# # —— โหลด CV score files เป็นตัวแปรตรง ๆ ——
# cv_score_normal           = joblib.load(os.path.join(directory, "cv_score_normal.pkl"))
# cv_score_topic            = joblib.load(os.path.join(directory, "cv_score_topic_model.pkl"))
# cv_score_polynorm_normal  = joblib.load(os.path.join(directory, "cv_score_smote_poly_normal.pkl"))
# cv_score_polynorm_topic   = joblib.load(os.path.join(directory, "cv_score_smote_poly_topic.pkl"))
# cv_score_prowsyn_normal   = joblib.load(os.path.join(directory, "cv_score_smote_prowsyn_normal.pkl"))
# cv_score_prowsyn_topic    = joblib.load(os.path.join(directory, "cv_score_smote_prowsyn_topic.pkl"))
#
# # —— โหลด Predict score files เป็นตัวแปรตรง ๆ ——
# predict_score_normal           = joblib.load(os.path.join(directory, "predict_score_normal.pkl"))
# predict_score_polynorm_normal  = joblib.load(os.path.join(directory, "predict_score_smote_poly_normal.pkl"))
# predict_score_polynorm_topic   = joblib.load(os.path.join(directory, "predict_score_smote_poly_topic.pkl"))
# predict_score_prowsyn_normal   = joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_normal.pkl"))
# predict_score_prowsyn_topic    = joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_topic.pkl"))
# predict_score_topic            = joblib.load(os.path.join(directory, "predict_score_topic_model.pkl"))

dataset_path = "C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/new_smote_result_04_5/"
datasets_normal = joblib.load(os.path.join(dataset_path, "x_y_fit_optuna.pkl"))
datasets_topic = joblib.load(os.path.join(dataset_path, "x_y_fit_topic_model_with_LDA_LSA.pkl"))
datasets_smote_poly_normal = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_normal_fit_polynom_transform.pkl"))
datasets_smote_prowsyn_normal = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_normal_fit_prowsyn_transform.pkl"))
datasets_smote_poly_topic = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_topic_model_polynom_transform.pkl"))
datasets_smote_prowsyn_topic = joblib.load(os.path.join(dataset_path, "x_y_SMOTE_topic_model_prowsyn_transform.pkl"))

