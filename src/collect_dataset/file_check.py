import os
import joblib

# โฟลเดอร์ที่เก็บไฟล์ .pkl ของคุณ
directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/"

# —— โหลด CV score files เป็นตัวแปรตรง ๆ ——
cv_score_normal           = joblib.load(os.path.join(directory, "cv_score_normal.pkl"))
cv_score_polynorm_normal  = joblib.load(os.path.join(directory, "cv_score_polynorm_normal.pkl"))
cv_score_polynorm_topic   = joblib.load(os.path.join(directory, "cv_score_polynorm_topic.pkl"))
cv_score_prowsyn_normal   = joblib.load(os.path.join(directory, "cv_score_prowsyn_normal.pkl"))
cv_score_prowsyn_topic    = joblib.load(os.path.join(directory, "cv_score_prowsyn_topic.pkl"))
cv_score_topic            = joblib.load(os.path.join(directory, "cv_score_topic.pkl"))

# —— โหลด Predict score files เป็นตัวแปรตรง ๆ ——
predict_score_normal           = joblib.load(os.path.join(directory, "predict_score_normal.pkl"))
predict_score_polynorm_normal  = joblib.load(os.path.join(directory, "predict_score_polynorm_normal.pkl"))
predict_score_polynorm_topic   = joblib.load(os.path.join(directory, "predict_score_polynorm_topic.pkl"))
predict_score_prowsyn_normal   = joblib.load(os.path.join(directory, "predict_score_prowsyn_normal.pkl"))
predict_score_prowsyn_topic    = joblib.load(os.path.join(directory, "predict_score_prowsyn_topic.pkl"))
predict_score_topic            = joblib.load(os.path.join(directory, "predict_score_topic.pkl"))

