import joblib

a = joblib.load("C:/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/train_test_20_loop/predict_20_runs/predict_20_loop_result_smote_poly_normal_start_at_60.pkl")
# Cut data at index 60 and the rest of the data
if len(a) > 60:
    a = a[60:]
    joblib.dump(a, "C:/repo/github_api_extractor/resources/tsdetect/test_smell_flink/optuna_result_flink/predict_20_loop_result_smote_poly_normal_start_at_60_only_20run_nofit.pkl")
