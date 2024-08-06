from datetime import datetime, timedelta, timezone
import logging

import joblib
import requests
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from repo.github_api_extractor.src.ml_trainging_one_file import train_cv

normal_result_for_train = joblib.load('../resources/result_0_0_2/x_y_fit_blind_transform_0_0_2.pkl')
normal_result_for_train_normalize_min_max = joblib.load(
    '../resources/result_0_0_2/normalize_x_y_fit_blind_transform_0_0_2_min_max_transform_0.0.2.pkl')
normal_result_for_train_normalize_log = joblib.load(
    '../resources/result_0_0_2/normalize_x_y_fit_blind_transform_0_0_2_log_transform_0.0.2.pkl')

SMOTE_result_for_train = joblib.load('../resources/result_0_0_2/x_y_fit_blind_SMOTE_transform_0_0_2.pkl')
SMOTE_result_for_train_normalize_min_max = joblib.load(
    '../resources/result_0_0_2/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_min_max_transform_0.0.2.pkl')
SMOTE_result_for_train_normalize_log = joblib.load(
    '../resources/result_0_0_2/normalize_x_y_fit_blind_SMOTE_transform_0_0_2_log_transform_0.0.2.pkl')


def objective(trial):
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

    result = train_cv(normal_result_for_train, gbm)
    return result


# Setting logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
line_url = 'https://notify-api.line.me/api/notify'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}
tz = timezone(timedelta(hours=7))

start_time = datetime.now(tz)
start_time_announce = start_time.strftime("%c")
start_noti = f"start to train CV and ML at: {start_time_announce}"
r = requests.post(line_url, headers=headers, data={'message': start_noti})
print(r.text, start_time_announce)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)
trial = study.best_trial
result = trial.value
best_params = trial.params
best_result = {
    'trial': trial,
    'result_value': result,
    'best_param': best_params
}

joblib.dump(best_result, '../resources/result_0_0_2/best_param_optuna.pkl')

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
optuna.visualization.plot_slice(study, params=['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
                                               'min_samples_leaf', 'subsample'])
optuna.visualization.plot_param_importances(study)

# normal result
# result_normal_1st_cv = train_cv(normal_result_for_train, gbm_model_1st_try)
# result_normal_1st_predict = train_ml(normal_result_for_train, gbm_model_1st_try)
#
# result_normal_2nd_cv = train_cv(normal_result_for_train, gbm_model_2nd)
# result_normal_2nd_predict = train_ml(normal_result_for_train, gbm_model_2nd)
#
# result_normal_3rd_cv = train_cv(normal_result_for_train, gbm_model_3rd)
# result_normal_3rd_predict = train_ml(normal_result_for_train, gbm_model_3rd)
#
# result_normal_ajkong_cv = train_cv(normal_result_for_train, gbm_model_ajarn_kong_rec)
# result_normal_ajkong_predict = train_ml(normal_result_for_train, gbm_model_ajarn_kong_rec)
# #
# # result_normal_normalize_min_max_cv = train_cv(normal_result_for_train_normalize_min_max)
# # result_normal_normalize_min_max_predict = train_ml(normal_result_for_train_normalize_min_max)
# #
# # result_normal_normalize_log_cv = train_cv(normal_result_for_train_normalize_log)
# # result_normal_normalize_log_predict = train_ml(normal_result_for_train_normalize_log)
#
# # SMOTE result
# result_SMOTE_cv_1st_model = train_cv(SMOTE_result_for_train, gbm_model_1st_try)
# result_SMOTE_predict_1st_model = train_ml(SMOTE_result_for_train, gbm_model_1st_try)
#
# result_SMOTE_cv_2nd_model = train_cv(SMOTE_result_for_train, gbm_model_2nd)
# result_SMOTE_predict_2nd_model = train_ml(SMOTE_result_for_train, gbm_model_2nd)
#
# result_SMOTE_cv_3rd_model = train_cv(SMOTE_result_for_train, gbm_model_3rd)
# result_SMOTE_predict_3rd_model = train_ml(SMOTE_result_for_train, gbm_model_3rd)
#
# result_SMOTE_cv_ajkong_model = train_cv(SMOTE_result_for_train, gbm_model_ajarn_kong_rec)
# result_SMOTE_predict_ajkong_model = train_ml(SMOTE_result_for_train, gbm_model_ajarn_kong_rec)
# result_SMOTE_normalize_min_max_cv = train_cv(SMOTE_result_for_train_normalize_min_max)
# result_SMOTE_normalize_min_max_predict = train_ml(SMOTE_result_for_train_normalize_min_max)
#
# result_SMOTE_normalize_log_cv = train_cv(SMOTE_result_for_train_normalize_log)
# result_SMOTE_normalize_log_predict = train_ml(SMOTE_result_for_train_normalize_log)

end_time = datetime.now(tz)
result_time = end_time - start_time
result_time_in_sec = result_time.total_seconds()
# Make it short to 2 decimal
in_minute = result_time_in_sec / 60
in_minute = "{:.2f}".format(in_minute)
# Make it short to 5 decimal
in_hour = result_time_in_sec / 3600
in_hour = "{:.5f}".format(round(in_hour, 2))
end_time_noti = f"Total time of CV and ML train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
print(r.text, end_time_noti)
