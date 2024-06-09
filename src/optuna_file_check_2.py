import logging
import warnings
import pickle
from datetime import timedelta, timezone, datetime

import joblib
import pandas as pd
import requests
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, confusion_matrix, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc

import static_method_class as smc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
line_url = 'https://notify-api.line.me/api/notify'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}
tz = timezone(timedelta(hours=7))


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


def loop_dict_optuna_list_to_df_but_one(dict_list, list_remover):
    temp = []
    new_dict = {}
    for a_dict in dict_list:
        temp_df_list = []
        if a_dict['combination'] == 'TfidfVectorizer_pre_process_porterstemmer_n_grams_1_1' and a_dict[
            'y_name'] == 'issue_in_test_step':
            count_vectorizer, pre_process, n_gram_first, n_gram_second = smc.parse_combination(a_dict['combination'])
            new_dict = {
                'count_vectorizer': count_vectorizer,
                'pre_process': pre_process,
                'n_gram': f"{n_gram_first}_{n_gram_second}"
            }
            new_dict.update(a_dict)
            for e in list_remover:
                new_dict.pop(e)
            best_params = new_dict.pop('best_params')
            try:
                best_params_series = pd.DataFrame(best_params, index=[0])
            except Exception as e:
                best_params_series = pd.DataFrame({'best_params': e}, index=[0])
            new_df = pd.DataFrame(new_dict, index=[0])
            new_df = pd.concat([new_df, best_params_series], axis=1)
            temp.append(new_df)

            del new_dict, new_df, best_params, best_params_series
    df = pd.concat(temp)
    return df


def loop_dict_optuna_list_to_df(dict_list, list_remover):
    temp = []
    new_dict = {}
    for a_dict in dict_list:
        temp_df_list = []
        count_vectorizer, pre_process, n_gram_first, n_gram_second = smc.parse_combination(a_dict['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}"
        }
        new_dict.update(a_dict)
        for e in list_remover:
            # check if key is in dict
            if e in new_dict:
                new_dict.pop(e)
        best_params = new_dict.pop('best_params')
        try:
            best_params_series = pd.DataFrame(best_params, index=[0])
        except Exception as e:
            best_params_series = pd.DataFrame({'best_params': e}, index=[0])
        new_df = pd.DataFrame(new_dict, index=[0])
        new_df = pd.concat([new_df, best_params_series], axis=1)
        temp.append(new_df)

        del new_dict, new_df, best_params, best_params_series
    df = pd.concat(temp)
    return df


# line noti start preparing df best param
start_time_prepare_df = datetime.now(tz)
start_time_announce = start_time_prepare_df.strftime("%c")
start_noti = f"start to preparing df best param at: {start_time_announce}"
r = requests.post(line_url, headers=headers, data={'message': start_noti})
print(r.text, start_time_announce)

# read all cv score from optuna result
cv_score_normal = joblib.load('../resources/optuna_result/cv_score_normal_dataset.pkl')
cv_score_smote_polynom_fit = joblib.load('../resources/optuna_result/cv_score_smote_dataset_polynom_fit.pkl')
cv_score_smote_prowsyn_fit = joblib.load('../resources/optuna_result/cv_score_smote_dataset_prowsyn_fit.pkl')

cv_score_normal_dataset_lda_lsa = joblib.load('../resources/optuna_result/cv_score_normal_dataset_lda_lsa.pkl')
cv_score_smote_dataset_lda_lsa_polynom = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_lda_lsa_polynom.pkl')
cv_score_smote_dataset_lda_lsa_prowsyn = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_lda_lsa_prowsyn.pkl')

cv_score_normal_datset_normalized = joblib.load('../resources/optuna_result/cv_score_normal_datset_normalized.pkl')
cv_score_normal_dataset_lda_lsa_normalized = joblib.load(
    '../resources/optuna_result/cv_score_normal_dataset_lda_lsa_normalized.pkl')
cv_score_smote_dataset_normalized_polynom = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_normalized_polynom.pkl')
cv_score_smote_dataset_normalized_prowsyn = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_normalized_prowsyn.pkl')
cv_score_smote_dataset_lda_lsa_normalized_polynom = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_lda_lsa_normalized_polynom.pkl')
cv_score_smote_dataset_lda_lsa_normalized_prowsyn = joblib.load(
    '../resources/optuna_result/cv_score_smote_dataset_lda_lsa_normalized_prowsyn.pkl')

# change optuna result into df
columns_i_dont_want = ['whole_x_fit', 'x_fit', 'x_blind_test', 'y_fit', 'y_blind_test', ]
cv_score_normal_df = loop_dict_optuna_list_to_df(cv_score_normal, columns_i_dont_want)
cv_score_smote_polynom_fit_df = loop_dict_optuna_list_to_df(cv_score_smote_polynom_fit, columns_i_dont_want)
cv_score_smote_prowsyn_fit_df = loop_dict_optuna_list_to_df(cv_score_smote_prowsyn_fit, columns_i_dont_want)

cv_score_normal_lda_lsa_df = loop_dict_optuna_list_to_df(cv_score_normal_dataset_lda_lsa, columns_i_dont_want)
cv_score_smote_polynom_lda_lsa_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_lda_lsa_polynom,
                                                                columns_i_dont_want)
cv_score_smote_prowsyn_lda_lsa_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_lda_lsa_prowsyn,
                                                                columns_i_dont_want)

cv_score_normal_datset_normalized_df = loop_dict_optuna_list_to_df(cv_score_normal_datset_normalized,
                                                                   columns_i_dont_want)
cv_score_normal_dataset_lda_lsa_normalized_df = loop_dict_optuna_list_to_df(cv_score_normal_dataset_lda_lsa_normalized,
                                                                            columns_i_dont_want)
cv_score_smote_dataset_normalized_polynom_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_normalized_polynom,
                                                                           columns_i_dont_want)
cv_score_smote_dataset_normalized_prowsyn_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_normalized_prowsyn,
                                                                           columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_normalized_polynom_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_lda_lsa_normalized_polynom, columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_lda_lsa_normalized_prowsyn, columns_i_dont_want)


# Get all best parameters and results of combination
def get_best_params_and_result(df: pd.DataFrame):
    best_params_and_result = df[['combination', 'y_name', 'result', 'n_estimators', 'learning_rate', 'max_depth',
                                 'min_samples_split', 'min_samples_leaf', 'subsample']]
    return best_params_and_result


best_param_of_normal = get_best_params_and_result(cv_score_normal_df)
joblib.dump(best_param_of_normal, '../resources/optuna_result_round_2/best_param_of_normal.pkl')
best_param_of_smote_polynom_fit = get_best_params_and_result(cv_score_smote_polynom_fit_df)
joblib.dump(best_param_of_smote_polynom_fit, '../resources/optuna_result_round_2/best_param_of_smote_polynom_fit.pkl')
best_param_of_smote_prowsyn_fit = get_best_params_and_result(cv_score_smote_prowsyn_fit_df)
joblib.dump(best_param_of_smote_prowsyn_fit, '../resources/optuna_result_round_2/best_param_of_smote_prowsyn_fit.pkl')

best_param_of_lda_lsa = get_best_params_and_result(cv_score_normal_lda_lsa_df)
joblib.dump(best_param_of_lda_lsa, '../resources/optuna_result_round_2/best_param_of_lda_lsa.pkl')
best_param_of_smote_polynom_lda_lsa = get_best_params_and_result(cv_score_smote_polynom_lda_lsa_df)
joblib.dump(best_param_of_smote_polynom_lda_lsa,
            '../resources/optuna_result_round_2/best_param_of_smote_polynom_lda_lsa.pkl')
best_param_of_smote_prowsyn_lda_lsa = get_best_params_and_result(cv_score_smote_prowsyn_lda_lsa_df)
joblib.dump(best_param_of_smote_prowsyn_lda_lsa,
            '../resources/optuna_result_round_2/best_param_of_smote_prowsyn_lda_lsa.pkl')

best_param_of_normal_normalized = get_best_params_and_result(cv_score_normal_datset_normalized_df)
joblib.dump(best_param_of_normal_normalized, '../resources/optuna_result_round_2/best_param_of_normal_normalized.pkl')
best_param_of_lda_lsa_normalized = get_best_params_and_result(cv_score_normal_dataset_lda_lsa_normalized_df)
joblib.dump(best_param_of_lda_lsa_normalized, '../resources/optuna_result_round_2/best_param_of_lda_lsa_normalized.pkl')
best_param_of_smote_normalized_polynom = get_best_params_and_result(cv_score_smote_dataset_normalized_polynom_df)
joblib.dump(best_param_of_smote_normalized_polynom,
            '../resources/optuna_result_round_2/best_param_of_smote_normalized_polynom.pkl')
best_param_of_smote_normalized_prowsyn = get_best_params_and_result(cv_score_smote_dataset_normalized_prowsyn_df)
joblib.dump(best_param_of_smote_normalized_prowsyn,
            '../resources/optuna_result_round_2/best_param_of_smote_normalized_prowsyn.pkl')
best_param_of_smote_lda_lsa_normalized_polynom = get_best_params_and_result(
    cv_score_smote_dataset_lda_lsa_normalized_polynom_df)
joblib.dump(best_param_of_smote_lda_lsa_normalized_polynom,
            '../resources/optuna_result_round_2/best_param_of_smote_lda_lsa_normalized_polynom.pkl')
best_param_of_smote_lda_lsa_normalized_prowsyn = get_best_params_and_result(
    cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df)
joblib.dump(best_param_of_smote_lda_lsa_normalized_prowsyn,
            '../resources/optuna_result_round_2/best_param_of_smote_lda_lsa_normalized_prowsyn.pkl')

# end line noti preparing df best param
end_time = datetime.now(tz)
result_time = end_time - start_time_prepare_df
result_time_in_sec = result_time.total_seconds()
# Make it short to 2 decimal
in_minute = result_time_in_sec / 60
in_minute = "{:.2f}".format(in_minute)
# Make it short to 5 decimal
in_hour = result_time_in_sec / 3600
in_hour = "{:.5f}".format(round(in_hour, 2))
end_time_noti = f"Total time of preparing df best param: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
print(r.text, end_time_noti)


# function to train cv model with list of the best parameters as df
def train_cv_model(df_parameters: pd.DataFrame, datasets):
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train cv at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_time_announce)
    count = 0
    for dataset in datasets:
        # find row in df_parameters by using combination text
        row_parameter = df_parameters[
            df_parameters['combination'].str.contains(dataset['combination'], case=False) & df_parameters[
                'y_name'].str.contains(dataset['y_name'], case=False)]
        # prepare variables
        n_estimators = row_parameter['n_estimators'].values[0]
        learning_rate = row_parameter['learning_rate'].values[0]
        max_depth = row_parameter['max_depth'].values[0]
        min_samples_split = row_parameter['min_samples_split'].values[0]
        min_samples_leaf = row_parameter['min_samples_leaf'].values[0]
        subsample = row_parameter['subsample'].values[0]
        x_fit = dataset['x_fit']
        y_fit = dataset['y_fit']
        # create model with best parameters
        gbm_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               subsample=subsample)
        scoring_metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
        count += 1
        print('start to cross validate... ', count)
        with warnings.catch_warnings(record=True) as w:
            cv_results = cross_validate(gbm_model, x_fit, y_fit, cv=5, n_jobs=-2,
                                        scoring=scoring_metrics)
            print(w)
            if len(w) > 0:
                for warning in w:
                    warning_message = ''
                    warning_message += f"{str(warning.message)}\n"  # Combine all warning messages
                    print(warning_message)
        print('done cross validate...')
        print("warning_message: ", warning_message)
        print('precision_macro:', cv_results['test_precision_macro'])
        print('recall_macro:', cv_results['test_recall_macro'])
        print('f1_macro:', cv_results['test_f1_macro'])
        print('roc_auc:', cv_results['test_roc_auc'])
        result_score = {
            'precision_macro': cv_results['test_precision_macro'].mean(),
            'recall_macro': cv_results['test_recall_macro'].mean(),
            'f1_macro': cv_results['test_f1_macro'].mean(),
            'roc_auc': cv_results['test_roc_auc'].mean(),
        }
        dataset.update(result_score)
        print('done')

    result_score_path = f"../resources/result_optuna_parameter_tuning_round_2/cv_score_{get_var_name(datasets)}.pkl"
    joblib.dump(datasets, result_score_path)
    text_logging = f"Result score of {get_var_name(datasets)} has been saved to {result_score_path}"
    r = requests.post(line_url, headers=headers, data={'message': text_logging})
    print(r.text, text_logging)
    end_time = datetime.now(tz)
    result_time = end_time - start_time
    result_time_in_sec = result_time.total_seconds()
    # Make it short to 2 decimal
    in_minute = result_time_in_sec / 60
    in_minute = "{:.2f}".format(in_minute)
    # Make it short to 5 decimal
    in_hour = result_time_in_sec / 3600
    in_hour = "{:.5f}".format(round(in_hour, 2))
    end_time_noti = f"Total time of CV train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)


# # line noti train cv model
# start_time_cv = datetime.now(tz)
# start_time_announce = start_time_cv.strftime("%c")
# start_noti = f"start to train cv at: {start_time_announce}"
# r = requests.post(line_url, headers=headers, data={'message': start_noti})
# print(r.text, start_time_announce)
# train_cv_model(best_param_of_normal, cv_score_normal)
# train_cv_model(best_param_of_smote_polynom_fit, cv_score_smote_polynom_fit)
# train_cv_model(best_param_of_smote_prowsyn_fit, cv_score_smote_prowsyn_fit)
#
# train_cv_model(best_param_of_lda_lsa, cv_score_normal_dataset_lda_lsa)
# train_cv_model(best_param_of_smote_polynom_lda_lsa, cv_score_smote_dataset_lda_lsa_polynom)
# train_cv_model(best_param_of_smote_prowsyn_lda_lsa, cv_score_smote_dataset_lda_lsa_prowsyn)
#
# train_cv_model(best_param_of_normal_normalized, cv_score_normal_datset_normalized)
# train_cv_model(best_param_of_lda_lsa_normalized, cv_score_normal_dataset_lda_lsa_normalized)
# train_cv_model(best_param_of_smote_normalized_polynom, cv_score_smote_dataset_normalized_polynom)
# train_cv_model(best_param_of_smote_normalized_prowsyn, cv_score_smote_dataset_normalized_prowsyn)
# train_cv_model(best_param_of_smote_lda_lsa_normalized_polynom, cv_score_smote_dataset_lda_lsa_normalized_polynom)
# train_cv_model(best_param_of_smote_lda_lsa_normalized_prowsyn, cv_score_smote_dataset_lda_lsa_normalized_prowsyn)
#
# # end line noti train cv
# end_time = datetime.now(tz)
# result_time = end_time - start_time_cv
# result_time_in_sec = result_time.total_seconds()
# # Make it short to 2 decimal
# in_minute = result_time_in_sec / 60
# in_minute = "{:.2f}".format(in_minute)
# # Make it short to 5 decimal
# in_hour = result_time_in_sec / 3600
# in_hour = "{:.5f}".format(round(in_hour, 2))
# end_time_noti = f"Total time of CV train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
# r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
# print(r.text, end_time_noti)


# function to train predict model with list of the best parameters as df
def train_predict_model(df_parameters: pd.DataFrame, datasets):
    start_time = datetime.now(tz)
    start_time_announce = start_time.strftime("%c")
    start_noti = f"start to train model at: {start_time_announce}" + '\n' + f"dataset: {get_var_name(datasets)}"
    r = requests.post(line_url, headers=headers, data={'message': start_noti})
    print(r.text, start_time_announce)
    for dataset in datasets:
        # find row in df_parameters by using combination text
        row_parameter = df_parameters[
            df_parameters['combination'].str.contains(dataset['combination'], case=False) & df_parameters[
                'y_name'].str.contains(dataset['y_name'], case=False)]
        # prepare variables
        n_estimators = row_parameter['n_estimators'].values[0]
        learning_rate = row_parameter['learning_rate'].values[0]
        max_depth = row_parameter['max_depth'].values[0]
        min_samples_split = row_parameter['min_samples_split'].values[0]
        min_samples_leaf = row_parameter['min_samples_leaf'].values[0]
        subsample = row_parameter['subsample'].values[0]
        x_fit = dataset['x_fit']
        x_blind_test = dataset['x_blind_test']
        y_fit = dataset['y_fit']
        y_blind_test = dataset['y_blind_test']
        # create model with best parameters
        gbm_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               subsample=subsample)
        gbm_model.fit(x_fit, y_fit)
        y_pred = gbm_model.predict(x_blind_test)
        y_pred_prob = gbm_model.predict_proba(x_blind_test)
        precision_test_score = precision_score(y_blind_test, y_pred)
        recall_test_score = recall_score(y_blind_test, y_pred)
        f1_test_score = f1_score(y_blind_test, y_pred)
        mcc = matthews_corrcoef(y_blind_test, y_pred)
        roc_auc_test_score = roc_auc_score(y_blind_test, y_pred_prob[:, 1])
        # prc = precision_recall_curve(y_blind_test, y_pred_prob[:, 1])

        result = {
            'precision_test_score': precision_test_score,
            'recall_test_score': recall_test_score,
            'f1_test_score': f1_test_score,
            'mcc': mcc,
            'roc_auc_test_score': roc_auc_test_score,
            # 'prc': prc
        }
        dataset.update(result)
    result_score_path = f"../resources/result_optuna_parameter_tuning_round_2/predict_score_{get_var_name(datasets)}.pkl"
    joblib.dump(datasets, result_score_path)
    text_logging = f"Result predict of {get_var_name(datasets)} has been saved to {result_score_path}"
    r = requests.post(line_url, headers=headers, data={'message': text_logging})
    print(r.text, text_logging)
    end_time = datetime.now(tz)
    result_time = end_time - start_time
    result_time_in_sec = result_time.total_seconds()
    # Make it short to 2 decimal
    in_minute = result_time_in_sec / 60
    in_minute = "{:.2f}".format(in_minute)
    # Make it short to 5 decimal
    in_hour = result_time_in_sec / 3600
    in_hour = "{:.5f}".format(round(in_hour, 2))
    end_time_noti = f"Total time of ML train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
    r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
    print(r.text, end_time_noti)


# # line noti train predict model
# start_time_predict = datetime.now(tz)
# start_time_announce = start_time_predict.strftime("%c")
# start_noti = f"start to train predict model at: {start_time_announce}"
# r = requests.post(line_url, headers=headers, data={'message': start_noti})
# print(r.text, start_time_announce)
# train_predict_model(best_param_of_normal, cv_score_normal)
# train_predict_model(best_param_of_smote_polynom_fit, cv_score_smote_polynom_fit)
# train_predict_model(best_param_of_smote_prowsyn_fit, cv_score_smote_prowsyn_fit)
#
# train_predict_model(best_param_of_lda_lsa, cv_score_normal_dataset_lda_lsa)
# train_predict_model(best_param_of_smote_polynom_lda_lsa, cv_score_smote_dataset_lda_lsa_polynom)
# train_predict_model(best_param_of_smote_prowsyn_lda_lsa, cv_score_smote_dataset_lda_lsa_prowsyn)
#
# train_predict_model(best_param_of_normal_normalized, cv_score_normal_datset_normalized)
# train_predict_model(best_param_of_lda_lsa_normalized, cv_score_normal_dataset_lda_lsa_normalized)
# train_predict_model(best_param_of_smote_normalized_polynom, cv_score_smote_dataset_normalized_polynom)
# train_predict_model(best_param_of_smote_normalized_prowsyn, cv_score_smote_dataset_normalized_prowsyn)
# train_predict_model(best_param_of_smote_lda_lsa_normalized_polynom, cv_score_smote_dataset_lda_lsa_normalized_polynom)
# train_predict_model(best_param_of_smote_lda_lsa_normalized_prowsyn, cv_score_smote_dataset_lda_lsa_normalized_prowsyn)
# # end line noti train predict model
# end_time = datetime.now(tz)
# result_time = end_time - start_time_predict
# result_time_in_sec = result_time.total_seconds()
# # Make it short to 2 decimal
# in_minute = result_time_in_sec / 60
# in_minute = "{:.2f}".format(in_minute)
# # Make it short to 5 decimal
# in_hour = result_time_in_sec / 3600
# in_hour = "{:.5f}".format(round(in_hour, 2))
# end_time_noti = f"Total time of ML train: {result_time_in_sec} seconds, {in_minute} minutes, {in_hour} hours"
# r = requests.post(line_url, headers=headers, data={'message': end_time_noti})
# print(r.text, end_time_noti)
