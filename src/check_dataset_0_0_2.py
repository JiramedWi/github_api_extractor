import os
from pathlib import Path

import joblib
import pandas as pd
import static_method_class as smc

data_dir = "../resources/result_optuna_parameter_tuning_round_2"

columns_i_dont_want = ['whole_x_fit', 'x_fit', 'x_blind_test', 'y_fit', 'y_blind_test', 'result']


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


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
    print(get_var_name(list_remover))
    return df


# read cv score
cv_score_normal = joblib.load(os.path.join(data_dir, "cv_score_cv_score_normal.pkl"))
cv_score_smote_polynom_fit = joblib.load(os.path.join(data_dir, "cv_score_cv_score_smote_polynom_fit.pkl"))
cv_score_smote_prowsyn_fit = joblib.load(os.path.join(data_dir, "cv_score_cv_score_smote_prowsyn_fit.pkl"))

cv_score_normal_dataset_lda_lsa = joblib.load(os.path.join(data_dir, "cv_score_cv_score_normal_dataset_lda_lsa.pkl"))
cv_score_smote_dataset_lda_lsa_polynom = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_lda_lsa_polynom.pkl"))
cv_score_smote_dataset_lda_lsa_prowsyn = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_lda_lsa_prowsyn.pkl"))

cv_score_normal_dataset_normalized = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_normal_datset_normalized.pkl"))
cv_score_normal_dataset_lda_lsa_normalized = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_normal_dataset_lda_lsa_normalized.pkl"))
cv_score_smote_dataset_normalized_polynom = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_normalized_polynom.pkl"))
cv_score_smote_dataset_normalized_prowsyn = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_normalized_prowsyn.pkl"))
cv_score_smote_dataset_lda_lsa_normalized_polynom = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_lda_lsa_normalized_polynom.pkl"))
cv_score_smote_dataset_lda_lsa_normalized_prowsyn = joblib.load(
    os.path.join(data_dir, "cv_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn.pkl"))


# read predict score
predict_score_cv_score_normal = joblib.load(os.path.join(data_dir, "predict_score_cv_score_normal.pkl"))
predict_score_cv_score_smote_polynom_fit = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_polynom_fit.pkl"))
predict_score_cv_score_smote_prowsyn_fit = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_prowsyn_fit.pkl"))

predict_score_cv_score_normal_dataset_lda_lsa = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_normal_dataset_lda_lsa.pkl"))
predict_score_cv_score_smote_dataset_lda_lsa_polynom = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_lda_lsa_polynom.pkl"))
predict_score_cv_score_smote_dataset_lda_lsa_prowsyn = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_lda_lsa_prowsyn.pkl"))

predict_score_cv_score_normal_dataset_normalized = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_normal_datset_normalized.pkl"))
predict_score_cv_score_normal_dataset_lda_lsa_normalized = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_normal_dataset_lda_lsa_normalized.pkl"))
predict_score_cv_score_smote_dataset_normalized_polynom = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_normalized_polynom.pkl"))
predict_score_cv_score_smote_dataset_normalized_prowsyn = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_normalized_prowsyn.pkl"))
predict_score_cv_score_smote_dataset_lda_lsa_normalized_polynom = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_lda_lsa_normalized_polynom.pkl"))
predict_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn = joblib.load(
    os.path.join(data_dir, "predict_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn.pkl"))

# turn cv into df
cv_score_normal_df = loop_dict_optuna_list_to_df(cv_score_normal.copy(), columns_i_dont_want)
cv_score_normal_df['smote'] = 'no'
cv_score_smote_polynom_fit_df = loop_dict_optuna_list_to_df(cv_score_smote_polynom_fit.copy(), columns_i_dont_want)
cv_score_smote_polynom_fit_df['smote'] = 'polynom_fit'
cv_score_smote_prowsyn_fit_df = loop_dict_optuna_list_to_df(cv_score_smote_prowsyn_fit.copy(), columns_i_dont_want)
cv_score_smote_prowsyn_fit_df['smote'] = 'prowsyn'

cv_score_normal_dataset_lda_lsa_df = loop_dict_optuna_list_to_df(cv_score_normal_dataset_lda_lsa.copy(),
                                                                 columns_i_dont_want)
cv_score_normal_dataset_lda_lsa_df['smote'] = 'no'
cv_score_smote_dataset_lda_lsa_polynom_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_lda_lsa_polynom.copy(),
                                                                        columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_polynom_df['smote'] = 'polynom_fit'
cv_score_smote_dataset_lda_lsa_prowsyn_df = loop_dict_optuna_list_to_df(cv_score_smote_dataset_lda_lsa_prowsyn.copy(),
                                                                        columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_prowsyn_df['smote'] = 'prowsyn'

cv_score_normal_dataset_normalized_df = loop_dict_optuna_list_to_df(cv_score_normal_dataset_normalized.copy(),
                                                                    columns_i_dont_want)
cv_score_normal_dataset_normalized_df['smote'] = 'no'
cv_score_normal_dataset_lda_lsa_normalized_df = loop_dict_optuna_list_to_df(
    cv_score_normal_dataset_lda_lsa_normalized.copy(),
    columns_i_dont_want)
cv_score_normal_dataset_lda_lsa_normalized_df['smote'] = 'no'
cv_score_smote_dataset_normalized_polynom_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_normalized_polynom.copy(),
    columns_i_dont_want)
cv_score_smote_dataset_normalized_polynom_df['smote'] = 'polynom_fit'
cv_score_smote_dataset_normalized_prowsyn_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_normalized_prowsyn.copy(),
    columns_i_dont_want)
cv_score_smote_dataset_normalized_prowsyn_df['smote'] = 'prowsyn'
cv_score_smote_dataset_lda_lsa_normalized_polynom_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_lda_lsa_normalized_polynom.copy(), columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_normalized_polynom_df['smote'] = 'polynom_fit'
cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df = loop_dict_optuna_list_to_df(
    cv_score_smote_dataset_lda_lsa_normalized_prowsyn.copy(), columns_i_dont_want)
cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df['smote'] = 'prowsyn'
dict_df_cv = {
    'cv_score_normal_df': cv_score_normal_df,
    'cv_score_smote_polynom_fit_df': cv_score_smote_polynom_fit_df,
    'cv_score_smote_prowsyn_fit_df': cv_score_smote_prowsyn_fit_df,
    'cv_score_normal_dataset_lda_lsa_df': cv_score_normal_dataset_lda_lsa_df,

    'cv_score_smote_dataset_lda_lsa_polynom_df': cv_score_smote_dataset_lda_lsa_polynom_df,
    'cv_score_smote_dataset_lda_lsa_prowsyn_df': cv_score_smote_dataset_lda_lsa_prowsyn_df,

    'cv_score_normal_dataset_normalized_df': cv_score_normal_dataset_normalized_df,
    'cv_score_normal_dataset_lda_lsa_normalized_df': cv_score_normal_dataset_lda_lsa_normalized_df,
    'cv_score_smote_dataset_normalized_polynom_df': cv_score_smote_dataset_normalized_polynom_df,
    'cv_score_smote_dataset_normalized_prowsyn_df': cv_score_smote_dataset_normalized_prowsyn_df,
    'cv_score_smote_dataset_lda_lsa_normalized_polynom_df': cv_score_smote_dataset_lda_lsa_normalized_polynom_df,
    'cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df': cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df
}

# turn predict into df
predict_score_cv_score_normal_df = loop_dict_optuna_list_to_df(predict_score_cv_score_normal.copy(),
                                                               columns_i_dont_want)
predict_score_cv_score_smote_polynom_fit_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_polynom_fit.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_prowsyn_fit_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_prowsyn_fit.copy(),
    columns_i_dont_want)

predict_score_cv_score_normal_dataset_lda_lsa_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_normal_dataset_lda_lsa.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_lda_lsa_polynom_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_lda_lsa_polynom.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_lda_lsa_prowsyn_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_lda_lsa_prowsyn.copy(),
    columns_i_dont_want)

predict_score_cv_score_normal_dataset_normalized_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_normal_dataset_normalized.copy(),
    columns_i_dont_want)
predict_score_cv_score_normal_dataset_lda_lsa_normalized_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_normal_dataset_lda_lsa_normalized.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_normalized_polynom_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_normalized_polynom.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_normalized_prowsyn_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_normalized_prowsyn.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_lda_lsa_normalized_polynom_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_lda_lsa_normalized_polynom.copy(),
    columns_i_dont_want)
predict_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df = loop_dict_optuna_list_to_df(
    predict_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn.copy(),
    columns_i_dont_want)

dict_df_predict = {
    'predict_score_normal_df': predict_score_cv_score_normal_df,
    'predict_score_smote_polynom_fit_df': predict_score_cv_score_smote_polynom_fit_df,
    'predict_score_smote_prowsyn_fit_df': predict_score_cv_score_smote_prowsyn_fit_df,
    'predict_score_normal_dataset_lda_lsa_df': predict_score_cv_score_normal_dataset_lda_lsa_df,
    'predict_score_smote_dataset_lda_lsa_polynom_df': predict_score_cv_score_smote_dataset_lda_lsa_polynom_df,
    'predict_score_smote_dataset_lda_lsa_prowsyn_df': predict_score_cv_score_smote_dataset_lda_lsa_prowsyn_df,
    'predict_score_normal_dataset_normalized_df': predict_score_cv_score_normal_dataset_normalized_df,
    'predict_score_normal_dataset_lda_lsa_normalized_df': predict_score_cv_score_normal_dataset_lda_lsa_normalized_df,
    'predict_score_smote_dataset_normalized_polynom_df': predict_score_cv_score_smote_dataset_normalized_polynom_df,
    'predict_score_smote_dataset_normalized_prowsyn_df': predict_score_cv_score_smote_dataset_normalized_prowsyn_df,
    'predict_score_smote_dataset_lda_lsa_normalized_polynom_df': predict_score_cv_score_smote_dataset_lda_lsa_normalized_polynom_df,
    'predict_score_smote_dataset_lda_lsa_normalized_prowsyn_df': predict_score_cv_score_smote_dataset_lda_lsa_normalized_prowsyn_df
}


def save_data_as_df(data_df, variable_name, data_dir=""):
    # Construct the full output file path
    if data_dir:
        output_filepath = os.path.join(data_dir, f"{variable_name}.pkl")
    else:
        output_filepath = f"{variable_name}.pkl"  # Save in current working directory if data_dir is empty

    # Save the DataFrame as a pickle file
    joblib.dump(data_df, output_filepath)
    print(f"Saved DataFrame to '{output_filepath}'.")


# Save the DataFrames as pickle files
directory_to_save = os.path.join(data_dir, "result_as_df")
for var_name, data in dict_df_cv.items():
    save_data_as_df(data, var_name, directory_to_save)

for var_name, data in dict_df_predict.items():
    save_data_as_df(data, var_name, directory_to_save)



