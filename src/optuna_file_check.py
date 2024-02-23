import pickle

import joblib
import pandas as pd
import static_method_class as smc


def loop_dict_optuna_list_to_df(dict_list, list_remover):
    temp = []
    new_dict = {}
    for a_dict in dict_list:
        temp_df_list = []
        if a_dict['combination'] == 'TfidfVectorizer_pre_process_porterstemmer_n_grams_1_1' and a_dict['y_name'] == 'issue_in_test_step':
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


cv_score_normal = joblib.load('../resources/optuna_result/cv_score_normal_result.pkl')
# cv_score_smote = joblib.load('../resources/optuna_result/cv_score_smote_result.pkl')

cv_score_normal_df = loop_dict_optuna_list_to_df(cv_score_normal, ['x_fit', 'x_blind_test'])
# cv_score_smote_df = loop_dict_optuna_list_to_df(cv_score_smote, ['x_fit', 'x_blind_test'])
