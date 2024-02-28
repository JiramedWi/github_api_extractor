import pandas as pd
from tabulate import tabulate


def parse_combination(combination_text):
    # Split the text using underscores
    words = combination_text.split('_')

    # Extract specific indices
    count_vectorizer = words[0]
    pre_process = words[3]
    n_gram_first = int(words[-2])
    n_gram_second = int(words[-1])

    # Return the components
    return count_vectorizer, pre_process, n_gram_first, n_gram_second


def loop_dict_normal_list_to_df(dict_list, list_remover):
    temp = []
    new_dict = {}
    for a_dict in dict_list:
        count_vectorizer, pre_process, n_gram_first, n_gram_second = parse_combination(a_dict['combination'])
        new_dict = {
            'count_vectorizer': count_vectorizer,
            'pre_process': pre_process,
            'n_gram': f"{n_gram_first}_{n_gram_second}"
        }
        new_dict.update(a_dict)
        for e in list_remover:
            new_dict.pop(e)
        new_df = pd.DataFrame(new_dict, index=[0])
        temp.append(new_df)
        del new_dict
    df = pd.concat(temp)
    return df


