
import joblib

# x_y_smote_prowsyn = joblib.load('../resources/result_0_0_3/x_y_fit_normal_smote_prowsyn.pkl')
x_y_normal = joblib.load('../resources/result_0_0_3/x_y_fit_normal_0_0_3.pkl')
x_y_LDA_LSA = joblib.load('../resources/result_0_0_3/x_y_fit_normal_with_LDA_LSA_old_same_size_xfit.pkl')


def check_y(list_of_dict_datasets):
    for i, a_dict in enumerate(list_of_dict_datasets):
        x_normal = x_y_normal[i]['x_fit']
        y_normal = x_y_normal[i]['y_fit']
        x_new = a_dict['x_fit']
        y_new = a_dict['y_fit']
        check_y_value = y_new.size / y_normal.size
        print(f'array at {i}')
        if check_y_value > 2:
            a_dict['is_y_over_2_times'] = f'yes it is, the y new is {y_new.size} and normal y is {y_normal.size}'
        else:
            a_dict['is_y_over_2_times'] = 'No'
            print(f'the y new is {y_new.size} and normal y is {y_normal.size}')
        check_x_size = x_new.size / x_normal.size
        a_dict['how_much_x_is_multiply'] = check_x_size
        print(f'X new shape {x_new.shape}')
        print(f'X normal shape {x_normal.shape}')
        print(f'X new more than X normal at {check_x_size} by X smote has size = {x_new.size} and normal X size = {x_normal.size}')
        print('\n')
    return list_of_dict_datasets


x_y_LDA_LSA = check_y(x_y_LDA_LSA)

# x_y_smote_polynom_fit = joblib.load('../resources/result_0_0_3/x_y_fit_normal_smote_polynom_fit.pkl')
