import nltk
import os
import numpy as np
import pandas as pd
import spacy
import joblib
import inspect
import time

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def pos_tagger_for_spacy(tag):
    # Mapping NLTK POS tags to spaCy POS tags
    tag_dict = {'N': 'NOUN', 'V': 'VERB', 'R': 'ADV', 'J': 'ADJ'}
    return tag_dict.get(tag, 'n')


nlp = spacy.load("en_core_web_sm")


def pre_process_spacy(s):
    doc = nlp(s)
    s = " ".join([token.lemma_ if token.pos_ in ['NOUN', 'VERB'] else token.text for token in doc if
                  token.pos_ in ['NOUN', 'VERB']])
    return s


def pre_process_textblob(s):
    blob = TextBlob(s)
    # Remove stopwords
    s = [word for word in blob.words if word not in nltk.corpus.stopwords.words('english')]
    s = " ".join(s)
    return s


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def pre_process_porterstemmer(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def pre_process_lemmatizer(s):
    s = word_tokenize(s)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    tags = nltk.pos_tag(s)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tags))
    s = [lemmatizer.lemmatize(word, tag) if tag == 'n' or tag == 'v' else None for word, tag in wordnet_tagged]
    s = list(filter(None, s))
    s = [w for w in s if w not in stop_dict]
    s = ' '.join(s)
    return s


sourceFile = ['../resources/tsdetect/all_test_smell/df_test_semantic_smell.csv',
              '../resources/tsdetect/all_test_smell/df_issue_in_test_step.csv',
              '../resources//tsdetect/all_test_smell/df_code_related.csv',
              '../resources/tsdetect/all_test_smell/df_dependencies.csv',
              '../resources/tsdetect/all_test_smell/df_test_execution.csv']


def compare_y_to_x(dfx, dfy):
    return dfy.loc[dfy['url'].isin(dfx['url'])]


def read_source(sourceFiles):
    path = os.path.dirname(__file__)
    filepath = Path(os.path.abspath(os.path.join(path, '../resources/clean_demo.pkl')))
    x = pd.read_pickle(filepath)
    data = pd.read_pickle(
        Path(os.path.abspath(os.path.join(path, '../resources/hive_use_for_run_pre_process.pkl'))))
    data = data[data['title_n_body'].notnull()]
    data.rename(columns={'title_n_body': 'title_n_body_not_clean'}, inplace=True)
    data = pd.concat([data, x.dropna()], axis=1)

    y_output = []
    for source_file in sourceFiles:
        tempTest = pd.read_csv(Path(os.path.abspath(os.path.join(path, source_file))))
        y_to_x = compare_y_to_x(data, tempTest)
        file_path, extension = os.path.splitext(source_file)
        parts = file_path.split('/')
        file_name_part = parts[-1]
        file_name = file_name_part.replace('df_', '')
        data_combination = {
            file_name: y_to_x['y']
        }
        y_output.append(data_combination)
    return x, y_output


x, y = read_source(sourceFile)
term_representations = [CountVectorizer, TfidfVectorizer]
pre_process_steps = [pre_process_porterstemmer, pre_process_lemmatizer, pre_process_textblob, pre_process_spacy]
n_grams_ranges = [(1, 1), (1, 2), (1, 3)]


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


#
def indexing_x():
    temp_x = []
    for term_representation in term_representations:
        for pre_process_step in pre_process_steps:
            for n_grams_range in n_grams_ranges:
                if term_representation == TfidfVectorizer:
                    term_x = term_representation(preprocessor=pre_process_step, ngram_range=n_grams_range,
                                                 use_idf=True)
                else:
                    term_x = term_representation(preprocessor=pre_process_step, ngram_range=n_grams_range)
                name = f"{get_var_name(term_representation)}_{get_var_name(pre_process_step)}_n_grams_{n_grams_range[0]}_{n_grams_range[1]}"
                data_combination = {
                    name: term_x
                }
                temp_x.append(data_combination)
    return temp_x


x_term_representation = indexing_x()


def x_fit_transform(x, y, terms_x):
    temp_x = []
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to fit transform at: {start_time_gmt}")
    print(f"Total of Terms_x: {len(terms_x)}")
    for y_dict in y:
        for y_name, y_value in y_dict.items():
            # x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(x, y_value, test_size=0.2)
            x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(x, y_value, test_size=0.2, stratify=y_value)
            for term_x in terms_x:
                for term_x_name, term_x_value in term_x.items():
                    # term_x_fit = term_x_value.fit(x)
                    term_x_value.fit(x)
                    term_x_value_train = term_x_value.transform(x_fit)
                    term_x_value_test = term_x_value.transform(x_blind_test)
                    data_combination = {
                        # term_x_name: term_x_fit,
                        "combination": term_x_name,
                        "x_fit": term_x_value_train,
                        "x_blind_test": term_x_value_test,
                        "y_fit": y_fit,
                        "y_blind_test": y_blind_test,
                    }
                    temp_x.append(data_combination)
                    count = len(temp_x)
                    print(f"Total process: {count}")
    joblib.dump(temp_x, f'../resources/x_y_fit_blind_transform.pkl')
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return temp_x


def set_smote(x_y_fit_blind_transform):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print("start to set smote at: " + start_time_gmt)
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        x_smote, y_smote = smote.fit_resample(x_y_fit_blind_transform_dict['x_fit'],
                                              x_y_fit_blind_transform_dict['y_fit'])
        x_y_fit_blind_transform_dict['x_fit'] = x_smote
        x_y_fit_blind_transform_dict['y_fit'] = y_smote
    joblib.dump(x_y_fit_blind_transform, f'../resources/x_y_fit_blind_SMOTE_transform.pkl')
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return x_y_fit_blind_transform


def scale_sparse_matrix(matrix):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(matrix.toarray())
    return csr_matrix(x_scaled)


def log_transform_tfidf(matrix):
    return np.log1p(matrix)


def normalize_x(x_y_fit_blind_transform, normalize_method):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(
        f"start to normalize at: {start_time_gmt} with {get_var_name(x_y_fit_blind_transform)} normalize method: {normalize_method}")
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        if normalize_method == 'min_max':
            x_y_fit_blind_transform_dict['x_fit'] = scale_sparse_matrix(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = scale_sparse_matrix(
                x_y_fit_blind_transform_dict['x_blind_test'])
        elif normalize_method == 'log':
            x_y_fit_blind_transform_dict['x_fit'] = log_transform_tfidf(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = log_transform_tfidf(
                x_y_fit_blind_transform_dict['x_blind_test'])
    joblib.dump(x_y_fit_blind_transform,
                f'../resources/x_y_fit_blind_{get_var_name(x_y_fit_blind_transform)}_{normalize_method}_transform.pkl')
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    return x_y_fit_blind_transform


# normal_result_for_train = x_fit_transform(x, y, x_term_representation)
normal_result_for_train = joblib.load(f'../resources/x_y_fit_blind_transform.pkl')
# SMOTE_result_for_train = set_smote(normal_result_for_train)
SMOTE_result_for_train = joblib.load(f'../resources/x_y_fit_blind_SMOTE_transform.pkl')

# normal_result_for_train_normalize_min_max = normalize_x(normal_result_for_train, 'min_max')
# normal_result_for_train_normalize_log = normalize_x(normal_result_for_train, 'log')
# SMOTE_result_for_train_normalize_min_max = normalize_x(SMOTE_result_for_train, 'min_max')
# SMOTE_result_for_train_normalize_log = normalize_x(SMOTE_result_for_train, 'log')
normal_result_for_train_normalize_min_max = joblib.load(
    f'../resources/x_y_fit_blind_normal_result_for_train_min_max_transform.pkl')
normal_result_for_train_normalize_log = joblib.load(
    f'../resources/x_y_fit_blind_normal_result_for_train_log_transform.pkl')
SMOTE_result_for_train_normalize_min_max = joblib.load(
    f'../resources/x_y_fit_blind_SMOTE_result_for_train_min_max_transform.pkl')
SMOTE_result_for_train_normalize_log = joblib.load(
    f'../resources/x_y_fit_blind_SMOTE_result_for_train_log_transform.pkl')

gbm_model = GradientBoostingClassifier(n_estimators=5000,
                                       learning_rate=0.05,
                                       max_depth=3,
                                       subsample=0.5,
                                       validation_fraction=0.1,
                                       n_iter_no_change=20,
                                       max_features='log2',
                                       verbose=1)


def train(datasets):
    start_time = time.time()
    start_time_gmt = time.gmtime(start_time)
    start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt)
    print(f"start to train at: {start_time_gmt}")
    results_predict = []
    result_cv = []
    for dataset in datasets:
        term_x_name = dataset['combination']
        x_fit = dataset['x_fit']
        x_blind_test = dataset['x_blind_test']
        y_fit = dataset['y_fit']
        y_blind_test = dataset['y_blind_test']
        # for term_x_name, x_fit, x_blind_test, y_fit, y_blind_test in dataset.items():
        # for term_x, x_fit, x_blind_test, y_fit, y_blind_test in dataset.items():
        # Train and evaluate the model
        scoring_metrics = ['precision_macro', 'recall_macro', 'f1_macro']
        cv_results = model_selection.cross_validate(gbm_model, x_fit, y_fit, cv=5, n_jobs=-2,
                                                    scoring=scoring_metrics)
        data_combination_cv_score = {
            "combination": term_x_name,
        }
        for metric in scoring_metrics:
            score = {
                metric: cv_results[f'test_{metric}'].mean()
            }
            data_combination_cv_score.update(score)
        result_cv.append(data_combination_cv_score)

        gbm_model.fit(x_fit, y_fit)
        predict = gbm_model.predict(x_blind_test)

        # Calculate metrics
        precision_test_score = precision_score(y_blind_test, predict)
        recall_test_score = recall_score(y_blind_test, predict)
        f1_test_score = f1_score(y_blind_test, predict)

        predicted_probabilities = gbm_model.predict_proba(x_blind_test)[:, 1]
        auc_test_score = roc_auc_score(y_blind_test, predicted_probabilities)

        data_combination_result_score = {
            "combination": term_x_name,
            "model": gbm_model,
            "precision": precision_test_score,
            "recall": recall_test_score,
            "f1": f1_test_score,
            "auc": auc_test_score
        }
        results_predict.append(data_combination_result_score)
        # Save the model and results path
        results_cv_path = f"../resources/cv_score_{get_var_name(dataset)}.pkl"
        results_predict_path = f"../resources/predict_{get_var_name(dataset)}.pkl"
        joblib.dump(result_cv, results_cv_path)
        joblib.dump(results_predict, results_predict_path)
    end_time = time.time()
    result_time = end_time - start_time
    result_time_gmt = time.gmtime(result_time)
    result_time = time.strftime("%H:%M:%S", result_time_gmt)
    print(f"Total time: {result_time}")
    result_cv = pd.DataFrame(result_cv)
    results_predict = pd.DataFrame(results_predict)

    return result_cv, results_predict


result_normal_cv, result_normal_predict = train(normal_result_for_train)
result_normal_normalize_min_max_cv, result_normal_predict_normalize_min_max_cv = train(
    normal_result_for_train_normalize_min_max)
result_normal_normalize_log_cv, result_normal_predict_normalize_log_cv = train(normal_result_for_train_normalize_log)

result_smote_cv, result_smote_predict = train(SMOTE_result_for_train)
result_smote_predict_normalize_min_max_cv, result_smote_predict_normalize_min_max_predict = train(
    SMOTE_result_for_train_normalize_min_max)
result_smote_result_for_train_normalize_log_cv, result_smote_result_for_train_normalize_log_predict = train(
    SMOTE_result_for_train_normalize_log)
