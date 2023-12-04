import nltk
import os
import numpy as np
import pandas as pd
import spacy
import joblib
import inspect

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

# TODO: Preparing pre-process,
# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")


def pos_tagger_for_spacy(tag):
    # Mapping NLTK POS tags to spaCy POS tags
    tag_dict = {'N': 'NOUN', 'V': 'VERB', 'R': 'ADV', 'J': 'ADJ'}
    return tag_dict.get(tag, 'n')


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


# TODO: Prepare X, Y as Same length

sourceFile = ['../resources/tsdetect/all_test_smell/df_test_semantic_smell.csv',
              '../resources/tsdetect/all_test_smell/df_issue_in_test_step.csv',
              '../resources//tsdetect/all_test_smell/df_code_related.csv',
              '../resources/tsdetect/all_test_smell/df_dependencies.csv',
              '../resources/tsdetect/all_test_smell/df_test_execution.csv']


def compare_y_to_x(dfx, dfy):
    return dfy.loc[dfy['url'].isin(dfx['url'])]


def read_source(sourceFiles):
    path = os.path.dirname(__file__)
    x_path = Path(os.path.abspath(os.path.join(path, '../resources/clean_demo.pkl')))
    x = pd.read_pickle(x_path)
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

# TODO: Prepare X: TF, TF-IDF, Ngram 1-3,
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


def indexing_x():
    temp_x = []
    for term_representation in term_representations:
        for pre_process_step in pre_process_steps:
            for n_grams_range in n_grams_ranges:
                term_x = term_representation(preprocessor=pre_process_step, ngram_range=n_grams_range)
                name = f"{get_var_name(term_representation)}_{get_var_name(pre_process_step)}_n_grams_{n_grams_range[0]}_{n_grams_range[1]}"
                data_combination = {
                    name: term_x
                }
                temp_x.append(data_combination)
    return temp_x


term_x_result = indexing_x()


# TODO: Method to apply 1,0 and Log(1+X) normalization
def scale_sparse_matrix(matrix):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(matrix.toarray())
    return csr_matrix(x_scaled)


def log_transform_tfidf(matrix):
    return np.log1p(matrix)


# TODO: Fit each X
# porter1
X_tf_train_porter_n_gram_1 = tf_vectorizer_porter_pre_n_gram_1.fit_transform(x)
X_tfidf_train_porter_n_gram_1 = tfidf_vectorizer_porter_pre_n_gram_1.fit_transform(x)
# lemma1
X_tf_train_lemma_n_gram_1 = tf_vectorizer_lemma_pre_n_gram_1.fit_transform(x)
X_tfidf_train_lemma_n_gram_1 = tfidf_vectorizer_lemma_pre_n_gram_1.fit_transform(x)
# spacy1
X_tf_train_spacy_n_gram_1 = tf_vectorizer_spacy_pre_n_gram_1.fit_transform(x)
X_tfidf_train_spacy_n_gram_1 = tfidf_vectorizer_spacy_pre_n_gram_1.fit_transform(x)
# textblob1
X_tf_train_textblob_n_gram_1 = tf_vectorizer_textblob_pre_n_gram_1.fit_transform(x)
X_tfidf_train_textblob_n_gram_1 = tfidf_vectorizer_textblob_pre_n_gram_1.fit_transform(x)

# porter2
X_tf_train_porter_n_gram_2 = tf_vectorizer_porter_pre_n_gram_2.fit_transform(x)
X_tfidf_train_porter_n_gram_2 = tfidf_vectorizer_porter_pre_n_gram_2.fit_transform(x)
# lemma2
X_tf_train_lemma_n_gram_2 = tf_vectorizer_lemma_pre_n_gram_2.fit_transform(x)
X_tfidf_train_lemma_n_gram_2 = tfidf_vectorizer_lemma_pre_n_gram_2.fit_transform(x)
# spacy2
X_tf_train_spacy_n_gram_2 = tf_vectorizer_spacy_pre_n_gram_2.fit_transform(x)
X_tfidf_train_spacy_n_gram_2 = tfidf_vectorizer_spacy_pre_n_gram_2.fit_transform(x)
# textblob2
X_tf_train_textblob_n_gram_2 = tf_vectorizer_textblob_pre_n_gram_2.fit_transform(x)
X_tfidf_train_textblob_n_gram_2 = tfidf_vectorizer_textblob_pre_n_gram_2.fit_transform(x)

# porter3
X_tf_train_porter_n_gram_3 = tf_vectorizer_porter_pre_n_gram_3.fit_transform(x)
X_tfidf_train_porter_n_gram_3 = tfidf_vectorizer_porter_pre_n_gram_3.fit_transform(x)
# lemma3
X_tf_train_lemma_n_gram_3 = tf_vectorizer_lemma_pre_n_gram_3.fit_transform(x)
X_tfidf_train_lemma_n_gram_3 = tfidf_vectorizer_lemma_pre_n_gram_3.fit_transform(x)
# spacy3
X_tf_train_spacy_n_gram_3 = tf_vectorizer_spacy_pre_n_gram_3.fit_transform(x)
X_tfidf_train_spacy_n_gram_3 = tfidf_vectorizer_spacy_pre_n_gram_3.fit_transform(x)
# textblob3
X_tf_train_textblob_n_gram_3 = tf_vectorizer_textblob_pre_n_gram_3.fit_transform(x)
X_tfidf_train_textblob_n_gram_3 = tfidf_vectorizer_textblob_pre_n_gram_3.fit_transform(x)

# TODO: Prepare X: Normalization (0-1) and Log(1+x) only TFIDF
# X_tfidf_train_porter_01 = scale_sparse_matrix(X_tfidf_train_porter)
# X_tfidf_train_porter_log = log_transform_tfidf(X_tfidf_train_porter)
# X_tfidf_test_porter_01 = scale_sparse_matrix(X_tfidf_test_porter)
# X_tfidf_test_porter_log = log_transform_tfidf(X_tfidf_test_porter)
# #
# X_tfidf_train_lemma_01 = scale_sparse_matrix(X_tfidf_train_lemma)
# X_tfidf_train_lemma_log = log_transform_tfidf(X_tfidf_train_lemma)
# X_tfidf_test_lemma_01 = scale_sparse_matrix(X_tfidf_test_lemma)
# X_tfidf_test_lemma_log = log_transform_tfidf(X_tfidf_test_lemma)
# #
# X_tfidf_train_spacy_01 = scale_sparse_matrix(X_tfidf_train_spacy)
# X_tfidf_train_spacy_log = log_transform_tfidf(X_tfidf_train_spacy)
# X_tfidf_test_spacy_01 = scale_sparse_matrix(X_tfidf_test_spacy)
# X_tfidf_test_spacy_log = log_transform_tfidf(X_tfidf_test_spacy)
# #
# X_tfidf_train_textblob_01 = scale_sparse_matrix(X_tfidf_train_textblob)
# X_tfidf_train_textblob_log = log_transform_tfidf(X_tfidf_train_textblob)
# X_tfidf_test_textblob_01 = scale_sparse_matrix(X_tfidf_test_textblob)
# X_tfidf_test_textblob_log = log_transform_tfidf(X_tfidf_test_textblob)

X_train_list = [X_tf_train_porter_n_gram_1, X_tf_train_porter_n_gram_2, X_tf_train_porter_n_gram_3,
                X_tfidf_train_porter_n_gram_1, X_tfidf_train_porter_n_gram_2, X_tfidf_train_porter_n_gram_3,
                X_tf_train_lemma_n_gram_1, X_tf_train_lemma_n_gram_2, X_tf_train_lemma_n_gram_3,
                X_tfidf_train_lemma_n_gram_1, X_tfidf_train_lemma_n_gram_2, X_tfidf_train_lemma_n_gram_3,
                X_tf_train_spacy_n_gram_1, X_tf_train_spacy_n_gram_2, X_tf_train_spacy_n_gram_3,
                X_tfidf_train_spacy_n_gram_1, X_tfidf_train_spacy_n_gram_2, X_tfidf_train_spacy_n_gram_3,
                X_tf_train_textblob_n_gram_1, X_tf_train_textblob_n_gram_2, X_tf_train_textblob_n_gram_3,
                X_tfidf_train_textblob_n_gram_1, X_tfidf_train_textblob_n_gram_2, X_tfidf_train_textblob_n_gram_3]
Y_train_list = [y_test_semantic_smell, y_issue_in_test_step, y_code_related, y_dependencies, y_test_execution]

X_train_dict = {
    "X_tf_train_porter_n_gram_1": X_tf_train_porter_n_gram_1,
    "X_tf_train_porter_n_gram_2": X_tf_train_porter_n_gram_2,
    "X_tf_train_porter_n_gram_3": X_tf_train_porter_n_gram_3,
    "X_tfidf_train_porter_n_gram_1": X_tfidf_train_porter_n_gram_1,
    "X_tfidf_train_porter_n_gram_2": X_tfidf_train_porter_n_gram_2,
    "X_tfidf_train_porter_n_gram_3": X_tfidf_train_porter_n_gram_3,
    "X_tf_train_lemma_n_gram_1": X_tf_train_lemma_n_gram_1,
    "X_tf_train_lemma_n_gram_2": X_tf_train_lemma_n_gram_2,
    "X_tf_train_lemma_n_gram_3": X_tf_train_lemma_n_gram_3,
    "X_tfidf_train_lemma_n_gram_1": X_tfidf_train_lemma_n_gram_1,
    "X_tfidf_train_lemma_n_gram_2": X_tfidf_train_lemma_n_gram_2,
    "X_tfidf_train_lemma_n_gram_3": X_tfidf_train_lemma_n_gram_3,
    "X_tf_train_spacy_n_gram_1": X_tf_train_spacy_n_gram_1,
    "X_tf_train_spacy_n_gram_2": X_tf_train_spacy_n_gram_2,
    "X_tf_train_spacy_n_gram_3": X_tf_train_spacy_n_gram_3,
    "X_tfidf_train_spacy_n_gram_1": X_tfidf_train_spacy_n_gram_1,
    "X_tfidf_train_spacy_n_gram_2": X_tfidf_train_spacy_n_gram_2,
    "X_tfidf_train_spacy_n_gram_3": X_tfidf_train_spacy_n_gram_3,
    "X_tf_train_textblob_n_gram_1": X_tf_train_textblob_n_gram_1,
    "X_tf_train_textblob_n_gram_2": X_tf_train_textblob_n_gram_2,
    "X_tf_train_textblob_n_gram_3": X_tf_train_textblob_n_gram_3,
    "X_tfidf_train_textblob_n_gram_1": X_tfidf_train_textblob_n_gram_1,
    "X_tfidf_train_textblob_n_gram_2": X_tfidf_train_textblob_n_gram_2,
    "X_tfidf_train_textblob_n_gram_3": X_tfidf_train_textblob_n_gram_3
}

Y_train_dict = {
    "y_test_semantic_smell": y_test_semantic_smell,
    "y_issue_in_test_step": y_issue_in_test_step,
    "y_code_related": y_code_related,
    "y_dependencies": y_dependencies,
    "y_test_execution": y_test_execution
}
#
# # Generate the text representation
# text_representation = "{\n"
# # for var in X_train_list:
# for var in Y_train_list:
#     var_name = [name for name, value in globals().items() if value is var]
#     if var_name:
#         var_name = var_name[0]
#         text_representation += f'"{var_name}": {var_name},\n'
#
# # Remove the trailing comma and add a closing curly brace
# text_representation = text_representation.rstrip(",\n") + "\n}"
# print(text_representation)

dirname = os.path.expanduser('~')
data_set_path = os.path.join(dirname, 'data_set')

print(dirname)
print(data_set_path)


# TODO: Prepare X,Y: Split80:20, SeT SMOTE
# x_fit, x_test = model_selection.train_test_split(x, test_size=0.2)
#
# y_for_train_test_semantic_smell, y_for_test_test_semantic_smell = model_selection.train_test_split(
#     y_test_semantic_smell, test_size=0.2)
# y_for_train_issue_in_test_step, y_for_test_issue_in_test_step = model_selection.train_test_split(y_issue_in_test_step,
#                                                                                                  test_size=0.2)
# y_for_train_code_related, y_for_test_code_related = model_selection.train_test_split(y_code_related, test_size=0.2)
# y_for_train_dependencies, y_for_test_dependencies = model_selection.train_test_split(y_dependencies, test_size=0.2)
# y_for_train_test_execution, y_for_test_test_execution = model_selection.train_test_split(y_test_execution,
#                                                                                          test_size=0.2)
def generate_data_combinations_default(X_train_dict, Y_train_dict, output_dir, normalization_method):
    data_combinations = []  # List to store data combinations

    for X_train_name, X_train in X_train_dict.items():
        for Y_train_name, Y_train in Y_train_dict.items():

            X_resampled, X_test, Y_resampled, Y_test = model_selection.train_test_split(X_train, Y_train,
                                                                                        test_size=0.2, random_state=42)
            if normalization_method == '0-1':
                X_resampled = scale_sparse_matrix(X_resampled)
            elif normalization_method == 'log1+x':
                X_resampled = log_transform_tfidf(X_resampled)
            else:
                pass
            # Store the combination as a dictionary
            data_combination = {
                "X_train": X_resampled,
                "Y_train": Y_resampled,
                "X_test": X_test,
                "Y_test": Y_test
            }
            data_combinations.append(data_combination)

            # Save the data combination to a file (optional)
            output_file = f"{output_dir}/data_combination_{X_train_name}_{Y_train_name}_normalize_{normalization_method}_default.pkl"
            joblib.dump(data_combination, output_file)
            print(f'{X_train_name} and {Y_train_name} in default are done')

    return data_combinations


data_combinations = generate_data_combinations_default(X_train_dict, Y_train_dict, data_set_path, '0-1')
data_combinations_log1 = generate_data_combinations_default(X_train_dict, Y_train_dict, data_set_path, 'log1+x')
data_combinations_default = generate_data_combinations_default(X_train_dict, Y_train_dict, data_set_path, 'donothing')


def generate_data_combinations_smote(X_train_dict, Y_train_dict, output_dir, normalization_method):
    data_combinations = []  # List to store data combinations

    for X_train_name, X_train in X_train_dict.items():
        for Y_train_name, Y_train in Y_train_dict.items():
            # Apply SMOTE
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

            X_resampled, X_test, Y_resampled, Y_test = train_test_split(X_resampled, Y_resampled,
                                                                        test_size=0.2, random_state=42)
            if normalization_method == '0-1':
                X_resampled = scale_sparse_matrix(X_resampled)
            elif normalization_method == 'log1+x':
                X_resampled = log_transform_tfidf(X_resampled)
            else:
                pass
            # Store the combination as a dictionary
            data_combination = {
                "X_train": X_resampled,
                "Y_train": Y_resampled,
                "X_test": X_test,
                "Y_test": Y_test
            }
            data_combinations.append(data_combination)

            # Save the data combination to a file (optional)
            output_file = f"{output_dir}/data_combination_{X_train_name}_{Y_train_name}_normalize_{normalization_method}_SMOTE.pkl"
            joblib.dump(data_combination, output_file)
            print(f'{X_train_name} and {Y_train_name} in SMOTE are done')
    return data_combinations


data_combinations_smote = generate_data_combinations_smote(X_train_dict, Y_train_dict, data_set_path, '0-1')
data_combinations_log1_smote = generate_data_combinations_smote(X_train_dict, Y_train_dict, data_set_path, 'log1+x')
data_combinations_default_smote = generate_data_combinations_smote(X_train_dict, Y_train_dict, data_set_path,
                                                                   'donothing')
print('done')
