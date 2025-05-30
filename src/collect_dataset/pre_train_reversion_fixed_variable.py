import logging
import time
import os
import numpy as np
import pandas as pd
import joblib
import platform
import spacy

from pathlib import Path
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from imblearn.over_sampling import SMOTE
from textblob import TextBlob
import smote_variants as sv
from scipy.sparse import csr_matrix

# Configure a custom log file path
LOG_FILE_PATH = "/home/pee/repo/github_api_extractor/resources/Logger/log_pre_train_reversion_fixed_variable_28_05.log"

logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    filemode='a',  # append mode
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Start to import libraries")

nlp = spacy.load("en_core_web_sm")

logging.info("Import libraries done")

def get_paths():
    logging.info("Start to get paths")
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR")

    if not input_directory or not output_directory:
        system_name = platform.system()
        logging.info(f"Detected OS: {system_name}")

        if system_name == "Linux":
            input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
            output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/new_pre_train_28_05"
        elif system_name == "Darwin":  # macOS
            input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
            output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        else:
            logging.error(f"Unsupported operating system: {system_name}")
            raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


def pos_tagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def pos_tagger_for_spacy(tag):
    tag_dict = {'N': 'NOUN', 'V': 'VERB', 'R': 'ADV', 'J': 'ADJ'}
    return tag_dict.get(tag, 'n')


def pre_process_spacy(s):
    doc = nlp(s)
    s = " ".join([token.lemma_ if token.pos_ in ['NOUN', 'VERB'] else token.text
                  for token in doc if token.pos_ in ['NOUN', 'VERB']])
    return s


def pre_process_textblob(s):
    blob = TextBlob(s)
    s = [word for word in blob.words if word not in stopwords.words('english')]
    s = " ".join(s)
    return s


def pre_process_porterstemmer(s):
    ps = PorterStemmer()
    tokens = word_tokenize(s)
    stopwords_set = set(stopwords.words('english'))
    s = [w for w in tokens if w not in stopwords_set]
    s = [ps.stem(w) for w in s]
    return ' '.join(s)


def pre_process_lemmatizer(s):
    tokens = word_tokenize(s)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    tags = pos_tag(tokens)
    wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in tags]
    s = [lemmatizer.lemmatize(word, tag) for word, tag in wordnet_tagged if tag in [wordnet.NOUN, wordnet.VERB]]
    s = [w for w in s if w not in stopwords_set]
    return ' '.join(s)


def get_var_name(var_value):
    global_vars = globals()
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    return None


def scale_sparse_matrix(matrix):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(matrix.toarray())
    return csr_matrix(x_scaled)


def log_transform_tfidf(matrix):
    return np.log1p(matrix)


def set_smote_variants(datasets, naming_file, smote_type):
    _, output_dir = get_paths()
    smote_variants = {
        'prowsyn': sv.ProWSyn(random_state=42),
        'polynom': sv.polynom_fit_SMOTE_poly(random_state=42),
    }
    if smote_type not in smote_variants:
        available_types = list(smote_variants.keys())
        logging.error(f"SMOTE type '{smote_type}' not supported. Available types: {available_types}")
        raise ValueError(f"SMOTE type '{smote_type}' not supported. Available types: {available_types}")

    selected_smote = smote_variants[smote_type]
    count = 0
    for dataset in datasets:
        count += 1
        logging.info(f"Applying {smote_type} SMOTE variant to data {count}...")
        x = dataset['x_fit'] if isinstance(dataset['x_fit'], np.ndarray) else dataset['x_fit'].toarray()
        y = dataset['y_fit']
        x_smote, y_smote = selected_smote.sample(x, y)

        logging.info(f"Original shapes: x={dataset['x_fit'].shape}, y={dataset['y_fit'].shape}")
        logging.info(f"SMOTE shapes: x={x_smote.shape}, y={y_smote.shape}")
        if x_smote.shape[0] <= dataset['x_fit'].shape[0] or y_smote.shape[0] <= dataset['y_fit'].shape[0]:
            logging.error(f"SMOTE variant {smote_type} failed to apply")
            raise Exception(f"SMOTE variant {smote_type} failed to apply")

        class_distribution_train_smote = pd.Series(y_smote).value_counts()
        logging.info(f"count_y_smote {class_distribution_train_smote.to_dict()}")
        ratio_class_1 = class_distribution_train_smote[1] / len(y_smote)
        ratio_class_0 = class_distribution_train_smote[0] / len(y_smote)
        logging.info(f"Ratio of class '1' in smote set: {ratio_class_1:.2%}")
        logging.info(f"Ratio of class '0' in smote set: {ratio_class_0:.2%}")

        dataset.update({'x_fit': x_smote, 'y_fit': y_smote, 'smote_variant': smote_type,
                        'y_smote_1_ratio': f"{ratio_class_1:.2%}", 'y_smote_0_ratio': f"{ratio_class_0:.2%}"})
        logging.info(f"Total process: {count}")

    output_file = output_dir / f'x_y_SMOTE_{naming_file}_{smote_type}_transform.pkl'
    joblib.dump(datasets, output_file)
    return datasets


def normalize_x(data_list, normalize_method):
    _, output_dir = get_paths()
    for item in data_list:
        if normalize_method == 'min_max':
            item['x_fit'] = scale_sparse_matrix(item['x_fit'])
            item['x_blind_test'] = scale_sparse_matrix(item['x_blind_test'])
        elif normalize_method == 'log':
            item['x_fit'] = log_transform_tfidf(item['x_fit'])
            item['x_blind_test'] = log_transform_tfidf(item['x_blind_test'])
    output_file = output_dir / f'normalize_{get_var_name(data_list)}_{normalize_method}_transform.pkl'
    joblib.dump(data_list, output_file)
    return data_list


class MachineLearningScript:
    def __init__(self, source_x: Path, source_y: Path, term_represented: list, pre_process_steps: list,
                 n_gram_range: list):
        self.input_dir, self.output_dir = get_paths()
        self.source_x = joblib.load(source_x)
        self.source_y = joblib.load(source_y)
        self.term_represented = term_represented
        self.pre_process_steps = pre_process_steps
        self.n_gram_range = n_gram_range
        self.x_y_fit_blind_transform = None

    def indexing_x(self):
        temp_x = []
        for term in self.term_represented:
            for preprocess in self.pre_process_steps:
                for ngram in self.n_gram_range:
                    vectorizer = CountVectorizer(preprocessor=preprocess, ngram_range=ngram) if term == CountVectorizer else TfidfVectorizer(preprocessor=preprocess, ngram_range=ngram, use_idf=True)
                    name = f"{get_var_name(term)}_{get_var_name(preprocess)}_n_grams_{ngram[0]}_{ngram[1]}"
                    temp_x.append({name: vectorizer})
        joblib.dump(temp_x, self.output_dir / 'indexing.pkl')
        return temp_x

    def data_fit_transform(self, terms_x):
        temp_x = []
        x_cleaned = self.source_x['cleaned_title_n_body']
        for y_dict in self.source_y:
            for y_name, y_value in y_dict.items():
                logging.info(f"start at y_dict name: {y_name}")
                x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(
                    x_cleaned, y_value, test_size=0.2, stratify=y_value)
                for desc, arr in zip(["train", "test"], [y_fit, y_blind_test]):
                    dist = pd.Series(arr).value_counts().to_dict()
                    logging.info(f"count_y_{desc} {dist}")
                ratios = {
                    'train': len(y_fit)/len(y_value),
                    'test': len(y_blind_test)/len(y_value)
                }
                for k, v in ratios.items():
                    logging.info(f"Ratio of {k} set: {v:.2%}")
                for preprocess_dict in terms_x:
                    for name, vec in preprocess_dict.items():
                        vec.fit(x_cleaned)
                        term_x_train = vec.transform(x_fit)
                        term_x_test = vec.transform(x_blind_test)
                        data_combination = {
                            "combination": name,
                            "x_fit": term_x_train,
                            "x_blind_test": term_x_test,
                            "y_name": y_name,
                            "y_fit": y_fit,
                            "y_blind_test": y_blind_test,
                            "y_amount": len(y_fit)
                        }
                        temp_x.append(data_combination)
        self.x_y_fit_blind_transform = temp_x
        joblib.dump(temp_x, self.output_dir / 'x_y_fit_optuna.pkl')
        return temp_x

    def set_lda_lsa(self, naming_file):
        _, output_dir = get_paths()
        start_time = time.time()
        start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))
        logging.info(f"Start to set LDA and LSA at: {start_time_gmt}")
        count = 0
        for item in self.x_y_fit_blind_transform:
            term_condition = item['combination'].split('_')[0]
            if term_condition == 'CountVectorizer':
                logging.info(f"Processing LDA for {term_condition}")
                count += 1
                lda = LatentDirichletAllocation(n_components=500, random_state=42)
                lda.fit(item['x_fit'])
                item['x_fit'] = lda.transform(item['x_fit'])
                item['x_blind_test'] = lda.transform(item['x_blind_test'])
                item['term'] = 'TF_LDA'
            elif term_condition == 'TfidfVectorizer':
                logging.info(f"Processing LSA for {term_condition}")
                count += 1
                lsa = TruncatedSVD(n_components=500, random_state=42)
                lsa.fit(item['x_fit'])
                item['x_fit'] = lsa.transform(item['x_fit'])
                item['x_blind_test'] = lsa.transform(item['x_blind_test'])
                item['term'] = 'TFidf_LSA'
        logging.info(f"Total transformations applied: {count}")
        joblib.dump(self.x_y_fit_blind_transform, output_dir / f'{naming_file}_with_LDA_LSA.pkl')
        total_time = time.time() - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
        logging.info(f"Done! Total time for LDA + LSA: {formatted_time}")
        return self.x_y_fit_blind_transform


def main():
    logging.info("Start to run main function")
    input_dir, output_dir = get_paths()
    # x_path = input_dir / 'x_for_pre_training.pkl'
    # y_source = input_dir / 'y_for_pre_training.pkl'
    # term_representations = [CountVectorizer, TfidfVectorizer]
    # pre_process_steps = [pre_process_porterstemmer, pre_process_lemmatizer, pre_process_textblob, pre_process_spacy]
    # n_grams_ranges = [(1, 1), (1, 2)]
    #
    # logging.info("Start to data fit transform soon")
    # run = MachineLearningScript(x_path, y_source, term_representations, pre_process_steps, n_grams_ranges)
    # indexer = run.indexing_x()
    # run.data_fit_transform(indexer)
    # run.set_lda_lsa('x_y_fit_topic_model')
    # logging.info("Done with data fit transform")

    logging.info("Start to set smote soon")
    time.sleep(10)
    # normal_fit_data = joblib.load(output_dir / 'x_y_fit_optuna.pkl')
    normal_fit_data = joblib.load(input_dir / 'x_y_fit_optuna.pkl')
    # topic_model_data = joblib.load(output_dir / f'x_y_fit_topic_model_with_LDA_LSA.pkl')
    topic_model_data = joblib.load(input_dir / f'x_y_fit_topic_model_with_LDA_LSA.pkl')
    set_smote_variants(normal_fit_data.copy(), 'normal_fit', 'prowsyn')
    set_smote_variants(normal_fit_data.copy(), 'normal_fit', 'polynom')
    set_smote_variants(topic_model_data.copy(), 'topic_model', 'prowsyn')
    set_smote_variants(topic_model_data.copy(), 'topic_model', 'polynom')
    logging.info("Done with SMOTE variants")

if __name__ == '__main__':
    main()  # TODO: Check the dataset compare in the calculation excel
