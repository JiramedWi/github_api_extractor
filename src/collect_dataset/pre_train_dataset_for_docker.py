print("Start to import libraries")
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
print("Import libraries done")

def get_paths():
    print("Start to get paths")
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR")

    if not input_directory or not output_directory:
        system_name = platform.system()
        print(f"Detected OS: {system_name}")

        if system_name == "Linux":
            input_directory = "/app/resources/tsdetect/test_smell_flink"
            output_directory = "/app/resources/tsdetect/test_smell_flink"
        elif system_name == "Darwin":  # macOS
            input_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
            output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        else:
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
    s = [word for word in blob.words if word not in stopwords.words('english')]
    s = " ".join(s)
    return s


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
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    stop_dict = {s: 1 for s in stopwords_set}
    tags = pos_tag(s)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tags))
    s = [lemmatizer.lemmatize(word, tag) if tag == 'n' or tag == 'v' else None for word, tag in wordnet_tagged]
    s = list(filter(None, s))
    s = [w for w in s if w not in stop_dict]
    s = ' '.join(s)
    return s


def get_var_name(var_value):
    # Use globals() to get the global symbol table
    global_vars = globals()

    # Iterate through items in the symbol table
    for var_name, value in global_vars.items():
        if value is var_value:
            return var_name
    # If the value is not found, return None
    return None


def scale_sparse_matrix(matrix):
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(matrix.toarray())
    return csr_matrix(x_scaled)


def log_transform_tfidf(matrix):
    return np.log1p(matrix)


def set_smote(x_y_fit_blind_transform):
    # Get the input and output directories
    _, output_dir = get_paths()

    count = 0
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        # count loop
        count += 1
        # Set SMOTE oversampling
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        x_smote, y_smote = smote.fit_resample(x_y_fit_blind_transform_dict['x_fit'],
                                              x_y_fit_blind_transform_dict['y_fit'])
        # Check value is smoted or not
        if x_smote.shape[0] > x_y_fit_blind_transform_dict['x_fit'].shape[0] and y_smote.shape[0] > \
                x_y_fit_blind_transform_dict['y_fit'].shape[0]:
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape[0]) + ", x value new = "
                  + str(x_smote.shape[0]) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape[0]) +
                  ", y value new = " + str(y_smote.shape[0]))
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape) + ", x value new = "
                  + str(x_smote.shape) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape) +
                  ", y value new = " + str(y_smote.shape))
            pass
        else:
            raise Exception("It not set smote")
        # Check ratio of Y_smote
        class_distribution_train_smote = pd.Series(y_smote).value_counts()
        print(f"count_y_smote {class_distribution_train_smote}")
        ratio_class_1_train_smote = class_distribution_train_smote[1] / len(y_smote)
        ratio_class_0_train_smote = class_distribution_train_smote[0] / len(y_smote)
        print(f"\nRatio of class '1' in the training smote set: {ratio_class_1_train_smote:.2%}")
        print(f"\nRatio of class '0' in the training smote set: {ratio_class_0_train_smote:.2%}")
        x_y_fit_blind_transform_dict['x_fit'] = x_smote
        x_y_fit_blind_transform_dict['y_fit'] = y_smote
        x_y_fit_blind_transform_dict['y_smote_1_ratio'] = f"{ratio_class_1_train_smote:.2%}"
        x_y_fit_blind_transform_dict['y_smote_0_ratio'] = f"{ratio_class_0_train_smote:.2%}"
        print(f"Total process: {count}")

    output_file = output_dir / 'x_y_fit_blind_SMOTE_transform_optuna.pkl'
    joblib.dump(x_y_fit_blind_transform, output_file)
    return x_y_fit_blind_transform


def set_smote_variants(x_y_fit_blind_transform, naming_file, smote_type):
    # Get the input and output directories
    _, output_dir = get_paths()

    # Dictionary of available SMOTE variants
    smote_variants = {
        'prowsyn': sv.ProWSyn(random_state=42),
        'polynom': sv.polynom_fit_SMOTE(random_state=42),
    }

    # Check if requested SMOTE type exists
    if smote_type not in smote_variants:
        available_types = list(smote_variants.keys())
        raise ValueError(f"SMOTE type '{smote_type}' not supported. Available types: {available_types}")

    # Select the specified SMOTE variant
    selected_smote = smote_variants[smote_type]

    count = 0
    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        # count loop
        count += 1

        print(f"Applying {smote_type} SMOTE variant to data {count}...")

        # Apply the selected SMOTE variant
        x_smote, y_smote = selected_smote.fit_resample(x_y_fit_blind_transform_dict['x_fit'],
                                                       x_y_fit_blind_transform_dict['y_fit'])

        # Check value is smoted or not
        if x_smote.shape[0] > x_y_fit_blind_transform_dict['x_fit'].shape[0] and y_smote.shape[0] > \
                x_y_fit_blind_transform_dict['y_fit'].shape[0]:
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape[0]) + ", x value new = "
                  + str(x_smote.shape[0]) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape[0]) +
                  ", y value new = " + str(y_smote.shape[0]))
            print("set Balanced: x value old = " + str(
                x_y_fit_blind_transform_dict['x_fit'].shape) + ", x value new = "
                  + str(x_smote.shape) + ", y value old = " + str(x_y_fit_blind_transform_dict['y_fit'].shape) +
                  ", y value new = " + str(y_smote.shape))
            pass
        else:
            raise Exception(f"SMOTE variant {smote_type} failed to apply")

        # Check ratio of Y_smote
        class_distribution_train_smote = pd.Series(y_smote).value_counts()
        print(f"count_y_smote {class_distribution_train_smote}")
        ratio_class_1_train_smote = class_distribution_train_smote[1] / len(y_smote)
        ratio_class_0_train_smote = class_distribution_train_smote[0] / len(y_smote)
        print(f"\nRatio of class '1' in the training smote set: {ratio_class_1_train_smote:.2%}")
        print(f"\nRatio of class '0' in the training smote set: {ratio_class_0_train_smote:.2%}")

        # Update dictionary with SMOTE results
        x_y_fit_blind_transform_dict['x_fit'] = x_smote
        x_y_fit_blind_transform_dict['y_fit'] = y_smote
        x_y_fit_blind_transform_dict['smote_variant'] = smote_type
        x_y_fit_blind_transform_dict['y_smote_1_ratio'] = f"{ratio_class_1_train_smote:.2%}"
        x_y_fit_blind_transform_dict['y_smote_0_ratio'] = f"{ratio_class_0_train_smote:.2%}"
        print(f"Total process: {count}")

    # Save to a different output file containing the SMOTE type in the filename
    output_file = output_dir / f'x_y_SMOTE_{naming_file}_{smote_type}_transform.pkl'
    joblib.dump(x_y_fit_blind_transform, output_file)
    return x_y_fit_blind_transform


def normalize_x(x_y_fit_blind_transform, normalize_method):
    # Get the input and output directories
    _, output_dir = get_paths()

    for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
        if normalize_method == 'min_max':
            x_y_fit_blind_transform_dict['x_fit'] = scale_sparse_matrix(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = scale_sparse_matrix(
                x_y_fit_blind_transform_dict['x_blind_test'])
        elif normalize_method == 'log':
            x_y_fit_blind_transform_dict['x_fit'] = log_transform_tfidf(x_y_fit_blind_transform_dict['x_fit'])
            x_y_fit_blind_transform_dict['x_blind_test'] = log_transform_tfidf(
                x_y_fit_blind_transform_dict['x_blind_test'])

    output_file = output_dir / f'normalize_{get_var_name(x_y_fit_blind_transform)}_{normalize_method}_transform.pkl'
    joblib.dump(x_y_fit_blind_transform, output_file)
    return x_y_fit_blind_transform


class MachineLearningScript:
    def __init__(self, source_x: Path, source_y: Path, term_represented: list, pre_process_steps: list,
                 n_gram_range: list):
        # Get the input and output directories
        self.input_dir, self.output_dir = get_paths()

        self.source_x = joblib.load(source_x)
        self.source_y = joblib.load(source_y)
        self.term_represented = term_represented
        self.pre_process_steps = pre_process_steps
        self.n_gram_range = n_gram_range
        self.x_y_fit_blind_transform = None

    def indexing_x(self):
        temp_x = []
        for term_represented in self.term_represented:
            for pre_process_step in self.pre_process_steps:
                for n_gram_range in self.n_gram_range:
                    if term_represented == CountVectorizer:
                        vectorizer = CountVectorizer(preprocessor=pre_process_step, ngram_range=n_gram_range, )
                    else:
                        vectorizer = TfidfVectorizer(preprocessor=pre_process_step, ngram_range=n_gram_range,
                                                     use_idf=True)
                    name = f"{get_var_name(term_represented)}_{get_var_name(pre_process_step)}_n_grams_{n_gram_range[0]}_{n_gram_range[1]}"
                    data_combined = {
                        name: vectorizer
                    }
                    temp_x.append(data_combined)

                    output_file = self.output_dir / 'indexing.pkl'
                    joblib.dump(temp_x, output_file)
        return temp_x

    def data_fit_transform(self, terms_x):
        temp_x = []
        x_cleaned = self.source_x['cleaned_title_n_body']
        y_list = self.source_y
        for y_dict in y_list:
            for y_name, y_value in y_dict.items():
                print(f"start at y_dict name: {y_name}")
                x_fit, x_blind_test, y_fit, y_blind_test = train_test_split(x_cleaned, y_value, test_size=0.2,
                                                                            stratify=y_value)
                # Reset index
                x_fit = x_fit.reset_index(drop=True)
                x_blind_test = x_blind_test.reset_index(drop=True)
                y_fit = y_fit.reset_index(drop=True)
                y_blind_test = y_blind_test.reset_index(drop=True)
                # Check ratio of Y
                class_distribution_train = pd.Series(y_fit).value_counts()
                print(f"count_y {class_distribution_train}")
                class_distribution_test = pd.Series(y_blind_test).value_counts()
                print(f"count_y_test {class_distribution_test}")
                ratio_class_train = len(y_fit) / len(y_value)
                ratio_class_test = len(y_blind_test) / len(y_value)
                ratio_class_1_train = class_distribution_train[1] / len(y_fit)
                ratio_class_1_test = class_distribution_test[1] / len(y_blind_test)
                ratio_class_0_train = class_distribution_train[0] / len(y_fit)
                ratio_class_0_test = class_distribution_test[0] / len(y_blind_test)
                print(f"\nRatio of class in the training set: {ratio_class_train:.2%}")
                print(f"\nRatio of class in the test set: {ratio_class_test:.2%}")
                print(f"\nRatio of class '1' in the training set: {ratio_class_1_train:.2%}")
                print(f"\nRatio of class '0' in the training set: {ratio_class_0_train:.2%}")
                print(f"\nRatio of class '1' in the test set: {ratio_class_1_test:.2%}")
                print(f"\nRatio of class '0' in the test set: {ratio_class_0_test:.2%}")
                for term_x in terms_x:
                    for term_x_name, term_x_value in term_x.items():
                        term_x_value.fit(x_cleaned)
                        term_x_train = term_x_value.transform(x_fit)
                        term_x_test = term_x_value.transform(x_blind_test)
                        data_combination = {
                            "combination": term_x_name,
                            "x_fit": term_x_train,
                            "x_blind_test": term_x_test,
                            "y_name": y_name,
                            "y_fit": y_fit,
                            "y_amount": len(y_fit),
                            "y_blind_test": y_blind_test,
                            "y_fit_ratio": f"{ratio_class_train:.2%}",
                            "y_blind_test_ratio": f"{ratio_class_test:.2%}",
                            "y_fit_1_ratio": f"{ratio_class_1_train:.2%}",
                            "y_fit_0_ratio": f"{ratio_class_0_train:.2%}",
                            "y_blind_test_1_ratio": f"{ratio_class_1_test:.2%}",
                            "y_blind_test_0_ratio": f"{ratio_class_0_test:.2%}",
                            "y_fit_ratio_0_1": f"{class_distribution_train[0]}:{class_distribution_train[1]}",
                            "y_blind_test_ratio_0_1": f"{class_distribution_test[0]}:{class_distribution_test[1]}"
                        }
                        temp_x.append(data_combination)
        self.x_y_fit_blind_transform = temp_x
        output_file = self.output_dir / 'x_y_fit_optuna.pkl'
        joblib.dump(temp_x, output_file)
        return temp_x

    def set_lda_lsa(self, naming_file):
        _, output_dir = get_paths()

        start_time = time.time()
        start_time_gmt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))
        print("Start to set LDA and LSA at:", start_time_gmt)

        count = 0
        x_y_fit_blind_transform = self.x_y_fit_blind_transform

        for x_y_fit_blind_transform_dict in x_y_fit_blind_transform:
            term_condition = x_y_fit_blind_transform_dict['combination'].split('_')[0]

            # ----------- LDA on TF ------------
            if term_condition == 'CountVectorizer':
                print(f"Processing LDA for {term_condition}")
                count += 1

                lda = LatentDirichletAllocation(n_components=500, random_state=42)

                # ✅ fit only on x_fit
                lda.fit(x_y_fit_blind_transform_dict['x_fit'])

                x_lda_fit = lda.transform(x_y_fit_blind_transform_dict['x_fit'])
                x_lda_blind_test = lda.transform(x_y_fit_blind_transform_dict['x_blind_test'])

                x_y_fit_blind_transform_dict['x_fit'] = x_lda_fit
                x_y_fit_blind_transform_dict['x_blind_test'] = x_lda_blind_test
                x_y_fit_blind_transform_dict['term'] = 'TF_LDA'

                del lda, x_lda_fit, x_lda_blind_test

            # ----------- LSA on TFIDF ------------
            elif term_condition == 'TfidfVectorizer':
                print(f"Processing LSA for {term_condition}")
                count += 1

                lsa = TruncatedSVD(n_components=500, random_state=42)

                # ✅ fit only on x_fit
                lsa.fit(x_y_fit_blind_transform_dict['x_fit'])

                x_lsa_fit = lsa.transform(x_y_fit_blind_transform_dict['x_fit'])
                x_lsa_blind_test = lsa.transform(x_y_fit_blind_transform_dict['x_blind_test'])

                x_y_fit_blind_transform_dict['x_fit'] = x_lsa_fit
                x_y_fit_blind_transform_dict['x_blind_test'] = x_lsa_blind_test
                x_y_fit_blind_transform_dict['term'] = 'TFidf_LSA'

                del lsa, x_lsa_fit, x_lsa_blind_test

        print(f"Total transformations applied: {count}")

        output_file = output_dir / f'{naming_file}_with_LDA_LSA.pkl'
        joblib.dump(x_y_fit_blind_transform, output_file)

        total_time = time.time() - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f"Done! Total time for LDA + LSA: {formatted_time}")

        return x_y_fit_blind_transform



def main():
    # Get the input and output directories
    input_dir, output_dir = get_paths()

    # Define input file paths using the input directory
    x_path = input_dir / 'x_for_pre_training.pkl'
    y_source = input_dir / 'y_for_pre_training.pkl'

    term_representations = [CountVectorizer, TfidfVectorizer]
    pre_process_steps = [pre_process_porterstemmer, pre_process_lemmatizer, pre_process_textblob, pre_process_spacy]
    n_grams_ranges = [(1, 1), (1, 2)]

    # To run datafit
    print("Start to data fit transform soon")
    run = MachineLearningScript(x_path, y_source, term_representations, pre_process_steps, n_grams_ranges)
    indexer = run.indexing_x()
    indexer = run.data_fit_transform(indexer)
    set_topic_model = run.set_lda_lsa('x_y_fit_topic_model')
    print("Done with data fit transform")

    # To run smote
    print("Start to set smote soon")
    time.sleep(10)
    # To run with a specific SMOTE variant
    input_dir, _ = get_paths()
    normal_fit = output_dir / 'x_y_fit_optuna.pkl'
    normal_fit_data = joblib.load(normal_fit)
    topic_model = output_dir / f'x_y_fit_topic_model_with_LDA_LSA.pkl'
    topic_model_data = joblib.load(topic_model)
    # Apply different SMOTE variants
    set_smote_variants(normal_fit_data.copy(), 'normal_fit', 'prowsyn')
    set_smote_variants(normal_fit_data.copy(), 'normal_fit', 'polynom')
    set_smote_variants(topic_model_data.copy(), 'topic_model', 'prowsyn')
    set_smote_variants(topic_model_data.copy(), 'topic_model', 'polynom')
    print("Done with SMOTE variants")


if __name__ == '__main__':
    print("Start to run main function")
    main()
    #TODO: Check the dataset compare in the calculation exel
