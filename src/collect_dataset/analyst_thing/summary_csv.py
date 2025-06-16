import pandas as pd
import os
import joblib

import pandas as pd

def parse_combination(combination_str):
    """
    Parses a combination string like:
    CountVectorizer_pre_process_porterstemmer_n_grams_1_2
    TFIDFVectorizer_pre_process_lemmatizer_n_grams_1_2
    Returns:
      - Textual feature
      - Stem lemma
      - N-gram
    """
    if not isinstance(combination_str, str) or not combination_str:
        return pd.Series({"Textual feature": None, "Stem lemma": None, "N-gram": None, "Topic modeling": None})

    # Default values
    textual_feature = None
    stem_lemma = None
    n_gram = None

    # Handle topic modeling (LDA/LSA) as suffix or embedded
    # Example: "TF_LDA_pre_process_spacy_n_grams_1_2"
    #          "TFIDF_LSA_pre_process_textblob_n_grams_1_1"
    if "_pre_process_" in combination_str and "_n_grams_" in combination_str:
        before_pre, after_pre = combination_str.split("_pre_process_")
        stem_and_ngram = after_pre.split("_n_grams_")
        stem_lemma = stem_and_ngram[0]
        n_gram_rest = stem_and_ngram[1]
        # Check for topic modeling as suffix in before_pre or after n_grams
        if "countvectorizer" in before_pre.lower():
            textual_feature = "TF"
        elif "tfidfvectorizer" in before_pre.lower():
            textual_feature = "TF-IDF"
        # If n_gram
        if n_gram_rest in ["11", "1_1"]:
            n_gram = "1"
        elif n_gram_rest in ["12", "1_2"]:
            n_gram = "2"
    else:
        textual_feature = combination_str
    # Standardize n_gram to "1" or "2" for 1_1 or 1_2 etc.

    return pd.Series({
        "Textual feature": textual_feature,
        "Stem lemma": stem_lemma,
        "N-gram": n_gram,
    })

def infer_imba_handling(source_name):
    s = source_name.lower()
    if "poly" in s:
        return "Polynomial Fit"
    elif "prowsyn" in s:
        return "ProWSyn"
    else:
        return "None"

def infer_topic_model(source_name, vectorizer_name):
    s = source_name.lower()
    v = vectorizer_name
    if "normal" in s:
        return "None"
    elif "topic" in s and v == "TF":
        return "LDA"
    elif "topic" in s and v == "TF-IDF":
        return "LSA"
    else:
        return "None"

def convert_list_result_to_csv_multi(list_of_tuples, csv_path):
    """
    list_of_tuples: list of (result_list, source_name)
    """
    all_rows = []
    for result_list, source_name in list_of_tuples:
        for d in result_list:
            row = {
                "combination": d.get("combination", ""),
                "y_name": d.get("y_name", ""),
                "cv_precision_macro": d.get("cv_precision_macro"),
                "cv_recall_macro": d.get("cv_recall_macro"),
                "cv_f1_macro": d.get("cv_f1_macro"),
                "cv_roc_auc": d.get("cv_roc_auc"),
                "test_precision": d.get("test_precision"),
                "test_recall": d.get("test_recall"),
                "test_f1": d.get("test_f1"),
                "test_roc_auc": d.get("test_roc_auc"),
                "result": d.get("result"),
                "source_name": source_name
            }
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    # Parse combination string to columns
    comb_df = df["combination"].apply(parse_combination)
    df = pd.concat([df, comb_df], axis=1)
    # Imba handling
    df["Imba handling"] = df["source_name"].apply(infer_imba_handling)
    # Topic modeling
    df["Topic modeling"] = df.apply(
        lambda row: infer_topic_model(row["source_name"], row["Textual feature"]),
        axis=1
    )
    # Save
    df.to_csv(csv_path, index=False)
    print(f"Saved merged summary CSV to {csv_path}")
    return df


directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/final_training"

datasets = [
    (joblib.load(os.path.join(directory, "predict_score_normal.pkl")),           "normal"),
    (joblib.load(os.path.join(directory, "predict_score_topic_model.pkl")),      "topic_model"),
    (joblib.load(os.path.join(directory, "predict_score_smote_poly_normal.pkl")),"smote_poly_normal"),
    (joblib.load(os.path.join(directory, "predict_score_smote_poly_topic.pkl")), "smote_poly_topic"),
    (joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_normal.pkl")),"smote_prowsyn_normal"),
    (joblib.load(os.path.join(directory, "predict_score_smote_prowsyn_topic.pkl")), "smote_prowsyn_topic"),
]

csv_path = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/merged_summary.csv"

convert_list_result_to_csv_multi(datasets, csv_path)

