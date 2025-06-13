import pandas as pd
import os
import joblib


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
                "source_name": source_name  # Add which dataset it came from
            }
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    # Parse combination string
    def parse_combination(combination_str):
        parts = combination_str.split("_pre_process_")
        if len(parts) < 2:
            return pd.Series({"Textual feature": None, "Stem lemma": None, "N-gram": None})
        textual_feature = parts[0]
        rest = parts[1]
        if "_n_grams_" in rest:
            stem_lemma, ngram_str = rest.split("_n_grams_")
            n_gram = "1" if ngram_str == "1_1" else "2" if ngram_str == "1_2" else ngram_str
        else:
            stem_lemma, n_gram = rest, None
        # Map vectorizer names
        textual_feature_map = {"CountVectorizer": "TF", "TFIDFVectorizer": "TF-IDF"}
        textual_feature = textual_feature_map.get(textual_feature, textual_feature)
        return pd.Series({
            "Textual feature": textual_feature,
            "Stem lemma": stem_lemma,
            "N-gram": n_gram
        })
    comb_df = df["combination"].apply(parse_combination)
    df = pd.concat([df, comb_df], axis=1)
    # Infer imbalance handling
    def infer_imba_handling(source_name):
        s = source_name.lower()
        if "poly" in s:
            return "Polynomial Fit SMOTE"
        elif "prowsyn" in s:
            return "ProWSyn SMOTE"
        else:
            return "None"
    df["Imba handling"] = df["source_name"].apply(infer_imba_handling)
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

