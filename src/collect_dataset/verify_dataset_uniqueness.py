# verify_uniqueness_inline.py
"""
Standalone script to verify uniqueness of x_fit and y_fit across dataset entries,
tracking duplicates by source file and entry index, and logging results.
Just run this file in Python; results will appear in console and log file.
"""
import joblib
import numpy as np
import logging
from pathlib import Path

# Configure logging
LOG_FILE_PATH = "/home/pee/repo/github_api_extractor/resources/Logger/uniqueness_verify.log"
logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logging.getLogger().addHandler(console)


def verify_uniqueness(entries):
    """
    entries: list of tuples (dataset_dict, filename, index_in_file)
    Logs and prints any identical x_fit & y_fit pairs, with context.
    """
    n = len(entries)
    duplicate_count = 0
    for i in range(n):
        data_i, file_i, idx_i = entries[i]
        x1 = data_i['x_fit']
        y1 = data_i['y_fit']
        arr1 = x1 if isinstance(x1, np.ndarray) else x1.toarray()
        for j in range(i+1, n):
            data_j, file_j, idx_j = entries[j]
            x2 = data_j['x_fit']
            y2 = data_j['y_fit']
            arr2 = x2 if isinstance(x2, np.ndarray) else x2.toarray()
            if np.array_equal(arr1, arr2) and np.array_equal(y1, y2):
                duplicate_count += 1
                msg = (f"Duplicate {duplicate_count}: "
                       f"{file_i}[{idx_i}] == {file_j}[{idx_j}] (identical x_fit & y_fit)")
                logging.warning(msg)
    if duplicate_count:
        summary = f"Total duplicate pairs found: {duplicate_count}"
        logging.info(summary)
    else:
        logging.info("No duplicates detected.")
    logging.info("Uniqueness verification complete.")


if __name__ == '__main__':
    # --- Customize these input files ---
    input_files = [
        '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_fit_optuna.pkl',
        # '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_SMOTE_normal_fit_polynom_transform.pkl',
        '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/x_y_SMOTE_normal_fit_prowsyn_transform.pkl',
    ]

    # Load entries with context
    entries = []
    for filepath in input_files:
        try:
            items = joblib.load(filepath)
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")
            continue
        if isinstance(items, list):
            for idx, ds in enumerate(items):
                entries.append((ds, filepath, idx))
            logging.info(f"Loaded {len(items)} entries from {filepath}")
        elif isinstance(items, dict):
            entries.append((items, filepath, 0))
            logging.info(f"Loaded single entry from {filepath}")
        else:
            logging.error(f"Unexpected format in {filepath}: {type(items)}")

    if not entries:
        logging.error("No dataset entries loaded. Aborting.")
    else:
        verify_uniqueness(entries)
