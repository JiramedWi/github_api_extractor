# pretty_print_dicts.py
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

def summarize_value(v):
    """
    - str, int, float (and numpy scalar equivalents): printed as-is
    - all other types: summarized by type/shape/dtype/len/etc.
    """
    # 1) direct-print types
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(v, (str,) + numeric_types):
        return str(v)

    # 2) common container types
    if isinstance(v, dict):
        return f"dict with {len(v)} keys"
    if isinstance(v, (list, tuple, set)):
        return f"{type(v).__name__} of length {len(v)}"

    # 3) numpy array
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape}, dtype={v.dtype}"

    # 4) pandas Series or DataFrame
    if isinstance(v, pd.Series):
        return f"Series length={len(v)}, dtype={v.dtype}"
    if isinstance(v, pd.DataFrame):
        dtypes = {col: str(dt) for col, dt in v.dtypes.items()}
        return f"DataFrame shape={v.shape}, dtypes={dtypes}"

    # 5) SciPy sparse matrix
    if isinstance(v, spmatrix):
        return f"{type(v).__name__} shape={v.shape}, dtype={v.dtype}, nnz={v.nnz}"

    # 6) any object with .shape
    if hasattr(v, "shape"):
        dtype = getattr(v, "dtype", None)
        msg = f"{type(v).__name__} shape={v.shape}"
        if dtype is not None:
            msg += f", dtype={dtype}"
        return msg

    # 7) fallback to length
    try:
        return f"{type(v).__name__} of length {len(v)}"
    except Exception:
        pass

    # 8) last resort
    return f"{type(v).__name__}: {repr(v)}"


def print_dict_list(dict_list, max_records=None):
    """
    Pretty-print up to `max_records` items from dict_list.
    """
    to_print = dict_list if max_records is None else dict_list[:max_records]
    for idx, record in enumerate(to_print):
        print(f"\n=== Record [{idx}] ===")
        for key, val in record.items():
            summary = summarize_value(val)
            print(f"{key:<20}: {summary}")

directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result/final_training"
cv_score_normal = joblib.load(os.path.join(directory, "predict_score_topic_model.pkl"))
print_dict_list(cv_score_normal, 1)