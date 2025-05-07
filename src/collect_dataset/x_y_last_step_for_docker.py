import pandas as pd
from joblib import dump, load
from pathlib import Path
import platform


def get_default_paths():
    system_name = platform.system()
    print(f"Detected OS: {system_name}")
    if system_name == "Linux":
        base_path = "/app/resources/tsdetect/test_smell_flink"
    elif system_name == "Darwin":  # macOS
        base_path = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")
    return Path(base_path)


# Get input/output paths
base_path = get_default_paths()

# Load data
x = pd.read_pickle(base_path / "flink_clean_description.pkl")
y = pd.read_pickle(base_path / "y_labeled_test_smells.pkl")

# Match rows based on 'pull_number'
x = x[x['pull_number'].isin(y['pull_number'])]

# Ensure sorting order
x = x.sort_values(by='pull_number').reset_index(drop=True)
y = y.sort_values(by='pull_number').reset_index(drop=True)

# Create a list of dictionaries
target_columns = [
    "label_test_semantic_smell",
    "label_issue_in_test_step",
    "label_code_related",
    "label_dependencies",
    "label_test_execution"
]
list_of_dicts = [{col: y[col]} for col in target_columns]

# Save datasets
dump(x, base_path / "x_for_pre_training.pkl")
dump(list_of_dicts, base_path / "y_for_pre_training.pkl")
