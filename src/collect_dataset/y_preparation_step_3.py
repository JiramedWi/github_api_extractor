import platform

import pandas as pd
from pathlib import Path

def get_default_paths():
    system_name = platform.system()
    if system_name == "Linux":
        input_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        output_directory = "/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink"
    elif system_name == "Darwin":  # macOS
        input_directory = "/path/to/mac/input"
        output_directory = "/path/to/mac/output"
    else:
        raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return input_directory, output_directory

def label_test_smells(open_version_path, closed_version_path, output_path):
    # Load the datasets
    open_version = pd.read_pickle(open_version_path)
    closed_version = pd.read_pickle(closed_version_path)

    # Align the datasets by `pull_number`
    merged_df = pd.merge(open_version, closed_version, on='pull_number', suffixes=('_open', '_closed'))

    # Test smell categories
    test_smell_categories = [
        'test_semantic_smell',
        'issue_in_test_step',
        'code_related',
        'dependencies',
        'test_execution'
    ]

    # Apply the labeling rules
    for category in test_smell_categories:
        merged_df[f'label_{category}'] = (
                merged_df[f'{category}_open'] < merged_df[f'{category}_closed']
        ).astype(int)

    # Keep only the relevant columns
    result = merged_df[['pull_number'] + [f'label_{category}' for category in test_smell_categories]]

    # Save the result to a new CSV
    result.to_pickle(output_path)

    return result


if __name__ == "__main__":
    # Replace with your actual file paths
    input_dir, output_dir = get_default_paths()
    open_version_path = Path(input_dir)  / 'y_open_versions_aggregated.pkl'
    closed_version_path = Path(input_dir) / 'y_closed_versions_aggregated.pkl'


    output_path = Path(output_dir) / 'y_labeled_test_smells.pkl'

    # Call the function
    labeled_data = label_test_smells(open_version_path, closed_version_path, output_path)

    # Print the result
    print(labeled_data)
