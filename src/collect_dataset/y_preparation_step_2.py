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

def aggregate_test_smells(data, categories):
    for category, smells in categories.items():
        data[category] = data[smells].sum(axis=1)

    # Aggregate by 'SHA' and 'pull_number'
    aggregated = data.groupby(['SHA', 'pull_number'], as_index=False).sum()
    # print head 10
    print(aggregated.head(10))

    # Keep only relevant columns
    relevant_columns = ['SHA', 'pull_number'] + list(categories.keys())
    return aggregated[relevant_columns]

# Main execution
if __name__ == "__main__":
    # Load your dataset
    input_dir, output_dir = get_default_paths()
    input_file_open_version = Path(input_dir) / "y_open_versions.pkl"
    input_file_closed_version = Path(input_dir) / "y_closed_versions.pkl"
    output_file_open_version = Path(output_dir) / "y_open_versions_aggregated.pkl"
    output_file_closed_version = Path(output_dir) / "y_closed_versions_aggregated.pkl"
    data_open_service = pd.read_pickle(input_file_open_version)
    data_closed_service = pd.read_pickle(input_file_closed_version)

    # Define the test smell categories
    test_smell_categories = {
        "test_semantic_smell": ['Assertion Roulette', 'Conditional Test Logic', 'Duplicate Assert'],
        "issue_in_test_step": ['Exception Catching Throwing', 'General Fixture', 'EmptyTest', 'Redundant Assertion', 'Unknown Test', 'Constructor Initialization'],
        "code_related": ['Magic Number Test', 'Print Statement', 'IgnoredTest', 'Verbose Test'],
        "dependencies": ['Mystery Guest', 'Resource Optimism'],
        "test_execution": ['Sensitive Equality', 'Sleepy Test']
    }

    # Aggregate test smells
    aggregated_result_open_version = aggregate_test_smells(data_open_service, test_smell_categories)
    aggregated_result_closed_version = aggregate_test_smells(data_closed_service, test_smell_categories)

    # Save the result
    aggregated_result_open_version.to_pickle(output_file_open_version)
    aggregated_result_closed_version.to_pickle(output_file_closed_version)
    print(f"Aggregated test smells saved to {output_file_open_version}")
    print(f"Aggregated test smells saved to {output_file_closed_version}")
