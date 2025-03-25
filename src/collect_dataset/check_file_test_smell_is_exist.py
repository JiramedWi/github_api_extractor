import os
from pathlib import Path

import pandas as pd
import re


def process_files_and_update_dataframe(directory_path, old_df):
    """
    Read files from directory, parse filenames, and update the dataframe with new columns.

    Parameters:
    -----------
    directory_path : str
        Path to the directory containing the files
    old_df : pandas.DataFrame
        Original dataframe with columns ['url', 'open', 'closed']

    Returns:
    --------
    pandas.DataFrame
        Updated dataframe with new columns
    """
    # Create a copy of the original dataframe
    df = old_df.copy()

    # Initialize new columns with default values
    df['is_has_test_smell'] = False
    df['open_file_path'] = None
    df['closed_file_path'] = None

    # Get all files in the directory
    try:
        files = os.listdir(directory_path)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return df

    # Dictionary to store file paths by SHA
    sha_to_file = {}

    # Parse filenames and extract SHA values
    for filename in files:
        # Check if file matches the expected pattern
        # Pattern: version_project_file_number_sha.csv
        pattern = r"^(open|closed)_(\w+)_file_(\d+)_([a-f0-9]+)\.csv$"
        match = re.match(pattern, filename)

        if match:
            version, project, number, sha = match.groups()
            # check sha = '82b628d4730eef32b2f7a022e3b73cb18f950e6e'
            if sha == '82b628d4730eef32b2f7a022e3b73cb18f950e6e':
                print(f"SHA: {sha}, filename: {filename}")
            # Store the file path with its SHA and version
            if sha not in sha_to_file:
                sha_to_file[sha] = {}

            sha_to_file[sha][version] = os.path.join(directory_path, filename)

    # Update the dataframe based on SHA matches
    for idx, row in df.iterrows():
        open_sha = row['open']
        closed_sha = row['closed']

        # Check if both open and closed SHAs have corresponding files
        open_exists = open_sha in sha_to_file and 'open' in sha_to_file[open_sha]
        closed_exists = closed_sha in sha_to_file and 'closed' in sha_to_file[closed_sha]

        if open_exists and closed_exists:
            df.at[idx, 'is_has_test_smell'] = True
            df.at[idx, 'open_file_path'] = sha_to_file[open_sha]['open']
            df.at[idx, 'closed_file_path'] = sha_to_file[closed_sha]['closed']
        elif open_exists:
            df.at[idx, 'open_file_path'] = sha_to_file[open_sha]['open']
        elif closed_exists:
            df.at[idx, 'closed_file_path'] = sha_to_file[closed_sha]['closed']

    return df

# Example usage:
# Assuming you have a dataframe with the structure ['url', 'open', 'closed']
# directory_path = "/path/to/your/files"
# old_dataframe = pd.DataFrame({
#     'url': ['https://github.com/apache/flink/pull/123', 'https://github.com/apache/flink/pull/456'],
#     'open': ['767bf99adba9d220437056c74759310a25d9686c', 'anothersha123456789'],
#     'closed': ['differentsha987654321', 'yetanothersha123456789']
# })
#
# updated_df = process_files_and_update_dataframe(directory_path, old_dataframe)
# print(updated_df)

old_df = pd.read_pickle(Path('/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_use_for_run_java.pkl').resolve())
updated_df = process_files_and_update_dataframe('/home/pee/repo/tmp_flink/csv', old_df)

df_run_tsdetect = updated_df[updated_df['is_has_test_smell'] == True].copy()
df_no_run_tsdetect_yet = updated_df[updated_df['is_has_test_smell'] == False].copy()

df_no_run_tsdetect_yet_from_dto_vm = pd.read_pickle(Path('/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_no_run_tsdetect_yet.pkl').resolve())
# pickle the dataframe
# df_no_run_tsdetect_yet.to_pickle(Path('/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_no_run_tsdetect_yet.pkl').resolve())

