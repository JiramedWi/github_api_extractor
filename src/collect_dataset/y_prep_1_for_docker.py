import os
import platform
import pandas as pd
from pathlib import Path


# Edit the path to the input and output directories as needed
def get_paths():
    """Get input and output directories from environment variables or default to system-specific paths."""
    input_directory = os.getenv("INPUT_DIR")
    output_directory = os.getenv("OUTPUT_DIR")

    if not input_directory or not output_directory:
        system_name = platform.system()
        print(f"Detected OS: {system_name}")

        if system_name == "Linux":
            input_directory = "/app/resources/tsdetect/test_smell_flink"
            output_directory = "/app/resources/tsdetect/test_smell_flink"
        elif system_name == "Darwin":  # macOS
            input_directory = "/Users/Jumma/temp/tsdetect/home/pee/repo/tmp_flink/tsdetect"
            output_directory = "/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink"
        else:
            raise EnvironmentError(f"Unsupported operating system: {system_name}")

    return Path(input_directory), Path(output_directory)


def process_and_pickle_files(input_directory, output_directory):
    open_dataframes = []
    closed_dataframes = []
    # Check amount of file .csv in directory
    input_directory_files = os.listdir(input_directory)
    print(f"Amount of file in directory: {len(input_directory_files)}")

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.startswith("tsdetect_output_") and filename.endswith(".csv"):
            # Parse the filename to determine its status ('open' or 'closed')
            parts = filename.split("_")
            status = parts[2]  # 'open' or 'closed'
            SHA = parts[4]  # Commit SHA
            pull_number = parts[8]  # Pull Request Number
            pull_number = pull_number.rstrip('.csv')

            # Construct the full file path
            file_path = os.path.join(input_directory, filename)

            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
                df["SHA"] = SHA
                df["pull_number"] = pull_number
                if status == "open":
                    open_dataframes.append(df)
                elif status == "closed":
                    closed_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Combine the DataFrames for "open" and "closed" versions
    open_df = pd.concat(open_dataframes, ignore_index=True) if open_dataframes else pd.DataFrame()
    closed_df = pd.concat(closed_dataframes, ignore_index=True) if closed_dataframes else pd.DataFrame()

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the DataFrames as pickle files
    open_pickle_path = os.path.join(output_directory, "y_open_versions.pkl")
    closed_pickle_path = os.path.join(output_directory, "y_closed_versions.pkl")

    open_df.to_pickle(open_pickle_path)
    closed_df.to_pickle(closed_pickle_path)

    print(f"Open versions DataFrame saved as: {open_pickle_path}")
    print(f"Closed versions DataFrame saved as: {closed_pickle_path}")

    return open_df, closed_df


# Main execution
if __name__ == "__main__":
    try:
        input_dir, output_dir = get_paths()
        print(f"Detected system: {platform.system()}")
        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {output_dir}")
        open_df, closed_df = process_and_pickle_files(input_dir, output_dir)

    except EnvironmentError as e:
        print(e)

