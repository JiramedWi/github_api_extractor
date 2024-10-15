import pandas as pd
from pathlib import Path
from git import Repo
import os, subprocess, platform
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(filename='process_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Determine the operating system and set paths accordingly
def initialize_paths():
    system = platform.system()
    paths = {}
    if system == 'Darwin':  # macOS
        paths['project_repo_path'] = '/path/to/your/project/repo/on/mac'
        paths['tsdetect_path'] = '/path/to/your/tsdetect/TestSmellDetector.jar/on/mac'
        paths['save_result_path'] = '/path/to/your/directory/on/mac'
        project_sha_path = '/path/to/your/directory/on/mac/hive_use_for_run_pre_process.pkl'
    elif system == 'Linux':
        paths['project_repo_path'] = '/home/pee/repo'
        paths['tsdetect_path'] = '/home/pee/repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar'
        paths['save_result_path'] = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive'
        project_sha_path = f"{paths['save_result_path']}/hive_use_for_run_pre_process.pkl"
    else:
        raise EnvironmentError('Unsupported operating system')

    paths['project_sha'] = pd.read_pickle(Path(project_sha_path))
    paths['project_sha'].reset_index(drop=True, inplace=True)

    logging.info('Initialized project repository and paths.')
    return paths


# Check if the given directory or file is related to tests
def is_test_directory(directory):
    return "test" in directory.lower().split("/") and "src" in directory.lower().split("/")


def is_test_file(filename):
    logging.info(f'Checking if {filename} is a test file.')
    return filename.endswith('.java') and ('test' in filename.lower() or 'testcase' in filename.lower())


# Collect all test files in the project directory
def collect_test_files(root_dir):
    test_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if is_test_directory(dirpath):
            for filename in filenames:
                if is_test_file(filename):
                    test_files.append(os.path.join(dirpath, filename))
                    logging.info(f'Collected test file: {os.path.join(dirpath, filename)}')
    return test_files


# Write the test files to a CSV that can be used with the TestSmellDetector.jar
def write_test_files_to_csv(project, save_result_path):
    test_files = collect_test_files(save_result_path)
    df = pd.DataFrame([(project, test_file) for test_file in test_files])
    logging.info(f'Created DataFrame for project {project} with test case files.')
    return df


# Clone the repository and checkout the specific SHA
def clone_and_checkout(repo_url, clone_path, sha):
    if os.path.exists(clone_path):
        logging.info(f'Removing existing directory: {clone_path}')
        subprocess.run(['rm', '-rf', clone_path])

    logging.info(f'Cloning repository from {repo_url} to {clone_path}')
    repo = Repo.clone_from(repo_url, clone_path, no_checkout=True)
    repo.git.checkout(sha)
    logging.info(f'Checked out SHA: {sha}')
    return repo


# Run the TestSmellDetector tool for the given SHA
def run_tsdetect(project_name, count, sha_type, sha, testfile_prefix, save_result_path, tsdetect_path):
    test_files = write_test_files_to_csv(project_name, save_result_path)
    testfile_path = f"{save_result_path}/csv/{testfile_prefix}_{project_name}_file_{count}_{sha}.csv"
    test_files.to_csv(testfile_path, index=False, header=None)
    logging.info(f'Saved test files for {sha_type} SHA to CSV: {testfile_prefix}_{project_name}_file_{count}_{sha}.csv')

    subprocess.run(['java', '-jar', tsdetect_path, testfile_path])
    logging.info(f'Ran TestSmellDetector for {sha_type} SHA.')


# Clone the repository, checkout both SHAs, and run the detector
def process_checkout(count, project_name, project_url, url, sha_opened, sha_closed, paths):
    clone_path = f"{paths['project_repo_path']}/tmp/clone_repo_{count}_{project_name}"

    # Process open SHA
    logging.info(f'Processing open SHA for URL: {project_url} at pull request {url}')
    repo = clone_and_checkout(project_url, clone_path, sha_opened)
    run_tsdetect(project_name, count, "open", sha_opened, 'open', paths['save_result_path'], paths['tsdetect_path'])

    # Remove cloned directory
    subprocess.run(['rm', '-rf', clone_path])

    # Process closed SHA
    logging.info(f'Processing closed SHA for URL: {project_url} at pull request {url}')
    repo = clone_and_checkout(project_url, clone_path, sha_closed)
    run_tsdetect(project_name, count, "closed", sha_closed, 'closed', paths['save_result_path'], paths['tsdetect_path'])

    # Clean up cloned directory again
    subprocess.run(['rm', '-rf', clone_path])


# Perform the auto-checkout for all SHAs in the project
def auto_checkout(project_name, project_url, paths):
    sha_opened = paths['project_sha']['open']
    sha_closed = paths['project_sha']['closed']
    urls = paths['project_sha']['url']

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(process_checkout, count, project_name, project_url, url, sha_opened[count],
                            sha_closed[count], paths)
            for count, url in enumerate(urls)
        ]
        for future in futures:
            future.result()

    logging.info('Completed auto-checkout process.')


# Main execution
if __name__ == "__main__":
    paths = initialize_paths()
    auto_checkout('hive', 'https://github.com/apache/hive.git', paths)
