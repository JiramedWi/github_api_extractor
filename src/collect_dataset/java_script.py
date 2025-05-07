import re

import pandas as pd
from pathlib import Path
from logging.handlers import RotatingFileHandler
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from git import Repo

import os, subprocess, platform
import logging
import shutil

# Configure logging
# Set up a rotating file handler that limits log file size to 10 MB with backup log files
log_handler = RotatingFileHandler('/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/log/process_log.log', maxBytes=10 * 1024 * 1024, backupCount=5)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

logging.info('Logging initialized with file rotation.')


# Check the disk space before running the process
def check_disk_usage(path, min_free_space_gb):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_space_gb:
        raise Exception(f"Low disk space. Only {free_gb:.2f} GB remaining.")
    logging.info(f"Disk space check passed: {free_gb:.2f} GB free.")


# Get  the pull request number from the URL
def get_pr_number(url: str):
    try:
        # Use regular expression to extract the pull request number from the URL
        match = re.search(r'/pulls/(\d+)', url)
        if match:
            return match.group(1)
        else:
            # Log an error if the URL is invalid
            raise ValueError("Invalid GitHub pull request URL")
    except ValueError as e:
        # Log the error message
        logging.error("Error: %s - URL: %s", e, url)
        raise  # Optionally re-raise the exception if needed


# Determine the operating system and set paths accordingly
def initialize_paths():
    try:
        system = platform.system()
        paths = {}
        if system == 'Darwin':  # macOS
            paths['project_repo_path'] = '/path/to/your/project/repo/on/mac'
            paths['tsdetect_path'] = '/path/to/your/tsdetect/TestSmellDetector.jar/on/mac'
            paths['save_result_path'] = '/path/to/your/directory/on/mac'
            project_sha_path = '/path/to/your/directory/on/mac/flink_use_for_run_pre_process.pkl'
        ## Fix path here
        elif system == 'Linux':
            paths['project_repo_path'] = '/home/pee/repo'
            paths['tsdetect_path'] = '/home/pee/repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar'
            paths['save_result_path'] = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink'
            project_sha_path = '/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_no_run_tsdetect_yet.pkl'
        else:
            raise EnvironmentError('Unsupported operating system')

        paths['project_sha'] = pd.read_pickle(Path(project_sha_path))
        paths['project_sha'].reset_index(drop=True, inplace=True)

        logging.info('Initialized project repository and paths.')
        return paths
    except Exception as e:
        logging.error(f'Error initializing paths: {e}', exc_info=True)
        raise


# Check if the given directory or file is related to tests
def is_test_directory(directory):
    try:
        return "test" in directory.lower().split("/") and "src" in directory.lower().split("/")
    except Exception as e:
        logging.error(f'Error checking test directory: {directory}, Error: {e}', exc_info=True)
        raise


def is_test_file(filename):
    try:
        # logging.info(f'Checking if {filename} is a test file.')
        return filename.endswith('.java') and ('test' in filename.lower() or 'testcase' in filename.lower())
    except Exception as e:
        logging.error(f'Error checking test file: {filename}, Error: {e}', exc_info=True)
        raise


# Collect all test files in the project directory
def collect_test_files(root_dir):
    try:
        test_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            # print walk through directories for debugging
            # print(dirpath, filenames)
            if is_test_directory(dirpath):
                for filename in filenames:
                    if is_test_file(filename):
                        test_files.append(os.path.join(dirpath, filename))
                        # logging.info(f'Collected test file: {os.path.join(dirpath, filename)}')
        return test_files
    except Exception as e:
        logging.error(f'Error collecting test files in directory: {root_dir}, Error: {e}', exc_info=True)
        raise


# Write the test files to a CSV that can be used with the TestSmellDetector.jar
def write_test_files_to_csv(project, directory_repo):
    try:
        test_files = collect_test_files(directory_repo)
        df = pd.DataFrame([(project, test_file) for test_file in test_files])
        logging.info(f'Created DataFrame for project {project} with test case files.')
        return df
    except Exception as e:
        logging.error(f'Error writing test files to CSV for project {project}, Error: {e}', exc_info=True)
        raise


# Clone the repository and checkout the specific SHA
def clone_and_checkout(repo_url, clone_path, sha):
    try:
        if not os.path.exists(clone_path):
            # Clone only if the repository doesn't exist
            logging.info(f'Cloning repository from {repo_url} to {clone_path}')
            Repo.clone_from(repo_url, clone_path)
        else:
            logging.info(f'Reusing existing repository at {clone_path}')

        repo = Repo(clone_path)

        # Reset any local changes (remove untracked and modified files)
        logging.info(f'Cleaning local changes in {clone_path}')
        repo.git.reset('--hard')
        repo.git.clean('-fd')

        # Checkout the desired SHA
        logging.info(f'Checking out SHA: {sha}')
        repo.git.checkout(sha)
        print(f"start checkout repo {repo_url} with sha at {sha} to {clone_path}")

        return repo
    except Exception as e:
        logging.error(f'Error during checkout and processing at {sha}, Error: {e}', exc_info=True)
        raise


# Run the TestSmellDetector tool for the given SHA
def run_tsdetect(project_url, pull_url, project_name, count, sha_type, sha, testfile_prefix, directory_repo,
                 save_result_path, tsdetect_path):
    try:
        test_files = write_test_files_to_csv(project_url, directory_repo)
        # check path if not exist create it
        testfile_path = f"{save_result_path}/csv/{testfile_prefix}_{project_name}_file_{count}_{sha}.csv"
        if not os.path.exists(f"{save_result_path}/csv"):
            os.makedirs(f"{save_result_path}/csv")

        test_files.to_csv(testfile_path, index=False, header=None)
        logging.info(
            f'Saved test files for {sha_type} SHA to CSV: {testfile_prefix}_{project_name}_file_{count}_{sha}.csv')

        pull_number = get_pr_number(pull_url)
        try:
            result = subprocess.run(['java', '-jar', tsdetect_path, testfile_path, sha_type, sha, pull_number], capture_output=True, text=True, check=True)
            logging.info(f'Ran TestSmellDetector for {sha_type} SHA.')

        except Exception as e:
            logging.error(f'''Error running TestSmellDetector for {sha_type} SHA: {sha},
            testfile_path:  {testfile_path}, 
            Error: {e}''', exc_info=True)
    except Exception as e:
        logging.error(f'''Error running TestSmellDetector for {sha_type} SHA: {sha},
                    Error: {e}''', exc_info=True)
        raise


# Clone the repository, checkout both SHAs, and run the detector
def process_checkout(count, project_name, project_url, pull_url, sha_opened, sha_closed, paths, clone_path):
    print("path to save the repo: ", clone_path)
    try:
        # Process open SHA
        logging.info(f'Processing open SHA for URL: {project_url} at pull request {pull_url}')
        clone_and_checkout(project_url, clone_path, sha_opened)
        save_result_path = f"{paths['project_repo_path']}/tmp_flink/"
        run_tsdetect(project_url, pull_url, project_name, count, "open", sha_opened, 'open', clone_path,
                     save_result_path,
                     paths['tsdetect_path'])

        # Remove cloned directory
        subprocess.run(['rm', '-rf', clone_path])

        # Process closed SHA
        logging.info(f'Processing closed SHA for URL: {project_url} at pull request {pull_url}')
        clone_and_checkout(project_url, clone_path, sha_closed)
        run_tsdetect(project_url, pull_url, project_name, count, "closed", sha_closed, 'closed', clone_path,
                     save_result_path,
                     paths['tsdetect_path'])

        # Clean up cloned directory again
        subprocess.run(['rm', '-rf', clone_path])
    except Exception as e:
        logging.error(f'Error during checkout and processing of SHAs for {project_name} at {pull_url}, Error: {e}',
                      exc_info=True)
        raise


def init_pool(shared_df):
    global project_sha_df
    project_sha_df = shared_df


def process_chunk(chunk, chunk_index, paths, project_name, project_url):
    global project_sha_df
    print("chunk_index", chunk_index)
    clone_path = f"{paths['project_repo_path']}/tmp_flink/clone_repo_{chunk_index}_{project_name}"
    for count in chunk:
        try:
            sha_opened = project_sha_df['open'][count]
            sha_closed = project_sha_df['closed'][count]
            pull_url = project_sha_df['url'][count]
            # Use the clone_path based on chunk_index to process each SHA
            process_checkout(count, project_name, project_url, pull_url, sha_opened, sha_closed, paths, clone_path)
        except Exception as e:
            logging.error(f'Error in process_chunk for count {count}: {e}', exc_info=True)


# Perform the auto-checkout for all SHAs in the project
def auto_checkout(project_name, project_url, paths):
    num_chunks = 6  # We want exactly 6 chunks
    total_items = len(paths['project_sha'])
    chunk_size = total_items // num_chunks

    # Divide the indices into 6 chunks, putting any excess in the last chunk
    chunks = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_chunks - 1)]
    chunks.append(list(range((num_chunks - 1) * chunk_size, total_items)))  # Last chunk gets the remainder

    with Pool(num_chunks, initializer=init_pool, initargs=(paths['project_sha'],)) as pool:
        pool.starmap(process_chunk, [(chunk, chunk_index, paths, project_name, project_url) for chunk_index, chunk in
                                     enumerate(chunks)])

    logging.info('Completed auto-checkout process.')


# # Process the checkout for each batch of data assigned to a worker
# def process_checkout_batch(worker_id, project_name, project_url, urls, sha_opened, sha_closed, paths):
#
#     try:
#         for count, url in enumerate(urls):
#             # Process individual SHAs within the chunk assigned to this worker
#             process_checkout(count, project_name, project_url, url, sha_opened.iloc[count], sha_closed.iloc[count],
#                              paths)
#
#         logging.info(f'Worker {worker_id} completed processing of its dataset chunk.')
#
#     except Exception as e:
#         logging.error(f'Error during batch processing by worker {worker_id}: {e}', exc_info=True)
#         raise


# Main execution
if __name__ == "__main__":
    paths = initialize_paths()
    auto_checkout('flink', 'https://github.com/apache/flink.git', paths)
    # auto_checkout('hive', 'https://github.com/apache/hive.git', paths)
    # try:
    #     paths = initialize_paths()
    #     auto_checkout('hive', 'https://github.com/apache/hive.git', paths)
    # except Exception as e:
    #     logging.error(f'Fatal error in main execution: {e}', exc_info=True)
    # try:
    #     paths = initialize_paths()
    #     auto_checkout('flink', 'https://github.com/apache/flink.git', paths)
    # except Exception as e:
    #     logging.error(f'Fatal error in main execution: {e}', exc_info=True)