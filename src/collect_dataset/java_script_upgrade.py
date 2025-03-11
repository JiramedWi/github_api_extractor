import re
import pandas as pd
from pathlib import Path
from logging.handlers import RotatingFileHandler
from multiprocessing import Pool, cpu_count, Manager, Lock
from git import Repo
import os, subprocess, platform
import threading
import logging
import shutil
import time
import json

# Configure logging
# Configure logging
log_handler = RotatingFileHandler(
    '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/log/process_log.log',
    maxBytes=10 * 1024 * 1024, backupCount=5
)

# Custom log formatter to include process ID, thread ID, and core ID (if applicable)
class CoreAwareFormatter(logging.Formatter):
    def format(self, record):
        record.process_id = os.getpid()
        record.thread_id = threading.get_ident()
        try:
            # Get CPU core affinity (Linux only)
            record.core_id = list(os.sched_getaffinity(0))[0] if hasattr(os, 'sched_getaffinity') else "N/A"
        except Exception:
            record.core_id = "N/A"
        return super().format(record)

log_formatter = CoreAwareFormatter('%(asctime)s - %(levelname)s - PID: %(process_id)d - Thread: %(thread_id)d - Core: %(core_id)s - %(message)s')
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


# Get the pull request number from the URL
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
        elif system == 'Linux':
            paths['project_repo_path'] = '/home/pee/repo'
            paths['tsdetect_path'] = '/home/pee/repo/tmp_flink/tsdetect/TestSmellDetector.jar'
            paths['save_result_path'] = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink'
            project_sha_path = '/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_use_for_run_java.pkl'
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
        print(f"Start checkout repo {repo_url} with sha at {sha} to {clone_path}")

        return repo
    except Exception as e:
        logging.error(f'Error during checkout and processing at {sha}, Error: {e}', exc_info=True)
        raise


# Run the TestSmellDetector tool for the given SHA
def run_tsdetect(project_url, pull_url, project_name, count, sha_type, sha, testfile_prefix, directory_repo,
                 save_result_path, tsdetect_path, jar_lock):
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
            # Acquire lock before running TestSmellDetector to prevent concurrent jar executions
            with jar_lock:
                logging.info(f'Worker acquired lock, running TestSmellDetector for {sha_type} SHA: {sha}')
                subprocess.run(['java', '-jar', tsdetect_path, testfile_path, sha_type, sha, pull_number],
                               check=True, timeout=1800, capture_output=True)  # 30-minute timeout
                logging.info(f'Successfully ran TestSmellDetector for {sha_type} SHA: {sha}')

        except subprocess.TimeoutExpired:
            logging.error(f'TestSmellDetector timed out after 30 minutes for {sha_type} SHA: {sha}')
        except subprocess.CalledProcessError as e:
            logging.error(f'''TestSmellDetector process returned non-zero exit code for {sha_type} SHA: {sha},
            testfile_path: {testfile_path}, 
            Error: {e.stderr}''', exc_info=True)
        except Exception as e:
            logging.error(f'''Error running TestSmellDetector for {sha_type} SHA: {sha},
            testfile_path: {testfile_path}, 
            Error: {e.stderr}''', exc_info=True)
    except Exception as e:
        logging.error(f'''Error running TestSmellDetector for {sha_type} SHA: {sha},
                    Error: {e}''', exc_info=True)
        # We're not re-raising to allow processing to continue to next SHA pair


# Clone the repository, checkout both SHAs, and run the detector
def process_checkout(count, project_name, project_url, pull_url, sha_opened, sha_closed, paths, clone_path, jar_lock,
                     completed_items):
    print(f"Processing item {count} - path to save the repo: {clone_path}")
    try:
        # Check if this item has already been processed
        if count in completed_items:
            logging.info(f'Skipping already processed item {count}')
            return

        # Process open SHA
        logging.info(f'Processing open SHA for URL: {project_url} at pull request {pull_url}')
        clone_and_checkout(project_url, clone_path, sha_opened)
        save_result_path = f"{paths['project_repo_path']}/tmp_flink"
        run_tsdetect(project_url, pull_url, project_name, count, "open", sha_opened, 'open', clone_path,
                     save_result_path, paths['tsdetect_path'], jar_lock)

        # Remove cloned directory if it exists
        if os.path.exists(clone_path):
            subprocess.run(['rm', '-rf', clone_path])

        # Process closed SHA
        logging.info(f'Processing closed SHA for URL: {project_url} at pull request {pull_url}')
        clone_and_checkout(project_url, clone_path, sha_closed)
        run_tsdetect(project_url, pull_url, project_name, count, "closed", sha_closed, 'closed', clone_path,
                     save_result_path, paths['tsdetect_path'], jar_lock)

        # Clean up cloned directory again if it exists
        if os.path.exists(clone_path):
            subprocess.run(['rm', '-rf', clone_path])

        # Mark this item as completed
        completed_items.append(count)
        save_checkpoint(paths['save_result_path'], project_name, completed_items)
        logging.info(f'Successfully completed processing item {count} and updated checkpoint')

    except Exception as e:
        logging.error(f'Error during checkout and processing of SHAs for {project_name} at {pull_url}, Error: {e}',
                      exc_info=True)
        # We continue to the next item without re-raising the exception


# Save checkpoint of completed items
def save_checkpoint(save_result_path, project_name, completed_items):
    checkpoint_file = f"{save_result_path}/checkpoint_{project_name}.json"
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed_items': sorted(completed_items)}, f)
    except Exception as e:
        logging.error(f'Error saving checkpoint file: {e}', exc_info=True)


# Load checkpoint of completed items
def load_checkpoint(save_result_path, project_name):
    checkpoint_file = f"{save_result_path}/checkpoint_{project_name}.json"
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('completed_items', [])
    except Exception as e:
        logging.error(f'Error loading checkpoint file: {e}', exc_info=True)
    return []


def init_worker(df, lock, items):
    global project_sha_df, jar_lock, completed_items
    project_sha_df = df
    jar_lock = lock
    completed_items = items


def process_chunk(chunk, chunk_index, paths, project_name, project_url):
    global project_sha_df, jar_lock, completed_items

    # Create a unique directory for this worker to avoid conflicts
    clone_path = f"{paths['project_repo_path']}/tmp_flink/clone_repo_{chunk_index}_{project_name}"

    for count in chunk:
        try:
            # Check if we have enough disk space before processing
            check_disk_usage(paths['project_repo_path'], 5)  # Ensure at least 5GB free

            sha_opened = project_sha_df['open'][count]
            sha_closed = project_sha_df['closed'][count]
            pull_url = project_sha_df['url'][count]

            # Process this SHA pair
            process_checkout(count, project_name, project_url, pull_url, sha_opened, sha_closed,
                             paths, clone_path, jar_lock, completed_items)

            # Small delay to avoid potential race conditions
            time.sleep(1)

        except Exception as e:
            logging.error(f'Error in process_chunk for count {count}: {e}', exc_info=True)
            # Continue to the next item without stopping


# Perform the auto-checkout for all SHAs in the project
def auto_checkout_improved(project_name, project_url, paths):
    # Load previously completed items
    completed_items = load_checkpoint(paths['save_result_path'], project_name)
    logging.info(f'Loaded {len(completed_items)} previously completed items from checkpoint')

    # Determine how many CPUs to use - use fewer than available to avoid overloading
    num_workers = max(2, min(cpu_count() - 1, 6))  # Between 2 and 6, but never more than CPU count - 1

    # Create a manager for sharing the lock and completed items list
    manager = Manager()
    jar_lock = manager.Lock()
    shared_completed_items = manager.list(completed_items)

    total_items = len(paths['project_sha'])

    # Filter out already completed items for processing
    all_indices = list(range(total_items))
    to_process = [idx for idx in all_indices if idx not in completed_items]

    logging.info(f'Will process {len(to_process)} out of {total_items} items with {num_workers} workers')

    # If everything is already completed, we're done
    if not to_process:
        logging.info('All items have already been processed. Nothing to do.')
        return

    # Divide the remaining indices into chunks for the workers
    chunk_size = len(to_process) // num_workers + (1 if len(to_process) % num_workers > 0 else 0)
    chunks = [to_process[i:i + chunk_size] for i in range(0, len(to_process), chunk_size)]

    with Pool(num_workers, initializer=init_worker,
              initargs=(paths['project_sha'], jar_lock, shared_completed_items)) as pool:
        pool.starmap(process_chunk, [(chunk, chunk_index, paths, project_name, project_url)
                                     for chunk_index, chunk in enumerate(chunks)])

    logging.info('Completed auto-checkout process.')


# Main execution
if __name__ == "__main__":
    try:
        # Ensure we have enough disk space before starting
        check_disk_usage('/home/pee/repo', 10)  # Ensure at least 10GB free

        # Initialize paths
        paths = initialize_paths()

        # Run the improved parallel process
        auto_checkout_improved('flink', 'https://github.com/apache/flink.git', paths)

        logging.info("Processing completed successfully")
    except Exception as e:
        logging.error(f'Fatal error in main execution: {e}', exc_info=True)