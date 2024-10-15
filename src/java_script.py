import pandas as pd
from pathlib import Path
from git import Repo
import os, re, subprocess, time, platform
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(filename='process_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the operating system
system = platform.system()

if system == 'Darwin':  # macOS
    project_repo_path = '/path/to/your/project/repo/on/mac'
    tsdetect_path = '/path/to/your/tsdetect/TestSmellDetector.jar/on/mac'
    save_result_path = '/path/to/your/directory/on/mac'
    project_sha = pd.read_pickle(
        Path(os.path.abspath('/path/to/your/directory/on/mac/hive_use_for_run_pre_process.pkl')))
    project_sha.reset_index(drop=True, inplace=True)
elif system == 'Linux':
    # Change path here
    project_repo_path = '/home/pee/repo'
    tsdetect_path = '/home/pee/repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar'
    save_result_path = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive'
    project_sha = pd.read_pickle(f'{save_result_path}/hive_use_for_run_pre_process.pkl')
    project_sha.reset_index(drop=True, inplace=True)
else:
    raise EnvironmentError('Unsupported operating system')

logging.info('Initialized project repository and paths.')


def is_test_directory(directory):
    directory_parts = directory.lower().split("/")
    return "test" in directory_parts and "src" in directory_parts


def is_test_file(filename):
    logging.info(f'Checking if {filename} is a test file.')
    return filename.endswith('.java') and ('test' in filename.lower() or 'testcase' in filename.lower())


def collect_test_files(root_dir):
    test_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if is_test_directory(dirpath):
            for filename in filenames:
                if is_test_file(filename):
                    file_path = os.path.join(dirpath, filename)
                    test_files.append(file_path)
                    logging.info(f'Collected test file: {file_path}')
    return test_files


def write_file_to_use_in_Jar(project):
    testcase_files = collect_test_files(save_result_path)
    df = pd.DataFrame([(project, testcase_file) for testcase_file in testcase_files])
    logging.info(f'Created DataFrame for project {project} with test case files.')
    return df


def clone_and_checkout(repo_url, clone_path, sha):
    if os.path.exists(clone_path):
        logging.info(f'Removing existing directory: {clone_path}')
        subprocess.run(['rm', '-rf', clone_path])
    logging.info(f'Cloning repository from {repo_url} to {clone_path}')
    print(repo_url)
    repo = Repo.clone_from(repo_url, clone_path, no_checkout=True)
    repo.git.checkout(sha)
    logging.info(f'Checked out SHA: {sha}')
    return repo


def run_tsdetect(project_name: str, count, sha_type, sha, testfile_prefix):
    testfiles = write_file_to_use_in_Jar(project_name)
    testfile_path = f"{save_result_path}/csv/{testfile_prefix}_{project_name}_file_{count}_{sha}.csv"
    testfiles.to_csv(testfile_path, index=False, header=None)
    logging.info(f'Saved test files for {sha_type} SHA to CSV: {testfile_prefix}_{project_name}_file_{count}_{sha}.csv')
    subprocess.run(['java', '-jar', tsdetect_path, testfile_path])
    logging.info(f'Ran TestSmellDetector for {sha_type} SHA.')


def auto_process_checkout(count, project_name, project_url, url, sha_opened, sha_closed):
    print(project_name)
    clone_path = f'{project_repo_path}/tmp/clone_repo_{count}_{project_name}'

    # Clone and checkout open SHA
    logging.info(f'Processing open SHA for URL: {project_url} at pull request {url}')
    repo = clone_and_checkout(project_url, clone_path, sha_opened)
    run_tsdetect(project_name, count, "open", sha_opened, 'open')

    # Remove cloned directory
    logging.info(f'Removing directory: {clone_path}')
    subprocess.run(['rm', '-rf', clone_path])

    # Clone and checkout closed SHA
    logging.info(f'Processing closed SHA for URL: {project_url} at pull request {url}')
    repo = clone_and_checkout(project_url, clone_path, sha_closed)
    run_tsdetect(project_name, count, "closed", sha_closed, 'closed')

    # Remove cloned directory again
    logging.info(f'Removing directory: {clone_path}')
    subprocess.run(['rm', '-rf', clone_path])


def auto_checkout(project_name: str, project_url: str):
    sha_opened = project_sha['open']
    sha_closed = project_sha['closed']
    urls = project_sha['url']

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(auto_process_checkout, count, project_name, project_url, url, sha_opened[count],
                            sha_closed[count]) for
            count, url in enumerate(urls)]
        for future in futures:
            future.result()

    logging.info('Completed auto checkout process.')


auto_checkout('hive', 'https://github.com/apache/hive.git')
