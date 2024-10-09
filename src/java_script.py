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
    directory_path = '/path/to/your/directory/on/mac'
elif system == 'Linux':
    project_repo_path = '/home/pee/IdeaProjects/hive'
    tsdetect_path = '/home/pee/repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar'
    directory_path = '/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_hive'
else:
    raise EnvironmentError('Unsupported operating system')

project_sha = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_pre_process.pkl')))
project_sha.reset_index(drop=True, inplace=True)

project_repo = Repo(project_repo_path)
checkout = project_repo.git.checkout
fetch = project_repo.git.fetch

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
    testcase_files = collect_test_files(directory_path)
    df = pd.DataFrame([(project, testcase_file) for testcase_file in testcase_files])
    logging.info(f'Created DataFrame for project {project} with test case files.')
    return df


# TODO: Add new sequence about cloning and delete after write run tsdetect
repo = Repo.clone_from('https://github.com/ozone-his/ozone.git', '/home/pee/IdeaProjects/ozone', no_checkout=True)
repo.git.checkout('7fa3cf1bef5af455e802467eac89cc0b73e097dc')


def process_sha(count, url, sha_opened, sha_closed):
    logging.info(f'Processing SHA for URL: {url}')
    logging.info(f'Checking out SHA opened: {sha_opened}')
    fetch()
    logging.info(f'fetching')
    checkout(sha_opened)
    logging.info('Checked out closed SHA.')
    testfiles = write_file_to_use_in_Jar('hive')
    testfiles.to_csv(os.path.abspath(f"../resources/tsdetect/open_hive_file_{count}_{sha_opened}.csv"), index=False,
                     header=None)
    logging.info(f'Saved test files for open SHA to CSV: open_hive_file_{count}_{sha_opened}.csv')
    time.sleep(5)
    subprocess.run(['java', '-jar', tsdetect_path,
                    f"/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/open_hive_file_{count}_{sha_opened}.csv"])
    logging.info('Ran TestSmellDetector for open SHA.')

    logging.info(f'Checking out SHA closed: {sha_closed}')
    fetch()
    checkout(sha_closed)
    logging.info('Checked out open SHA.')
    testfiles = write_file_to_use_in_Jar('hive')
    testfiles.to_csv(os.path.abspath(f"../resources/tsdetect/closed_hive_file_{count}_{sha_closed}.csv"), index=False,
                     header=None)
    logging.info(f'Saved test files for closed SHA to CSV: closed_hive_file_{count}_{sha_closed}.csv')
    time.sleep(5)
    subprocess.run(['java', '-jar', tsdetect_path,
                    f"/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/closed_hive_file_{count}_{sha_closed}.csv"])
    logging.info('Ran TestSmellDetector for closed SHA.')
    time.sleep(5)


def auto_checkout():
    sha_opened = project_sha['closed']
    sha_closed = project_sha['open']
    urls = project_sha['url']

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_sha, count, url, sha_opened[count], sha_closed[count]) for count, url in
                   enumerate(urls)]
        for future in futures:
            future.result()

    logging.info('Completed auto checkout process.')

# auto_checkout()
#
