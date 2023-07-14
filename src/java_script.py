import pandas as pd
from pathlib import Path
from git import Repo
import os, re, subprocess, gc, time

hive_sha = pd.read_pickle(Path(os.path.abspath('../resources/hive_use_for_run_java.pkl')))
tsdetect = os.path.abspath('../resources/tsdetect/TestSmellDetector.jar')

hive_repo = Repo('/Users/Jumma/github_repo/hive/.git')
checkout = hive_repo.git.checkout
fetch = hive_repo.git.fetch()

# Specify the root directory of your Java project
directory_path = '/Users/Jumma/github_repo/hive'


# TODO Junit4 or more must in test directory
# def collect_testcase_filenames(directory):
#     testcase_filepath = []
#     test_pattern = re.compile(r'^Test.*\.java$', re.IGNORECASE)
#
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if test_pattern.match(file):
#                 filepath = os.path.join(root, file)
#                 testcase_filepath.append(filepath)
#
#     return testcase_filepath


def is_test_directory(directory):
    directory_parts = directory.lower().split("/")
    return "test" in directory_parts and "src" in directory_parts


def is_test_file(filename):
    # return filename.endswith('.java')
    return filename.endswith('.java') and ('test' in filename.lower() or 'testcase' in filename.lower())


def collect_test_files(root_dir):
    test_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory contains test files
        if is_test_directory(dirpath):
            for filename in filenames:
                if is_test_file(filename):
                    file_path = os.path.join(dirpath, filename)
                    test_files.append(file_path)
                    # test_files.append((filename, file_path))

    return test_files


# Print the collected test files
# for filepath in test_files:
#     # print(f"Filename: {filename}")
#     print(f"Filepath: {filepath}")
#     print()


def write_file_to_use_in_Jar(project):
    testcase_files = collect_test_files(directory_path)
    df = pd.DataFrame([(project, testcase_file) for testcase_file in testcase_files])
    return df


# TODO loop for checkout commit automatic
def auto_checkout():
    sha_opened = hive_sha['closed']
    sha_closed = hive_sha['opened']
    for count, url in enumerate(hive_sha['url'].head(1)):
        print(url)
        # checkout open version
        print(sha_opened[count])
        checkout(sha_opened[count])
        fetch
        print('checked closed')
        testfiles = write_file_to_use_in_Jar('hive')
        testfiles.to_csv(os.path.abspath(f"../resources/tsdetect/open_hive_file_{count}_{sha_opened[count]}.csv"),
                         index=False, header=None)
        time.sleep(5)
        # call tsdetect to find test smell
        subprocess.run(
                    ['java', '-jar', '/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar',
                     f"/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/open_hive_file_{count}_{sha_opened[count]}.csv"])

        print('Done open')
        # checkout open version
        print(sha_closed[count])
        checkout(sha_closed[count])
        fetch
        print('checked open')
        testfiles = write_file_to_use_in_Jar('hive')
        testfiles.to_csv(os.path.abspath(f"../resources/tsdetect/closed_hive_file_{count}_{sha_opened[count]}.csv"),
                         index=False, header=None)
        time.sleep(5)
        # call tsdetect to find test smell
        subprocess.run(
            ['java', '-jar', '/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/TestSmellDetector.jar',
             f"/Users/Jumma/github_repo/github_api_extractor/resources/tsdetect/closed_hive_file_{count}_{sha_opened[count]}.csv"])
        time.sleep(5)
        print('Done open')
    print('Done process')


# auto_checkout()
