import pickle
import re

import pandas
import pandas as pd
import os
import time
from pathlib import Path

import requests

from get_github_api import github_api
line_url = 'https://notify-api.line.me/api/notify'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer ' + 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'}

class collect_pulls:
    def __init__(self, url):
        self.url = github_api.extract_url(url)
        self.token = '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'
        # TODO: write new file path to store the file
        self.save_path = Path(os.path.abspath('../resources'))
        self.logger_path = Path(os.path.abspath('../resources/Logger'))
        self.result_dataframe = {'project_name': [], 'issue': [], 'file_target': [], 'before_change': [],
                                 'after_change': [], 'commit_id_before': [],
                                 'commit_id_after': []}
        self.request_count = 0

    def save_pull_request(self, state):
        eurl = self.url
        temp = []
        page = 1
        while True:
            print(eurl[0], eurl[1], state, 100, page)
            pulls = github_api.get_pulls_in_detail(eurl[0], eurl[1], state, 100, page, self.token)
            if not pulls:
                result = pd.concat(temp)
                result = result.reset_index(drop=True)
                result.to_pickle(self.save_path / f"{eurl[0]}_{eurl[1]}_all_{state}_requests.pkl")
                break
            df = pandas.json_normalize(pulls)
            temp.append(df)
            page += 1
        return result

    def test_get_null(self):
        eurl = self.url
        pulls = github_api.get_pulls_in_detail(eurl[0], eurl[1], 'open', 100, 100)
        return pulls

    def check_and_wait(self):
        """Check the rate limit and wait if necessary."""
        remaining, reset_time = github_api.check_rate_limit(self.token)
        if remaining < 500:  # If close to the rate limit, pause
            current_time = time.time()
            wait_time = max(reset_time - current_time, 0) + 60  # Add 1 minute buffer
            print(f"Approaching rate limit. Sleeping for {wait_time/60:.2f} minutes.")
            # line noti
            message = f"Approaching rate limit. Sleeping for {wait_time/60:.2f} minutes."
            payload = {'message': message}
            r = requests.post(line_url, headers=headers, params=payload)
            print(r.text)
            time.sleep(wait_time)
            self.check_and_wait()
        else:
            print(f"Rate limit OK. Remaining requests: {remaining}")
            # line noti
            message = f"Rate limit OK. Remaining requests: {remaining}"
            payload = {'message': message}
            r = requests.post(line_url, headers=headers, params=payload)
            print(r.text)
            self.request_count = 0


    def save_testing_pulls(self, df):
        temp = []
        for url in df['url']:
            print(url)
            eurl = github_api.extract_url_pulls(url)
            testing_pull = github_api.get_files_change(eurl[0], eurl[1], eurl[2], self.token)
            self.request_count += 1
            df = pandas.json_normalize(testing_pull)
            df = df.dropna()
            # Check rate limit only every 4000 requests
            if self.request_count >= 4500:
                self.check_and_wait()
            # Reset the request count after checking
            if testing_pull and df['filename'].str.contains(pat=r'\btest\b', regex=True).any():
                df['pulls_url'] = url
                print('it has')
                temp.append(df)
            if len(temp) != 0 and len(temp) % 30 == 0:
                temp_save = pd.concat(temp)
                temp_save['pulls_url'].dropna(axis=0, inplace=True)
                temp_save = temp_save[temp_save['filename'].str.contains(pat=r'\btest\b', regex=True) == True]
                temp_save = temp_save[temp_save['filename'].str.contains(pat=r'\bjava\b', regex=True) == True]
                temp_save.to_pickle(
                    self.logger_path / f"{eurl[0]}_{eurl[1]}_testing_pulls_at_{eurl[2]}_at{len(temp)}.pkl")
                print('save')
        result = pd.concat(temp)
        result['pulls_url'].dropna(axis=0, inplace=True)
        result.to_pickle(self.save_path / f"{eurl[0]}_{eurl[1]}_testing_merged_pulls.pkl")
        result = result.set_index('pulls_url')
        return result


if __name__ == '__main__':
    # s = creat_pulls('https://github.com/coder/vscode-coder')
    s = collect_pulls('https://github.com/apache/flink')
    # s = collect_pulls('https://github.com/apache/dubbo')
    # s = collect_pulls('https://github.com/apache/flink')
    # s = collect_pulls('https://github.com/json-iterator/java')
    # df = s.save_pull_request('open')
    # a = s.test_get_null()
    # df1 = s.check_testing_pulls(os.path.abspath('../resources/coder_vscode-coder_all_closed_requests.pkl'))
    flink_pull_request = "../resources/pull_request_projects/flink_pulls.pkl"
    flink_df = pd.read_pickle(flink_pull_request)
    flink_df = github_api.remove_invalid_rows(flink_df, ['url', 'title', 'body', 'base.sha', 'merge_commit_sha'])
    flink_df = s.save_testing_pulls(flink_df)
    cassandra_pull_request = "../resources/pull_request_projects/cassandra_pulls.pkl"
    cassandra_df = pd.read_pickle(cassandra_pull_request)

    # df1 = s.save_testing_pulls(os.path.abspath('../resources/apache_hive_all_Merged_requests.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl'))
    check = github_api.check_rate_limit('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')


