import time
import requests

import pandas
import pandas as pd
import os
from pathlib import Path

line_token = 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'  # env line token


class CollectPulls:

    @staticmethod
    def extract_url(url):  # use for extract url to user and repo
        temp = []
        if 'github.com' in url:
            eurl = url.split('/')
            owner = eurl[-2]
            repo = eurl[-1]
            temp.append(owner)
            temp.append(repo)
        return temp

    @staticmethod
    def extract_url_pulls(url):  # use for extract url to user, repo and pull number from pull request url
        temp = []
        if 'github.com' in url:
            eurl = url.split('/')
            owner = eurl[-4]
            repo = eurl[-3]
            pull_number = eurl[-1]
            temp.append(owner)
            temp.append(repo)
            temp.append(pull_number)
        return temp

    @staticmethod
    def get_pulls_in_detail(owner, repo, state, per_page, page, token):  # use for get detail in pull request url
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}&page={page}'
        headers = {"Accept": "application/vnd.github.v3.star+json",
                   'Authorization': 'Bearer ' + token}
        response = requests.get(url, headers=headers)
        return response.json()

    @staticmethod
    def get_files_change(owner, repo, pull_number, token):  # use for get file change in each pull request
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files'
        headers = {'Authorization': 'Bearer ' + token}
        response = requests.get(url, headers=headers)
        return response.json()

    def __init__(self, url):  # init thing to use i class
        self.url = CollectPulls.extract_url(url)
        self.token = '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'
        # TODO: write new file path to store the file
        self.save_path = Path(os.path.abspath('../../../home/jiramed_withun/resource_pull_request'))
        self.logger_path = Path(os.path.abspath('../../../home/jiramed_withun/logger_pull/'))
        self.result_dataframe = {'project_name': [], 'issue': [], 'file_target': [], 'before_change': [],
                                 'after_change': [], 'commit_id_before': [],
                                 'commit_id_after': []}
        self.nums_loop = 3000
        self.pause_time = 3600
        self.line_url = 'https://notify-api.line.me/api/notify'
        self.headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + line_token}

    def save_pull_request(self, state):  # use for save all pull request detail
        eurl = self.url
        temp = []
        page = 1
        while True:
            pr = f'we are at {eurl[0], eurl[1], state, 100, page}'
            r = requests.post(self.line_url, headers=self.headers, data={'message': pr})
            print(r.text)
            time.sleep(1)
            pulls = CollectPulls.get_pulls_in_detail(eurl[0], eurl[1], state, 100, page, self.token)
            time.sleep(2)
            if not pulls:
                result = pd.concat(temp)
                result = result.reset_index(drop=True)
                result.to_pickle(self.save_path / f"{eurl[0]}_{eurl[1]}_all_{state}_pull_requests.pkl")
                break
            df = pandas.json_normalize(pulls)
            temp.append(df)
            page += 1
        return result

    def test_get_null(self):  # for testing get null json form api
        eurl = self.url
        print(eurl)
        pulls = CollectPulls.get_pulls_in_detail(eurl[0], eurl[1], 'closed', 100, 300, self.token)
        time.sleep(2)
        if not pulls:
            return True
        else:
            return False

    def save_testing_pulls(self, path):  # use for filter pull request data use only pull request which is edit on test
        df = pd.read_pickle(path)
        temp = []
        # Line noti
        row, col = df.shape
        rows = f'all closed pull request row equal {row}'
        r = requests.post(self.line_url, headers=self.headers, data={'message': rows})
        print(r.text)
        # end
        # Counter for the number of loops completed
        loop_count = 0
        # Time to pause in seconds (1 hour = 3600 seconds)
        pause_time = 1800
        # Number of loops you want to run
        num_loops = 1000
        try:
            for url in df['url']:
                loop_count += 1
                eurl = CollectPulls.extract_url_pulls(url)
                testing_pull = CollectPulls.get_files_change(eurl[0], eurl[1], eurl[2], self.token)
                time.sleep(2)
                df = pandas.json_normalize(testing_pull)
                df = df.dropna()
                if testing_pull:
                    if df['filename'].str.contains(pat=r'\btest\b', regex=True).any():
                        df['pulls_url'] = url
                        # Line noti
                        msg = f'this {url} has test class at iteration {loop_count}'
                        r = requests.post(self.line_url, headers=self.headers, data={'message': msg})
                        print(r.text)
                        #
                        temp.append(df)
                if len(temp) != 0 and len(temp) % 300 == 0:
                    temp_save = pd.concat(temp)
                    temp_save['pulls_url'].dropna(axis=0, inplace=True)
                    temp_save = temp_save[temp_save['filename'].str.contains(pat=r'\btest\b', regex=True) == True]
                    temp_save = temp_save[temp_save['filename'].str.contains(pat=r'\bjava\b', regex=True) == True]
                    temp_save.to_pickle(
                        self.logger_path / f"{eurl[0]}_{eurl[1]}_testing_pulls_at_{eurl[2]}_at{len(temp)}.pkl")
                    print('save')
                if loop_count % num_loops == 0:
                    wait = f'Pausing for 30 mins at {url}, iteration at {loop_count}'
                    r = requests.post(self.line_url, headers=self.headers, data={'message': wait})
                    print(r.text)
                    time.sleep(pause_time)
                    resume = f'Resuming at {url}, iteration at {loop_count}'
                    r = requests.post(self.line_url, headers=self.headers, data={'message': resume})
                    print(r.text)
        except Exception as e:
            error = f"An error occurred: {e}"
            r = requests.post(self.line_url, headers=self.headers, data={'message': error})
            print(r.text)

        result = pd.concat(temp)
        result['pulls_url'].dropna(axis=0, inplace=True)
        result.set_index('pulls_url')
        result.to_csv(self.save_path / f"{eurl[0]}_{eurl[1]}_testing_merged_pulls.csv")
        return result


if __name__ == '__main__':
    s = CollectPulls('https://github.com/apache/flink')
    # s.save_pull_request('closed')
    # time.sleep(10)
    # # a = s.test_get_null()
    # s.save_testing_pulls(
    #     os.path.abspath('../../../home/jiramed_withun/resource_pull_request/apache_flink_all_closed_pull_requests.pkl'))
    s.save_testing_pulls(
        os.path.abspath('../../../home/jiramed_withun/resource_pull_request/apache_flink_all_closed_pull_requests_start_at_17731.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl'))
