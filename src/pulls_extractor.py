import pickle
import re

import pandas
import pandas as pd
import os
from pathlib import Path

from get_github_api import github_api


class collect_pulls:
    def __init__(self, url):
        self.url = github_api.extract_url(url)
        self.token = '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'
        self.save_path = Path(os.path.abspath('../resources'))
        self.logger_path = Path(os.path.abspath('../resources/Logger'))
        self.result_dataframe = {'project_name': [], 'issue': [], 'file_target': [], 'before_change': [],
                                 'after_change': [], 'commit_id_before': [],
                                 'commit_id_after': []}

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

    def save_testing_pulls(self, path):
        df = pd.read_pickle(path)
        temp = []
        for url in df['url']:
            print(url)
            eurl = github_api.extract_url_pulls(url)
            testing_pull = github_api.get_files_change(eurl[0], eurl[1], eurl[2], self.token)
            df = pandas.json_normalize(testing_pull)
            df = df.dropna()
            if testing_pull:
                if df['filename'].str.contains(pat=r'\btest\b', regex=True).any():
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
    s = collect_pulls('https://github.com/apache/hive')
    # s = collect_pulls('https://github.com/apache/dubbo')
    # s = collect_pulls('https://github.com/apache/flink')
    # s = collect_pulls('https://github.com/json-iterator/java')
    # df = s.save_pull_request('open')
    # a = s.test_get_null()
    # df1 = s.check_testing_pulls(os.path.abspath('../resources/coder_vscode-coder_all_closed_requests.pkl'))
    df1 = s.save_testing_pulls(os.path.abspath('../resources/apache_hive_all_Merged_requests.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/json-iterator_java_all_closed_requests.pkl'))
    # result = s.save_testing_pulls(os.path.abspath('../resources/apache_flink_all_closed_requests.pkl'))
    check = github_api.check_rate_limit('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')
    check = pandas.json_normalize(check)
