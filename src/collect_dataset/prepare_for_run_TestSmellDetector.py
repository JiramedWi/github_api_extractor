import pandas as pd
from pathlib import Path
import os


class ProjectDataProcessorForTSdetector:
    def __init__(self, test_file_path, pulls_file_path, output_dir):
        self.project_test_pulls = pd.read_pickle(Path(test_file_path).resolve())
        self.all_pulls = pd.read_pickle(Path(pulls_file_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save(self, output_filename, columns, rename_map):
        url = self.project_test_pulls['pulls_url'].drop_duplicates()
        project_filtered = self.all_pulls[self.all_pulls['url'].isin(url)].dropna(subset=['merged_at'])
        project_filtered = project_filtered.filter(columns).rename(columns=rename_map)
        project_filtered = project_filtered.reset_index(drop=True)
        output_path = self.output_dir / output_filename
        project_filtered.to_pickle(output_path)
        print(f"Data saved to {output_path}")
        return project_filtered

    def extract_sha_and_save(self, output_filename):
        return self.extract_and_save(
            output_filename,
            ['url', 'base.sha', 'merge_commit_sha'],
            {'base.sha': 'open', 'merge_commit_sha': 'closed'}
        )

    def extract_issue_description_and_save(self, output_filename):
        self.all_pulls['title_n_body'] = self.all_pulls['title'] + " " + self.all_pulls['body']
        return self.extract_and_save(
            output_filename,
            ['url', 'body', 'base.sha', 'merge_commit_sha', 'title_n_body'],
            {'base.sha': 'open', 'merge_commit_sha': 'closed'}
        )


def process_project(test_file, pulls_file, output_dir, sha_filename, desc_filename):
    processor = ProjectDataProcessorForTSdetector(test_file, pulls_file, output_dir)
    processor.extract_sha_and_save(sha_filename)
    processor.extract_issue_description_and_save(desc_filename)


# Process multiple projects
# projects = [
#     ('flink_testing_pulls.pkl', 'flink_pulls.pkl', 'flink_use_for_run_java.pkl', 'flink_use_for_run_pre_process.pkl'),
#     ('cassandra_testing_pulls.pkl', 'cassandra_pulls.pkl', 'cassandra_use_for_run_java.pkl', 'cassandra_use_for_run_pre_process.pkl')
# ]
#
# base_dir = '/home/pee/repo/github_api_extractor/resources/pull_request_projects'
# for test_file, pulls_file, sha_filename, desc_filename in projects:
#     process_project(
#         os.path.join(base_dir, test_file),
#         os.path.join(base_dir, pulls_file),
#         base_dir,
#         sha_filename,
#         desc_filename
#     )

# read all file to check
flink_use_for_run_java = pd.read_pickle(Path('/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_use_for_run_java.pkl').resolve())
flink_use_for_run_pre_process = pd.read_pickle(Path('/home/pee/repo/github_api_extractor/resources/pull_request_projects/flink_use_for_run_pre_process.pkl').resolve())

index_list = flink_use_for_run_java.query("url == 'https://api.github.com/repos/apache/flink/pulls/12640'").index.tolist()
print(index_list)