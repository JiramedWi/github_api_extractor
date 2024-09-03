import pandas as pd
from pathlib import Path
import os


class ProjectDataProcessorForTSdetector:
    def __init__(self, test_file_path, pulls_file_path, output_dir):
        self.project_test_pulls = pd.read_pickle(Path(os.path.abspath(test_file_path)))
        self.all_pulls = pd.read_pickle(Path(os.path.abspath(pulls_file_path)))
        self.output_dir = Path(os.path.abspath(output_dir))
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_sha_and_save(self, output_filename):
        # Extract the SHA from pull request
        url = self.project_test_pulls['pulls_url'].drop_duplicates()
        project_issue_filtered = self.all_pulls[self.all_pulls['url'].isin(url)]
        columns = ['url', 'base.sha', 'merge_commit_sha']
        project_issue_filtered = project_issue_filtered.filter(columns)
        project_issue_filtered = project_issue_filtered.rename(columns={'base.sha': 'open', 'merge_commit_sha': 'closed'})
        project_issue_filtered = project_issue_filtered.reset_index(drop=True)
        output_path = self.output_dir / output_filename
        project_issue_filtered.to_pickle(output_path)
        print(f"SHA data saved to {output_path}")
        return project_issue_filtered

    def extract_issue_description_and_save(self, output_filename):
        # Extract issue description
        url = self.project_test_pulls['pulls_url'].drop_duplicates()
        project_issue_description = self.all_pulls[self.all_pulls['url'].isin(url)]
        project_issue_description['title_n_body'] = project_issue_description['title'] + " " + project_issue_description['body']
        columns = ['url', 'body', 'base.sha', 'merge_commit_sha', 'title_n_body']
        project_issue_filtered = project_issue_description.filter(columns)
        project_issue_filtered = project_issue_filtered.rename(
            columns={'body': 'body', 'base.sha': 'open', 'merge_commit_sha': 'closed'})
        project_issue_filtered = project_issue_filtered.reset_index(drop=True)
        output_path = self.output_dir / output_filename
        project_issue_filtered.to_pickle(output_path)
        print(f"Issue description data saved to {output_path}")
        return project_issue_filtered


# Usage example:
# for flink project
processor = ProjectDataProcessorForTSdetector('../resources/pull_request_projects/flink_testing_pulls.pkl',
                                              '../resources/pull_request_projects/flink_pulls.pkl',
                                              '../resources/pull_request_projects/')
save_java_flink = processor.extract_sha_and_save('flink_use_for_run_java.pkl')
pre_process_flink = processor.extract_issue_description_and_save('flink_use_for_run_pre_process.pkl')

# for cassandra project
processor = ProjectDataProcessorForTSdetector('../resources/pull_request_projects/cassandra_testing_pulls.pkl',
                                              '../resources/pull_request_projects/cassandra_pulls.pkl',
                                              '../resources/pull_request_projects/')

save_java_cassandra = processor.extract_sha_and_save('cassandra_use_for_run_java.pkl')
pre_process_cassandra = processor.extract_issue_description_and_save('cassandra_use_for_run_pre_process.pkl')

