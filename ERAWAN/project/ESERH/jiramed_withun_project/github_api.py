import requests


class github_api:
    headers = {"Accept": "application/vnd.github.v3.star+json",
               'Authorization': 'Bearer ' + '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'}

    @staticmethod
    def extract_url(url):
        temp = []
        if 'github.com' in url:
            eurl = url.split('/')
            owner = eurl[-2]
            repo = eurl[-1]
            temp.append(owner)
            temp.append(repo)
        return temp

    @staticmethod
    def extract_url_pulls(url):
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
    def get_detail(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_issues(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/issues'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_pulls(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_pulls_in_detail(owner, repo, state, per_page, page, token):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}&page={page}'
        headers = {"Accept": "application/vnd.github.v3.star+json",
                   'Authorization': 'Bearer ' + token}
        response = requests.get(url, headers=headers)
        return response.json()

    @staticmethod
    def get_commits(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/commits'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_contributors(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/contributors'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_stargazers(owner, repo, page):
        url = f'https://api.github.com/repos/{owner}/{repo}/stargazers?per_page=30&page={page}'
        headers = {"Accept": "application/vnd.github.v3.star+json",
                   'Authorization': 'Bearer ' + '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'}
        response = requests.get(url, headers=headers)
        return response.json()

    @staticmethod
    def get_watchers(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/subscribers'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_forks(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/forks'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_releases(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/releases'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_branches(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/branches'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_tags(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/tags'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_languages(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/languages'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_readme(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/readme'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_license(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/license'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_events(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/events'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_collaborators(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/collaborators'
        response = requests.get(url)
        return response.json()

    @staticmethod
    def get_starred_at(owner, repo):
        url = f'https://api.github.com/repos/{owner}/{repo}/stargazers'
        headers = {"Accept": "application/vnd.github.v3.star+json",
                   'Authorization': 'Bearer ' + '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'}
        response = requests.get(url, headers=headers)
        return response.json()

    @staticmethod
    def get_files_change(owner, repo, pull_number, token):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files'
        headers = {'Authorization': 'Bearer ' + token}
        response = requests.get(url, headers=headers)
        return response.json()

    @staticmethod
    def check_rate_limit(token):
        url = "https://api.github.com/rate_limit"
        headers = {"Accept": "application/vnd.github.v3.star+json"
            , 'Authorization': 'Bearer ' + token}
        response = requests.get(url, headers=headers)
        return response.json()
