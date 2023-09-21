import os
import pandas as pd
import requests
from pathlib import Path
from getpass import getpass


class StudyInfoLoader:
    def __init__(self, config):
        self.api_url = config["api"]["endpoint_url"]
        self.api_login_url = config["api"]["login_url"]
        self.storage_path_root = config["storage"]["storage_path_root"]
        self.filename = config["storage"]["filename"]
        self.prompt_username = config["authentication"]["prompt_username"]
        self.prompt_password = config["authentication"]["prompt_password"]

    def authenticate(self):
        if self.prompt_username:
            self.username = input("Enter your username: ")
        if self.prompt_password:
            self.password = getpass("Enter your password: ")

        loginheaders = {
            "accept": "text/plain",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        logindata = {"password": self.password, "username": self.username}

        response = requests.post(
            self.api_login_url, headers=loginheaders, data=logindata
        )
        if response.status_code == 200:
            self.token = response.text
        else:
            raise Exception("Authentication Failed")

    def fetch_studies(self):
        headers = {
            "accept": "application/json",
            "Content-type": "application/json",
            "Authorization": "Bearer " + self.token,
        }
        studies = requests.get(self.api_url, headers=headers).json()
        self.df_studyinfo = pd.json_normalize(studies, max_level=0)

    def save_studies(self):
        os.makedirs(self.storage_path_root, exist_ok=True)
        fp_studies = Path(self.storage_path_root, self.filename)
        self.df_studyinfo.to_csv(fp_studies, index=False)

    def load_studies(self):
        fp_studies = Path(self.storage_path_root, self.filename)
        if os.path.isfile(fp_studies):
            self.df_studyinfo = pd.read_csv(fp_studies)
        else:
            self.authenticate()
            self.fetch_studies()
            self.save_studies()
