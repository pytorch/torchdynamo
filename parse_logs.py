import argparse
import csv
from genericpath import isfile
import requests
import json
import os
import re
import configparser


GITHUB_API = "https://api.github.com"
GIST_URL = "https://gist.github.com"
def upload_gist(gh_conf, file_data):
    gh_user, gh_apikey = gh_conf["user"], gh_conf["apikey"]
    url = GITHUB_API+"/gists"
    headers = {'Authorization': 'token %s' % gh_apikey}
    params = {'scope': 'gist'}
    payload = {
        "description": "Torchbench logs auto upload",
        "public": True,
        "files": { fname: {"content": file_data[fname]} for fname in file_data },
    }
    res = requests.post(url, headers=headers, params=params, data=json.dumps(payload))
    j = json.loads(res.text)
    url = j['url']
    id = j['id']
    return {
        fname: f"{GIST_URL}/{gh_user}/{id}#file-{fname.lower().replace('.',  '-')}" for fname in file_data
    }


def parse_log(log_lines):
    log_str = "".join(log_lines)
    code = re.findall(r"PASS|FAIL", log_str)
    errors = re.findall(r".*Error:.*", log_str)
    if len(code) < 1:
        code = "ERROR"
    else:
        assert len(code) == 1
        code = code[0]
    return {
        'code': code,
        'errors': errors,
    }


def get_github_creds():
    gh_user = input('Github username: ')
    gh_apikey = input('Github API key (with user gist create permission): ')
    return gh_user, gh_apikey


def create_config(conf_file):
    config = configparser.ConfigParser()
    gh_user, gh_apikey = get_github_creds()
    config["github"] = {}
    config["github"]["user"] = gh_user
    config["github"]["apikey"] = gh_apikey
    with open(conf_file, 'w') as f:
        config.write(f)


def get_config(conf_file):
    if not os.path.isfile(conf_file):
        create_config(conf_file)

    config = configparser.ConfigParser()
    config.read(conf_file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("csvfile")
    args = parser.parse_args()

    conf = get_config(f"{os.path.expanduser('~')}/.tblogparse.conf")
    gh_conf =  conf["github"]

    file_data = {}
    parsed = {}
    for fname in os.listdir(args.logdir):
        with open(os.path.join(args.logdir, fname), 'r') as f:
            file_data[fname] = f.read()
            parsed[fname] = parse_log(file_data[fname])

    file_urls = upload_gist(gh_conf, file_data)

    with open(args.csvfile, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Model", "Code", "Errors", "Log URL"])
        csv_writer.writerows([
            [
                fname,
                parsed[fname]["code"],
                parsed[fname]["errors"],
                file_urls[fname],
            ] for fname in file_data
        ])