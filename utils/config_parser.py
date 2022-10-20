import json


def get_config_file(path: str = '../config.json'):
    with open(path, "r") as config_file:
        return json.load(config_file)
