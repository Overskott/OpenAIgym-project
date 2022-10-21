import json

config_file_path = 'config.json'
data = {}

with open(config_file_path, "r") as config_file:
    data = json.load(config_file)
