import json
from pathlib import Path

current_path = Path.cwd()
config_file_name = 'config.json'
parent_dir = current_path.parents[0]
config_file_path = (parent_dir / config_file_name).resolve()

data = {}

with open(config_file_path, "r") as config_file:
    data = json.load(config_file)

# C:\Users\overs\Python\OpenAIgym-project\src\ca_environment.py

print("Config loaded")
