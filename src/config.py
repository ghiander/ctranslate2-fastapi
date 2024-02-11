import os
import yaml


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{current_dir}/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.BaseLoader)


def get_app_title():
    return config["title"]


def get_app_description():
    return config["description"]
