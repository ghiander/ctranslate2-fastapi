import os
import yaml
import logging


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{current_dir}/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.BaseLoader)


def get_app_title():
    return config["title"]


def get_app_description():
    return config["description"]

def get_logger(name):
    """
    Creates a logger where level is
    determined by the env LOGGING_LEVEL.

    """
    log = logging.getLogger(name)
    if os.environ.get(
            "LOGGING_LEVEL", "INFO").upper() == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
        log.debug("Loaded logging with DEBUG level")
    else:
        logging.basicConfig(level=logging.INFO)
    return log
