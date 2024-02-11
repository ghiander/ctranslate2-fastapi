import os
import json


class ModelLoadException(Exception):
    pass


def get_artifact_dir():
    return _get_artifact_dir_from_env()


def _get_artifact_dir_from_env():
    try:
        return os.environ["LLM_ARTIFACT_DIR"]
    except KeyError:
        raise ModelLoadException("Configure the variable 'LLM_ARTIFACT_DIR' "
                                 "to point to the model folder before "
                                 "importing 'languagemodels'")


def load_bootstrap_config():
    artifact_dir = _get_artifact_dir_from_env()
    with open(f"{artifact_dir}/bootstrap_config.json", "r") as f:
        return json.load(f)
