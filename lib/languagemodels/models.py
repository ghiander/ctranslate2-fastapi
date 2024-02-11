import re
import ctranslate2

from tokenizers import Tokenizer
from languagemodels.config import config, models


class ModelException(Exception):
    pass


def get_model_info():
    """Gets info about the current model in use.
    """
    model_name = config["name"]
    m = [m for m in models if m["name"] == model_name][0]
    param_bits = int(re.search(r"\d+", m["quantization"]).group(0))
    m["size_gb"] = m["params"] * param_bits / 8 / 1e9
    return m


def get_artifacts(artifact_dir, model_info):
    """Loads tokenizer and model from an artifact path.
    """
    compute_type = model_info["quantization"]
    model = ctranslate2.Translator(artifact_dir, "cpu",
                                   compute_type=compute_type)
    tokenizer = Tokenizer.from_file(f"{artifact_dir}/tokenizer.json")
    cached_artifacts = (tokenizer, model)
    return cached_artifacts
