"""Global model and inference configuration

This module manages the global configuration object shared between other
modules in the package. It implements a dictionary with data validation
on the keys and values.

Note that this module provides access to many implementation details
that are not expected to be used by average users. Specific models that
have never been the default for the package may be removed at any time.
"""

import re
import os

from collections import namedtuple
from languagemodels.bootstrap import load_bootstrap_config

ConfigItem = namedtuple("ConfigItem", "initfn default")


class ModelFilterException(Exception):
    pass


class Config(dict):
    """
    Store configuration information for the package.

    This is a dictionary that provides data basic data validation.

    Only appropriate keys and values are allowed to be set.

    >>> c = Config({'max_ram': '4gb'})
    >>> c
    {...'max_ram': 4.0...}

    >>> c = Config({'instruct_model': 'flan-t5-small'})
    >>> c
    {...'instruct_model': 'flan-t5-small'...}

    >>> c = Config({'model_license': 'apache|mit|bsd'})
    >>> c
    {...'model_license': re.compile('apache|mit|bsd')...}

    >>> c = Config({'instruct_model': 'flan-t5-bad'})
    Traceback (most recent call last):
      ...
    KeyError: 'flan-t5-bad'

    >>> c = Config({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    >>> c = Config()
    >>> c.update({'bad_value': 1})
    Traceback (most recent call last):
      ...
    KeyError: 'bad_value'

    """
    def __init__(self, config={}):
        # Defaults are loaded first
        for key in Config.schema:
            self[key] = self.schema[key].default

        # Environment variables override defaults
        for key in Config.schema:
            value = os.environ.get(f"LANGUAGEMODELS_{key.upper()}")
            if value:
                self[key] = value

        # Any values passed in the config dict override environment vars
        for key in config.keys():
            self[key] = config[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, Config.schema[key].initfn(value))

    def update(self, other):
        for key in other:
            self[key] = other[key]

    @staticmethod
    def validate_model(model_name):
        return model_names[model_name]["name"]

    @staticmethod
    def validate_device(device):
        assert device in ["auto", "cpu"]
        return device

    @staticmethod
    def convert_to_gb(space):
        """Convert max RAM string to int

        Output will be in gigabytes

        If not specified, input is assumed to be in gigabytes

        >>> Config.convert_to_gb("512")
        512.0

        >>> Config.convert_to_gb(".5")
        0.5

        >>> Config.convert_to_gb("4G")
        4.0

        >>> Config.convert_to_gb("256mb")
        0.25

        >>> Config.convert_to_gb("256M")
        0.25

        >>> Config.convert_to_gb("small")
        0.2

        >>> Config.convert_to_gb("base")
        0.48

        >>> Config.convert_to_gb("large")
        1.0

        >>> Config.convert_to_gb("xl")
        4.0

        >>> Config.convert_to_gb("xxl")
        16.0
        """

        if isinstance(space, int) or isinstance(space, float):
            return float(space)

        size_names = {
            "small": 0.2,
            "base": 0.48,
            "large": 1.0,
            "xl": 4.0,
            "xxl": 16.0,
        }

        if space.lower().strip() in size_names:
            return size_names[space.lower().strip()]

        multipliers = {
            "g": 1.0,
            "m": 2 ** -10,
        }

        space = space.lower()
        space = space.rstrip("b")

        if space[-1] in multipliers:
            return float(space[:-1]) * multipliers[space[-1]]
        else:
            return float(space)


# Load the bootstrap config for the model passed via env var
models = [load_bootstrap_config()]
model_names = {m["name"]: m for m in models}

Config.schema = {
    "max_ram": ConfigItem(Config.convert_to_gb, 0.48),
    "max_tokens": ConfigItem(int, 200),
    "device": ConfigItem(Config.validate_device, "cpu"),
    "model_license": ConfigItem(re.compile, ".*")
}


# Map models to their config
for name in model_names:
    Config.schema["name"] = ConfigItem(Config.validate_model, name)
    # Config.schema["name"] = name

config = Config()

if "COLAB_GPU" in os.environ:
    if len(os.environ["COLAB_GPU"]) > 0:
        # We have a Colab GPU, so default to using it
        config["device"] = "auto"
