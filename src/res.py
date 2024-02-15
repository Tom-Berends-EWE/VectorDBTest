__all__ = ['load_env',
           'resource_path',
           'load_connection_config',
           'ResourceLocation',
           'CSVResourceLocation',
           'QUALIFIERS',
           'ISOTROPIC_QUALIFIERS',
           'CUSTOMER_INPUTS',
           'PCA_RESULTS']

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv, dotenv_values


def load_env():
    load_dotenv('../.env.shared')
    load_dotenv('../.env.secret')


load_env()

_RES_DIR = os.environ['RES_DIR']

np.set_printoptions(threshold=sys.maxsize)


def resource_path(relative_path: str) -> str:
    return _RES_DIR + relative_path


def load_connection_config(env_file) -> dict[str, str]:
    config = dict(os.environ.items())
    config.update(dotenv_values(env_file))
    return config


class ResourceLocation:
    def __init__(self, rel_path: str, load_function, save_function, load_function_is_method: bool = False,
                 save_function_is_method: bool = False):
        self._path = resource_path(rel_path)
        self._load_function = load_function
        self._save_function = save_function
        self._load_function_is_method = load_function_is_method
        self._save_function_is_method = save_function_is_method

    def call_function(self, function, is_method: bool, *args, **kwargs):
        if is_method:
            if len(args) == 0:
                raise ValueError('Missing instance argument')
            instance = args[0]
            new_args = args[1:] if len(args) > 1 else tuple()
            return function(instance, self._path, *new_args, **kwargs)
        else:
            return function(self._path, *args, **kwargs)

    def load(self, *args, **kwargs):
        return self.call_function(self._load_function, self._load_function_is_method, *args, **kwargs)

    def save(self, *args, **kwargs):
        return self.call_function(self._save_function, self._save_function_is_method, *args, **kwargs)


class CSVResourceLocation(ResourceLocation):
    def __init__(self, rel_path: str):
        super().__init__(rel_path, pd.read_csv, pd.DataFrame.to_csv, save_function_is_method=True)


QUALIFIERS = CSVResourceLocation('qualifiers.csv')
ISOTROPIC_QUALIFIERS = CSVResourceLocation('isotropic-qualifiers.csv')
CUSTOMER_INPUTS = CSVResourceLocation('customer-inputs.csv')
PCA_RESULTS = ResourceLocation('pca-results.npz', np.load, np.savez)
