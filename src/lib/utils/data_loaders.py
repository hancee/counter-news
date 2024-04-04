import pandas as pd
from src.lib.utils.path_finder import PROJECT_DIRECTORY
from pathlib import Path


def load_data(file_path, kwargs=None):
    ext = file_path.split(".")[-1]
    if ext.upper() == "CSV":
        data = pd.read_csv(Path.joinpath(PROJECT_DIRECTORY, file_path), **kwargs)
    elif ext.upper() == "TSV":
        data = pd.read_csv(Path.joinpath(PROJECT_DIRECTORY, file_path), delimiter="\t", **kwargs)
    return data
