import pandas as pd


def load_data(file_path, kwargs=None):
    ext = file_path.split(".")[-1]
    if ext.upper() == "CSV":
        data = pd.read_csv(file_path, **kwargs)
    elif ext.upper() == "TSV":
        data = pd.read_csv(file_path, delimiter="\t", **kwargs)
    return data
