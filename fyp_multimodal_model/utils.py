
import os
import json
import pandas as pd

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    # Normalize BOM in header if present
    if "\ufeffFILENAME" in df.columns:
        df = df.rename(columns={"\ufeffFILENAME": "FILENAME"})
    return df
