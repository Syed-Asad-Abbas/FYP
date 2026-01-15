
import os
import json
import pandas as pd
import re

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    # Normalize BOM in header if present
    if "\ufeffFILENAME" in df.columns:
        df = df.rename(columns={"\ufeffFILENAME": "FILENAME"})
    return df


def build_dom_tokens(row):
    """
    Build DOM token list from row data (reused from train_dom_doc2vec_lgbm.py)
    Used for both training and inference
    """
    BOOLEAN_PREFIXES = ("Has",)
    COUNT_PREFIXES = ("NoOf",)
    KEYWORDS = {"Bank", "Pay", "Crypto"}
    
    tokens = []
    for col, val in row.items():
        # boolean-style DOM flags
        if col.startswith(BOOLEAN_PREFIXES) and col not in ("HasObfuscation",):
            try:
                if int(val) == 1:
                    tokens.append(col.lower())
            except Exception:
                pass

        # count-style DOM stats
        if col.startswith(COUNT_PREFIXES):
            try:
                cnt = int(val)
                reps = min(cnt, 10)
                base = re.sub(r"^NoOf", "count_", col).lower()
                tokens.extend([base] * reps)
            except Exception:
                pass

        # keyword cues
        if col in KEYWORDS:
            try:
                if int(val) == 1:
                    tokens.append(f"kw_{col.lower()}")
            except Exception:
                pass

    # Fallback: TLD
    for aux in ("TLD",):
        if aux in row and isinstance(row[aux], str):
            tokens.append(f"tld_{row[aux].lower()}")

    return tokens if tokens else ["empty"]
