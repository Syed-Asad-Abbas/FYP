"""
Ablation study for DOM Doc2Vec + LightGBM model.

- DOES NOT touch your main DOM model files.
- Trains multiple variants with different:
    * Doc2Vec epochs: 15, 20, 25, 30, 35, 40, 45, 50
    * Vector sizes: 32, 64, 128
    * Token settings:
        - FullTokens   : use keyword + TLD tokens
        - NoKW_TLD     : no keyword tokens, no TLD tokens
- Saves results to: models/dom_ablation.json

Run:
  python ablate_dom_doc2vec_lgbm.py --config config.json

Note: This is computationally heavy (48 variants).
"""

import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from utils import load_config, load_dataset

# Heuristics from main DOM script
BOOLEAN_PREFIXES = ("Has",)
COUNT_PREFIXES = ("NoOf",)
KEYWORDS = {"Bank", "Pay", "Crypto"}


def build_dom_tokens_variant(row, use_keywords=True, use_tld=True):
    """
    Same logic as main build_dom_tokens, but with switches
    to disable keyword tokens or TLD tokens for ablation.
    """
    tokens = []

    # boolean-style DOM flags
    for col, val in row.items():
        if col.startswith(BOOLEAN_PREFIXES) and col not in ("HasObfuscation",):
            try:
                if int(val) == 1:
                    tokens.append(col.lower())
            except Exception:
                pass

    # count-style DOM stats
    for col, val in row.items():
        if col.startswith(COUNT_PREFIXES):
            try:
                cnt = int(val)
                reps = min(cnt, 10)
                base = re.sub(r"^NoOf", "count_", col).lower()
                tokens.extend([base] * reps)
            except Exception:
                pass

    # optional keyword cues
    if use_keywords:
        for col, val in row.items():
            if col in KEYWORDS:
                try:
                    if int(val) == 1:
                        tokens.append(f"kw_{col.lower()}")
                except Exception:
                    pass

    # optional TLD tokens
    if use_tld:
        if "TLD" in row and isinstance(row["TLD"], str):
            tokens.append(f"tld_{row['TLD'].lower()}")

    return tokens if tokens else ["empty"]


def run_variant(
    df,
    name,
    vector_size=64,
    epochs=30,
    use_keywords=True,
    use_tld=True,
):
    """
    Train one Doc2Vec+LGBM variant and return summary metrics.
    Does NOT save any model, only returns metrics for ablation table.
    """
    print(f"\n[ABLATION] === Variant: {name} ===")
    print(
        f"[ABLATION] Settings: vec_size={vector_size}, epochs={epochs}, "
        f"use_keywords={use_keywords}, use_tld={use_tld}"
    )

    labels = df["label"].astype(int).values.tolist()
    rows = df.to_dict(orient="records")

    # Build token documents
    documents = []
    for i, row in enumerate(
        tqdm(rows, desc=f"[{name}] Building DOM tokens", unit="row")
    ):
        tokens = build_dom_tokens_variant(
            row,
            use_keywords=use_keywords,
            use_tld=use_tld,
        )
        documents.append(TaggedDocument(words=tokens, tags=[str(i)]))

    # Train Doc2Vec (same style as main DOM script)
    print(f"[{name}] Training Doc2Vec...")
    model_d2v = Doc2Vec(
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=epochs,
        dm=1,
        seed=42,
    )
    model_d2v.build_vocab(documents)
    model_d2v.train(
        documents,
        total_examples=model_d2v.corpus_count,
        epochs=epochs,
    )

    # Infer embeddings
    print(f"[{name}] Inferring embeddings...")
    embeddings = np.vstack(
        [model_d2v.infer_vector(doc.words) for doc in documents]
    )

    # Train/test split (same random_state & stratify for fairness)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=48,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    print(f"[{name}] Training LightGBM...")
    clf.fit(X_train, y_train)

    print(f"[{name}] Evaluating...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Phishing (label=1) precision/recall/F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    prec_phish = float(prec[0])
    rec_phish = float(rec[0])
    f1_phish = float(f1[0])

    print(
        f"[{name}] Accuracy={acc:.4f}, "
        f"Phish_Prec={prec_phish:.4f}, Phish_Rec={rec_phish:.4f}, Phish_F1={f1_phish:.4f}"
    )

    return {
        "variant": name,
        "vector_size": vector_size,
        "epochs": epochs,
        "use_keywords": use_keywords,
        "use_tld": use_tld,
        "accuracy": float(acc),
        "phish_precision": prec_phish,
        "phish_recall": rec_phish,
        "phish_f1": f1_phish,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    print("[ABLATION] Loading config and dataset...")
    cfg = load_config(args.config)
    df = load_dataset(cfg["dataset_csv"]).copy()
    print(f"[ABLATION] Dataset: {len(df)} rows, {len(df.columns)} columns")
    print("[ABLATION] This will run MANY variants (epochs × vec_size × tokens).")

    epoch_list = [ 20, 30, 40, 50]
    vec_sizes = [32, 64, 128]

    # token modes:
    # - FullTokens: use_keywords=True, use_tld=True
    # - NoKW_TLD : use_keywords=False, use_tld=False
    token_modes = [
        ("FullTokens", True, True),
        ("NoKW_TLD", False, False),
    ]

    results = []

    for epochs in epoch_list:
        for vec in vec_sizes:
            for token_name, use_kw, use_tld in token_modes:
                variant_name = f"Ep{epochs}_Vec{vec}_{token_name}"
                res = run_variant(
                    df,
                    name=variant_name,
                    vector_size=vec,
                    epochs=epochs,
                    use_keywords=use_kw,
                    use_tld=use_tld,
                )
                results.append(res)

    # Save results next to your models, without touching main files
    os.makedirs(cfg["models_dir"], exist_ok=True)
    out_path = os.path.join(cfg["models_dir"], "dom_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ABLATION] Saved ablation results to {out_path}")
    print("[ABLATION] Done ✔")


if __name__ == "__main__":
    main()
