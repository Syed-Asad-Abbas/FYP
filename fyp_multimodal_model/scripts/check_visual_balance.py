import argparse
import os
import json
from collections import Counter

import pandas as pd


def load_config(path):
    """Minimal config loader (expects dataset_csv and image_dir keys)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_screenshot(fname, image_dir):
    """
    Check if a screenshot exists for the given filename in image_dir.

    Mirrors the logic from ScreenshotDataset:
    - strips .txt
    - tries stem.png / stem.jpg / stem.jpeg and stem.txt.png / stem.txt.jpg / ...
    """
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    stem = os.path.splitext(str(fname))[0]

    for ext in exts:
        candidates = [f"{stem}{ext}", f"{stem}.txt{ext}"]
        for cand in candidates:
            p = os.path.join(image_dir, cand)
            if os.path.exists(p):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check phishing vs benign balance for rows that actually have screenshots."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), '../config.json'),
        help="Path to config.json (must contain dataset_csv and image_dir).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to dataset CSV (used if --config is not given).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing screenshots (used if --config is not given).",
    )

    args = parser.parse_args()

    # Decide how to get csv and image_dir
    if args.config is not None:
        cfg = load_config(args.config)
        csv_path = cfg["dataset_csv"]
        image_dir = cfg["image_dir"]
    else:
        if args.csv is None or args.image_dir is None:
            parser.error("Either --config OR both --csv and --image_dir must be provided.")
        csv_path = args.csv
        image_dir = args.image_dir

    print(f"[INFO] Using CSV: {csv_path}")
    print(f"[INFO] Using image dir: {image_dir}")

    # Load dataset
    df = pd.read_csv(csv_path)

    if "FILENAME" not in df.columns or "label" not in df.columns:
        raise RuntimeError(
            'CSV must contain at least "FILENAME" and "label" columns.'
        )

    # Mark rows that have an actual screenshot on disk
    print("[INFO] Scanning for existing screenshots on disk (this may take a bit)...")
    df["has_img"] = df["FILENAME"].apply(lambda x: has_screenshot(x, image_dir))

    sub = df[df["has_img"]]

    total_rows = len(df)
    total_with_img = len(sub)

    print("\n===== Screenshot Coverage =====")
    print(f"Total rows in CSV:          {total_rows}")
    print(f"Rows with screenshots:      {total_with_img}")
    if total_rows > 0:
        print(f"Coverage (has_img == True): {total_with_img / total_rows:.2%}")

    if total_with_img == 0:
        print("\n[WARN] No rows with screenshots found. Check image_dir and naming.")
        return

    # Class balance among rows that have screenshots
    counts = sub["label"].value_counts().sort_index()
    props = sub["label"].value_counts(normalize=True).sort_index()

    print("\n===== Class Balance (only rows that HAVE screenshots) =====")
    for lab in counts.index:
        count = counts[lab]
        prop = props[lab]
        print(f"Label {lab}: {count} ({prop:.2%})")

    print("\nHint:")
    print(" - ~50/50 is perfectly balanced.")
    print(" - ~60/40 is okay, mild imbalance.")
    print(" - ~70/30 or worse = consider class weights / sampler in training.")


if __name__ == "__main__":
    main()
