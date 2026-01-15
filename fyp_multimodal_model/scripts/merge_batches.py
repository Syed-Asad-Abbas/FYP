# merge_batches.py
import pandas as pd

paths = [f"batches/batch_{i}.csv" for i in range(1, 8)]
dfs = [pd.read_csv(p) for p in paths]
merged = pd.concat(dfs, ignore_index=True)

print("Total rows:", len(merged))
print("Label counts:\n", merged["label"].value_counts())

merged.to_csv("rich_visual_subset.csv", index=False)
print("Saved to rich_visual_subset.csv")
