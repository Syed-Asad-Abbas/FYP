# analyze_feature_importance.py
import os, sys


import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# --- YOUR PATHS ---
DATASET_CSV = r"E:\FYP\fyp_multimodal_model\data\PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH  = r"E:\FYP\fyp_multimodal_model\models\url_lgbm.joblib"
if not os.path.exists(DATASET_CSV):
    sys.exit(f"Dataset not found: {DATASET_CSV}")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Model not found: {MODEL_PATH}")

# --- SETTINGS ---
TEST_SIZE    = 0.20
RANDOM_STATE = 42
PERM_SAMPLES = 10000   # reduce if needed
PERM_REPEATS = 5

# --- LOAD MODEL + DATA ---
bundle = joblib.load(MODEL_PATH)
model: LGBMClassifier = bundle["model"]
scaler                = bundle["scaler"]
used_features         = bundle["feature_names"]

df = pd.read_csv(DATASET_CSV)
X  = df[used_features].values
y  = df["label"].astype(int).values

# --- SPLIT ---
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# use the saved scaler from training
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# --- LIGHTGBM BUILT-IN IMPORTANCES ---
booster  = model.booster_
gain_imp = booster.feature_importance(importance_type="gain")
split_imp= booster.feature_importance(importance_type="split")

gain_pct  = (gain_imp  / gain_imp.sum()  * 100) if gain_imp.sum()  > 0 else np.zeros_like(gain_imp)
split_pct = (split_imp / split_imp.sum() * 100) if split_imp.sum() > 0 else np.zeros_like(split_imp)

# --- PERMUTATION IMPORTANCE (robustness check) ---
n   = X_te_s.shape[0]
idx = np.random.RandomState(RANDOM_STATE).choice(n, size=min(PERM_SAMPLES, n), replace=False)
perm = permutation_importance(
    model, X_te_s[idx], y_te[idx],
    scoring="accuracy", n_repeats=PERM_REPEATS,
    random_state=RANDOM_STATE, n_jobs=-1
)
perm_mean = perm.importances_mean
perm_pct  = (perm_mean / perm_mean.sum() * 100) if perm_mean.sum() != 0 else np.zeros_like(perm_mean)

# --- COMBINE & SAVE ---
out = pd.DataFrame({
    "Feature": used_features,
    "Gain %":  gain_pct,
    "Split %": split_pct,
    "Permutation ΔAcc %": perm_pct
}).sort_values("Gain %", ascending=False).reset_index(drop=True)

os.makedirs("analysis_artifacts", exist_ok=True)
csv_path = os.path.join("analysis_artifacts", "url_feature_contributions.csv")
png_path = os.path.join("analysis_artifacts", "url_feature_contributions_gain.png")

out.round(3).to_csv(csv_path, index=False)

plt.figure()
plt.bar(out["Feature"], out["Gain %"])
plt.title("URL Feature Contribution (LightGBM Gain %)")
plt.xlabel("Feature")
plt.ylabel("Contribution (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(png_path, dpi=200)
plt.close()

print("Saved:", csv_path)
print("Saved:", png_path)
print("\nTop 5 by Gain %:\n", out.head(5).to_string(index=False))

