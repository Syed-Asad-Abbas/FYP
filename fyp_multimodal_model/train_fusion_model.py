"""
Fusion Layer Model Training
- Combines outputs from URL, DOM, and Visual modalities
- Uses meta-classifier (LightGBM) with dynamic weighting
- Handles missing modalities gracefully
- Includes ablation study to validate fusion improvement

Run:
  python train_fusion_model.py --config config.json
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_config, load_dataset, build_dom_tokens
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, matthews_corrcoef
)
from lightgbm import LGBMClassifier
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
def load_url_model(models_dir):
    """Load trained URL model (Production version)"""
    path = os.path.join(models_dir, "url_lgbm_production.joblib")
    data = joblib.load(path)
    return data["model"], data["scaler"], data["feature_names"]



def load_dom_model(models_dir):
    """Load trained DOM model"""
    path = os.path.join(models_dir, "dom_doc2vec_lgbm.joblib")
    data = joblib.load(path)
    return data["doc2vec"], data["model"]


def load_visual_model(models_dir, device):
    """Load trained Visual model"""
    path = os.path.join(models_dir, "visual_resnet50.pt")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_url_prediction(url_model, scaler, feature_names, row):
    """Get URL modality prediction"""
    try:
        features = [row[fn] for fn in feature_names]
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        proba = url_model.predict_proba(X_scaled)[0]  # [p_benign, p_phish]
        return proba[1], max(proba)  # phishing prob, confidence
    except Exception as e:
        return -1.0, 0.0  # missing


def get_dom_prediction(doc2vec_model, dom_model, row):
    """Get DOM modality prediction"""
    try:
        tokens = build_dom_tokens(row)
        embedding = doc2vec_model.infer_vector(tokens)
        proba = dom_model.predict_proba([embedding])[0]
        return proba[1], max(proba)
    except Exception as e:
        return -1.0, 0.0


def get_visual_prediction(visual_model, image_path, device):
    """Get Visual modality prediction"""
    try:
        if not os.path.exists(image_path):
            return -1.0, 0.0
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = visual_model(img_tensor)
            probs = torch.softmax(out, dim=1)[0]
            p_phish = probs[1].item()
            confidence = max(probs[0].item(), probs[1].item())
        
        return p_phish, confidence
    except Exception as e:
        return -1.0, 0.0


def build_fusion_features(df, url_model, url_scaler, url_features,
                          doc2vec, dom_model, visual_model, 
                          image_dir, device):
    """
    Build fusion training data from all modalities
    Returns: X (fusion features), y (labels), metadata
    """
    fusion_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building fusion features"):
        # URL modality
        p_url, conf_url = get_url_prediction(url_model, url_scaler, url_features, row)
        
        # DOM modality
        p_dom, conf_dom = get_dom_prediction(doc2vec, dom_model, row)
        
        # Visual modality
        filename = str(row["FILENAME"])
        stem = os.path.splitext(filename)[0]
        img_path = None
        for ext in [".png", ".jpg", ".PNG", ".JPG"]:
            for variant in [f"{stem}{ext}", f"{stem}.txt{ext}"]:
                candidate = os.path.join(image_dir, variant)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path:
                break
        
        p_visual, conf_visual = get_visual_prediction(visual_model, img_path, device) if img_path else (-1.0, 0.0)
        
        # Build fusion feature vector
        features = [
            p_url,           # URL phishing probability
            p_dom,           # DOM phishing probability
            p_visual,        # Visual phishing probability
            conf_url,        # URL confidence
            conf_dom,        # DOM confidence
            conf_visual,     # Visual confidence
            1.0 if p_url >= 0 else 0.0,    # has_url flag
            1.0 if p_dom >= 0 else 0.0,    # has_dom flag
            1.0 if p_visual >= 0 else 0.0, # has_visual flag
        ]
        
        fusion_data.append({
            "features": features,
            "label": int(row["label"]),
            "filename": filename,
            "has_visual": p_visual >= 0
        })
    
    X = np.array([d["features"] for d in fusion_data])
    y = np.array([d["label"] for d in fusion_data])
    metadata = pd.DataFrame([{
        "filename": d["filename"],
        "label": d["label"],
        "has_visual": d["has_visual"]
    } for d in fusion_data])
    
    return X, y, metadata


def train_fusion_classifier(X_train, y_train, X_test, y_test):
    """Train meta-classifier for fusion"""
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    metrics = {
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "MCC": float(mcc),
        "ROC_AUC": float(roc_auc)
    }
    
    return clf, metrics


def ablation_study(X, y, metadata):
    """
    Run ablation study comparing different modality combinations
    Returns dict with results for each combination
    """
    print("\n[Fusion] Running ablation study...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # Define feature indices
    # [p_url, p_dom, p_visual, conf_url, conf_dom, conf_visual, has_url, has_dom, has_visual]
    combinations = {
        "url_only": [0, 3, 6],           # p_url, conf_url, has_url
        "dom_only": [1, 4, 7],           # p_dom, conf_dom, has_dom
        "visual_only": [2, 5, 8],        # p_visual, conf_visual, has_visual
        "url_dom": [0, 1, 3, 4, 6, 7],
        "url_visual": [0, 2, 3, 5, 6, 8],
        "dom_visual": [1, 2, 4, 5, 7, 8],
        "all_modalities": list(range(9))
    }
    
    for name, indices in combinations.items():
        print(f"  Testing: {name}...")
        
        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]
        
        # For visual-only, filter to samples with screenshots
        if name == "visual_only":
            has_visual_test = X_test[:, 8] > 0
            if has_visual_test.sum() == 0:
                results[name] = {"accuracy": 0.0, "note": "No visual samples in test set"}
                continue
            X_test_sub = X_test_sub[has_visual_test]
            y_test_sub = y_test[has_visual_test]
        else:
            y_test_sub = y_test
        
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train_sub, y_train)
        
        y_pred = clf.predict(X_test_sub)
        acc = accuracy_score(y_test_sub, y_pred)
        report = classification_report(y_test_sub, y_pred, output_dict=True)
        
        results[name] = {
            "accuracy": float(acc),
            "precision_phish": float(report["1"]["precision"]),
            "recall_phish": float(report["1"]["recall"]),
            "f1_phish": float(report["1"]["f1-score"])
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sample_size", type=int, default=None, 
                       help="Use subset for faster testing (e.g., 10000)")
    args = parser.parse_args()
    
    print("[Fusion] Loading configuration...")
    cfg = load_config(args.config)
    models_dir = cfg["models_dir"]
    image_dir = cfg["image_dir"]
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Fusion] Using device: {device}")
    
    # Load dataset
    print("[Fusion] Loading dataset...")
    df = load_dataset(cfg["dataset_csv"])
    
    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(f"[Fusion] Using sample size: {len(df)}")
    
    # Load all three trained models
    print("[Fusion] Loading URL model...")
    url_model, url_scaler, url_features = load_url_model(models_dir)
    
    print("[Fusion] Loading DOM model...")
    doc2vec, dom_model = load_dom_model(models_dir)
    
    print("[Fusion] Loading Visual model...")
    visual_model = load_visual_model(models_dir, device)
    
    # Build fusion training data
    print("[Fusion] Building fusion features from all modalities...")
    X, y, metadata = build_fusion_features(
        df, url_model, url_scaler, url_features,
        doc2vec, dom_model, visual_model,
        image_dir, device
    )
    
    print(f"[Fusion] Built fusion dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[Fusion] Samples with visual data: {metadata['has_visual'].sum()} / {len(metadata)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train fusion classifier
    print("[Fusion] Training fusion meta-classifier...")
    fusion_clf, fusion_metrics = train_fusion_classifier(X_train, y_train, X_test, y_test)
    
    print(f"\n[Fusion] Fusion Model Performance:")
    print(f"  Accuracy: {fusion_metrics['accuracy']:.4f}")
    print(f"  Precision (phish): {fusion_metrics['report']['1']['precision']:.4f}")
    print(f"  Recall (phish): {fusion_metrics['report']['1']['recall']:.4f}")
    print(f"  F1 (phish): {fusion_metrics['report']['1']['f1-score']:.4f}")
    print(f"  ROC-AUC: {fusion_metrics['ROC_AUC']:.4f}")
    print(f"  FPR: {fusion_metrics['FPR']:.4f}")
    print(f"  FNR: {fusion_metrics['FNR']:.4f}")
    
    # Ablation study
    ablation_results = ablation_study(X, y, metadata)
    
    print("\n[Fusion] Ablation Study Results:")
    for name, metrics in ablation_results.items():
        print(f"  {name:20s}: Acc={metrics.get('accuracy', 0):.4f}")
    
    # Save fusion model
    os.makedirs(models_dir, exist_ok=True)
    fusion_path = os.path.join(models_dir, "fusion_lgbm.joblib")
    joblib.dump({"model": fusion_clf}, fusion_path)
    
    # Save metrics
    metrics_path = os.path.join(models_dir, "fusion_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(fusion_metrics, f, indent=2)
    
    # Save ablation results
    ablation_path = os.path.join(models_dir, "fusion_ablation.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n[Fusion] Saved fusion model to {fusion_path}")
    print(f"[Fusion] Saved metrics to {metrics_path}")
    print(f"[Fusion] Saved ablation results to {ablation_path}")
    print("[Fusion] Training complete! ✔")


if __name__ == "__main__":
    main()
