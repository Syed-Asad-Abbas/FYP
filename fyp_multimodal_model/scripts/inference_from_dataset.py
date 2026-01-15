"""
Inference using Dataset Features (for demonstration/testing)
Looks up pre-computed features from PhiUSIIL dataset instead of calculating them
"""

import pandas as pd
import numpy as np
import joblib
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os


# URL features used by the URL model
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", 
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio"
]


def load_dataset(csv_path):
    """Load PhiUSIIL dataset"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    return df


def get_url_features_from_dataset(url, dataset_df):
    """
    Look up pre-computed URL features from dataset
    Returns features in the order expected by the trained model
    """
    # Find matching URL
    matches = dataset_df[dataset_df['url'] == url]
    
    if len(matches) == 0:
        raise ValueError(f"URL not found in dataset: {url}\nThis function only works for URLs in PhiUSIIL dataset.")
    
    row = matches.iloc[0]
    
    # Extract features in correct order
    features = [row[feat] for feat in URL_FEATURES]
    
    return features, row['label']


def predict_url_with_dataset_features(url, dataset_df, url_model, url_scaler):
    """Get URL prediction using dataset features"""
    try:
        features, true_label = get_url_features_from_dataset(url, dataset_df)
        X = np.array(features).reshape(1, -1)
        X_scaled = url_scaler.transform(X)
        proba = url_model.predict_proba(X_scaled)[0]
        return proba[1], max(proba), True, true_label
    except Exception as e:
        print(f"[URL Dataset Lookup] Error: {e}")
        return -1.0, 0.0, False, None


def test_url_with_dataset_features(url, dataset_path, models_dir):
    """
    Test a single URL using dataset features
    Returns: prediction result with true label comparison
    """
    # Load dataset
    dataset_df = load_dataset(dataset_path)
    
    # Load URL model
    url_data = joblib.load(os.path.join(models_dir, 'url_lgbm.joblib'))
    url_model = url_data['model']
    url_scaler = url_data['scaler']
    
    # Get prediction
    p_url, conf_url, success, true_label = predict_url_with_dataset_features(
        url, dataset_df, url_model, url_scaler
    )
    
    if not success:
        return {"error": "URL not found in dataset"}
    
    predicted_label = 1 if p_url > 0.5 else 0
    
    result = {
        "url": url,
        "true_label": "PHISHING" if true_label == 1 else "BENIGN",
        "predicted_label": "PHISHING" if predicted_label == 1 else "BENIGN",
        "phishing_probability": float(p_url),
        "confidence": float(conf_url),
        "correct": predicted_label == true_label
    }
    
    return result


def batch_test_dataset_urls(urls, dataset_path, models_dir):
    """
    Test multiple URLs using dataset features
    """
    # Load dataset
    dataset_df = load_dataset(dataset_path)
    
    # Load URL model
    url_data = joblib.load(os.path.join(models_dir, 'url_lgbm.joblib'))
    url_model = url_data['model']
    url_scaler = url_data['scaler']
    
    results = []
    
    print("="*70)
    print("TESTING URLS WITH DATASET FEATURES")
    print("="*70)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Testing: {url}")
        print("-" * 70)
        
        p_url, conf_url, success, true_label = predict_url_with_dataset_features(
            url, dataset_df, url_model, url_scaler
        )
        
        if success:
            predicted_label = 1 if p_url > 0.5 else 0
            correct = predicted_label == true_label
            
            true_str = "PHISHING" if true_label == 1 else "BENIGN"
            pred_str = "PHISHING" if predicted_label == 1 else "BENIGN"
            status = "✅ CORRECT" if correct else "❌ WRONG"
            
            print(f"True Label:      {true_str}")
            print(f"Predicted:       {pred_str}")
            print(f"Phishing Prob:   {p_url:.4f}")
            print(f"Confidence:      {conf_url:.2%}")
            print(f"Status:          {status}")
            
            results.append({
                "url": url,
                "true_label": true_str,
                "predicted": pred_str,
                "phishing_prob": float(p_url),
                "confidence": float(conf_url),
                "correct": bool(correct)  # Convert numpy bool to Python bool
            })
        else:
            print("❌ URL not found in dataset")
            results.append({
                "url": url,
                "error": "Not in dataset"
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    correct_count = sum(1 for r in results if r.get('correct', False))
    total = len([r for r in results if 'error' not in r])
    
    print(f"\nTotal Tested: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count/total*100:.2f}%" if total > 0 else "N/A")
    
    return results


if __name__ == "__main__":
    # Test the 10 phishing URLs
    urls = [
        "https://www.atelierozmoz.be",
        "https://www.diemon.com",
        "https://www.wausauschools.org",
        "https://www.paademode.com",
        "https://www.boxturtles.com",
        "https://www.mmstadium.com",
        "https://www.brswimwear.com",
        "https://www.leathercouncil.org",
        "https://www.historync.org",
        "https://www.toshin.com"
    ]
    
    # Use paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../data/PhiUSIIL_Phishing_URL_Dataset.csv")
    models_dir = os.path.join(base_dir, "../models")
    
    results = batch_test_dataset_urls(urls, dataset_path, models_dir)
    
    # Save results
    import json
    with open("dataset_lookup_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: dataset_lookup_test_results.json")
    print("="*70)
