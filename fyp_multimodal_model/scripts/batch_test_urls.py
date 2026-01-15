"""
Batch URL Testing Script
Tests multiple URLs and compiles results
"""
import sys
import os
import json

# Add parent directory to path to import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference_pipeline import predict_fusion, load_all_models
from utils import load_config
import torch

# URLs to test
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

print("="*70)
print("BATCH URL TESTING - Phishing Detection")
print("="*70)

# Load configuration and models
config_path = os.path.join(os.path.dirname(__file__), '../config.json')
cfg = load_config(config_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[Loading Models] Using device: {device}")
models = load_all_models(cfg["models_dir"], device)
print("[Loading Models] All models loaded successfully!\n")

# Test each URL
results = []
for i, url in enumerate(urls, 1):
    print(f"\n[{i}/10] Testing: {url}")
    print("-" * 70)
    
    try:
        result = predict_fusion(
            url_string=url,
            screenshot_path=None,
            models=models,
            device=device
        )
        
        pred = result['prediction'].upper()
        conf = result['confidence']
        
        # Color coding for terminal (won't work on all terminals)
        if pred == "PHISHING":
            symbol = "⚠️"
            status = "PHISHING"
        else:
            symbol = "✅"
            status = "BENIGN"
        
        print(f"{symbol} Prediction: {status}")
        print(f"   Confidence: {conf:.2%}")
        print(f"   URL Score: {result['modality_scores']['url']:.4f}" if result['modality_scores']['url'] else "   URL Score: N/A")
        print(f"   DOM Score: {result['modality_scores']['dom']:.4f}" if result['modality_scores']['dom'] else "   DOM Score: N/A")
        print(f"   Visual Score: {result['modality_scores']['visual']:.4f}" if result['modality_scores']['visual'] else "   Visual Score: N/A")
        
        results.append({
            "url": url,
            "prediction": pred,
            "confidence": f"{conf:.2%}",
            "url_score": result['modality_scores']['url'],
            "dom_score": result['modality_scores']['dom'],
            "visual_score": result['modality_scores']['visual']
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append({
            "url": url,
            "prediction": "ERROR",
            "error": str(e)
        })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

phishing_count = sum(1 for r in results if r['prediction'] == 'PHISHING')
benign_count = sum(1 for r in results if r['prediction'] == 'BENIGN')
error_count = sum(1 for r in results if r['prediction'] == 'ERROR')

print(f"\nTotal URLs Tested: {len(urls)}")
print(f"  🚨 Detected as PHISHING: {phishing_count}")
print(f"  ✅ Detected as BENIGN: {benign_count}")
print(f"  ❌ Errors: {error_count}")

print("\n" + "="*70)
print("DETAILED RESULTS TABLE")
print("="*70)
print(f"{'#':<4} {'URL':<35} {'Prediction':<12} {'Confidence':<12}")
print("-" * 70)
for i, r in enumerate(results, 1):
    url_short = r['url'][:33] + ".." if len(r['url']) > 35 else r['url']
    print(f"{i:<4} {url_short:<35} {r['prediction']:<12} {r.get('confidence', 'N/A'):<12}")

# Save to JSON
output_file = "batch_test_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print("="*70)
