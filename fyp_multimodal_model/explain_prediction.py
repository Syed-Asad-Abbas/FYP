"""
Explainability Module
- Generates SHAP feature importance for URL and DOM modalities
- Creates natural language explanations using LLM
- Provides detailed breakdown of prediction reasoning

Usage:
  python explain_prediction.py --url "https://example.com" --config config.json
  python explain_prediction.py --url "https://phishing.com" --screenshot image.png --output explanation.json
"""

import argparse
import os
import json
import joblib
import numpy as np
import shap
from inference_pipeline import predict_fusion, load_all_models
from url_feature_extractor import extract_url_features_from_string
from utils import load_config, build_dom_tokens


def get_shap_url_explanations(url_features, models, feature_names, top_k=5):
    """
    Get SHAP explanations for URL modality
    Returns top-k contributing features
    """
    try:
        X = np.array(url_features).reshape(1, -1)
        X_scaled = models["url_scaler"].transform(X)
        
        # Create SHAP explainer for URL model
        explainer = shap.TreeExplainer(models["url_model"])
        shap_values = explainer.shap_values(X_scaled)
        
        # Get SHAP values for phishing class (class 1)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # class 1
        else:
            shap_vals = shap_values[0]
        
        # Get top-k features by absolute SHAP value
        indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        
        top_features = []
        for idx in indices:
            top_features.append({
                "feature": feature_names[idx],
                "value": float(url_features[idx]),
                "shap_impact": float(shap_vals[idx]),
                "abs_impact": float(abs(shap_vals[idx]))
            })
        
        return top_features
    except Exception as e:
        print(f"[SHAP-URL] Error: {e}")
        return []


def get_shap_fusion_explanations(fusion_features, models):
    """
    Get SHAP explanations for fusion layer
    Shows which modality contributed most
    """
    try:
        X = np.array(fusion_features).reshape(1, -1)
        
        explainer = shap.TreeExplainer(models["fusion_model"])
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
        
        # Feature names for fusion
        fusion_feature_names = [
            "p_url", "p_dom", "p_visual",
            "conf_url", "conf_dom", "conf_visual",
            "has_url", "has_dom", "has_visual"
        ]
        
        # Aggregate by modality
        modality_contributions = {
            "url": float(shap_vals[0] + shap_vals[3] + shap_vals[6]),
            "dom": float(shap_vals[1] + shap_vals[4] + shap_vals[7]),
            "visual": float(shap_vals[2] + shap_vals[5] + shap_vals[8])
        }
        
        # Normalize to weights (0-1)
        total = sum(abs(v) for v in modality_contributions.values())
        if total > 0:
            modality_weights = {k: abs(v)/total for k, v in modality_contributions.items()}
        else:
            modality_weights = {"url": 0.33, "dom": 0.33, "visual": 0.33}
        
        return {
            "modality_contributions": modality_contributions,
            "modality_weights": modality_weights
        }
    except Exception as e:
        print(f"[SHAP-Fusion] Error: {e}")
        return {
            "modality_contributions": {"url": 0, "dom": 0, "visual": 0},
            "modality_weights": {"url": 0.33, "dom": 0.33, "visual": 0.33}
        }


def generate_llm_explanation(prediction_result, shap_url_features, shap_fusion):
    """
    Generate natural language explanation using template-based approach
    In production, replace with actual LLM API call (OpenAI/Ollama)
    """
    pred = prediction_result['prediction']
    conf = prediction_result['confidence']
    url = prediction_result['url']
    
    # Build explanation based on top SHAP features
    explanation_parts = []
    
    # Header
    if pred == "phishing":
        explanation_parts.append(f"This URL is classified as **PHISHING** with {conf:.1%} confidence.")
    else:
        explanation_parts.append(f"This URL appears to be **BENIGN** with {conf:.1%} confidence.")
    
    # URL feature contributions
    if shap_url_features:
        explanation_parts.append("\n**Key URL indicators:**")
        for feat in shap_url_features[:3]:
            feature_name = feat['feature']
            value = feat['value']
            impact = feat['shap_impact']
            
            # Interpret common features
            if feature_name == "URLLength":
                if value > 100:
                    explanation_parts.append(f"- URL is unusually long ({int(value)} characters), which is suspicious.")
                else:
                    explanation_parts.append(f"- URL length ({int(value)} chars) is normal.")
            elif feature_name == "NoOfSubDomain":
                if value >= 3:
                    explanation_parts.append(f"- Excessive subdomains ({int(value)}), often used in phishing.")
                elif value == 0:
                    explanation_parts.append(f"- No subdomains, typical of legitimate sites.")
            elif feature_name == "HasObfuscation":
                if value == 1:
                    explanation_parts.append(f"- URL contains obfuscation characters (e.g., @, %), commonly used to deceive users.")
            elif feature_name == "TLDLegitimateProb":
                if value < 0.5:
                    explanation_parts.append(f"- Domain uses uncommon TLD (legitimacy score: {value:.2f}).")
    
    # Modality weights
    if shap_fusion:
        weights = shap_fusion['modality_weights']
        top_modality = max(weights, key=weights.get)
        explanation_parts.append(f"\n**Decision primarily based on:** {top_modality.upper()} analysis ({weights[top_modality]:.1%} contribution).")
    
    # Footer
    if pred == "phishing":
        explanation_parts.append("\n⚠️ **Recommendation:** Do not enter sensitive information on this page.")
    else:
        explanation_parts.append("\n✅ **Recommendation:** This URL appears safe based on our analysis.")
    
    return "\n".join(explanation_parts)


def explain_prediction(url_string, screenshot_path=None, models=None, device="cpu"):
    """
    Generate full explanation for a prediction
    
    Returns:
        Dict with prediction, SHAP values, and natural language explanation
    """
    # Get prediction
    result = predict_fusion(
        url_string=url_string,
        screenshot_path=screenshot_path,
        models=models,
        device=device
    )
    
    # Extract URL features for SHAP
    url_features = extract_url_features_from_string(url_string, models["url_features"])
    
    # Get SHAP explanations for URL
    shap_url = get_shap_url_explanations(url_features, models, models["url_features"])
    
    # Build fusion features
    p_url = result['modality_scores']['url'] if result['modality_scores']['url'] is not None else -1.0
    p_dom = result['modality_scores']['dom'] if result['modality_scores']['dom'] is not None else -1.0
    p_visual = result['modality_scores']['visual'] if result['modality_scores']['visual'] is not None else -1.0
    
    conf_url = result['modality_confidence']['url'] if result['modality_confidence']['url'] is not None else 0.0
    conf_dom = result['modality_confidence']['dom'] if result['modality_confidence']['dom'] is not None else 0.0
    conf_visual = result['modality_confidence']['visual'] if result['modality_confidence']['visual'] is not None else 0.0
    
    fusion_features = [
        p_url, p_dom, p_visual,
        conf_url, conf_dom, conf_visual,
        1.0 if result['modality_available']['url'] else 0.0,
        1.0 if result['modality_available']['dom'] else 0.0,
        1.0 if result['modality_available']['visual'] else 0.0
    ]
    
    # Get SHAP explanations for fusion
    shap_fusion = get_shap_fusion_explanations(fusion_features, models)
    
    # Generate natural language explanation
    llm_explanation = generate_llm_explanation(result, shap_url, shap_fusion)
    
    # Combine everything
    result['explanation'] = {
        "summary": llm_explanation,
        "shap_url_features": shap_url,
        "shap_fusion": shap_fusion
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL to analyze")
    parser.add_argument("--screenshot", default=None, help="Path to screenshot (optional)")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default=None, help="Save explanation to JSON file")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    models_dir = cfg["models_dir"]
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    print("[Explain] Loading models...")
    models = load_all_models(models_dir, device)
    print("[Explain] Models loaded.")
    
    # Generate explanation
    print(f"\n[Explain] Analyzing and explaining: {args.url}\n")
    result = explain_prediction(
        url_string=args.url,
        screenshot_path=args.screenshot,
        models=models,
        device=device
    )
    
    # Display explanation
    print("="*70)
    print("PREDICTION WITH EXPLANATION")
    print("="*70)
    print(f"URL: {result['url']}\n")
    print(result['explanation']['summary'])
    print("\n" + "="*70)
    print("TECHNICAL DETAILS")
    print("="*70)
    print(f"\nTop URL Features (SHAP):")
    for feat in result['explanation']['shap_url_features'][:5]:
        print(f"  {feat['feature']:20s} = {feat['value']:8.2f}  (SHAP: {feat['shap_impact']:+.4f})")
    
    print(f"\nModality Contribution Weights:")
    for mod, weight in result['explanation']['shap_fusion']['modality_weights'].items():
        print(f"  {mod.upper():10s}: {weight:.2%}")
    print("="*70)
    
    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Explain] Full result saved to {args.output}")
    
    return result


if __name__ == "__main__":
    import torch  # Import here to avoid circular dependency
    main()
