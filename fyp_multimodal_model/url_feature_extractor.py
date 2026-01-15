"""
Proper URL Feature Extraction
Extracts the same features used during training from PhiUSIIL dataset
"""

import re
from urllib.parse import urlparse
from collections import Counter


def extract_url_features_from_string(url_string, feature_names):
    """
    Extract URL features matching the PhiUSIIL dataset format
    Returns list of feature values in the same order as feature_names
    """
    try:
        parsed = urlparse(url_string)
        domain = parsed.netloc
        path = parsed.path
        full_url = url_string
        
        # Initialize features dict
        features = {}
        
        # 1. URLLength - total length of URL
        features["URLLength"] = len(full_url)
        
        # 2. DomainLength - length of domain/netloc
        features["DomainLength"] = len(domain)
        
        # 3. IsDomainIP - is domain an IP address?
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        features["IsDomainIP"] = 1 if re.match(ip_pattern, domain) else 0
        
        # 4. URLSimilarityIndex - ratio of unique chars to total chars
        if len(full_url) > 0:
            unique_chars = len(set(full_url))
            features["URLSimilarityIndex"] = unique_chars / len(full_url)
        else:
            features["URLSimilarityIndex"] = 0.0
        
        # 5. CharContinuationRate - max consecutive same char / total length
        if len(full_url) > 0:
            max_consecutive = 1
            current_consecutive = 1
            for i in range(1, len(full_url)):
                if full_url[i] == full_url[i-1]:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
            features["CharContinuationRate"] = max_consecutive / len(full_url)
        else:
            features["CharContinuationRate"] = 0.0
        
        # 6. TLDLegitimateProb - TLD legitimacy (heuristic based on common TLDs)
        tld = domain.split('.')[-1] if '.' in domain else ''
        legitimate_tlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'uk', 'de', 'fr', 'jp']
        suspicious_tlds = [
            'tk', 'ml', 'ga', 'cf', 'gq', 'zip', 'top', 'xyz', 
            'icu', 'cyou', 'sbs', 'cfd', 'site', 'live', 'shop', 
            'store', 'online', 'tech', 'website', 'space', 'vip', 
            'work', 'click', 'cloud', 'buzz', 'monster', 'asia'
        ]
        
        if tld in legitimate_tlds:
            features["TLDLegitimateProb"] = 100
        elif tld in suspicious_tlds:
            features["TLDLegitimateProb"] = 10
        else:
            features["TLDLegitimateProb"] = 50
        
        # 7. URLCharProb - ratio of alphanumeric to total chars
        if len(full_url) > 0:
            alphanumeric = sum(c.isalnum() for c in full_url)
            features["URLCharProb"] = alphanumeric / len(full_url)
        else:
            features["URLCharProb"] = 0.0
        
        # 8. TLDLength - length of TLD
        features["TLDLength"] = len(tld)
        
        # 9. NoOfSubDomain - number of subdomains
        domain_parts = domain.split('.')
        # Subtract 2 for domain + TLD (e.g., example.com = 0 subdomains)
        features["NoOfSubDomain"] = max(0, len(domain_parts) - 2)
        
        # 10. HasObfuscation - presence of obfuscation characters
        obfuscation_chars = ['@', '%20', '%', '\\', '///', '..']
        features["HasObfuscation"] = 1 if any(char in full_url for char in obfuscation_chars) else 0
        
        # 11. NoOfObfuscatedChar - count of obfuscation characters
        obfuscation_count = sum(full_url.count(char) for char in ['@', '%', '\\'])
        features["NoOfObfuscatedChar"] = obfuscation_count
        
        # 12. ObfuscationRatio - ratio of obfuscated chars to total
        if len(full_url) > 0:
            features["ObfuscationRatio"] = obfuscation_count / len(full_url)
        else:
            features["ObfuscationRatio"] = 0.0
        
        # Return features in the exact order specified by feature_names
        return [features.get(fn, 0) for fn in feature_names]
        
    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        # Return zeros if extraction fails
        return [0.0] * len(feature_names)


# Optional: Function to extract features and return as dict for debugging
def extract_url_features_dict(url_string):
    """
    Extract URL features and return as dictionary
    Useful for debugging and understanding feature values
    """
    feature_names = [
        "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", 
        "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
        "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio"
    ]
    
    values = extract_url_features_from_string(url_string, feature_names)
    return dict(zip(feature_names, values))


if __name__ == "__main__":
    # Test the feature extractor
    test_urls = [
        "https://www.atelierozmoz.be",  # From user's test (labeled phishing in dataset)
        "https://www.paypal.com",       # Legitimate
        "https://paypal-login-verify.tk/@secure",  # Obvious phishing
    ]
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = extract_url_features_dict(url)
        for k, v in features.items():
            print(f"  {k:25s}: {v}")
