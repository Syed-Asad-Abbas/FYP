# Test URLs for Live Fetching - Multimodal Phishing Detection

## ✅ Legitimate Websites (Should Predict BENIGN)

### Major Brands
- https://www.google.com
- https://www.github.com
- https://www.microsoft.com
- https://www.amazon.com
- https://www.facebook.com
- https://www.wikipedia.org
- https://www.reddit.com
- https://www.stackoverflow.com

### Financial Institutions (Real)
- https://www.paypal.com
- https://www.chase.com
- https://www.bankofamerica.com
- https://www.wellsfargo.com

### Organizations
- https://www.whitehouse.gov
- https://www.nasa.gov
- https://www.un.org
- https://www.redcross.org

---

## ⚠️ Suspicious/Testing URLs

### Suspicious Patterns (May or may not be active phishing)

**Note:** These are synthetic examples of suspicious patterns. DO NOT visit these in a normal browser!

#### URL Obfuscation Patterns
- Use https://iplogger.org (link shortener - track clicks)
- Use https://bit.ly shortened links (could hide destination)

#### Suspicious TLDs
Look for sites using suspicious TLDs like:
- .tk (Tokelau - often free, used by phishers)
- .ml (Mali - free registrations)
- .ga (Gabon - free registrations)
- .cf (Central African Republic - free)
- .gq (Equatorial Guinea - free)

---

## 🎯 How to Get REAL Active Phishing URLs for Testing

### Option 1: PhishTank (RECOMMENDED)

1. Visit: https://www.phishtank.com/
2. Click "Browse"
3. Filter by:
   - Status: "Online" or "Valid"
   - Verified: "Yes"
4. Copy URLs from the last 24-48 hours

**Example process:**
```bash
# Visit PhishTank → Filter "Online" → Copy fresh URLs
# Then test:
python inference_complete.py --url "PHISHING_URL_FROM_PHISHTANK"
```

### Option 2: OpenPhish

1. Visit: https://openphish.com/feed.txt
2. Download the feed (updated every 15 minutes)
3. Pick URLs from the recent feed

```bash
# Download feed
curl https://openphish.com/feed.txt > phishing_feed.txt

# Test first URL
python inference_complete.py --url "$(head -n 1 phishing_feed.txt)"
```

### Option 3: URLhaus (Malware URLs)

1. Visit: https://urlhaus.abuse.ch/browse/
2. Filter by "online"
3. Pick recent URLs (be careful - these may host malware!)

---

## 🧪 Quick Test Script

I've created a test script for you:

```bash
# Test benign URLs
python test_live_urls.py --benign

# Test your own URL list
python test_live_urls.py --file my_urls.txt

# Test with timeout
python test_live_urls.py --benign --timeout 15
```

---

## ⚠️ SAFETY WARNINGS

1. **Never visit suspected phishing URLs in your normal browser**
2. **Use the headless fetching system ONLY**
3. **Don't enter credentials on test sites**
4. **PhishTank/OpenPhish URLs are dangerous - use programmatically only**
5. **Test in isolated environment if possible**

---

## 📊 Expected Results

### Legitimate Sites
```
URL: https://www.google.com
Prediction: BENIGN
Confidence: 99%+
Modalities: 3/3
```

### Phishing Sites (from PhishTank)
```
URL: [phishing URL]
Prediction: PHISHING
Confidence: 85-99%
Modalities: 2/3 or 3/3 (some may fail to load)
```

---

## 🎓 For FYP Demo

**Suggested demo flow:**

1. **Show legitimate site:**
   ```bash
   python inference_complete.py --url "https://www.github.com"
   # Should: BENIGN, 99%+ confidence
   ```

2. **Show known phishing from PhishTank:**
   ```bash
   # Get fresh URL from PhishTank (verified, online)
   python inference_complete.py --url "[FRESH_PHISHING_URL]"
   # Should: PHISHING, high confidence
   ```

3. **Show explanation:**
   ```bash
   python explain_prediction.py --url "[PHISHING_URL]"
   # Shows SHAP features + natural language
   ```

---

## 📝 Creating Your Own Test List

Create `my_test_urls.txt`:
```
https://www.google.com
https://www.github.com
[Add PhishTank URLs here]
```

Then test:
```bash
python test_live_urls.py --file my_test_urls.txt
```
