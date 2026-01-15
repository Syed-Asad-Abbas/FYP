import pandas as pd
import sys

# Load dataset
print("Loading dataset...")
df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv', encoding='utf-8')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:15]}")

# Check label column (should be last column)
label_col = df.columns[-1]
print(f"\nLabel column: {label_col}")
print(f"Phishing count: {(df[label_col] == 1).sum()}")
print(f"Benign count: {(df[label_col] == 0).sum()}")

# Get the row for atelierozmoz.be
test_url = "https://www.atelierozmoz.be"
print(f"\nSearching for: {test_url}")

# Check if URL column exists
url_col = None
for col in df.columns:
    if 'url' in col.lower() or col == 'URL':
        url_col = col
        break

if url_col:
    print(f"Found URL column: {url_col}")
    matching = df[df[url_col] == test_url]
    if len(matching) > 0:
        row = matching.iloc[0]
        print(f"\n✓ Found URL in dataset!")
        print(f"Label: {row[label_col]} ({'PHISHING' if row[label_col]==1 else 'BENIGN'})")
        print(f"\nFeatures:")
        feature_cols = ['URLLength', 'DomainLength', 'NoOfSubDomain', 'TLDLegitimateProb', 'URLCharProb']
        for fc in feature_cols:
            if fc in df.columns:
                print(f"  {fc}: {row[fc]}")
    else:
        print(f"✗ URL not found in dataset!")
        print(f"\nShowing first few URLs from dataset:")
        print(df[url_col].head(10).tolist())
else:
    print("✗ No URL column found!")
    print(f"Available columns: {df.columns.tolist()}")
