import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm  # progress bar

# ==== CONFIG ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../data/PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "../screenshots")
WAIT_TIME = 15      # seconds to wait for page load
BATCH_SIZE = 5000   # number of screenshots per batch
HEADLESS = True
# ================

# Make output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH, encoding="utf-8", engine="python")
df.columns = [c.lower().strip() for c in df.columns]
print(f"✅ Loaded dataset with {len(df)} rows")

# Selenium setup
chrome_options = Options()
if HEADLESS:
    chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=1280,1024")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--ignore-ssl-errors")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.set_page_load_timeout(WAIT_TIME)

# ===== Resume Logic =====
last_saved = input("\nEnter the LAST saved filename (e.g., 'batch_3_2499.png') or press Enter to start fresh: ").strip()
start_index = 0

if last_saved:
    fname_no_ext = os.path.splitext(last_saved)[0]
    match_idx = df.index[df["filename"].astype(str).str.contains(fname_no_ext, case=False, regex=False)]
    if not match_idx.empty:
        start_index = match_idx[0] + 1
        print(f"✅ Resuming from index {start_index} (after {last_saved})")
    else:
        print(f"⚠️ '{last_saved}' not found in CSV, starting from beginning.")
else:
    print("🆕 Starting fresh from index 0...")

total_urls = len(df)
total_batches = (total_urls - start_index + BATCH_SIZE - 1) // BATCH_SIZE
print(f"📦 Total of {total_batches} batches (each of {BATCH_SIZE} screenshots).")

count, errors = 0, []

# ===== Batch Loop =====
for batch_num in range(total_batches):
    batch_start = start_index + batch_num * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, total_urls)
    if batch_start >= total_urls:
        break

    batch_df = df.iloc[batch_start:batch_end]
    print(f"\n🚀 Starting Batch {batch_num + 1}/{total_batches}  →  Processing rows {batch_start} to {batch_end - 1}")

    # Progress bar
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_num + 1}", ncols=100):
        url = row["url"]
        fname = os.path.splitext(str(row["filename"]))[0] + ".png"
        out_path = os.path.join(OUTPUT_DIR, fname)

        # Skip if already exists
        if os.path.exists(out_path):
            continue

        try:
            driver.get(url)
            time.sleep(3)
            driver.save_screenshot(out_path)
            count += 1
        except (TimeoutException, WebDriverException) as e:
            errors.append((url, str(e)))
            continue

    # ===== Batch Summary =====
    print(f"\n✅ Batch {batch_num + 1}/{total_batches} completed!")
    print(f"   🖼️ Total screenshots so far: {count}")
    print(f"   🔁 Remaining batches: {total_batches - (batch_num + 1)}")

    # ===== Error Log =====
    if errors:
        log_file = os.path.join(OUTPUT_DIR, f"errors_batch_{batch_num + 1}.log")
        with open(log_file, "w", encoding="utf-8") as f:
            for url, err in errors:
                f.write(f"{url}\t{err}\n")
        print(f"⚠️ Errors logged in {log_file}")
        errors = []  # reset for next batch

    # ===== Cooldown =====
    if batch_num + 1 < total_batches:
        print("⏳ Cooling down before next batch...")
        for sec in range(5, 0, -1):
            print(f"   Next batch starts in {sec} sec...", end="\r")
            time.sleep(1)
        print()

driver.quit()

print(f"\n🎉 ALL DONE! Total screenshots captured: {count}")
print("📘 Check the screenshots folder and error logs for details.")
