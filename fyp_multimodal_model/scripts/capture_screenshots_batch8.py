import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------- CONFIG ----------
OUTPUT_ROOT = "screenshots"
TIMEOUT = 15
HEADLESS = True
MAX_PARALLEL = min(6, cpu_count())
# -----------------------------

def setup_driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-gpu")
    return webdriver.Chrome(options=options)

def capture_single(args):
    url, out_dir = args
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:180] + ".png"
    save_path = os.path.join(out_dir, filename)

    if os.path.exists(save_path):
        return ("skip", url)

    driver = None
    try:
        driver = setup_driver()
        driver.set_page_load_timeout(TIMEOUT)
        driver.get(url)
        time.sleep(2)
        driver.save_screenshot(save_path)
        return ("success", url)
    except TimeoutException:
        return ("timeout", url)
    except WebDriverException:
        return ("fail", url)
    finally:
        if driver:
            driver.quit()

def process_batch(batch_file, resume=False):
    batch_name = os.path.splitext(os.path.basename(batch_file))[0]
    out_dir = os.path.join(OUTPUT_ROOT, batch_name, "valuable_phish")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(batch_file)

    # Automatically find URL column
    url_col = next((c for c in df.columns if c.strip().lower() == "url"), None)
    if not url_col:
        raise KeyError(f"No column named 'url' found. Available: {list(df.columns)}")

    urls = df[url_col].dropna().unique().tolist()

    failed_log_path = os.path.join(out_dir, f"failed_{batch_name}.txt")

    # Resume logic
    completed = set()
    if resume:
        print(f"♻️ Resume mode ON — scanning {out_dir}")
        for file in os.listdir(out_dir):
            if file.endswith(".png"):
                completed.add(file.split(".png")[0])
        if os.path.exists(failed_log_path):
            with open(failed_log_path, "r", encoding="utf-8") as f:
                completed.update(line.strip() for line in f.readlines())

    urls_to_process = [u for u in urls if not any(x in u for x in completed)]

    print(f"\n📦 Starting {batch_name} — {len(urls_to_process)} URLs (resume={resume})")

    failed_urls = []
    with Pool(processes=MAX_PARALLEL) as pool:
        for status, url in tqdm(pool.imap_unordered(capture_single, [(u, out_dir) for u in urls_to_process]), total=len(urls_to_process)):
            if status in ("fail", "timeout"):
                failed_urls.append(url)

    # Log failed URLs
    if failed_urls:
        with open(failed_log_path, "a", encoding="utf-8") as f:
            for u in failed_urls:
                f.write(u + "\n")
        print(f"⚠️ Logged {len(failed_urls)} failed URLs to {failed_log_path}")

    print(f"✅ Completed {batch_name} — all screenshots saved in {out_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screenshot valuable phishing URLs with resume support")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume mode (skip completed)")
    args = parser.parse_args()

    batch_file = f"batches/batch_{args.batch}.csv"
    if not os.path.exists(batch_file):
        print(f"❌ File not found: {batch_file}")
    else:
        process_batch(batch_file, resume=args.resume)
