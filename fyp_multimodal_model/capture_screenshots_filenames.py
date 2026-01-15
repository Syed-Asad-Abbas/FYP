# capture_screenshots_filenames.py
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------- CONFIGURATION ----------
OUTPUT_ROOT = "screenshots"   # <-- single folder for ALL screenshots
TIMEOUT = 15           # seconds before giving up on a page
HEADLESS = True        # set to False to see browser
MAX_PARALLEL = min(6, cpu_count())  # parallel threads
# -----------------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)


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
    """Captures screenshot of a single URL into a given path."""
    url, save_path = args

    # Skip if already done
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
    """
    Processes a batch file and captures screenshots.

    Batch CSV must have columns:
        - 'url'
        - 'FILENAME'  (same ID as in your main dataset CSV)
    """
    batch_name = os.path.splitext(os.path.basename(batch_file))[0]

    df = pd.read_csv(batch_file)
    df = df.dropna(subset=["url", "FILENAME"])

    urls = df["url"].tolist()
    filenames = df["FILENAME"].astype(str).tolist()

    # Build (url, save_path) tasks
    tasks = []
    for url, fname in zip(urls, filenames):
        stem = os.path.splitext(fname)[0]  # remove .txt, .html, etc.
        save_name = f"{stem}.png"
        save_path = os.path.join(OUTPUT_ROOT, save_name)
        tasks.append((url, save_path))

    if resume:
        tasks = [(u, sp) for (u, sp) in tasks if not os.path.exists(sp)]

    print(f"\n📦 Starting {batch_name} — {len(tasks)} URLs (Resume={resume})")

    failed_urls = []
    with Pool(processes=MAX_PARALLEL) as pool:
        for status, url in tqdm(pool.imap_unordered(capture_single, tasks), total=len(tasks)):
            if status in ("fail", "timeout"):
                failed_urls.append(url)

    failed_log_path = os.path.join(OUTPUT_ROOT, f"failed_{batch_name}.txt")
    if failed_urls:
        with open(failed_log_path, "a", encoding="utf-8") as f:
            for url in failed_urls:
                f.write(url + "\n")
        print(f"⚠️ Logged {len(failed_urls)} failed URLs to {failed_log_path}")

    print(f"✅ Completed {batch_name} — Screenshots saved in {OUTPUT_ROOT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel URL screenshot capture (FILENAME-based)")
    parser.add_argument("--batch", type=int, required=True, help="Batch number (1–7)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (skip existing screenshots)")
    args = parser.parse_args()

    batch_file = f"batches/batch_{args.batch}.csv"

    if not os.path.exists(batch_file):
        print(f"❌ Batch file not found: {batch_file}")
    else:
        process_batch(batch_file, resume=args.resume)
