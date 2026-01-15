import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------- CONFIGURATION ----------
OUTPUT_ROOT = "screenshots"
TIMEOUT = 15           # seconds before giving up on a page
HEADLESS = True        # set to False to see browser
MAX_PARALLEL = min(6, cpu_count())  # parallel threads
# -----------------------------------

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
    """Captures screenshot of a single URL"""
    url, out_dir = args
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:180] + ".png"
    save_path = os.path.join(out_dir, filename)

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
    """Processes a batch file and captures screenshots"""
    batch_name = os.path.splitext(os.path.basename(batch_file))[0]
    out_dir = os.path.join(OUTPUT_ROOT, batch_name)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(batch_file)
    urls = df["url"].dropna().unique().tolist()

    # Load completed and failed logs if resuming
    completed_urls = set()
    failed_log_path = os.path.join(out_dir, f"failed_{batch_name}.txt")

    if resume:
        print(f"♻️ Resume mode ON — checking existing screenshots and failed logs")
        for file in os.listdir(out_dir):
            if file.endswith(".png"):
                completed_urls.add(file.split(".png")[0].replace("_", "/"))
        if os.path.exists(failed_log_path):
            with open(failed_log_path, "r") as f:
                completed_urls.update(line.strip() for line in f.readlines())

    # Filter remaining URLs
    urls_to_process = [u for u in urls if not any(x in u for x in completed_urls)]

    print(f"\n📦 Starting {batch_name} — {len(urls_to_process)} URLs (Resume={resume})")

    failed_urls = []
    with Pool(processes=MAX_PARALLEL) as pool:
        for status, url in tqdm(pool.imap_unordered(capture_single, [(url, out_dir) for url in urls_to_process]), total=len(urls_to_process)):
            if status in ("fail", "timeout"):
                failed_urls.append(url)

    # Log failed URLs
    if failed_urls:
        with open(failed_log_path, "a", encoding="utf-8") as f:
            for url in failed_urls:
                f.write(url + "\n")
        print(f"⚠️ Logged {len(failed_urls)} failed URLs to {failed_log_path}")

    print(f"✅ Completed {batch_name} — Screenshots saved in {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel URL screenshot capture (with resume + logging)")
    parser.add_argument("--batch", type=int, required=True, help="Batch number (1–7)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (skips completed URLs)")
    args = parser.parse_args()

    batch_file = f"batches/batch_{args.batch}.csv"

    if not os.path.exists(batch_file):
        print(f"❌ Batch file not found: {batch_file}")
    else:
        process_batch(batch_file, resume=args.resume)
