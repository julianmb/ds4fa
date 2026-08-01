#!/usr/bin/env python3
# Persistent downloader for DeepSeek-V4-Flash-0731 GGUF

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10"
MODEL_FILE = "DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
DEST_FILE = GGUF_DIR / "DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf"

DSPARK_REPO = "sm54/deepseek-v4-flash-0731-gguf"
DSPARK_FILE = "DeepSeek-V4-Flash-0731-DSpark-support.gguf"
DEST_DSPARK = GGUF_DIR / "DeepSeek-V4-Flash-0731-DSpark-support.gguf"

LOG_FILE = GGUF_DIR / "download_0731.log"

def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def download_with_retry(repo_id: str, filename: str, target: Path):
    if target.exists():
        log(f"Target file {target.name} already exists. Skipping download.")
        return

    log(f"Starting download: {filename} from {repo_id}...")
    for attempt in range(1, 10):
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(GGUF_DIR)
            )
            downloaded = Path(downloaded_path)
            if downloaded != target and downloaded.exists():
                downloaded.rename(target)
            log(f"Successfully downloaded {target.name} ({target.stat().st_size / 1e9:.2f} GB)!")
            return
        except Exception as e:
            log(f"Attempt {attempt} failed with error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    log(f"Failed to download {filename} after 10 attempts.")

if __name__ == "__main__":
    log("=== DeepSeek-V4-Flash-0731 Persistent Downloader Started ===")
    download_with_retry(MODEL_REPO, MODEL_FILE, DEST_FILE)
    download_with_retry(DSPARK_REPO, DSPARK_FILE, DEST_DSPARK)

    # Link to ds4flash.gguf
    symlink = ROOT / "ds4flash.gguf"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(DEST_FILE)
    log(f"Linked {symlink} -> {DEST_FILE}")
    log("=== All downloads complete! ===")
