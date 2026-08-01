#!/usr/bin/env python3
# Download DeepSeek-V4-Flash-0731 UD-IQ2_XXS GGUF from Unsloth (~90.8 GB)

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
FILES = [
    "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf",
    "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf",
    "UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00003-of-00003.gguf",
]

LOG_FILE = GGUF_DIR / "download_0731.log"

def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

if __name__ == "__main__":
    log("=== DeepSeek-V4-Flash-0731 Unsloth UD-IQ2_XXS Downloader Started ===")
    
    token = None
    token_file = Path.home() / ".cache/huggingface/token"
    if token_file.exists():
        token = token_file.read_text().strip()

    downloaded_files = []
    for filename in FILES:
        target_local = GGUF_DIR / Path(filename).name
        if target_local.exists():
            log(f"File {target_local.name} already exists. Skipping download.")
            downloaded_files.append(target_local)
            continue

        log(f"Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(GGUF_DIR),
                token=token
            )
            downloaded = Path(downloaded_path)
            log(f"Downloaded {downloaded.name} ({downloaded.stat().st_size / 1e9:.2f} GB)!")
            downloaded_files.append(downloaded)
        except Exception as e:
            log(f"Download failed for {filename}: {e}")
            sys.exit(1)

    # Link first shard to ds4flash.gguf
    first_shard = GGUF_DIR / Path(FILES[0]).name
    symlink = ROOT / "ds4flash.gguf"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(first_shard)
    log(f"Linked {symlink} -> {first_shard}")
    log("=== All downloads complete! ===")
