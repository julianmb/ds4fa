#!/bin/bash
# Multi-connection aria2c downloader for DeepSeek-V4-Flash-0731 GGUF

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_DIR="${ROOT}/gguf"
mkdir -p "$GGUF_DIR"

MODEL_URL="https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10/resolve/main/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
MODEL_FILE="${GGUF_DIR}/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf"

DSPARK_URL="https://huggingface.co/sm54/deepseek-v4-flash-0731-gguf/resolve/main/DeepSeek-V4-Flash-0731-DSpark-support.gguf"
DSPARK_FILE="${GGUF_DIR}/DeepSeek-V4-Flash-0731-DSpark-support.gguf"

LOG_FILE="${GGUF_DIR}/download_0731.log"

log() {
    timestamp="$(date '+[%Y-%m-%d %H:%M:%S]')"
    echo "$timestamp $1" | tee -a "$LOG_FILE"
}

log "=== Starting 16-Connection Parallel aria2c Download ==="

if [ ! -f "$MODEL_FILE" ]; then
    log "Downloading target model (86.7 GB) with 16 parallel connections..."
    aria2c -x 16 -s 16 -k 1M -j 16 -c \
      --file-allocation=none \
      --summary-interval=5 \
      --console-log-level=info \
      -dir="$GGUF_DIR" \
      -o "DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf" \
      "$MODEL_URL" 2>&1 | tee -a "$LOG_FILE"
else
    log "Target model already downloaded at $MODEL_FILE"
fi

if [ ! -f "$DSPARK_FILE" ]; then
    log "Downloading DSpark draft model (5.9 GB) with 16 parallel connections..."
    aria2c -x 16 -s 16 -k 1M -j 16 -c \
      --file-allocation=none \
      --summary-interval=5 \
      --console-log-level=info \
      -dir="$GGUF_DIR" \
      -o "DeepSeek-V4-Flash-0731-DSpark-support.gguf" \
      "$DSPARK_URL" 2>&1 | tee -a "$LOG_FILE"
else
    log "DSpark draft model already downloaded at $DSPARK_FILE"
fi

# Link to ds4flash.gguf
ln -sfn "$MODEL_FILE" "${ROOT}/ds4flash.gguf"
log "Linked ${ROOT}/ds4flash.gguf -> $MODEL_FILE"
log "=== All downloads complete! ==="
