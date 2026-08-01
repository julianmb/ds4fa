#!/bin/bash
# Download and setup DeepSeek-V4-Flash-0731 (July 31 Official Release) for Strix Halo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/gguf"
mkdir -p "$OUT_DIR"

MODEL_REPO="tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10"
MODEL_FILE="DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
DEST_FILE="${OUT_DIR}/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf"

DSPARK_REPO="sm54/deepseek-v4-flash-0731-gguf"
DSPARK_FILE="DeepSeek-V4-Flash-0731-DSpark-support.gguf"
DEST_DSPARK="${OUT_DIR}/DeepSeek-V4-Flash-0731-DSpark-support.gguf"

HF_CMD="$(command -v hf || command -v huggingface-cli || echo "${HOME}/.local/bin/hf")"

echo "=== 1. Downloading DeepSeek-V4-Flash-0731 GGUF (~86.7 GB) ==="
if [ ! -f "$DEST_FILE" ]; then
    echo "Downloading target GGUF from $MODEL_REPO..."
    "$HF_CMD" download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$OUT_DIR"
    mv "${OUT_DIR}/${MODEL_FILE}" "$DEST_FILE"
else
    echo "Target GGUF already present at $DEST_FILE"
fi

echo "=== 2. Downloading DeepSeek-V4-Flash-0731 DSpark Support GGUF (~5.9 GB) ==="
if [ ! -f "$DEST_DSPARK" ]; then
    echo "Downloading DSpark draft GGUF from $DSPARK_REPO..."
    "$HF_CMD" download "$DSPARK_REPO" "$DSPARK_FILE" --local-dir "$OUT_DIR"
else
    echo "DSpark draft GGUF already present at $DEST_DSPARK"
fi

echo "=== 3. Linking to ds4flash.gguf ==="
ln -sfn "$DEST_FILE" "${ROOT}/ds4flash.gguf"

echo "Done! DeepSeek-V4-Flash-0731 downloaded successfully."
echo "Target Model: $DEST_FILE (86.7 GB)"
echo "Draft Model:  $DEST_DSPARK (5.9 GB)"
echo ""
echo "Run it with 32 tok/s LocalMaxxing speed:"
echo "  ./run-deepseek-v4.sh --model $DEST_FILE --draft $DEST_DSPARK"
