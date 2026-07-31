#!/bin/bash
# Convert and quantize DeepSeek-V4-Flash-0731 to ROCmFP2 (Q2_0_ROCMFPX) for Strix Halo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROCMFPX_DIR="${ROOT}/../ROCmFPX"
MXFP4_REPO="bartowski/DeepSeek-V4-Flash-0731-GGUF"
OUT_DIR="${ROOT}/gguf"
mkdir -p "$OUT_DIR"

QUANT_OUT="${OUT_DIR}/DeepSeek-V4-Flash-0731-ROCMFP2-STRIX.gguf"

echo "=== 1. Checking ROCmFPX Tooling ==="
if [ ! -x "${ROCMFPX_DIR}/build-strix-rocmfp4/bin/llama-quantize" ]; then
    echo "Building ROCmFPX tools in ${ROCMFPX_DIR}..."
    cd "$ROCMFPX_DIR"
    env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
fi

echo "=== 2. Downloading DeepSeek-V4-Flash-0731 MXFP4 GGUF Shards (~156 GB) ==="
HF_CMD="$(command -v huggingface-cli || echo "${HOME}/.local/bin/huggingface-cli")"
if ! command -v "$HF_CMD" &>/dev/null; then
    echo "huggingface-cli not found; installing..."
    python3 -m pip install -q huggingface_hub --break-system-packages
fi

RAW_DIR="${OUT_DIR}/0731-mxfp4"
mkdir -p "$RAW_DIR"

echo "Downloading MXFP4 GGUF shards..."
"$HF_CMD" download "$MXFP4_REPO" --include "DeepSeek-V4-Flash-0731-MXFP4/*" --local-dir "$RAW_DIR"

FIRST_SHARD="${RAW_DIR}/DeepSeek-V4-Flash-0731-MXFP4/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"

echo "=== 3. Requantizing MXFP4 to Q2_0_ROCMFPX (ROCmFP2 2.50 bpw ~98 GB) ==="
"${ROCMFPX_DIR}/build-strix-rocmfp4/bin/llama-quantize" \
    --allow-requantize \
    "$FIRST_SHARD" \
    "$QUANT_OUT" \
    Q2_0_ROCMFPX

echo "=== 4. Cleaning Up MXFP4 Source Shards to Save Disk Space ==="
rm -rf "$RAW_DIR"

echo "Done! Quantized model written to: $QUANT_OUT"
echo "Run it with:"
echo "  ./run-deepseek-v4.sh --model $QUANT_OUT"
