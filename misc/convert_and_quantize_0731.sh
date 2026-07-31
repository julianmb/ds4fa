#!/bin/bash
# Convert and quantize DeepSeek-V4-Flash-0731 to ROCmFP2 (Q2_0_ROCMFPX) for Strix Halo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROCMFPX_DIR="${ROOT}/../ROCmFPX"
HF_MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
OUT_DIR="${ROOT}/gguf"
mkdir -p "$OUT_DIR"

QUANT_OUT="${OUT_DIR}/DeepSeek-V4-Flash-0731-ROCMFP2-STRIX.gguf"

echo "=== 1. Checking ROCmFPX Tooling ==="
if [ ! -x "${ROCMFPX_DIR}/build-strix-rocmfp4/bin/llama-quantize" ]; then
    echo "Building ROCmFPX tools in ${ROCMFPX_DIR}..."
    cd "$ROCMFPX_DIR"
    env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh
fi

echo "=== 2. Downloading DeepSeek-V4-Flash-0731 from Hugging Face ==="
HF_CMD="$(command -v huggingface-cli || echo "${HOME}/.local/bin/huggingface-cli")"
if ! command -v "$HF_CMD" &>/dev/null; then
    echo "huggingface-cli not found; installing..."
    python3 -m pip install -q huggingface_hub --break-system-packages
fi

RAW_DIR="${OUT_DIR}/0731-hf"
mkdir -p "$RAW_DIR"

echo "Downloading safetensors shards..."
"$HF_CMD" download "$HF_MODEL" --local-dir "$RAW_DIR"

echo "=== 3. Converting Safetensors to Base GGUF ==="
BASE_GGUF="${OUT_DIR}/DeepSeek-V4-Flash-0731-BF16.gguf"
python3 "${ROCMFPX_DIR}/scripts/convert_deepseek_v4_modular.py" \
    "$RAW_DIR" \
    --outfile "$BASE_GGUF" \
    --deepseek4-include-mtp

echo "=== 4. Quantizing to Q2_0_ROCMFPX (ROCmFP2 2.50 bpw) ==="
"${ROCMFPX_DIR}/build-strix-rocmfp4/bin/llama-quantize" \
    "$BASE_GGUF" \
    "$QUANT_OUT" \
    Q2_0_ROCMFPX

echo "=== 5. Cleaning Up Base GGUF to Save Disk Space ==="
rm -f "$BASE_GGUF"

echo "Done! Quantized model written to: $QUANT_OUT"
echo "Run it with:"
echo "  ./run-deepseek-v4.sh --model $QUANT_OUT"
