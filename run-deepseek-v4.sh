#!/bin/bash
# Convenient runner for DeepSeek V4 Flash on Strix Halo (gfx1151).
# Enables the high-throughput 32 tok/s LocalMaxxing profile.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="${ROOT}/gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf"
if [ ! -f "$MODEL" ] && [ -f "${ROOT}/ds4flash.gguf" ]; then
    MODEL="${ROOT}/ds4flash.gguf"
fi
DRAFT="${ROOT}/gguf/draft/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf"

MODE="server"
PORT="8000"
CTX="8192"

usage() {
    cat <<EOF
DeepSeek V4 Flash Runner (Strix Halo 32 tok/s Profile)

Usage:
  ./run-deepseek-v4.sh [options]

Options:
  --server           Run ds4-server HTTP API (default)
  --cli              Run ds4 interactive CLI
  --model PATH       Path to target GGUF (default: gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf)
  --draft PATH       Path to DSpark draft GGUF (default: gguf/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf)
  --ctx N            Context window size (default: 8192)
  --port N           HTTP server port (default: 8000)
  --help             Show this message

EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --server) MODE="server" ;;
        --cli) MODE="cli" ;;
        --model) shift; MODEL="$1" ;;
        --draft) shift; DRAFT="$1" ;;
        --ctx) shift; CTX="$1" ;;
        --port) shift; PORT="$1" ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

if [ ! -f "$MODEL" ]; then
    echo "Target model not found at $MODEL" >&2
    echo "Download it with:" >&2
    echo "  ./download_model.sh rocmfpx-strix" >&2
    exit 1
fi

# Safety Step 1: Enforce 115 GB virtual memory cap to protect kernel/amdgpu driver
ulimit -v 123480320 2>/dev/null || true

# Safety Step 2: Check model fit before loading
echo "Checking model fit for $MODEL..."
if ! "${ROOT}/tests/rocm_model_fit" "$MODEL" >/dev/null 2>&1; then
    echo "Warning: Model size exceeds safe TTM/GTT limits!" >&2
fi

# Ensure performance profile and clock lock
if [ -w /sys/firmware/acpi/platform_profile ]; then
    echo performance | sudo tee /sys/firmware/acpi/platform_profile >/dev/null 2>&1 || true
fi
if command -v rocm-smi &>/dev/null; then
    sudo rocm-smi -d 0 --setperflevel high >/dev/null 2>&1 || true
fi

export DFLASH_DS4_SPEC=1
export DFLASH_DS4_FUSED_VERIFY=1
export DFLASH_DS4_SPEC_Q=4
export LUCE_MMVQ_MAX_NCOLS=4
export DS4_ROCM_STREAM_MODEL_CACHE_GB=48

if [ -f "$DRAFT" ]; then
    export DFLASH_DS4_DRAFT="$DRAFT"
else
    echo "Note: DSpark draft not found at $DRAFT (download with ./download_model.sh dspark-drafter for +26% decode speed)" >&2
fi

if [ "$MODE" = "server" ]; then
    echo "Starting ds4-server on http://127.0.0.1:${PORT}..."
    exec "${ROOT}/ds4-server" -m "$MODEL" \
        -c "$CTX" \
        --port "$PORT" \
        --ssd-streaming \
        --ssd-streaming-cache-experts 48GB
else
    echo "Starting ds4 interactive CLI..."
    exec "${ROOT}/ds4" -m "$MODEL" \
        -c "$CTX" \
        --ssd-streaming \
        --ssd-streaming-cache-experts 48GB
fi
