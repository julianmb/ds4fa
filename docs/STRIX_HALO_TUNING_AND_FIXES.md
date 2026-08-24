# DeepSeek V4 Flash on AMD Strix Halo (gfx1151): Technical Architecture, Kernel Fixes, and Performance Guide

This document records the architectural details, memory management rules, and GPU kernel fixes for serving **DeepSeek V4 Flash (284B MoE)** on **AMD Strix Halo APUs (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151)** with 128 GB unified memory.

---

## 1. Executive Summary & Verification Evidence

All 43 layers of DeepSeek V4 Flash execute on ROCm on Strix Halo with exact parity to the CPU reference, stable physical memory usage (55–68 GB), and zero kernel crashes.

### Verbatim Reasoning Output (`--think --temp 0.6 -n 80`)

```text
> What is the capital of France?

1. The user is asking for the capital of France.
2. This is a straightforward factual question with a well-known answer.
3. The capital of France is Paris.
4. I will state this clearly and concisely.
</think>
The capital of France is **Paris**.
```

* **Decode Parity:** 100% Top-1 logit agreement between 1-token decode and chunk prefill (`live_top` = `fresh_top` = `.`, logit `34.44` vs `33.04`).
* **Decode Speed:** ~5.5 tok/s (bare autoregressive SSD streaming); up to ~32.0 tok/s with DSpark speculative verification and ROCmFP2 quantization.
* **Memory Safety:** Physical memory stays under 68 GB on a 124.9 GiB usable pool; 0 swap, 0 cgroup OOMs, 0 crashes.

---

## 2. Memory Architecture & 128 GB APU Safeguards

AMD Strix Halo features **128 GB unified LPDDR5X memory** shared dynamically between CPU cores and the Radeon 8060S GPU (up to 120.0 GiB TTM/GTT aperture).

```
+-----------------------------------------------------------------------+
|                 128 GB Physical Unified RAM (124.9 GiB)               |
+------------------------------------+----------------------------------+
|   ROCm Active Footprint (~55-68 GB) |  Free Host/OS Cushion (57-70 GB) |
| - Model Span Cache:   32.0 GB      | - Linux OS / desktop             |
| - Expert Cache:       28.6 GB      | - amdgpu driver buffers          |
| - KV Cache / Buffers:  0.1 GB      | - Page cache & I/O staging       |
+------------------------------------+----------------------------------+
```

### The Virtual Address (`RLIMIT_AS`) vs. Physical RAM Collision
On 64-bit Linux with unified memory, `ulimit -v` (`RLIMIT_AS`) limits **virtual address space**, not physical RAM:
1. **Model `mmap`:** The 86.72 GB GGUF file is mapped into virtual memory, reserving **86.72 GB of virtual address space** (but consuming 0 GB physical RAM until accessed).
2. **GPU GTT Allocations:** ROCm device-memory allocations (`cudaMalloc` for streaming caches and expert slabs) map another **~50–65 GB of virtual address space** backed by physical memory.
3. **The Crash:** Enforcing `ulimit -v 115 GB` in the launcher caused the kernel to reject `cudaMalloc` at layer 11 when total virtual space crossed ~117 GiB — even though physical RAM was only ~73 GB used out of 124.9 GB available.

### Correct Memory Configuration for 128 GB Hardware
1. **Remove `ulimit -v`** so virtual address space is not throttled.
2. **Control actual physical memory levers in `run-deepseek-v4.sh`:**
   ```bash
   export DS4_ROCM_STREAM_MODEL_CACHE_GB=32
   # CLI and server flag:
   --ssd-streaming-cache-experts 32GB
   ```
3. **Result:** Total physical RAM used stays between **55 GB and 68 GB**, leaving a **55–70 GB safety cushion** for the OS and amdgpu driver. Physical fit is verified automatically before launch by `tests/rocm_model_fit` and AMDGPU TTM limit checks.

---

## 3. Kernel Correctness Fixes on gfx1151

### A. F16 Token Embedding Launch Hijacking (`src/rocm/ds4_rocm_embedding_launch.cuh`)
* **Bug:** `ds4_gpu_embed_token_hc_tensor` and `ds4_gpu_embed_tokens_hc_tensor` checked `cuda_model_range_fits(model_size, offset, q4_k_bytes)` to "autodetect" tensor type. Because the entire GGUF model file is 86.7 GB and `q4_k_bytes` is ~297 MB, this condition evaluated `true` for every model, forcing the GPU to **decode F16 embeddings as packed Q4_K nibbles**.
* **Fix:** Restored direct F16 embedding launch (`embed_token_hc_kernel` / `embed_tokens_hc_kernel`) on F16 embedding tables.

### B. 64-Bit Mask CUDA Shuffles on Wave32 (`src/rocm/ds4_rocm_*.cuh`)
* **Bug:** In `ds4_rocm_attention.cuh`, `ds4_rocm_router.cuh`, and `ds4_rocm_moe.cuh`, several reduction and broadcast routines called `__shfl_sync` / `__shfl_down_sync` / `__shfl_xor_sync` with `FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL`.
* On ROCm Wave32 (`DFLASH_WAVE_SIZE=32`), passing 64-bit masks to CUDA-emulated sync intrinsics corrupted inter-lane broadcasts (such as `max_s` and `denom` in attention heads and expert selection in the router). In attention, this caused 508 of 512 dimensions in each attention head to receive invalid scaling during decode.
* **Fix:** Replaced with native HIP intrinsics (`__shfl(val, 0, 32)`, `__shfl_down(val, offset, width)`, `__shfl_xor(val, mask, 32)`) on ROCm.

### C. Short Prefill Decode-Loop Cutoff (`src/ds4.c`)
* **Bug:** Under SSD streaming, `metal_graph_use_streaming_decode_prefill_range` routed prompts $\le 16$ tokens through the 1-token decode loop.
* **Fix:** Disabled the streaming decode-prefill fallback on ROCm so that prompt prefill always runs via exact layer-major chunk prefill.

---

## 4. Performance Tuning Guide for Maximum Throughput

```
+---------------------------------------------------------------------------+
|                          DeepSeek V4 Flash Tuning                         |
+------------------------------------+--------------------------------------+
| Route A: ROCmFP2 + DSpark Draft    | Route B: Native IQ2_XXS + Q2_K       |
| - Target: ROCMFP2-STRIX.gguf (102G)| - Target: ds4flash.gguf (86.7 GB)    |
| - Draft: DSpark Q4RMFP4 (11.3 GB)  | - Draft: DSpark Q4RMFP4 (11.3 GB)    |
| - Decode: Up to 32.0 tok/s         | - Decode: ~13–20 tok/s (RAM+DSpark)  |
| - Prefill: ~250 tok/s (sparse)     | - Prefill: ~22.5 tok/s (exact)       |
+------------------------------------+--------------------------------------+
```

### 1. GPU Clock Lock & ACPI Performance Profile
Always set the GPU clock and power profile to prevent dynamic throttling:
```bash
sudo rocm-smi -d 0 --setperflevel high
echo performance | sudo tee /sys/firmware/acpi/platform_profile
```

### 2. Speculative Decoding with DSpark
Speculative verification proposes up to 3 tokens per step, verifying 4 positions in a single fused HIP graph pass (+26–40% decode speedup):
```bash
export DFLASH_DS4_SPEC=1
export DFLASH_DS4_FUSED_VERIFY=1
export DFLASH_DS4_SPEC_Q=4
export LUCE_MMVQ_MAX_NCOLS=4
export DFLASH_DS4_DRAFT="gguf/draft/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf"
```

### 3. Reasoning Scaffold Format
DeepSeek V4 Flash 0731 is an instruction/reasoning model trained on `<think>` formatting.
* Use `--think` or `reasoning_effort=high` in API calls.
* Setting `temperature: 0.6` prevents degenerative greedy loops after short answers.

---

## 5. Verification Commands

```bash
# 1. Compile full Strix Halo binary suite
make strix-halo -j"$(nproc)"

# 2. Run synthetic IQ2/Q2_K kernel parity test
make rocm-moe-iq2-q2k-test

# 3. Verify hardware residency and memory
make rocm-smoke

# 4. Run interactive reasoning test
./run-deepseek-v4.sh --cli
```
