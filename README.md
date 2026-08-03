<p align="center">
  <strong>DeepSeek V4 Flash on AMD Strix Halo</strong><br>
  <em>Tuned Strix Halo (gfx1151) fork of antirez/ds4 with ROCm 7.2.x diagnostics, SSD expert streaming, and DeepSeek-V4-Flash-0731 support</em>
</p>

<p align="center">
  <a href="https://github.com/julianmb/ds4fa/blob/main/LICENSE"><img src="https://img.shields.io/github/license/julianmb/ds4fa?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-AMD%20Strix%20Halo%20gfx1151-blue?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/ROCm-7.2.x-e95420?style=flat-square" alt="ROCm">
  <img src="https://img.shields.io/badge/Ubuntu-24.04%20HWE-orange?style=flat-square" alt="Ubuntu">
  <img src="https://img.shields.io/badge/model-DeepSeek%20V4%20Flash%200731-purple?style=flat-square" alt="Model">
</p>

---

A focused Strix Halo (`gfx1151`) fork of [antirez/ds4](https://github.com/antirez/ds4) optimized for **AMD Ryzen AI MAX+** systems (128 GB unified memory / Radeon 8060S). It adds ROCm 7.2.x hardware diagnostics, SSD expert streaming fixes, GPU kernel additions, and native support for the official **DeepSeek-V4-Flash-0731** release.

> **Target hardware:** 128 GB Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S) with Ubuntu 24.04 HWE kernel (6.18.4+).

---

## What We Improved Over Original `antirez/ds4`

The original `antirez/ds4` repository provided the initial ROCm backend for DeepSeek V4 Flash. This fork (`ds4fa`) adds the following core improvements for Strix Halo hardware:

### 1. Fixed SSD Expert Streaming Slab Allocator (`ds4.c`)
* **Problem in upstream**: Mixed-precision GGUFs (like 0731) have varying per-expert byte sizes across layers (e.g. Layer 26 uses `IQ2_S` gate/up experts at 82 B/256 vs `IQ2_XXS` at 66 B/256). Upstream pinned the SSD expert streaming cache slab size to the *first* routed layer's byte size (`7,077,888 B`), causing Layer 26 to be rejected from the slab pool and forced into pageable mapped model views. On ROCm systems without unified pageable migration enabled, accessing mapped views during GPU execution triggered MMU page faults.
* **Fix**: Updated `ds4_streaming_routed_expert_bytes` to set the slab size class to the **maximum** per-expert size across all 43 layers (`8,126,464 B`), and updated the uniformity check to `bytes <= base`.
* **Result**: **0 out of 43 layers off the slab size class!** 100% of routed expert layers are served directly from the SSD expert cache with zero MMU faults.

### 2. Added `Q4_K` Token Embedding GPU Kernels (`rocm/ds4_rocm_common.cuh` & `rocm/ds4_rocm_embedding_launch.cuh`)
* Added `embed_token_hc_q4_k_kernel` and `embed_tokens_hc_q4_k_kernel` to allow loading models with `Q4_K` token embedding weights directly on GPU without host fallback.

### 3. Added `Q4_K` & `BF16` Dense Matmul GPU Kernels (`rocm/ds4_rocm_matmul.cuh` & `ds4_rocm_compat.cu`)
* Added `matmul_q4_k_f32_sharedx_warp_rows_w32_kernel` and `matmul_q4_k_f32_batch_warp8_kernel` for `Q4_K` dense matrix multiplications.
* Added `matmul_bf16_ordered_chunks_kernel` for `BF16` dense matrix multiplications.

### 4. Built `MXFP4`-to-`Q2_K` Requantization Tool (`gguf-tools/requant_down_q2k.c`)
* Created an in-place GGUF requantization tool with bit-exact `MXFP4` (GGUF type 39) dequantization and correct interleaved element ordering to convert `IQ3_XXS` and `MXFP4` down experts into custom `Q2_K` (type 10) in under 2 minutes.

### 5. Hardware Diagnostics & TTM/GTT Auto-Sizing
* Added `make rocm-diag`, `make rocm-doctor`, `make rocm-smoke`, `make rocm-bench-quick`, and `DS4_ROCM_TTM_AUTORAISE=1` to detect non-`gfx1151` architectures, low TTM limits, and automatically size the GTT aperture to ~90% of visible RAM.

---

## DeepSeek-V4-Flash-0731 Model Architecture & Recipe

**DeepSeek-V4-Flash-0731** is a 284B parameter Mixture-of-Experts (MoE) model featuring:
* **43 routed layers**
* **256 total experts per layer** with **6 active experts** per token
* **4-stream hyper-connections** and **shared FFN experts**
* **DeepSeek reasoning mode** (`<think>...</think>` tags)

### Recommended Model Recipe

To run on `ds4fa` without modifying GPU kernels, the model tensors must match the engine's kernel expectations:

```
tekosML/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf (86.72 GB)
```

| Tensor Group | GGUF Tensor Name | Quantization Type | Status |
| :--- | :--- | :--- | :--- |
| **Attention Projections** | `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a/b` | `Q8_0` (type 8) | ✅ Matched |
| **Shared Experts** | `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` | `Q8_0` (type 8) | ✅ Matched |
| **Output Language Head** | `output.weight` | `Q8_0` (type 8) | ✅ Matched |
| **Token Embedding** | `token_embd.weight` | `F16` (type 1) | ✅ Matched |
| **Routed Gate & Up Experts**| `ffn_gate_exps`, `ffn_up_exps` | `IQ2_XXS` (type 16) | ✅ Matched |
| **Routed Down Experts** | `ffn_down_exps` | `Q2_K` (type 10) | ✅ Matched |

---

## Step-by-Step Installation & Setup

### 1. Clone the Repository

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
```

### 2. Run One-Shot System Setup

Run the automated setup script to configure GRUB parameters, udev rules, and tuned profiles:

```sh
bash misc/strix-halo-setup.sh
```

> **Note:** Reboot your machine after running `strix-halo-setup.sh` for the GRUB `amdgpu.gttsize` and `ttm.pages_limit` parameters to take effect.

### 3. Install System Dependencies

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev aria2

# Copy rocWMMA internal headers if needed by your ROCm version:
git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA
sudo cp -a /tmp/rocWMMA/library/include/rocwmma /usr/local/include/
```

### 4. Build for Strix Halo (`gfx1151`)

```sh
make strix-halo -j"$(nproc)"
```

This compiles `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent` specifically for the `gfx1151` (Radeon 8060S) architecture.

### 5. Verify Hardware & TTM Limit

```sh
make rocm-doctor
```

Confirm `gcnArchName=gfx1151` and that your TTM/GTT limit is near 120 GiB. If a warning is printed, raise the limit:

```sh
sudo amd-ttm --set-pages 8126464
```

---

## Downloading & Running the Model

### 1. Download the `tekosML` 0731 GGUF Model (**86.72 GB**)

Use 16-connection parallel `aria2c` for fast download (~110 MB/s):

```sh
cd /home/user/source/ds4-strix-halo
mkdir -p gguf
aria2c -x 16 -s 16 -k 1M -j 16 -c --file-allocation=none \
  -d gguf -o DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf \
  "https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10/resolve/main/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
```

### 2. Set Up the Symlink

```sh
ln -sf gguf/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf ds4flash.gguf
```

### 3. Run Interactive CLI Inference

```sh
DS4_ROCM_STREAM_MODEL_CACHE_GB=48 ./ds4 -m ds4flash.gguf -c 512 \
  --ssd-streaming --ssd-streaming-cache-experts 32GB \
  -p "What is the capital of France?" --think --tokens 60
```

### 4. Run OpenAI-Compatible HTTP Server (`ds4-server`)

```sh
DS4_ROCM_STREAM_MODEL_CACHE_GB=48 ./ds4-server -m ds4flash.gguf -c 8192 \
  --port 8000 --ssd-streaming --ssd-streaming-cache-experts 32GB
```

Send a completion request to `http://127.0.0.1:8000/v1/chat/completions`:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds4flash",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

---

## Troubleshooting & Environment Variables

| Variable | Description |
| :--- | :--- |
| `DS4_ROCM_STREAM_MODEL_CACHE_GB` | Set GPU device model cache size in GiB (default: `48` for 128 GB systems) |
| `DS4_ROCM_TTM_PAGES` | Override TTM/GTT mapping limit in 4 KiB pages |
| `DS4_ROCM_TTM_AUTORAISE` | Auto-raise TTM limit via `amd-ttm` on startup (requires root) |
| `ROCM_SMOKE_STRICT` | Fail smoke tests and CI on any hardware warning |

| Common Issue | Cause & Solution |
| :--- | :--- |
| **`raw KV batch store failed`** | VRAM exhausted when loading without `--ssd-streaming`. Set `DS4_ROCM_STREAM_MODEL_CACHE_GB=48` and pass `--ssd-streaming`. |
| **`pageable-memory access disabled`** | Raise TTM limit via `sudo amd-ttm --set-pages 8126464` or `DS4_ROCM_TTM_AUTORAISE=1`. |
| **Garbage output text** | Ensure you pass `--think` for reasoning mode or set `temperature=0.6` in API requests so DeepSeek V4 reasoning formatting is followed. |

---

## Documentation

| Document | Description |
| :--- | :--- |
| [STRIXHALO.md](STRIXHALO.md) | ROCm install, GRUB params, TTM priority, hardware notes |
| [FORK_NOTES.md](FORK_NOTES.md) | Audit of what was retained/rejected from upstream |

---

## License

This project is a fork of [antirez/ds4](https://github.com/antirez/ds4). See `LICENSE` for terms.
