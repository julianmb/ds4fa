---
license: mit
tags:
  - deepseek
  - moe
  - amd
  - strix-halo
  - rocm
  - gguf
  - rocmfpx
pipeline_tag: text-generation
---

<p align="center">
  <strong>⚡ DeepSeek V4 Flash on AMD Strix Halo</strong><br>
  <em>Up to 32 tok/s decode — a tuned gfx1151 fork of antirez/ds4 with ROCmFPX tooling, SSD expert streaming, and native DeepSeek-V4-Flash-0731 support</em>
</p>

<p align="center">
  <a href="https://github.com/julianmb/ds4fa/blob/main/LICENSE"><img src="https://img.shields.io/github/license/julianmb/ds4fa?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-AMD%20Strix%20Halo%20gfx1151-blue?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/ROCm-7.2.x-e95420?style=flat-square" alt="ROCm">
  <img src="https://img.shields.io/badge/Ubuntu-24.04%20HWE-orange?style=flat-square" alt="Ubuntu">
  <img src="https://img.shields.io/badge/model-DeepSeek%20V4%20Flash%200731-purple?style=flat-square" alt="Model">
</p>

<p align="center">
  <strong>284B MoE · 43 routed layers · 256 experts/layer · 128 GB unified memory · 100% local</strong>
</p>

---

## 🚀 The Short Version

Run a **284B-parameter** model at **up to 32 tok/s** on a single AMD APU — no cloud, no discrete GPU, no per-token bill. This fork turns a Strix Halo mini-PC (Ryzen AI MAX+ 395 / Radeon 8060S) into a local DeepSeek V4 Flash inference machine.

That number is not a fantasy. It's the same hardware configuration that took the **LocalMaxxing** DeepSeek V4 Flash record:

| Engine / Quant | Decode (tok/s) |
| :--- | ---: |
| DwarfStar · Q2_K | 15.60 |
| HipFire · MQ2 + MTP | 18.99 |
| **This stack · ROCmFPX + DSpark** | **32.00** |

That's **2.05× faster** than the previous unified-memory leader and **68.5% ahead** of the runner-up — measured on the exact silicon this repo targets.

---

## 🧠 What Makes ROCmFPX So Good

### The format

**ROCmFPX is not one quantization format — it's a family of block formats designed for AMD ROCm/HIP silicon.** Each block holds **32 weights** as packed low-bit codes plus one or two tiny scales, and the GPU kernels are written for exactly that byte layout:

| Variant | Block size | Bits / weight | Typical use |
| :--- | ---: | ---: | --- |
| **ROCmFP2** | 10 B / 32 | **2.50** | Routed expert gate & up matrices (the biggest tensors) |
| **ROCmFP3** | — | **3.50** | Expert down projections |
| **ROCmFP4** | — | **4.25** | Dense / sensitive projections |

A Strix-specific mixed-precision recipe combines them: the enormous routed-expert gate/up matrices at ROCmFP2, expert down at ROCmFP3, and dense projections at ROCmFP4+. With an importance matrix during quantization, the full DeepSeek V4 Flash target lands at **~2.88 bits per parameter** in a **102.3 GB** file — just under 95.3 GiB, so the whole model fits in Strix Halo's 128 GB unified pool with room to spare.

### Why it's fast

The format is inseparable from the kernel that eats it:

1. **Register-resident codebooks.** Kernels expand the tiny packed codebooks **in GPU registers** using AMD's byte-permute instruction (`v_perm_b32`) instead of doing a separate gather from memory. No indirection, no extra loads.
2. **Integer dot products.** Packed blocks feed `dp4a`-style integer dot products directly — the fastest path on RDNA3/3.5 — instead of dequantizing to floats first.
3. **Designed as one path.** The file layout and the HIP kernel are specified together, so decode is a straight memory-traffic-bound stream of weights through fixed-purpose hardware.

At batch one, every generated token streams the active experts across all 43 layers, so **decode is memory-traffic-limited** — and ROCmFPX is built to maximize useful bytes per load.

### The engine side

ROCmFPX only handles the weight traffic. The full 32 tok/s profile also uses:

- **DSpark draft + fused q=4 verification** — a small 3-layer drafter proposes up to 3 tokens; the 284B target verifies 4 positions in one fused HIP graph pass (26.4% faster than autoregressive).
- **Weight reuse across verification columns** — each packed dense weight is decoded once and applied to all 4 verify columns (+2.1–2.3%).
- **Indexed sparse prefill** — ~250 tok/s on 8K prompts via DeepSeek's learned indexer.

---

## 🛠️ What This Fork Improves Over Upstream `antirez/ds4`

The upstream repo brought the initial DeepSeek V4 ROCm backend. This fork makes it actually run *well* on Strix Halo:

### 1. Fixed the SSD Expert Streaming Slab Allocator (`ds4.c`)
Mixed-precision GGUFs have layers with different per-expert sizes (e.g. 0731's Layer 26 uses `IQ2_S` gate/up at 82 B/256 vs `IQ2_XXS` at 66 B/256). Upstream pinned the streaming cache to the *first* layer's size, which bounced Layer 26 into pageable mapped views — an MMU fault on ROCm. Now the slab is sized to the **maximum** across all 43 layers, so **0/43 layers fall off the fast path**.

### 2. New GPU kernels (`rocm/`)
- **`Q4_K` token embedding** kernels (`embed_token_hc_q4_k_kernel`, `embed_tokens_hc_q4_k_kernel`)
- **`Q4_K` dense matmul** kernels (`matmul_q4_k_f32_sharedx_warp_rows_w32_kernel`, `matmul_q4_k_f32_batch_warp8_kernel`)
- **`BF16` dense matmul** kernel (`matmul_bf16_ordered_chunks_kernel`)

### 3. `MXFP4` → `Q2_K` requantization tool (`gguf-tools/requant_down_q2k.c`)
In-place GGUF converter with bit-exact `MXFP4` dequantization and correct element interleaving — converts `IQ3_XXS`/`MXFP4` down experts to custom `Q2_K` in under 2 minutes.

### 4. ROCm 7.2.x diagnostics & TTM auto-sizing
`make rocm-diag`, `make rocm-doctor`, `make rocm-smoke`, `make rocm-bench-quick`, and `DS4_ROCM_TTM_AUTORAISE=1` — detect bad configs and fix them automatically.

---

## 📦 Two Ways to Run DeepSeek-V4-Flash-0731

### Route A — High-throughput ROCmFPX (~32 tok/s)

The ROCmFPX/ROCmFP2 target (`Q2_0_ROCMFPX`, ~98–102 GB) plus the DSpark drafter. This is the LocalMaxxing record path.

```sh
./download_model.sh rocmfpx-strix    # 102.3 GB ROCmFP2-STRIX target
./download_model.sh dspark-drafter   # 11.3 GB DSpark draft
./run-deepseek-v4.sh                 # 32 tok/s high-throughput server
```

### Route B — Native ds4fa quant recipe (~13 tok/s, zero kernel gaps)

Uses standard GGML quants that the ds4fa engine supports natively:

- **Model**: `tekosML/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf` (**86.72 GB**)
- **HF model page**: [tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10](https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10)

| Tensor group | Type | Status |
| :--- | :--- | :--- |
| Attention projections | `Q8_0` | ✅ |
| Shared experts | `Q8_0` | ✅ |
| Output language head | `Q8_0` | ✅ |
| Token embedding | `F16` | ✅ |
| Routed gate/up experts | `IQ2_XXS` | ✅ |
| Routed down experts | `Q2_K` | ✅ |

---

## 🔧 Install & Run (Self-Hosted)

### 1. Clone & one-shot setup

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
bash misc/strix-halo-setup.sh     # GRUB gttsize/pages_limit, udev, tuned profile — reboot after
```

### 2. Install the toolchain

```sh
sudo apt-get update
sudo apt-get install -y hipcc rocminfo rocm-smi libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev aria2

git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA
sudo cp -a /tmp/rocWMMA/library/include/rocwmma /usr/local/include/
```

### 3. Build for gfx1151

```sh
make strix-halo -j"$(nproc)"
make rocm-doctor       # verify TTM/GTT limit; warns + suggests the exact amd-ttm fix
```

### 4. Download the 0731 model (86.72 GB, ~110 MB/s with 16 connections)

Models live under `gguf/` in organized subdirectories (see [gguf/README.md](gguf/README.md)):

```
gguf/
├── deepseek-v4-flash-0731/          # native route target
└── draft/                           # speculative-draft models
```

```sh
aria2c -x 16 -s 16 -k 1M -j 16 -c --file-allocation=none \
  -d gguf/deepseek-v4-flash-0731 -o DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf \
  "https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10/resolve/main/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
ln -sf gguf/deepseek-v4-flash-0731/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf ds4flash.gguf
```

### 5. Chat

**Interactive CLI:**

```sh
DS4_ROCM_STREAM_MODEL_CACHE_GB=48 ./ds4 -m ds4flash.gguf -c 512 \
  --ssd-streaming --ssd-streaming-cache-experts 32GB \
  -p "What is the capital of France?" --think --tokens 60
```

**OpenAI-compatible server:**

```sh
DS4_ROCM_STREAM_MODEL_CACHE_GB=48 ./ds4-server -m ds4flash.gguf -c 8192 \
  --port 8000 --ssd-streaming --ssd-streaming-cache-experts 32GB
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ds4flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

---

## 📊 Performance Reference

| Stage | Measured |
| :--- | ---: |
| Decode (ROCmFPX + DSpark q=4, top-k 4) | **32.0 tok/s** |
| Decode (ROCmFPX autoregressive) | 25.3 tok/s |
| Sparse prefill (indexed, 8K) | **~250 tok/s** |
| Exact prefill (short prompts) | 22.5–23 tok/s |

*Measured July 2026 on Ryzen AI MAX+ 395, ROCm 7.2.4, Radeon `high` (2.9 GHz), context 8,192, temp 0.*

---

## ❓ Troubleshooting

| Problem | Fix |
| :--- | :--- |
| `raw KV batch store failed` | Set `DS4_ROCM_STREAM_MODEL_CACHE_GB=48` and pass `--ssd-streaming` |
| `pageable-memory access disabled` | `sudo amd-ttm --set-pages 8126464` or `DS4_ROCM_TTM_AUTORAISE=1` |
| Garbage output | Use `--think` (or `temperature: 0.6`) so DeepSeek reasoning format is respected |
| `missing gfx1151` | Build with `make strix-halo`; ensure ROCm 7.2.x |
| rocWMMA header errors | Install the matching `rocwmma` tree (step 2) |

---

## 📚 Documentation

| Document | Description |
| :--- | :--- |
| [STRIXHALO.md](STRIXHALO.md) | ROCm install, GRUB params, TTM priority, hardware notes |
| [FORK_NOTES.md](FORK_NOTES.md) | Audit of what was retained/rejected from upstream |

## 🤝 Acknowledgements

DeepSeek V4 Flash · [antirez/ds4](https://github.com/antirez/ds4) · [Lucebox / ROCmFPX](https://github.com/Luce-Org/lucebox) · [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp) · [tekosML](https://huggingface.co/tekosML)

*Local AI should be the default, not a privilege.*
