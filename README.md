# ds4fa: DeepSeek V4 Flash Inference Engine

`ds4fa` (ds4 for AMD) is a high-performance fork of Antirez's original [ds4](https://github.com/antirez/ds4) engine, adapted specifically for **AMD ROCm** and the **Strix Halo (Ryzen AI Max, gfx1151)** APU architecture. While it is highly tuned for Strix Halo, it should work on other modern AMD GPUs (like RDNA 3/CDNA 3) via ROCm, though this remains untested.

You can try the DeepSeek V4 Flash experience powered by this architecture online at [au.privchat.ai](https://au.privchat.ai).

## Why DeepSeek v4 Flash?

1. **Efficiency**: Faster inference due to fewer active parameters (MoE).
2. **Dynamic Reasoning**: Proportional thinking section length based on problem complexity.
3. **Massive Context**: Features a context window of **1 million tokens**.
4. **Knowledge Depth**: Superior performance in specific knowledge domains due to its 284B total parameters.
5. **Compressed KV Cache**: Incredible compression allowing for **on-disk KV cache persistence**.
6. **Optimized Quantization**: Works exceptionally well with specialized 2-bit quantization (routed experts only).

## Architecture & Backends

### 1. Apple Silicon (Metal)
The main path for macOS users. It exploits unified memory and Metal graph
executors to achieve massive performance on MacBooks and Mac Studios.

### 2. AMD Strix Halo (ROCm/HIP)
A specialized port for AMD's unified memory APU architecture (`gfx1151`). It
achieves 100% feature parity with the Metal backend, fully eradicating all C-level 
stubs to map the entire DeepSeek generation logic to the GPU. It achieves near-native 
performance on Linux by leveraging:
*   **Zero-Copy APU Memory**: Direct mapping of host RAM to the iGPU via `hipHostMallocMapped`.
*   **Hardware Matrix Cores (WMMA)**: MatMul acceleration using RDNA 3.5 AI accelerators.
*   **Single-Cycle 2-bit Decoding**: Bitfield extraction intrinsics (`ubfe`) for `IQ2_XXS` weights.
*   **HIP Graph Capture**: Elimination of CPU kernel dispatch overhead.
*   **Stream Ordered Memory (ROCm 7.3 SOMA)**: Fully asynchronous `hipMallocAsync` workflow.

### 3. NPU Hybrid Architecture (XDNA 2)
A foundational framework for **Multi-Token Prediction (MTP)** speculative decoding using the Ryzen AI NPU. The engine is architected to offload DeepSeek's native sequential MTP draft modules to the XDNA 2 NPU via the XRT API.
*   **Zero-Copy SVA**: Uses Shared Virtual Addressing so the CPU, 40 CU iGPU, and XDNA 2 NPU all read/write the exact same token memory buffers without PCIe transfers.
*   **Draft & Verify**: The NPU predicts multiple future tokens asynchronously, while the massive iGPU performs a single parallel batch verification to drastically increase tokens/second.

## Performance Tuning (Strix Halo)

The `ds4fa` engine includes several Strix Halo-specific optimizations enabled by 
default when using the ROCm backend:

*   **Memory Pre-warming**: On startup, the engine launches a dedicated GPU kernel to 
    touch every page of the 80GB model map. This forces physical memory pinning 
    and warms the Ryzen Infinity Fabric cache, eliminating first-token stutters.
*   **Cache Hints**: Utilizes `hipMemAdviseSetReadMostly` to signal the Strix Halo 
    memory controller to prioritize model weights in the System Level Cache (SLC).
*   **Wave32 WMMA**: All MatMul and Attention kernels are compiled for Wave32 mode 
    to maximize CU occupancy on RDNA 3.5.

### Advanced Improvements
For developers looking to push the engine further:
1.  **Fused MoE Routing**: Fusing the `ds4_hip_router_select` and `ds4_hip_routed_moe` 
    kernels to keep routing weights in L2.
2.  **register-level ASM**: Hand-tuning the `IQ2_XXS` bit-extraction using raw 
    GCN/RDNA assembly for even lower latency.
3.  **io_uring Support**: Implementing asynchronous disk I/O for the `--kv-disk-dir` 
    path to overlap context swapping with active generation.

---

## Requirements

### macOS (Metal)
*   **Hardware**: Apple Silicon (M1/M2/M3/M4 Max or Ultra).
*   **Memory**: 128GB (for `q2` model) or 512GB (for `q4` model).
*   **OS**: macOS 14.x or newer.

### Linux (AMD ROCm)
*   **Hardware**: AMD Ryzen AI Max Series (**Strix Halo**, `gfx1151`).
*   **Memory**: 64GB - 128GB of LPDDR5X.
*   **ROCm**: **ROCm 7.1+** (7.2.2/7.3 preferred).
*   **Kernel**: Linux **6.18+** or **7.x** (7.1+ for NPU support).
*   **NPU Software**: Xilinx Runtime 2026.1+ (`xrt_coreutil`) for hybrid features.

---

## Quickstart Guide

This guide covers the AMD Strix Halo (ROCm) path on Ubuntu 24.04+. If you are using macOS, simply use `make` instead of `make BACKEND=rocm`.

### 1. Install Dependencies
Ensure you have the ROCm toolkit installed. 
```bash
# Ubuntu 24.04
sudo apt update
sudo apt install build-essential git
sudo apt install rocm-core rocm-hip-sdk  # Requires AMD repository setup
```

### 2. Clone & Build
Clone the repository and compile the engine using the ROCm backend:
```bash
git clone https://github.com/yourusername/ds4fa.git
cd ds4fa

# Build the binaries (ds4 and ds4-server)
make BACKEND=rocm
```

### 3. Verify Hardware
Check if the engine can successfully communicate with your AMD XDNA 2 NPU and iGPU:
```bash
./ds4 --npu-test
```

### 4. Download Model Weights
You **must** use the specialized GGUFs from `https://huggingface.co/antirez/deepseek-v4-gguf`. Generic GGUFs will fail.
```bash
# For 128 GB RAM machines (Downloads ~81GB)
./download_model.sh q2

# OR: For >= 256 GB RAM machines
./download_model.sh q4
```

### 5. Run the Engine
You can now start an interactive chat session:
```bash
./ds4
```

---

## Usage

### CLI (Interactive Chat)
The CLI keeps a rendered token transcript and live KV checkpoint. Useful 
commands: `/help`, `/think`, `/think-max`, `/nothink`, `/ctx N`, `/read FILE`.

```bash
# Standard mode
./ds4

# AMD NPU-accelerated Speculative Decoding (Experimental)
./ds4 --npu-xclbin mtp_draft.xclbin -p "Your prompt"
```

### Server (OpenAI/Anthropic API)
Start a local server with **SSD-backed KV Cache** to handle 1M token contexts without re-prefilling:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```
Point agents to `http://127.0.0.1:8000/v1` (API key: `dsv4-local`).

---

## Speed

Performance numbers utilizing the ROCm backend on AMD APUs, measured with 
`--ctx 32768`, `--nothink`, greedy decoding, and `-n 256`:

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| Dinference's ClawRig 128GB | q2 | short | 433.32 t/s | 29.23 t/s |

*Note: The `q4` quantization numbers are coming soon. Testing requires a 256GB unified memory system.*

---

## Acknowledgements

`ds4fa` exists thanks to the path opened by the **llama.cpp** project and the 
kernels, quantization formats, and engineering knowledge developed there. We 
are deeply thankful to Georgi Gerganov and the GGML contributors.

This software was developed with **strong assistance from GPT 5.5** and with 
humans leading the ideas, testing, and debugging.

## License

MIT License. Original engine by Antirez. Port and optimizations by local AI enthusiasts.
