# ds4fa: Running DeepSeek V4 Flash on AMD Strix Halo (ROCm / XDNA 2)

`ds4fa` (ds4 for AMD) is a high-performance native inference engine dedicated to running the **DeepSeek V4 Flash 284B MoE** model locally on AMD's unified memory APUs. It is a highly optimized fork of Antirez's original [ds4](https://github.com/antirez/ds4) engine, adapted specifically for **AMD ROCm** and the **Strix Halo (Ryzen AI Max, gfx1151)** architecture. While strictly tuned for Strix Halo, it should work on other modern AMD GPUs (like RDNA 3/CDNA 3) via ROCm, though this remains untested.

You can try the DeepSeek V4 Flash experience powered by this architecture online at [au.privchat.ai](https://au.privchat.ai).

## Why DeepSeek v4 Flash?

1. **Efficiency**: Faster inference due to fewer active parameters (MoE).
2. **Dynamic Reasoning**: Proportional thinking section length based on problem complexity.
3. **Massive Context**: Features a context window of **1 million tokens**.
4. **Knowledge Depth**: Superior performance in specific knowledge domains due to its 284B total parameters.
5. **Compressed KV Cache**: Incredible compression allowing for **on-disk KV cache persistence**.
6. **Optimized Quantization**: Works exceptionally well with specialized 2-bit quantization (routed experts only).

## Architecture & Backends

### 1. AMD ROCm / HIP Backend
A specialized port for AMD's unified memory APU architecture (`gfx1151`). It
targets feature parity with the original implementation, but this fork should
still be treated as a work in progress. The current priority is making the ROCm
path honest and debuggable before re-enabling more aggressive graph-level
optimizations. It leverages:
*   **Zero-Copy APU Memory**: Direct mapping of host RAM to the iGPU via `hipHostMallocMapped` for transient tensors and validated `hipHostRegisterMapped` for model mmap ranges when HIP exposes an identical device pointer.
*   **Hardware Matrix Cores (WMMA)**: MatMul acceleration using RDNA 3.5 AI accelerators.
*   **Single-Cycle 2-bit Decoding**: Bitfield extraction intrinsics (`ubfe`) for `IQ2_XXS` weights.
*   **Plain HIP Stream Submission**: The previous graph-capture shim has been disabled because it captured temporary host buffers unsafely. HIP graphs should only be reintroduced with explicit graph-safe parameter updates.
*   **Strict Memory & Execution Bounds**: Bounds-checked tensor views, synchronous host I/O where required, and fail-fast diagnostics for incomplete cache paths.
*   **Stream Ordered Memory (ROCm 7.2.x SOMA)**: Uses `hipMallocAsync` only when the runtime reports HIP memory-pool support, with an automatic `hipMalloc` fallback for unsupported or mismatched stacks.

### 2. NPU Hybrid Architecture (XDNA 2)
A foundational framework for **Multi-Token Prediction (MTP)** speculative decoding using the Ryzen AI NPU. The engine is architected to offload DeepSeek's native sequential MTP draft modules to the XDNA 2 NPU via the XRT API.
*   **Zero-Copy SVA**: Uses Shared Virtual Addressing so the CPU, 40 CU iGPU, and XDNA 2 NPU all read/write the exact same token memory buffers without PCIe transfers.
*   **Draft & Verify**: The NPU predicts multiple future tokens asynchronously, while the massive iGPU performs a single parallel batch verification to drastically increase tokens/second.

## Performance Tuning (Strix Halo)

The `ds4fa` engine includes several Strix Halo-specific optimizations enabled by 
default when using the ROCm backend:

*   **Memory Pre-warming**: On startup, the engine launches a dedicated GPU kernel to
    touch each safely registered model-map page. This warms address translation and
    cache state without forcing oversized model ranges to be pinned.
*   **Cache Hints**: Utilizes `hipMemAdviseSetReadMostly` to signal the Strix Halo 
    memory controller to prioritize model weights in the System Level Cache (SLC).
*   **Wave32 WMMA**: All MatMul and Attention kernels are compiled for Wave32 mode 
    to maximize CU occupancy on RDNA 3.5.
*   **RDNA3.5 Shared-Memory Checks**: At runtime the ROCm backend reports system RAM,
    HIP-visible memory, and `/sys/module/ttm/parameters/pages_limit`, then warns when
    a mapped model range is near the current TTM/GTT limit.

### Advanced Improvements & Network Pipeline Parallelism (Q4 Models)
Strix Halo's 128GB memory ceiling restricts it natively to the `q2` quant (80GB). For developers looking to push the engine further to run the `q4` models (160GB), we have added foundational **Network Pipeline Parallelism** via the `ds4_rpc` module.

Because USB4 (40Gbps) and 10GbE networking have high latency compared to local PCIe, standard Tensor Parallelism (AllReduce per-layer) is non-viable. Instead, `ds4fa` utilizes **Layer Sharding (Pipeline Parallelism)**:
*   **HMM Memory Fallback**: `ds4fa` checks whether a mapped model range can be safely registered. If not, it skips `hipHostRegisterMapped` and relies on ROCm's Heterogeneous Memory Management (HMM). This lets two 128GB nodes memory-map a 160GB q4 file without trying to pin the whole file on each node.
*   **Master Node**: Maps the first 30 layers (`--rpc-role master`). Executes the forward pass, performs a `hipMemcpyDtoHAsync` of the intermediate activation state, and transmits it via a raw TCP socket (`ds4_rpc_tx`).
*   **Worker Node**: Maps the final 31 layers (`--rpc-role worker`). Receives the activation, executes the remaining network graph, and returns the logits to the Master.

The C sockets and headers are implemented in `ds4_rpc.c`. To deploy this over your 10GbE network:
1. Integrate the `ds4_rpc_tx` and `ds4_rpc_rx` hooks within `rocm_graph_eval_token_raw_swa` in `ds4.c`.
2. Run on Machine B (Worker): `./ds4-server --rpc-role worker --rpc-port 8000`
3. Run on Machine A (Master): `./ds4 -m ds4flash-q4.gguf --rpc-role master --rpc-ip 192.168.1.100 --rpc-port 8000 -p "Your prompt"`

---

## Requirements

*   **Hardware**: AMD Ryzen AI Max Series (**Strix Halo**, `gfx1151`).
*   **Memory**: 64GB - 128GB of LPDDR5X.
*   **ROCm**: **ROCm 7.2.3** is the current development baseline. The HIP runtime reports as HIP 7.2.x inside ROCm 7.2.3; ROCm 7.1+ is no longer the assumed target for Strix Halo.
*   **Kernel**: For Ryzen AI Max (`gfx1151`), use a kernel with AMD's RDNA3.5 KFD fixes: Ubuntu 24.04 HWE `6.17.0-19.19~24.04.2+`, Ubuntu OEM `6.14.0-1018+`, or other distributions `6.18.4+`.
*   **NPU Software**: Xilinx Runtime 2026.1+ (`xrt_coreutil`) for hybrid features.

---

## Quickstart Guide

This guide covers the AMD Strix Halo (ROCm) path on Ubuntu 24.04+.

### 1. Install Dependencies
Ensure you have the ROCm toolkit installed. 
```bash
# Ubuntu 24.04
sudo apt update
sudo apt install build-essential git
sudo apt install rocm-core rocm-hip-sdk  # Requires AMD repository setup
```

The current known-good toolchain is ROCm 7.2.3. Confirm `hipcc --version`
reports a 7.2.3 ROCm install before chasing backend issues. On Strix Halo, also
check the shared GPU mapping limit:

```bash
cat /sys/module/ttm/parameters/pages_limit
```

AMD recommends keeping the BIOS VRAM reservation small and increasing the Linux
TTM/GTT limit for large shared-memory workloads. The backend prints this limit at
startup and warns when a model map is close to it.

### 2. Clone & Build
Clone the repository and compile the engine using the ROCm backend:
```bash
git clone https://github.com/yourusername/ds4fa.git
cd ds4fa

# Build the binaries for Strix Halo (default gfx1151)
make

# Or build for another AMD architecture (e.g. RX 7900 XTX / gfx1100)
make GPU_ARCH=gfx1100
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

## Testing

The default ROCm fork test target currently runs the local server/parser/cache
unit tests without requiring the model weights:

```bash
make test
```

Full ROCm inference validation still requires the DeepSeek V4 Flash GGUF and
the prompt/logprob vector tests to be ported from the old Metal-specific test
path. Known incomplete ROCm host-layer paths now fail fast with a diagnostic
instead of silently returning incorrect cache state.

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


## License

MIT License. Original engine by Antirez. Port and optimizations by local AI enthusiasts.
