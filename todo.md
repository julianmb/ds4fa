## 1. Environment & Project Setup (Done)
- [x] Configure AMD ROCm build environment (`Makefile`).
- [x] Setup `ds4_hip.cpp`, `ds4_hip.h`, and `hip/*.hip` scaffolds.

## 2. Kernel Implementation (ROCm/HIP C++) (Done)
- [x] `argsort.hip`
- [x] `bin.hip`
- [x] `concat.hip`
- [x] `cpy.hip`
- [x] `dense.hip` (Integrate WMMA Matrix Cores)
- [x] `dsv4_hc.hip`
- [x] `dsv4_kv.hip`
- [x] `dsv4_misc.hip`
- [x] `dsv4_rope.hip`
- [x] `flash_attn.hip` (Integrate WMMA Matrix Cores)
- [x] `get_rows.hip`
- [x] `glu.hip`
- [x] `moe.hip` (Integrate `IQ2_XXS` bitfield extraction via `__builtin_amdgcn_ubfe`)
- [x] `norm.hip`
- [x] `repeat.hip`
- [x] `set_rows.hip`
- [x] `softmax.hip`
- [x] `sum_rows.hip`
- [x] `unary.hip`

## 3. Implement HIP Graphs for CPU Overhead Reduction (Open)
- [ ] Reintroduce HIP graph capture with graph-safe parameter updates.
- [ ] Avoid captured memcpy nodes from stack or short-lived host buffers.
- [ ] Add a ROCm graph replay regression test before enabling this path by default.

## 4. Host-to-Device Wiring (`ds4_hip.cpp`) (Done)
- [x] APU Zero-copy unified memory (`hipHostMallocMapped`).
- [x] Map all 19 HIP kernels to host C wrappers.
- [x] ROCm 7.x Stream Ordered Memory Allocator (SOMA).

## 5. C-Engine Integration (`ds4.c`) (In Progress)
- [ ] Remove or implement the remaining ROCm host-layer stubs in `ds4_hip.cpp`.
- [x] Hook the main ROCm kernels into the engine flow.

## 6. NPU (XDNA 2) Speculative Decoding (Experimental)
- [x] Add optional XRT initialization scaffolding.
- [ ] Provide a real DeepSeek MTP XDNA graph/xclbin contract.
- [ ] Add an end-to-end NPU draft/verify test.
