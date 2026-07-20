# ds4-strix-halo

A Strix Halo (`gfx1151`) tuned fork of [DwarfStar (ds4)](https://github.com/antirez/ds4).

This repository tracks upstream DS4 inference development and carries forward the
Strix Halo / ROCm tuning and diagnostics for AMD Ryzen AI MAX+ ("Strix Halo")
machines. It is **not** a separate inference backend: it uses upstream's ROCm
backend and adds hardware-focused build targets, smoke tests, and configuration
guidance.

## Scope and relationship to upstream

- Tracks `antirez/ds4` upstream and merges in its improvements.
- Validates and tunes the ROCm backend for Strix Halo.
- Uses **ROCm 7.2.3** as the tested baseline.
- Detects common Strix Halo memory and driver misconfigurations at startup.
- Ships a Strix Halo hardware smoke test (`make rocm-smoke`).
- Keeps detailed feature documentation in the upstream project and the
  repository's subsystem guides (see [STRIXHALO.md](STRIXHALO.md),
  [FORK_NOTES.md](FORK_NOTES.md), and upstream `README.md`).

Everything else (model support, server, agent, SSD streaming, speculative
decoding) is upstream DS4 and is documented upstream.

## What this fork changes

- `ds4_rocm.h`: includes `<hip/hip_version.h>` for build-version diagnostics.
- `rocm/ds4_rocm_runtime.cuh`: prints a Strix Halo runtime profile during
  `ds4_gpu_init()` (device name, `gcnArchName`, HIP/ROCm versions,
  managed/concurrent-managed/pageable memory support, HIP-visible memory,
  system RAM, and the TTM/GTT mapping limit) and warns about:
  - non-`gfx1151` architectures,
  - HIP older than the 7.2.x baseline,
  - a TTM/GTT limit below 75% of system RAM,
  - a model sized too close to the TTM/GTT limit.
- `tests/rocm_smoke.c` + `Makefile` `rocm-smoke` target: a hardware smoke test
  against the current `ds4_gpu` API (allocation, fill, copy, managed tensors,
  page-aligned host model range mapping, model-range caching, and ordered
  cleanup).

## Tested hardware and ROCm baseline

- AMD Strix Halo (`gfx1151`), e.g. Framework Desktop / Ryzen AI MAX+.
- ROCm **7.2.3** (HIP 7.2.x, AMD clang). Newer ROCm 7.2 series is fine.
- Linux with the Strix Halo KFD fixes. Ubuntu 26.04 includes them; otherwise
  use Linux **6.18.4** or newer unless the fixes are backported.

## BIOS guidance

Strix Halo uses unified physical memory. A large BIOS dedicated-VRAM carveout
**permanently removes RAM from the operating system** and helps compute very
little. AMD recommends a small reservation, preferably **512 MB**, and a larger
dynamic **TTM/GTT** mapping limit instead. After booting, check the GTT limit:

```sh
cat /sys/module/ttm/parameters/pages_limit      # or amdttm / amd_ttm
```

GTT is a **dynamic mapping limit**, not permanently reserved memory. Raise it
with AMD's `amd-ttm` helper (preferred) or a kernel parameter as a fallback.
See [STRIXHALO.md](STRIXHALO.md).

## Headless machines and AMD SMI

On headless Strix Halo boxes, `amd-smi` GPU utilization and some power/clock
queries may be limited or require specific driver builds. The startup
diagnostics in this fork read the kernel TTM/GTT limit and HIP device
properties directly, which do not depend on `amd-smi`.

## Device permissions

Ensure your user can access the ROCm devices:

```sh
ls -l /dev/kfd /dev/dri/render*     # should be readable/writable by your user
groups                              # should include 'render' and/or 'video'
```

Add your user to the `render` and `video` groups if needed, then re-login.

## Building

Install ROCm 7.2.3 and the `hipblas` / `hipblaslt` libraries, then:

```sh
make strix-halo -j"$(nproc)"     # or: make rocm
```

This builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`
for `gfx1151`.

## Running the smoke test

```sh
make rocm-smoke
```

It initializes the ROCm backend, exercises device allocation/copy/managed
tensors, maps a page-aligned host model range, caches a model range, and checks
cleanup ordering. On a healthy machine it prints `rocm-smoke: PASSED` and a
runtime profile. On a misconfigured machine it also prints actionable
warnings (e.g. a low TTM/GTT limit).

You can also run `make rocm-diag` to print only the runtime profile (no model
needed), or set `DS4_ROCM_DIAG=FILE` on any run to also append a machine-readable
`key=value` summary for CI and bug reports.

## Environment variables (ROCm / Strix Halo)

- `DS4_ROCM_TTM_PAGES` — override the TTM/GTT mapping limit in 4 KiB pages
  (e.g. `8126464` ≈ 31 GiB). Useful without changing the system.
- `DS4_ROCM_TTM_AUTORAISE` — if set (non-`0`), the engine attempts to raise the
  limit via `amd-ttm --set-pages` when a model would not fit. Requires root.
- `DS4_ROCM_DIAG` — path to a file where the startup profile is appended as
  `key=value` lines.
- `DS4_TEST_MODEL` — when set for `make rocm-smoke`, the smoke test additionally
  maps and caches the given GGUF, acting as a "can I load a model" gate.

## Running a model

Download a supported GGUF (DeepSeek V4 Flash, GLM 5.2, or a smaller model) and
run it the same way as upstream:

```sh
./download_model.sh            # or provide your own GGUF
./ds4-server ds4flash.gguf      # start the HTTP server
./ds4 ds4flash.gguf             # interactive CLI
```

### SSD streaming

On machines where the model does not fully fit in the GTT-mapped memory, upstream
SSD streaming lets you run larger models from a fast local NVMe. See the upstream
README for the streaming model set and `./ds4 --help`.

## Troubleshooting

- **Low visible memory / OOM:** raise the TTM/GTT limit (see BIOS guidance) and
  keep the dedicated VRAM carveout small (512 MB).
- **Missing `gfx1151`:** confirm you built with `make strix-halo` (which uses
  `--offload-arch=gfx1151`) and that your ROCm build supports gfx1151.
- **rocWMMA issues:** ensure ROCm 7.2.x and the matching `rocwmma` package are
  installed; the build pulls `rocwmma` headers via the standard include path.
- **Driver/KFD errors:** use a kernel with the Strix Halo KFD fixes
  (Ubuntu 26.04, or Linux 6.18.4+).

## Syncing future upstream changes

This fork is meant to stay close to upstream:

```sh
git fetch upstream
git merge upstream/main          # or rebase your Strix Halo work on top
```

Preferred policy: keep Strix Halo additions confined to `rocm/`, `ds4_rocm.h`,
`tests/rocm_smoke.c`, the `Makefile` ROCm targets, and this repository's docs.
Do **not** reintroduce the old `ds4fa` `ds4_hip.cpp` backend, NPU scaffolding,
or RPC implementation; upstream's ROCm backend and distributed/MTP work are
newer and more complete. See [FORK_NOTES.md](FORK_NOTES.md).

## Attribution and license

This project is a fork of [antirez/ds4](https://github.com/antirez/ds4), which
in turn builds on the path opened by [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp).
See the upstream `README.md` and `LICENSE` for full acknowledgements and the
MIT license terms. Upstream DS4 was developed with strong AI assistance (see
upstream README); this fork follows the same disclosure.
