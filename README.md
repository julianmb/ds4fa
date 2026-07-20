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

## How it works

Strix Halo is a unified-memory part: the CPU and GPU share one physical RAM
pool, and the GPU reaches that memory through a **TTM/GTT** mapping limit rather
than a fixed VRAM allocation. The single biggest source of "it won't load my
model" or "it OOMs" problems on this hardware is a GTT limit that is too small
relative to RAM — usually caused by an oversized BIOS dedicated-VRAM carveout.

This fork does **not** change how DS4 runs models. It adds a small, ROCm-only
diagnostics layer that runs at startup (`ds4_gpu_init`) and answers three
questions:

1. **Is this the right hardware/driver?** It prints `gcnArchName` (must be
   `gfx1151`), the HIP/ROCm build/runtime/driver versions (baseline 7.2.3), and
   the memory capabilities (managed, concurrent-managed, pageable access).
2. **Is there enough mapped memory?** It reads the TTM/GTT `pages_limit` from
   `/sys/module/{ttm,amdttm,amd_ttm}/parameters/pages_limit`, compares it to
   total system RAM, and warns when it is below 75%. It also warns when a loaded
   model would leave less than ~8 GiB of that limit for runtime state, and it
   cross-checks the `amdgpu.gttsize` boot parameter against the live limit so a
   boot param that "didn't take" is visible.
3. **What is the exact fix?** Instead of only warning, it prints the precise
   `sudo amd-ttm --set-pages N` command that raises the limit to ~90% of RAM.

The diagnostics are **advisory**: they print to stderr and (optionally, via
`DS4_ROCM_DIAG`) to a file. They never alter how inference executes. Two escape
hatches let you act on them: `DS4_ROCM_TTM_PAGES` overrides the limit for a
single run, and `DS4_ROCM_TTM_AUTORAISE=1` lets the engine call `amd-ttm`
itself (as root) when a model wouldn't fit.

The smoke/bench tests are ordinary C programs that link the `ds4_gpu` API the
engine itself uses, so they exercise the *same* allocation, copy, mapping, and
kernel paths a real model load would — without needing any weights.

## Run it yourself

This is the full loop on a fresh Strix Halo machine.

### 1. Install the toolchain

On Ubuntu 26.04 (which already has the Strix Halo KFD fixes):

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
# rocWMMA is missing its internal headers on 26.04; add a matching tree:
git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA
sudo cp -a /tmp/rocWMMA/library/include/rocwmma /usr/local/include/
```

(Other distros: use Linux 6.18.4+ or backport the KFD fixes; see
[STRIXHALO.md](STRIXHALO.md).)

### 2. Clone and build

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
make strix-halo -j"$(nproc)"      # builds ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent for gfx1151
```

### 3. Check the hardware (no model needed)

```sh
make rocm-diag
```

This prints the startup profile. On a healthy machine you'll see something like
`gcnArchName=gfx1151` and a TTM/GTT limit near your RAM size. On a misconfigured
one you'll get a `WARNING` plus a `SUGGESTED: sudo amd-ttm --set-pages ...` line.

### 4. Fix the GTT limit (if warned)

```sh
sudo amd-ttm --set-pages 8126464     # 8126464 * 4 KiB ≈ 31 GiB; pick ~90% of your RAM
# or, for a single run without touching the system:
DS4_ROCM_TTM_PAGES=8126464 ./ds4 -m your-model.gguf
```

Re-run `make rocm-diag` to confirm the limit rose and the warning is gone.

### 5. Run the smoke and quick-bench tests

```sh
make rocm-smoke            # allocation/copy/mapping + optional real-model gate
make rocm-bench-quick      # confirms gfx1151 kernels execute; prints bandwidth
```

Both should print `PASSED`. To make CI fail on any config warning:

```sh
ROCM_SMOKE_STRICT=1 make rocm-smoke
```

### 6. Run a model

```sh
./download_model.sh                       # or drop in your own GGUF
./ds4-server ds4flash.gguf                # HTTP server
# or
./ds4 ds4flash.gguf                       # interactive CLI
```

If the model is larger than the GTT-mapped memory allows, use upstream SSD
streaming:

```sh
./ds4 --ssd-streaming -m your-large-model.gguf
```

### 7. Capture a report for help/CI

```sh
DS4_ROCM_DIAG=./diag.txt DS4_ROCM_DIAG_JSON=1 ./ds4 -m ds4flash.gguf
cat ./diag.txt          # machine-readable profile (JSON), attach to issues
```

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

## Tests and CI

The `make rocm-smoke`, `make rocm-diag`, `make rocm-bench-quick`, `make rocm-model-fit`,
`make rocm-doctor`, and `make ci` targets are documented in the **Run it yourself**
section above. In short: `rocm-smoke` exercises allocation/copy/mapping (and a real
model when `DS4_TEST_MODEL` is set) and checks for device-memory leaks across
alloc/free cycles *and* repeated model swaps; `rocm-diag` prints only the profile;
`rocm-bench-quick` runs a fill+copy + a managed-tensor round-trip on gfx1151 and
reports bandwidth; `rocm-model-fit` reports whether `DS4_TEST_MODEL` fits the
TTM/GTT limit (exit 0 = fits, 1 = does not fit, 2 = no model set); `rocm-doctor`
is a one-screen triage for "is this box set up correctly?"; `ci` runs the strict
smoke test, the quick bench, and `misc/sync-check.sh`. Set `DS4_ROCM_DIAG=FILE` on
any run to also write a machine-readable `key=value` (or JSON with
`DS4_ROCM_DIAG_JSON=1`) summary for CI and bug reports.

## Environment variables (ROCm / Strix Halo)

- `DS4_ROCM_TTM_PAGES` — override the TTM/GTT mapping limit in 4 KiB pages
  (e.g. `8126464` ≈ 31 GiB). Useful without changing the system. Highest
  priority: beats the live `pages_limit`.
- `DS4_ROCM_TTM_AUTORAISE` — if set (non-`0`), the engine attempts to raise the
  limit via `amd-ttm --set-pages` when a model would not fit. Requires root.
- `DS4_ROCM_AUTO_RAISE_ONCE` — pair with `AUTORAISE` so `amd-ttm` is invoked at
  most once per process (avoids repeated calls on eval startup / server reload).
- `DS4_ROCM_DIAG` — path to a file where the startup profile is written (truncated
  each run) as `key=value` lines.
- `DS4_ROCM_DIAG_JSON` — if set (non-`0`), `DS4_ROCM_DIAG` is written as JSON
  instead of `key=value`.
- `DS4_ROCM_DIAG_FIELDS` — `basic` | `full` (default) | `all`. `basic` = device,
  arch, versions, TTM limit. `full` = + memory flags, HIP-visible memory,
  system RAM. `all` = + GGUF header, cmdline `gttsize`, model-fit estimate.
- `DS4_TEST_MODEL` — when set for `make rocm-smoke`, the smoke test additionally
  maps and caches the given GGUF, acting as a "can I load a model" gate.
- `ROCM_SMOKE_STRICT` — if set (non-`0`), `make rocm-smoke` / `make ci` fails when
  any configuration warning (e.g. low TTM/GTT limit) is emitted.

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
