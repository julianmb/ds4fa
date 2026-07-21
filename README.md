# ds4-strix-halo

A Strix Halo (`gfx1151`) tuned **adaptation** of [DwarfStar (ds4)](https://github.com/antirez/ds4).

This repository tracks upstream DS4 inference development and carries forward the
Strix Halo / ROCm tuning and diagnostics for AMD Ryzen AI MAX+ ("Strix Halo")
machines. It is **not** a separate inference backend: it uses upstream's ROCm
backend and adds hardware-focused build targets, smoke tests, and configuration
guidance.

## What this fork adds

This fork keeps upstream's ROCm inference backend intact and adds a Strix Halo
diagnostics and configuration layer. All diagnostics are **advisory** — they print
to stderr and never alter inference.

| What | How |
|------|-----|
| Startup hardware profile | `ds4_gpu_init()` prints gfx1151 arch, HIP/ROCm versions, memory flags, TTM/GTT limit |
| Misconfiguration warnings | Non-gfx1151, old HIP, low TTM limit, model too close to limit, `gttsize` mismatch |
| Actionable fix | Prints exact `sudo amd-ttm --set-pages N` command |
| Per-model fit verdict | `make rocm-model-fit DS4_TEST_MODEL=x.gguf` |
| Machine-readable diag | `DS4_ROCM_DIAG=FILE DS4_ROCM_DIAG_JSON=1` |
| Hardware smoke test | `make rocm-smoke` (allocation/copy/mapping, no weights needed) |
| Quick bench | `make rocm-bench-quick` (confirms gfx1151 kernels, reports bandwidth) |
| 10-step doctor | `make rocm-doctor` (OS, permissions, arch, ROCm, TTM, tuned, bench, model-fit) |
| One-shot setup | `misc/strix-halo-setup.sh` (GRUB, modprobe, udev, tuned, ROCm install) |
| CI guard | `make ci` (strict smoke + bench + upstream sync check) |

### Files changed vs upstream

- `ds4_rocm.h` — `<hip/hip_version.h>` for build-version diagnostics
- `rocm/ds4_rocm_runtime.cuh` — startup profile, warnings, TTM autoraise, model-fit estimate, machine-readable diag
- `ds4_gpu.h` — model-fit estimate struct, `ds4_gpu_hip_free_bytes()`
- `tests/rocm_smoke.c`, `tests/rocm_bench_quick.c`, `tests/rocm_model_fit.c` — hardware validation suite
- `misc/strix-halo-setup.sh`, `misc/99-amd-kfd.rules`, `misc/rocm-doctor.sh` — setup and triage

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

### 0. Clone and one-shot setup (Ubuntu 24.04 HWE)

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
bash misc/strix-halo-setup.sh     # configures GRUB/udev/tuned; reboot after
```

The script applies RAM-sized GRUB `gttsize`/`pages_limit`, CWSR off, `modprobe.d`
tuning, udev GPU-access rules, `render`/`video` group membership, and the
`tuned accelerator-performance` profile. It sizes the GTT aperture to ~90% of
visible RAM. **Requires Ubuntu 24.04** with HWE kernel (6.18.4+ with KFD fixes).

### 1. Install the toolchain

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
# rocWMMA may be missing internal headers; add a matching tree:
git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA
sudo cp -a /tmp/rocWMMA/library/include/rocwmma /usr/local/include/
```

### 2. Build

```sh
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

## Performance tuning

The one-shot setup script applies the `accelerator-performance` tuned profile.
To verify or re-apply manually:

```sh
sudo systemctl enable --now tuned
sudo tuned-adm profile accelerator-performance
tuned-adm active   # should show accelerator-performance
```

Key parameters:
- **GRUB**: `amdgpu.gttsize=<MiB> ttm.pages_limit=<4KiB-pages> ttm.page_pool_size=<4KiB-pages> amdgpu.cwsr_enable=0`
- **modprobe.d**: `/etc/modprobe.d/amdgpu_strix_halo.conf` (same values)
- **udev**: `/etc/udev/rules.d/99-amd-kfd.rules` (render group access)
- **Groups**: user must be in `render` and `video` groups

## Model size recommendations

This fork targets **128 GB Strix Halo** systems (Ryzen AI MAX+ 395 / Radeon
8060S) running **DeepSeek V4 Flash (284B)**. After a 512 MB BIOS VRAM carveout,
expect ~120 GiB usable RAM. The TTM/GTT mapping limit should be raised to ~90%
of RAM (~108 GiB) via the setup script or `amd-ttm`.

### DeepSeek V4 Flash (284B) GGUF sizes

| Quant | Size | Fits in 120 GiB? | Speed | Notes |
|-------|------|-------------------|-------|-------|
| UD-IQ2_XXS | ~91 GB | Yes | ~13 t/s | Verified capacity proof |
| IQ2_XXS | ~100 GB | Yes | ~12 t/s | Low quality |
| UD-Q2_K | ~110 GB | Tight | ~10 t/s | May need SSD streaming |
| Q4_K_M | ~200 GB | No | — | Requires multi-GPU or SSD streaming |

Use `make rocm-model-fit DS4_TEST_MODEL=your.gguf` to check whether a specific
quant fits your TTM/GTT limit before loading it.

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

## Troubleshooting

- **Low visible memory / OOM:** raise the TTM/GTT limit (see BIOS guidance) and
  keep the dedicated VRAM carveout small (512 MB).
- **Missing `gfx1151`:** confirm you built with `make strix-halo` (which uses
  `--offload-arch=gfx1151`) and that your ROCm build supports gfx1151.
- **rocWMMA issues:** ensure ROCm 7.2.x and the matching `rocwmma` package are
  installed; the build pulls `rocwmma` headers via the standard include path.
- **Driver/KFD errors:** use a kernel with the Strix Halo KFD fixes
  (Ubuntu 24.04 with HWE kernel, or Linux 6.18.4+).

## Syncing future upstream changes

This fork is meant to stay close to upstream:

```sh
git fetch upstream
git merge upstream/main          # or rebase your Strix Halo work on top
```

Preferred policy: keep Strix Halo additions confined to `rocm/`, `ds4_rocm.h`,
`tests/rocm_smoke.c`, the `Makefile` ROCm targets, and this repository's docs.
Do **not** reintroduce the old `ds4fa` backend or RPC implementation; upstream's
ROCm backend and distributed/MTP work are newer and more complete. See
[FORK_NOTES.md](FORK_NOTES.md).

## Attribution and license

This project is a fork of [antirez/ds4](https://github.com/antirez/ds4), which
in turn builds on the path opened by [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp).
See the upstream `README.md` and `LICENSE` for full acknowledgements and the
MIT license terms. Upstream DS4 was developed with strong AI assistance (see
upstream README); this fork follows the same disclosure.
