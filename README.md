<p align="center">
  <strong>DeepSeek V4 Flash on AMD Strix Halo</strong><br>
  <em>ROCm 7.2 diagnostics, smoke tests, and one-shot setup for antirez/ds4</em>
</p>

<p align="center">
  <a href="https://github.com/julianmb/ds4fa/blob/main/LICENSE"><img src="https://img.shields.io/github/license/julianmb/ds4fa?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-AMD%20Strix%20Halo%20gfx1151-blue?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/ROCm-7.2.x-e95420?style=flat-square" alt="ROCm">
  <img src="https://img.shields.io/badge/Ubuntu-24.04%20HWE-orange?style=flat-square" alt="Ubuntu">
  <img src="https://img.shields.io/badge/model-DeepSeek%20V4%20Flash%20284B-purple?style=flat-square" alt="Model">
</p>

---

A tuned fork of [antirez/ds4](https://github.com/antirez/ds4) for **AMD Ryzen AI MAX+** ("Strix Halo") hardware. This repository does not change how DS4 runs models — it adds hardware diagnostics, smoke tests, and one-shot configuration on top of upstream's ROCm backend.

> **Target hardware:** 128 GB Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S) with Ubuntu 24.04 HWE kernel.

## What's in the box

| Feature | Status | Description |
|---------|--------|-------------|
| **Startup diagnostics** | `make rocm-diag` | Prints gfx1151 arch, HIP/ROCm versions, memory flags, TTM/GTT limit |
| **Misconfiguration warnings** | auto | Detects non-gfx1151, old HIP, low TTM, model too close to limit |
| **Actionable fix** | auto | Prints exact `sudo amd-ttm --set-pages N` command |
| **Model fit verdict** | `make rocm-model-fit` | Pass/fail gate for `DS4_TEST_MODEL` |
| **Machine-readable diag** | `DS4_ROCM_DIAG` | key=value or JSON output for CI and bug reports |
| **Hardware smoke test** | `make rocm-smoke` | Allocation/copy/mapping — no weights needed |
| **Quick bench** | `make rocm-bench-quick` | Confirms gfx1151 kernels execute; reports bandwidth |
| **10-step doctor** | `make rocm-doctor` | OS, permissions, arch, ROCm, TTM, tuned, bench, model-fit |
| **One-shot setup** | `misc/strix-halo-setup.sh` | GRUB, modprobe, udev, tuned, ROCm install |
| **CI guard** | `make ci` | Strict smoke + bench + upstream sync check |

## Quick start

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
bash misc/strix-halo-setup.sh        # configure system; reboot after
make strix-halo -j"$(nproc)"         # build for gfx1151
make rocm-doctor                     # verify setup
./download_model.sh rocmfpx-strix    # download DeepSeek V4 Flash ROCmFPX (~102 GB)
./download_model.sh dspark-drafter   # download DSpark draft model (~11 GB)
./run-deepseek-v4.sh                 # run 32 tok/s high-throughput server
```

<details>
<summary><strong>Full step-by-step guide</strong></summary>

### 1. Clone and one-shot setup (Ubuntu 24.04 HWE)

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
bash misc/strix-halo-setup.sh     # configures GRUB/udev/tuned; reboot after
```

The script applies RAM-sized GRUB `gttsize`/`pages_limit`, CWSR off, `modprobe.d`
tuning, udev GPU-access rules, `render`/`video` group membership, and the
`tuned accelerator-performance` profile. It sizes the GTT aperture to ~90% of
visible RAM. **Requires Ubuntu 24.04** with HWE kernel (6.18.4+ with KFD fixes).

### 2. Install the toolchain

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

### 3. Build

```sh
make strix-halo -j"$(nproc)"      # builds ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent for gfx1151
```

### 4. Check the hardware (no model needed)

```sh
make rocm-diag
```

On a healthy machine you'll see `gcnArchName=gfx1151` and a TTM/GTT limit near
your RAM size. On a misconfigured one you'll get a `WARNING` plus a
`SUGGESTED: sudo amd-ttm --set-pages ...` line.

### 5. Fix the GTT limit (if warned)

```sh
sudo amd-ttm --set-pages 8126464     # 8126464 * 4 KiB ≈ 31 GiB; pick ~90% of your RAM
# or, for a single run without touching the system:
DS4_ROCM_TTM_PAGES=8126464 ./ds4 -m your-model.gguf
```

Re-run `make rocm-diag` to confirm the limit rose and the warning is gone.

### 6. Run the smoke and quick-bench tests

```sh
make rocm-smoke            # allocation/copy/mapping + optional real-model gate
make rocm-bench-quick      # confirms gfx1151 kernels execute; prints bandwidth
```

Both should print `PASSED`. To make CI fail on any config warning:

```sh
ROCM_SMOKE_STRICT=1 make rocm-smoke
```

### 7. Run a model

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

### 8. Capture a report for help/CI

```sh
DS4_ROCM_DIAG=./diag.txt DS4_ROCM_DIAG_JSON=1 ./ds4 -m ds4flash.gguf
cat ./diag.txt          # machine-readable profile (JSON), attach to issues
```

</details>

## How it works

Strix Halo is a **unified-memory** part: the CPU and GPU share one physical RAM
pool, and the GPU reaches that memory through a **TTM/GTT** mapping limit rather
than a fixed VRAM allocation. The single biggest source of "it won't load my
model" or "it OOMs" problems on this hardware is a GTT limit that is too small
relative to RAM — usually caused by an oversized BIOS dedicated-VRAM carveout.

This fork adds a small, ROCm-only diagnostics layer that runs at startup
(`ds4_gpu_init`) and answers three questions:

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Is this the right hardware/driver?                             │
│     → gfx1151 arch, HIP/ROCm versions, memory capabilities         │
│                                                                     │
│  2. Is there enough mapped memory?                                  │
│     → reads TTM/GTT pages_limit, compares to RAM, warns if < 75%   │
│     → cross-checks amdgpu.gttsize boot param vs live limit          │
│                                                                     │
│  3. What is the exact fix?                                          │
│     → prints: sudo amd-ttm --set-pages <N>                         │
└─────────────────────────────────────────────────────────────────────┘
```

All diagnostics are **advisory** — they print to stderr and never alter inference.
Two escape hatches let you act on them:

- `DS4_ROCM_TTM_PAGES` — override the limit for a single run
- `DS4_ROCM_TTM_AUTORAISE=1` — let the engine call `amd-ttm` itself (as root)

## DeepSeek V4 Flash model sizes

This fork targets **128 GB Strix Halo** systems running **DeepSeek V4 Flash** (including the new **DeepSeek-V4-Flash-0731** release). After a 512 MB BIOS VRAM carveout, expect ~120 GiB usable RAM.

| Model / Quant | Size | Fits in 120 GiB? | Speed | Notes |
|---------------|------|:-----------------:|-------|-------|
| DeepSeek-V4-Flash-0731 (ROCmFP2) | ~98 GB | :white_check_mark: | **32.0 t/s** | **July 31 official release** (TerminalBench 82.7, DeepSWE 54.4) |
| ROCmFPX STRIX (284B) | ~102 GB | :white_check_mark: | **32.0 t/s** | **High-throughput LocalMaxxing route** (with DSpark draft) |
| UD-IQ2_XXS | ~91 GB | :white_check_mark: | ~13 t/s | Capacity proof route |
| IQ2_XXS | ~100 GB | :white_check_mark: | ~12 t/s | Low quality |
| UD-Q2_K | ~110 GB | :warning: | ~10 t/s | May need SSD streaming |
| Q4_K_M | ~200 GB | :x: | — | Requires multi-GPU or SSD streaming |

### Reaching 32 tok/s LocalMaxxing Speed

1. Download ROCmFPX target model + DSpark drafter:
   ```sh
   ./download_model.sh rocmfpx-strix
   ./download_model.sh dspark-drafter
   ```
2. Lock GPU clocks & set performance profile:
   ```sh
   echo performance | sudo tee /sys/firmware/acpi/platform_profile
   sudo rocm-smi -d 0 --setperflevel high
   ```
3. Run with fused DSpark verification and sparse prefill (~250 tok/s prefill):
   ```sh
   ./run-deepseek-v4.sh   # convenience launcher
   # or manually:
   DFLASH_DS4_SPEC=1 DFLASH_DS4_FUSED_VERIFY=1 DFLASH_DS4_SPEC_Q=4 LUCE_MMVQ_MAX_NCOLS=4 \
     ./ds4-server gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
     --ds4-draft gguf/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf \
     --ds4-prefill sparse --ds4-fused-decode --ds4-expert-top-k 4 --max-ctx 8192
   ```

Check fit before loading:

```sh
make rocm-model-fit DS4_TEST_MODEL=your.gguf
```

## BIOS guidance

Strix Halo uses unified physical memory. A large BIOS dedicated-VRAM carveout
**permanently removes RAM from the operating system** and helps compute very
little. AMD recommends a small reservation, preferably **512 MB**, and a larger
dynamic **TTM/GTT** mapping limit instead.

```sh
cat /sys/module/ttm/parameters/pages_limit      # check current limit
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

## Environment variables

| Variable | Description |
|----------|-------------|
| `DS4_ROCM_TTM_PAGES` | Override TTM/GTT limit in 4 KiB pages (e.g. `8126464` ≈ 31 GiB) |
| `DS4_ROCM_TTM_AUTORAISE` | Auto-raise limit via `amd-ttm` when model wouldn't fit (requires root) |
| `DS4_ROCM_AUTO_RAISE_ONCE` | Bound autoraise to one call per process |
| `DS4_ROCM_DIAG` | Write startup profile to file (key=value) |
| `DS4_ROCM_DIAG_JSON` | Write diag as JSON instead of key=value |
| `DS4_ROCM_DIAG_FIELDS` | `basic` \| `full` (default) \| `all` |
| `DS4_TEST_MODEL` | GGUF path for `make rocm-smoke` model gate |
| `ROCM_SMOKE_STRICT` | Fail smoke/CI on any config warning |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Low visible memory / OOM | Raise TTM/GTT limit; keep BIOS VRAM carveout at 512 MB |
| Missing `gfx1151` | Build with `make strix-halo`; verify ROCm supports gfx1151 |
| rocWMMA issues | Ensure ROCm 7.2.x + matching `rocwmma` package installed |
| Driver/KFD errors | Use Ubuntu 24.04 HWE kernel (6.18.4+) with KFD fixes |

## Syncing upstream

This fork stays close to [antirez/ds4](https://github.com/antirez/ds4):

```sh
git fetch upstream
git merge upstream/main          # or rebase your Strix Halo work on top
```

Preferred policy: keep Strix Halo additions confined to `rocm/`, `ds4_rocm.h`,
`tests/rocm_smoke.c`, the `Makefile` ROCm targets, and this repository's docs.
Do **not** reintroduce the old `ds4fa` backend or RPC implementation. See
[FORK_NOTES.md](FORK_NOTES.md).

## Documentation

| Document | Description |
|----------|-------------|
| [STRIXHALO.md](STRIXHALO.md) | ROCm install, GRUB params, TTM priority, hardware notes |
| [FORK_NOTES.md](FORK_NOTES.md) | Audit of what was retained/rejected from upstream |

## License

This project is a fork of [antirez/ds4](https://github.com/antirez/ds4), which
in turn builds on the path opened by [llama.cpp / GGML](https://github.com/ggml-org/llama.cpp).
See the upstream `README.md` and `LICENSE` for full acknowledgements and the
MIT license terms.
