# ds4-strix-halo

A Strix Halo (`gfx1151`) tuned **adaptation** of [DwarfStar (ds4)](https://github.com/antirez/ds4).

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

## Adaptation: improvements over upstream ds4

This repository is an **adaptation** of `antirez/ds4` for AMD Strix Halo
(`gfx1151`) hardware. It keeps upstream's ROCm inference backend intact and adds a
Strix-Halo-specific configuration, diagnostics, and validation layer on top. The
improvements over stock ds4 are:

- **Startup hardware profile.** On `ds4_gpu_init()` the engine prints a Strix Halo
  profile — device name, `gcnArchName` (must be `gfx1151`), HIP/ROCm build/runtime/
  driver versions (baseline ROCm 7.2.3), managed / concurrent-managed / pageable
  memory support, HIP-visible memory, system RAM, and the live TTM/GTT mapping
  limit. Upstream ds4 prints none of this.
- **Misconfiguration detection at startup.** The engine warns on a non-`gfx1151`
  arch, HIP older than the 7.2.x baseline, a TTM/GTT limit below 75% of system RAM,
  and a model sized too close to that limit (less than ~8 GiB headroom). It also
  cross-checks the `amdgpu.gttsize` boot parameter against the live limit so a boot
  param that "didn't take" is visible.
- **Actionable fix, not just a warning.** When the limit is low it prints the exact
  `sudo amd-ttm --set-pages N` command that raises it to ~90% of RAM, and it
  suggests a value derived from the box's own RAM.
- **Runtime TTM control.** `DS4_ROCM_TTM_PAGES` overrides the limit for a single run
  (highest priority), and `DS4_ROCM_TTM_AUTORAISE=1` lets the engine call `amd-ttm`
  itself (as root) when a model wouldn't fit. `DS4_ROCM_AUTO_RAISE_ONCE=1` bounds
  that to one raise per process across server reloads / eval restarts.
- **Per-model fit verdict.** The engine records, for every model map, whether it
  fits the GTT limit (headroom, would-OOM, and GGUF header info), exposed via
  `ds4_rocm_last_model_load_estimate()`. `make rocm-model-fit` turns that into a
  pass/fail gate for `DS4_TEST_MODEL`.
- **Machine-readable diagnostics.** `DS4_ROCM_DIAG` writes the profile to a file as
  `key=value` (or JSON with `DS4_ROCM_DIAG_JSON=1`), scoped by `DS4_ROCM_DIAG_FIELDS`
  (`basic` / `full` / `all`) for CI and bug reports. These build-version includes
  come from `ds4_rocm.h`'s `<hip/hip_version.h>` pull-in.
- **Hardware smoke + quick-bench tests.** `make rocm-smoke` exercises the real
  `ds4_gpu` allocation/copy/mapping/managed-tensor paths (plus an optional real-GGUF
  gate and a model-swap leak check) without needing weights; `make rocm-bench-quick`
  confirms gfx1151 kernels actually execute and reports bandwidth; `make rocm-doctor`
  is a one-screen triage (OS, permissions, gfx1151, TTM/GTT, amd-ttm, tuned, udev,
  BIOS VRAM, profile, bench, model-fit).
- **One-shot Ubuntu 24.04 setup.** `misc/strix-halo-setup.sh` applies the
  persistent GRUB/modprobe/udev/tuned configuration (RAM-sized, CWSR off) and
  enforces the 24.04 requirement; `misc/99-amd-kfd.rules` ships the GPU-access udev
  rules. Installs ROCm 7.2.x if missing.
- **CI policy guard.** `make ci` runs the strict smoke test, the quick bench, and
  `misc/sync-check.sh` (which keeps the fork close to upstream ds4).

Everything diagnostic is **advisory**: it prints to stderr and (optionally) a file,
and never alters how inference executes. The adaptation adds no new inference path —
it reuses upstream's ROCm backend.

### What upstream ds4 has vs this fork

| Feature | Upstream ds4 | This fork |
|---------|-------------|-----------|
| Inference backends | CUDA, Metal, ROCm | Same (unchanged) |
| ROCm smoke test | No | `make rocm-smoke` |
| ROCm quick-bench | No | `make rocm-bench-quick` |
| Model fit verdict | No | `make rocm-model-fit` |
| Hardware doctor | No | `make rocm-doctor` (10-step triage) |
| TTM/GTT diagnostics | No | Startup profile + warnings |
| Auto-raise TTM limit | No | `DS4_ROCM_TTM_AUTORAISE` |
| Machine-readable diag | No | `DS4_ROCM_DIAG` (key=value/JSON) |
| Ubuntu 24.04 setup | No | `misc/strix-halo-setup.sh` |
| udev rules | No | `misc/99-amd-kfd.rules` |
| CI policy guard | No | `make ci` + `misc/sync-check.sh` |

The fork adds **validation and configuration tooling** around the same inference
engine — it does not change how models run.

Specific files changed vs upstream:

- `ds4_rocm.h`: includes `<hip/hip_version.h>` for build-version diagnostics.
- `rocm/ds4_rocm_runtime.cuh`: the startup profile, warnings, TTM checks/autoraise,
  model-fit estimate, and machine-readable diag described above.
- `ds4_gpu.h`: declares the model-fit estimate struct and `ds4_gpu_hip_free_bytes()`.
- `tests/rocm_smoke.c`, `tests/rocm_bench_quick.c`, `tests/rocm_model_fit.c` +
  `Makefile` `rocm-*` targets: the hardware validation suite.
- `misc/strix-halo-ubuntu26-setup.sh`, `misc/99-amd-kfd.rules`, `misc/rocm-doctor.sh`:
  the Ubuntu 26.04 setup and triage tooling.

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

### 0. One-shot system setup (Ubuntu 24.04 HWE)

A single script applies the persistent configuration the backend needs — RAM-sized
GRUB `gttsize`/`pages_limit`, CWSR off, `modprobe.d` tuning, udev GPU-access rules,
`render`/`video` group membership, and the `tuned accelerator-performance` profile.
It **requires Ubuntu 24.04** with HWE kernel (6.18.4+ with KFD fixes); other
distros need the same kernel and must apply the settings by hand. Run it as your
normal user (it uses `sudo` internally), then reboot:

```sh
git clone https://github.com/julianmb/ds4fa.git ds4-strix-halo
cd ds4-strix-halo
bash misc/strix-halo-setup.sh     # configures GRUB/udev/tuned; reboot after
# copy the udev rules manually if you skipped the script:
sudo cp misc/99-amd-kfd.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger
```

The script sizes the GTT aperture to ~90% of visible RAM, so the same command works
on both 31 GiB and 124 GiB boxes. `rocm-doctor` re-checks every item it sets.

### 1. Install the toolchain

On Ubuntu 24.04 with HWE kernel (currently 7.x):

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

(Other distros: use Linux 6.18.4+ with the KFD fixes; see
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
- Ubuntu 24.04 LTS with HWE kernel (currently 7.x) includes the Strix Halo
  KFD fixes. Other distros need Linux 6.18.4+ or backported fixes.

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

On a 32 GB Strix Halo system (with ~31 GiB visible after BIOS carveout):

| Model class | Quant | Size | Fits? | Notes |
|-------------|-------|------|-------|-------|
| 7B dense | Q4_K_M | ~4 GB | Yes | Fast, good quality |
| 13B dense | Q4_K_M | ~7 GB | Yes | Balanced speed/quality |
| 30B MoE | Q4_K_M | ~18 GB | Yes | Best speed/quality ratio |
| 70B dense | Q4_K_M | ~35 GB | Marginal | Needs TTM override or SSD streaming |
| 120B MoE | UD-IQ4_XS | ~50 GB | No | Requires SSD streaming |

Use `make rocm-model-fit DS4_TEST_MODEL=your.gguf` to check whether a specific
model fits your TTM/GTT limit before loading it.

## Device permissions

The one-shot setup script installs `misc/99-amd-kfd.rules` and adds you to the
`render`/`video` groups, so most users need do nothing by hand. Verify:

```sh
ls -l /dev/kfd /dev/dri/render*     # readable/writable by group 'render'
groups                              # should include 'render' and 'video'
make rocm-doctor                   # step 1 + 4 re-check permissions, udev, groups
```

If you skipped the script, apply the rules manually:

```sh
sudo cp misc/99-amd-kfd.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG render,video "$USER"   # then log out/in
```

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
