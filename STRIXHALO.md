# DS4 on Strix Halo

This is the minimal setup for DS4 ROCm inference on a
Strix Halo machine with 128 GB RAM and Radeon 8060S (`gfx1151`).

## 1. Install ROCm

On Ubuntu 24.04 with HWE kernel (currently 7.x), install the ROCm compiler/runtime and libraries used by the Strix Halo backend:

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi \
  libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev \
  librocblas-dev \
  librocwmma-dev \
  libhipcub-dev
```

The backend uses rocWMMA. On Ubuntu 24.04, `librocwmma-dev`
installs the top-level rocWMMA headers but misses `rocwmma/internal/`.
No Ubuntu package currently provides those internal headers. Install a complete
matching rocWMMA header tree:

```sh
git clone --depth 1 --branch rocm-7.2.3 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA-rocm-7.2.3
sudo mkdir -p /usr/local/include
sudo cp -a /tmp/rocWMMA-rocm-7.2.3/library/include/rocwmma /usr/local/include/
```

If ROCm is installed under `/usr` but tooling expects `/opt/rocm`, add these
compatibility links:

```sh
sudo mkdir -p /opt/rocm/bin
sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc
sudo ln -sfn /usr/lib/x86_64-linux-gnu /opt/rocm/lib
sudo ln -sfn /usr/include /opt/rocm/include
```

## 2. Enable ROCm access

The user running DS4 must be able to open `/dev/kfd` and the DRM render node:

```sh
sudo usermod -aG render,video "$USER"
```

Log out and back in, or reboot. Verify:

```sh
rocminfo | grep -A80 'Name:                    gfx1151'
```

If DS4 says `no ROCm-capable device is detected`, check that `rocminfo` can open
`/dev/kfd` and that `groups` includes `render`.

## 3. Increase GPU-visible memory (TTM/GTT limit)

Strix Halo uses unified physical memory. A large **BIOS dedicated-VRAM carveout**
permanently removes RAM from the operating system and helps compute very little.
AMD recommends a small reservation (for example **512 MB**) and a larger dynamic
**TTM/GTT** mapping limit instead.

> GTT is a **dynamic mapping limit**, not permanently reserved memory. Raising it
> does not carve out RAM; it raises the ceiling the GPU can map on demand.

### Preferred: AMD `amd-ttm`

AMD provides the `amd-ttm` helper to inspect or set the limit without rebooting
or editing boot parameters:

```sh
amd-ttm --show
sudo amd-ttm --set-pages <pages>     # 4 KiB pages; e.g. 32505856 ≈ 124 GiB
```

### Fallback: kernel parameters

If `amd-ttm` is unavailable, raise the limit via boot parameters. A 128 GB
Strix Halo system may otherwise expose only about 62 GB of GPU-visible memory;

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

On Ubuntu with GRUB:

```sh
sudo cp /etc/default/grub /etc/default/grub.bak
sudoedit /etc/default/grub
```

Set:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
```

Then:

```sh
sudo update-grub
sudo reboot
```

After reboot (or after `amd-ttm --set-pages`), verify:

```sh
cat /proc/cmdline
sudo dmesg | grep -Ei 'GTT|gttsize|TTM|VRAM'
rocminfo | grep -A80 'Name:                    gfx1151'
cat /sys/module/ttm/parameters/pages_limit      # or amdttm / amd_ttm
```

Expected signs:

```text
amdgpu:  126976M of GTT memory ready
rocminfo gfx1151 pool: 130023424 KB
```

The `ds4-strix-halo` startup diagnostics print the TTM/GTT mapping limit and
warn when it is below 75% of system RAM or when a model is sized too close to it.

### Worked example (reference machine)

On the reference Strix Halo (32 GB RAM reported to the OS after a large BIOS
VRAM carveout), the default limit was far too low:

```text
ds4: ROCm TTM/GTT mapping limit: 15.5 GiB
ds4: ROCm WARNING: TTM/GTT mapping limit is 50.0% of system RAM (15.5/31.0 GiB)...
```

Fix: keep the dedicated VRAM carveout at 512 MB in the BIOS, then raise the GTT
limit. With 31 GiB of system RAM and an ~24 GiB model plus runtime buffers, a
31 GiB GTT ceiling is appropriate:

```sh
sudo amd-ttm --set-pages 8126464     # 8126464 * 4 KiB = 31 GiB
cat /sys/module/ttm/parameters/pages_limit
# or, if you prefer not to grant the engine root, set it once at boot via the
# ttm.pages_limit kernel parameter (see the fallback above).
```

You can also override the limit for a single run without changing the system:

```sh
DS4_ROCM_TTM_PAGES=8126464 ./ds4 -m your-model.gguf
# and to let the engine try to raise it via amd-ttm itself (run as root):
DS4_ROCM_TTM_AUTORAISE=1 ./ds4 -m your-model.gguf
# across multiple set_model_map calls (e.g. eval startup, server reload):
DS4_ROCM_TTM_AUTORAISE=1 DS4_ROCM_AUTO_RAISE_ONCE=1 ./ds4 -m your-model.gguf
```

**Priority order:** `DS4_ROCM_TTM_PAGES` > live `pages_limit` from the kernel
(possibly raised by `amd-ttm`). The env var always wins when set, so a single
run can use a higher limit than the rest of the system sees. `DS4_ROCM_TTM_AUTORAISE`
is a write path (calls `amd-ttm` and changes the system-wide limit); the env
override is a read path (per-process only, no system change). `DS4_ROCM_AUTO_RAISE_ONCE=1`
makes the engine call `amd-ttm` at most once per process, so multiple
`set_model_map` calls (eval startup, server reload) do not repeatedly invoke it.

After raising it, the diagnostic should show the full mapped ceiling and no
warning:

```text
ds4: ROCm TTM/GTT mapping limit: 31.0 GiB
```

## 4. Build DS4

Use the normal Strix Halo target. It builds the standard binary names:

```sh
make strix-halo -j"$(nproc)"
```

`make rocm` is an alias for `make strix-halo`.

## 5. Model Selection & High-Throughput (32 tok/s) Setup

### Standard Capacity Route
For baseline evaluation and capacity proof:
```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

### High-Throughput ROCmFPX Route (Up to 32 tok/s decode, ~250 tok/s prefill)
To achieve maximum local throughput on 128 GB Strix Halo (Radeon 8060S):

1. **Download ROCmFPX 2.88-bit block model and DSpark drafter**:
   ```sh
   ./download_model.sh rocmfpx-strix
   ./download_model.sh dspark-drafter
   ```

2. **Lock GPU clocks and set performance platform profile**:
   ```sh
   echo performance | sudo tee /sys/firmware/acpi/platform_profile
   sudo rocm-smi -d 0 --setperflevel high
   ```
   *Note: Locking GPU clocks at 2.9 GHz prevents dynamic frequency scaling drops.*

3. **Run with DSpark speculation, fused verification, and sparse prefill**:
   ```sh
   DFLASH_DS4_SPEC=1 \
   DFLASH_DS4_FUSED_VERIFY=1 \
   DFLASH_DS4_SPEC_Q=4 \
   LUCE_MMVQ_MAX_NCOLS=4 \
   ./ds4-server gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
     --ds4-draft gguf/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf \
     --ds4-prefill sparse \
     --ds4-fused-decode \
     --ds4-expert-top-k 4 \
     --max-ctx 8192 --port 8000
   ```

### Key Performance Innovations Behind 32 tok/s:
- **ROCmFPX Block Quantization**: 2.88-bit mixed precision (102.3 GB total). Expert gate/up matrices in ROCmFP2, down in ROCmFP3, dense in ROCmFP4. Kernels dequantize directly in registers via AMD byte-permute instructions (`v_perm_b32`).
- **DSpark Speculative Verification (`q=4`)**: DSpark draft proposes up to 3 tokens; target verifies 4 positions in one fused pass. Fused decode unpacks packed dense weights ONCE in registers across all 4 verification columns (+2.3% gain).
- **Sparse Prefill**: DeepSeek V4 learned indexer limits compressed-history attention, reaching **~250 tok/s prefill**.
- **Top-4 Experts Option**: `--ds4-expert-top-k 4` uses 4 experts instead of 6, trading slight quality margin for a ~25% decode speedup.

## 6. Run DS4

Run it normally:

```sh
./ds4 -m gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf
```

The ROCm build uses the Strix Halo backend automatically.

## 7. Known limitations

### `pageable-access=0`

On some Strix Halo setups the device reports `pageable-access=0` in the startup
profile. The engine's managed-KV-cache path (`ds4_gpu_should_use_managed_kv_cache`)
assumes demand-paged, pageable host memory for good throughput on very large KV
caches. Without pageable access:

- Large KV caches may be slower, or may consume more device-resident memory than
  necessary.
- You may need to rely on device-only allocation (smaller context) or SSD
  streaming for models that would otherwise use the managed path.

This is a driver/platform property, not a DS4 bug. If you see it, check your
kernel/KFD build (Ubuntu 24.04 with HWE kernel, or Linux 6.18.4+ with the Strix Halo fixes) and
the ROCm release notes for the `pageable-memory-access` capability on gfx1151.

### GTT is dynamic, not reserved

Raising `amdgpu.gttsize` / `ttm.pages_limit` does **not** carve RAM away from the
OS; it only raises the ceiling the GPU can map on demand. If the diagnostic shows
`amdgpu.gttsize=N MiB` set but the live TTM/GTT limit is much lower, the boot
parameter did not take effect (re-check GRUB) or a large BIOS VRAM carveout is
reducing the available GTT.

