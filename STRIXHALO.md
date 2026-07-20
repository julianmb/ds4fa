# DS4 on Strix Halo

This is the minimal setup for DS4 ROCm inference on a
Strix Halo machine with 128 GB RAM and Radeon 8060S (`gfx1151`).

## 1. Install ROCm

On Ubuntu 26.04 LTS, install the ROCm compiler/runtime and libraries used by the Strix Halo backend:

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

The backend uses rocWMMA. On this Ubuntu 26.04 setup, `librocwmma-dev`
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
```

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

## 5. Use the right GGUF

Use the standard IQ2XXS/Q2K/Q8 imatrix GGUF:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Avoid the mixed IQ2/IQ4 or IQ2/Q4 GGUFs on this machine for now. They put much
more memory pressure on the ROCm path and can trigger system OOM instead of a
clean DS4 failure.

## 6. Run DS4

Run it normally:

```sh
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

The ROCm build uses the Strix Halo backend automatically.
