# Model Files

This directory holds the GGUF model files used by `ds4fa`. Download scripts place
files here; the layout is organized by route so it stays clear at a glance.

## Layout

```
gguf/
├── deepseek-v4-flash-0731/          # Native ds4fa route — DeepSeek V4 Flash 0731 target
│   └── DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf   (86.72 GB)
├── draft/                           # Speculative-draft models
│   └── DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf   (11.3 GB)
└── logs/                            # Download / quantization logs
```

## Files

### Native route (default, no kernel gaps)

**`deepseek-v4-flash-0731/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf`** (86.72 GB)

* Source: [tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10](https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10)
* Recipe: attention/shared/output = `Q8_0`, token embedding = `F16`,
  routed gate/up = `IQ2_XXS`, routed down = `Q2_K`
* Matches the `ds4fa` engine natively — no requantization needed.

### DSpark draft (speculative decode)

**`draft/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf`** (11.3 GB)

* Source: [Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF)
* Used by the high-throughput ROCmFPX route for ~+26% decode speed.

## How to download

```sh
# Native route model + symlink
aria2c -x 16 -s 16 -k 1M -j 16 -c --file-allocation=none \
  -d gguf/deepseek-v4-flash-0731 -o DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf \
  "https://huggingface.co/tekosML/DeepSeek-V4-Flash-0731-GGUF-GX10/resolve/main/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
ln -sf gguf/deepseek-v4-flash-0731/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf ds4flash.gguf
```

> The `ds4flash.gguf` symlink at the repository root points at the active model,
> so `./ds4 -m ds4flash.gguf` always uses the latest downloaded target.
