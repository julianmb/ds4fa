# AGENTS.md — ds4fa

## What this repo is
DeepSeek V4 Flash 284B MoE on AMD Strix Halo (gfx1151):
github.com/julianmb/ds4fa. ROCmFPX-based serving, up to ~32 tok/s decode /
~250 tok/s prefill via IQ2_XXS-class quants.

## Upstream sync status (resolved 2026-08-21)
The long-stalled merge of upstream `antirez/ds4` (@84cc882, DSpark/MXFP4 era)
was completed on branch `sync-upstream-test`, fast-forwarded to `main`, and
pushed. Key resolution decisions:

- **Tensor type IDs:** upstream's `MXFP4 = 39` kept (official GGML
  numbering); our local `IQ2_M` moved to custom ID **100** (it was a
  half-baked addition: inconsistent block sizes, no dequant kernel).
- **README:** our Strix Halo product README is canonical; upstream's full
  document preserved at `docs/UPSTREAM_README.md`.
- **Makefile:** `-I src` include paths kept (src/ layout), upstream's new
  deps (`ds4_gpu_args.o`, `cuda/mmq/ds4_mmq.h`) adopted.
- The old `ds4-strix-halo/` duplicate checkout has been retired — its two
  commits were already on origin/main; only the `ulimit -v` safety cap had
  been lost in the src/-reorg and was re-ported (b68ca1c).

## Layout notes
- Engine source lives under `src/` (moved there by commit 0f80087).
- Upstream of the engine: `~/source/ds4-upstream` (antirez/ds4 clone).
- Weights for this model live in the workshop:
  `~/source/halofpx-research/models/deepseek-v4/`.
