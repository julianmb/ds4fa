# Route A on Ubuntu 26.04.1 and ROCm 10.1

Prepared contribution branch: `codex/ubuntu26-rocm10.1`, based on `797a35f`.
This is a tested platform and model-support port, not an upstream release or a
reproduction of the advertised 32 tok/s result. The change inventory below
separates build compatibility, required model support, correctness/memory fixes,
and kernel tuning so maintainers can decide whether to keep or split any part.

## Artifact and format

The tested target is `gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`:

- Size: 102,320,631,200 bytes.
- SHA256: `8fa6c30d9badd8e72f83c62952fe24b2e8cd647e103aff086bece328cb9c9208`.
- The hash matches the publisher's Hugging Face LFS hash.
- GGUF types 101/104/107 use 17/14/10 bytes per 32 values respectively.

The Route B compatibility check used `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (86,720,111,488 bytes; SHA256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`).

Type 107 is the original Lucebox FP2 encoding: `{-1, 0, 1, 2}`, with two
signed UE4M3 half-block scales. Bit 7 negates a half-block's scale. S40
`{-4, -1, 1, 4}`, affine, and adaptive codebook variants reuse this custom ID
elsewhere and are incompatible with this implementation. The ID alone cannot
identify those variants. Do not substitute another custom-format GGUF without
checking its encoding.

Reference: [publisher's format declarations](https://github.com/Luce-Org/lucebox/blob/main/server/deps/llama.cpp/ggml/rocmfpx/rocmfpx.h)
and [reference decoder](https://github.com/Luce-Org/lucebox/blob/main/server/deps/llama.cpp/ggml/rocmfpx/rocmfpx.c).

## Build and launch

On the tested host, `hipcc` and runtime libraries resolve to ROCm core 10.1.
The package version is 10.1.0-3; its internal HIP version is 7162.63.85. The
startup message's static "tested baseline ROCm 7.2.3" text is not the installed
version. No ROCm or OS downgrade was made.

```sh
make -j8 strix-halo
make rocm-fpx-test
./ds4_test --server
./ds4-server -m gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
    -c 8192 --port 8001
```

Run GPU tests and large-model processes serially. The target has 95.29 GiB of
tensor spans; context 8192 plans another 0.53 GiB. Actual system use after load
was about 103 GiB on the 122.2 GiB host, with a 120 GiB TTM/GTT limit.

The current native DSpark interface is:

```sh
./ds4-server -m gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
    --mtp gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
    --dspark -c 8192 --port 8001
```

The current source does not implement the runner's `DFLASH_DS4_SPEC`,
`DFLASH_DS4_DRAFT`, or fused q=4 verification controls. Its 11.3 GB standalone
drafter is not the native 5.99 GB DSpark support artifact. Native `--mtp` also
cannot be combined with `--ssd-streaming`. The old runner therefore does not
reproduce the published 32 tok/s configuration in this checkout.

## Change inventory

### Linux and ROCm build compatibility

- Add `ROCM_PATH`-based HIP include paths and use the selected `HIPCC` and
  `ROCM_CFLAGS` consistently for compile and link commands.
- The unmodified upstream revision failed to compile on the tested host because
  HIP headers from `/usr/include` were mixed with ROCm core 10.1's LLVM headers.
  Supplying `/opt/rocm/include` and `/opt/rocm/include/hip` resolved the missing
  device-math declarations. The same include paths worked in an isolated copy of
  the unmodified revision.

### Model compatibility required by the tested routes

- Route B's IQ2XXS model uses tensor formats already supported by the base code;
  this change does not alter its decoding. The include-path fix makes the base
  ROCm build available on the tested 10.1 stack.
- Route A's ROCmFP2 model requires custom GGUF tensor types 101/104/107 and
  routed layouts. Register and validate these formats and dispatch their dense
  projections, compressor APE tables, and routed FP2/FP3 operations.
- Decode the model's Q6_K target and draft embeddings through the existing host
  reference path. The FP4 dense projection type is also used by compressor and
  sensitive dense weights in the target.
- Add an independent ROCm regression target for FP4 dense/compressor paths and
  FP2/FP3 routed calculations.

### Correctness and model-fit fixes

- Correct FP3 packed values at byte-boundary bit offsets 6 and 7, and restrict
  the FP4 reduction result to the single lane that owns the output.
- Dispatch projections using their actual tensor type and use typed fallback
  paths where fused F16-only operations do not apply.
- Allocate large expert/cache spans tightly. On this host the previous arena
  policy stranded about 21 GiB and could not fit the tested model under the
  120 GiB TTM/GTT limit.

### Kernel execution and tuning included in this branch

- Parallelize routed FP2/FP3 dot products across wave32 lanes.
- Process eight packed blocks per wave and reuse unpacked scales across values
  to reduce loop and memory-transaction overhead.
- These kernel choices are included because they are the current tested
  implementation. They are tuning as well as execution changes; maintainers may
  prefer to review or split them separately. The short measurements below are
  diagnostic data, not a claim that this change improves performance generally.

## Speed checkpoint at interruption

Same host, ROCm backend, target artifact, `speed-bench/promessi_sposi.txt`,
context allocation 8192, frontiers 64/128, increment 64, 32 greedy decode tokens
per frontier, no DSpark, no profiler:

| Kernel version | Context | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Wave32 experts, original dense reduction | 64 | 7.56 | 5.50 |
| Wave32 experts, original dense reduction | 128 | 7.55 | 5.47 |
| Wave32 dense and experts | 64 | 8.09 | 5.64 |
| Wave32 dense and experts | 128 | 8.07 | 5.61 |
| Packed blocks across wave32 (current) | 64 | 28.42 | 17.48 |
| Packed blocks across wave32 (current) | 128 | 28.59 | 17.27 |

```sh
./ds4-bench -m gguf/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf \
    --backend rocm --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 64 --ctx-max 128 --step-incr 64 --ctx-alloc 8192 \
    --gen-tokens 32 --csv /tmp/routea.csv
```

These short frontiers establish a promising improvement, not a formal
reproduction of the advertised result. The final packed-block kernels pass the
numerical regression and the native-DSpark arithmetic smoke documented below.
The earlier wave32 version returned correct arithmetic with native DSpark at
4.82 tok/s, slower than its ordinary decode; native-DSpark throughput was not
measured again with the final packed-block kernels.

## Route B before/after speed check

The original source revision built with only the ROCm include-path compatibility
flags, so the already-supported IQ2XXS Route B model provides a useful
before/after regression check. Both binaries ran on the same Strix Halo host,
ROCm backend, IQ2XXS model file, prompt `speed-bench/promessi_sposi.txt`,
context allocation (8192), context frontiers (64 and 128), and 32-token greedy
generation. The runs were serial. The second
run had a warmer filesystem cache; model startup time is therefore excluded
from the comparison.

| Source | Context | Prefill tok/s | Generation tok/s |
|---|---:|---:|---:|
| Base `797a35f` | 64 | 23.41 | 16.69 |
| Contribution | 64 | 24.01 | 16.73 |
| Base `797a35f` | 128 | 58.10 | 16.55 |
| Contribution | 128 | 63.82 | 16.57 |

This is one short sweep, not a general performance claim. It shows no decode
regression in this check; the modest prefill differences are not conclusive.

## Validation and limitations

`make rocm-fpx-test` compares dense FP4, ratio-4/128 FP4 compressor state,
and FP2/FP3 routed MoE output with independent references. It covers negative
FP2 half-scales, packed FP3 byte crossings, nonconsecutive expert IDs, and
single/multiple input tokens. The ROCm build, server regression, and a CPU-only
build in an isolated directory passed.

On this contribution branch, the Route A ROCmFP2 target plus the native DSpark
support GGUF loaded with an 8192 context. The arithmetic smoke request returned
`391` for 17 times 23 (17 prompt tokens, 58 completion tokens); DSpark target
hidden capture initialized at layers 40-42. Earlier development checks also
returned the expected weather tool arguments and a working `reverse_string`
implementation. These are functional smokes, not a broad quality evaluation.

The full default `make test` target was not run because it builds CUDA-specific
test targets on this ROCm-only host. The ROCm build, ROCmFPX regression, and
`./ds4_test --server` passed. These checks do not establish broad model quality
or long-context correctness. The short speed sweeps are disclosed above; they
do not establish general performance gains. Earlier IQ2 results accidentally
called Route A must not be used as ROCmFPX measurements. No GGUF was
requantized or replaced.
