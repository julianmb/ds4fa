# Fork Notes: ds4-strix-halo

A focused Strix Halo (`gfx1151`) fork of [antirez/ds4](https://github.com/antirez/ds4).
This document records the base, the audit, what was retained, what was rejected,
and the policy for future upstream merges.

## Upstream base commit

The fork starts from current upstream `main` (the clone at
`/home/user/source/ds4-strix-halo`). The original ROCm work lived on the deleted
upstream `new-rocm` branch and is already merged into `main`, so `main` is the
correct Strix Halo base.

## ds4fa audit

`julianmb/ds4fa` was compared against upstream. The common ancestor is:

```
baa084482020263a17a3b33a238f6f0809b425fd
```

Since that point:

- Upstream has **323** commits not present in `ds4fa`.
- `ds4fa` has **15** commits not present in upstream.

A wholesale rebase or merge of `ds4fa` is therefore the wrong approach: it would
remove or replace a large amount of newer upstream code, including the upstream
Metal backend and the current ROCm architecture.

## What was retained (adapted to upstream)

- **ROCm/HIP startup diagnostics**: build/runtime/driver HIP versions, device
  name and `gcnArchName`, managed/concurrent-managed/pageable memory support,
  HIP-visible memory, total system RAM, and the TTM/GTT `pages_limit`.
- **Strix Halo memory diagnostics**: warning when the TTM/GTT limit is below 75%
  of system RAM, and warning when a model is sized too close to the limit.
- **A current ROCm smoke test** (`tests/rocm_smoke.c`) against the `ds4_gpu`
  API, exposed as `make rocm-smoke`.
- **A Strix Halo-focused README** plus updated `STRIXHALO.md` (ROCm 7.2.3,
  preferring `amd-ttm` over hard-coded boot parameters).

## What was rejected and why

- **`ds4_hip.cpp` backend and `hip/` kernel tree**: upstream's ROCm backend is
  substantially newer and more complete.
- **Removal of the Metal backend**: this fork keeps upstream backends intact.
- **`q4_guard.py`**: its useful checks belong in the runtime (now the startup
  diagnostics), not as a separate script.
- **The old `ds4fa` README**: describes incomplete or superseded behavior.

## Policy for future upstream merges

- Track `antirez/ds4` `main`; merge or rebase regularly.
- Keep Strix Halo additions confined to:
  - `rocm/ds4_rocm_runtime.cuh` (diagnostics only),
  - `ds4_rocm.h` (version include),
  - `tests/rocm_smoke.c` and the `Makefile` ROCm targets,
  - this repository's docs (`README.md`, `STRIXHALO.md`, `FORK_NOTES.md`).
- Do **not** reintroduce rejected `ds4fa` components.
- Do **not** replace upstream allocation/cache lifetimes (e.g. with
  `hipMallocAsync`) without profiling and broad regression testing.

### Tooling

- `make rocm-smoke` — allocation/copy/cleanup + optional real-model smoke.
- `make rocm-diag` — print only the runtime profile.
- `make rocm-bench-quick` — confirm gfx1151 kernels execute and report bandwidth.
- `make ci` — strict smoke test + quick bench + `misc/sync-check.sh`.
- `misc/sync-check.sh` — enforces the divergence policy (fails if too far behind
  upstream, or too far ahead without a `FORK_NOTES.md` update).
- `misc/sync.sh` — fetches upstream, runs the policy check, and rebases the
  current branch onto `upstream/main`.

