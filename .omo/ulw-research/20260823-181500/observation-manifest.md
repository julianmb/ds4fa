# Observation Manifest

| observation_id | source | layer | observer group | independence basis | observer | observed_at | valid_at | artifact | anchor | contamination |
|---|---|---|---|---|---|---|---|---|---|---|
| O1 | `tests/test_moe_iq2_q2k_batch.cpp` | local source/runtime | path-matched oracle | Independent CPU Q2_K-by-float implementation | Sisyphus | 2026-08-23 | current checkout | `report.md` | Experimental Findings 1 | Synthetic random weights |
| O2 | Default IQ2 gate WMMA run | local runtime | red test | Explicit 16-token hot-threshold reproduction | Sisyphus | 2026-08-23 | ROCm 7.2.5 gfx1151 | `report.md` | Regression Coverage | No full model |
| O3 | Gate-WMMA-disabled run | local runtime | cause toggle | Sorted gate/down paths retained; only gate overlay disabled | Sisyphus | 2026-08-23 | ROCm 7.2.5 gfx1151 | `report.md` | Experimental Findings 3 | Synthetic random weights |
| O4 | `DS4_ROCM_ENABLE_IQ2_GATE_WMMA=1` after patch | local runtime | reverse toggle | Explicit opt-in restores the same failure | Sisyphus | 2026-08-23 | ROCm 7.2.5 gfx1151 | `report.md` | Regression Coverage | No full model |
| O5 | `make strix-halo`; `make rocm-smoke` | local build/runtime | integration QA | Full gfx1151 build plus HIP smoke | Sisyphus | 2026-08-23 | current checkout | `report.md` | Verification | Model-free only |
