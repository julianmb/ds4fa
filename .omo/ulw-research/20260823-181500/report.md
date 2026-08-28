# ROCm IQ2/Q2_K Prefill Corruption: Root Cause and Fix

## Executive Summary

Issue #2 was reduced to a model-free routed-MoE parity failure on AMD gfx1151. The failure is not in IQ2_XXS dequantization, pair sorting, scalar Q2_K float-down, FP16 down-result storage, or generic HIP stream ordering. It begins when an expert reaches the hot threshold of eight token-slot pairs and the IQ2 gate/up WMMA overlay replaces the scalar result.

The correctness-first patch disables that gate/up WMMA overlay by default and retains `DS4_ROCM_ENABLE_IQ2_GATE_WMMA=1` as an explicit benchmark opt-in. Sorted dispatch and the Q2_K down WMMA path remain enabled.

The synthetic regression changed from a 16-token maximum absolute error of `6.284668` to `0.401123`; token counts 1, 2, 4, and 16 now pass their path-matched CPU oracles. A full-model confirmation was deliberately not run because prior model diagnostics hard-hung the host.

## Scope and Safety

- Hardware/runtime: Radeon gfx1151, ROCm build/runtime/driver `7025.32.11`.
- Reproduction: synthetic IQ2_XXS gate/up and Q2_K down weights only.
- Prohibited during this investigation: loading the 86 GB GGUF or running full-model diagnostics.
- User-visible symptom retained as prior evidence: CPU output coherent; ROCm output deterministic and incoherent.

## Experimental Findings

### 1. The original batch oracle was mismatched to the optimized path

The rollback path quantizes each mid activation to Q8_K before Q2_K down projection. The optimized float-down path intentionally consumes unquantized float mids. Comparing both paths to the Q8_K oracle produced thousands of differences at 2 and 4 tokens, but those differences were not corruption.

A path-matched CPU Q2_K-by-float oracle showed:

| Tokens | Float-oracle RMSE | Maximum absolute error | Verdict |
|---:|---:|---:|---|
| 2 | `0.000062` | `0.000244` | Scalar float-down correct |
| 4 | `0.000063` | `0.000252` | Scalar float-down correct |
| 16, hot overlays active | `1.438592` | `6.276001` | Incorrect |

This refutes pair sorting and scalar float-down as the primary cause.

### 2. FP16 down-result storage is not causal

Forcing float output storage while retaining the hot overlays left the 16-token error essentially unchanged: `16038/32768` outputs exceeded `1.0`, with `max_abs=6.276001`. The normal FP16-output run reached `max_abs=6.284668`.

### 3. The IQ2 gate/up WMMA overlay is causal

Disabling only gate/up WMMA while retaining sorted dispatch and down WMMA produced:

- Float output: zero errors over `1.0`, `rmse=0.072447`, `max_abs=0.341537`.
- Normal FP16 down storage: zero errors over `1.0`, `rmse=0.090038`, `max_abs=0.401123`.

Re-enabling gate/up WMMA restores the failure. This satisfies the cause-toggle requirement.

### 4. Gate and down hot paths have a storage contract

The gate overlay writes hot mids to FP16 storage while the scalar gate path writes float mids. Disabling down WMMA alone while leaving gate WMMA active caused scalar down to read unwritten float mids and output zeros for hot experts. This is not evidence that scalar down is wrong; it demonstrates that the two hot overlays are coupled through mid-storage selection.

## Patch

In `src/rocm/ds4_rocm_moe_launch.cuh`, `use_iq2_gate_wmma` now requires:

```c
getenv("DS4_ROCM_ENABLE_IQ2_GATE_WMMA") != NULL
```

Consequences:

- Safe scalar sorted gate/up is the default.
- Q2_K float-down and down WMMA remain enabled.
- The unsafe overlay remains available for targeted benchmarking and future repair.
- Existing `DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED` rollback remains available but is no longer needed to avoid this confirmed defect.

## Regression Coverage

`make rocm-moe-iq2-q2k-test` builds and runs a synthetic routed-MoE test with path-matched CPU references.

| Tokens | Oracle | Post-fix maximum absolute error | Limit |
|---:|---|---:|---:|
| 1 | Q8_K rollback path | `0.000153` | `0.010000` |
| 2 | Q2_K-by-float | `0.199165` | `0.500000` |
| 4 | Q2_K-by-float | `0.205505` | `0.500000` |
| 16 | Q2_K-by-float | `0.401123` | `0.500000` |

Red/green proof:

- Before/default overlay: 16 tokens failed at `max_abs=6.284668`.
- After/default scalar gate: all token counts passed.
- After with `DS4_ROCM_ENABLE_IQ2_GATE_WMMA=1`: 16-token failure returned at `max_abs=6.284668`.

## External Evidence

- HIP operations issued to the same stream execute in order, so a generic missing same-stream barrier was not the leading explanation: [HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html).
- Upstream PR #484 independently disabled an IQ2/Q2 float-down WMMA optimization on affected AMD architectures. Its architecture guard is not directly transferable to gfx1151, but it supports treating new WMMA overlays as opt-in until parity is demonstrated: [antirez/ds4#484](https://github.com/antirez/ds4/pull/484).
- Upstream PR #602 concerns multi-row hyper-connection normalization. This fork uses an all-row fused normalization launch, and the isolated MoE toggle proves the present defect occurs earlier: [antirez/ds4#602](https://github.com/antirez/ds4/pull/602).

## Verification

- `make rocm-moe-iq2-q2k-test`: passed all four token counts.
- `make strix-halo -j"$(nproc)"`: completed successfully.
- `make rocm-smoke`: `rocm-smoke: PASSED`.
- `DS4_ROCM_ENABLE_IQ2_GATE_WMMA=1 ./tests/test_moe_iq2_q2k_batch`: reproduced the expected 16-token regression.

## Residual Risk

The synthetic harness confirms the corrected routed-MoE dispatch mechanism, but the original full-model prompt was not rerun due the documented hard-hang risk. Full-model semantic recovery therefore remains unconfirmed. If the user later approves that risk, the first model-level check should use the new safe default with strict memory limits and a single short prompt; it should not enable IQ2 gate WMMA.

The overlay itself remains defective. A future optimization repair should compare its gate/up mid tensor directly against the scalar kernel before down projection, including counts at 7, 8, 9, and larger hot batches.
