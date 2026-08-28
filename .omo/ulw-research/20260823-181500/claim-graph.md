# Claim Graph

## Verified Claims

- C2: the IQ2 gate/up hot WMMA overlay is incorrect on gfx1151 once an expert count reaches eight.
- C3: defaulting to scalar sorted gate/up removes the isolated failure while retaining optimized Q2_K down.

| claim_id | statement | type | risk | scope | intent ids | supporting observations | contradicting observations | independent groups | convergence | counter-search | primary source | dependencies | status | synthesis location |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | ROCm corruption occurs before coherent generation is established | causal | high | full model | I1 | Prior run logs | Full-model rerun prohibited | 1 | partial | complete | local runtime | Safe model-level confirmation | partial | report.md |
| C2 | The IQ2 hot gate/up WMMA overlay is incorrect on gfx1151 | causal | high | isolated harness | I2 | Default/disable/enable toggle; path-matched oracle | Scalar float-down parity | 2 | converged | complete | local source/runtime | None | verified | report.md |
| C3 | Gate WMMA should be opt-in until repaired | recommendation | high | ds4fa | I3 | Red-green regression; full Strix build; ROCm smoke | Full-model recovery unconfirmed | 2 | converged | complete | local source/runtime | C2 | verified | report.md |
