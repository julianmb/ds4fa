# Intent Diff

| intent_id | expected truth | observed reality | diff | violated invariant | intent source | supporting observations | status | claim ids |
|---|---|---|---|---|---|---|---|---|
| I1 | ROCm prefill and decode preserve CPU-equivalent model semantics | Full-model ROCm output was incoherent while CPU output was reported coherent | Model-level recovery not rerun due hard-hang risk | Backend parity | User request and issue #2 | Prior logs plus isolated root cause | partial | C1 |
| I2 | Multi-token MoE matches its active numerical path's CPU reference | Default gate WMMA exceeds the oracle limit at the hot threshold; scalar gate passes | Unsafe hot overlay | Batch kernel parity | Local runtime evidence | Red/green synthetic test | resolved | C2 |
| I3 | The fix is supported by a safe cause toggle | Default safe path passes; explicit gate-WMMA opt-in restores failure | None in isolated scope | Debugging evidence | User request | Repeated toggle proof | resolved | C3 |
