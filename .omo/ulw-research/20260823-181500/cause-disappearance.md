# Cause Disappearance

| cause_id | expected truth | previous observation | last_seen | disconfirming observation | replacement cause | status | violation gone |
|---|---|---|---|---|---|---|---|
| CA1 | Disabling all WMMA should restore correctness if any WMMA is causal | `--quality` still produced incoherent output | 2026-08-23 | Quality mode does not disable sorted float-down and was not a precise gate-overlay toggle | CA2 | superseded | no |
| CA2 | Disabling IQ2 gate/up WMMA should restore path-matched MoE parity | 16-token `max_abs=6.284668` with overlay | 2026-08-23 | Gate-disabled `max_abs=0.401123`; explicit re-enable restores failure | IQ2 gate/up hot WMMA overlay | confirmed | yes |
