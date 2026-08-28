# Verification Economics

| claim | risk | error cost | verification cost/time | chosen path | decision | outcome | residual risk |
|---|---|---|---|---|---|---|---|
| Multi-token MoE path is broken | high | Patch wrong subsystem | Low | Add a path-matched CPU oracle and independent hot-path toggles | verify | Gate/up WMMA isolated and guarded | Synthetic weights differ from production weights |
| Full-model output remains broken | high | Machine hard hang | Very high | Do not rerun; use existing evidence | defer | Not rerun | Model-level semantic recovery unconfirmed |
