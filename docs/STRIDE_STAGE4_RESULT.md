# STRIDE-LNS Stage 4 result

Stage 4 completed the preregistered five-fold, map-grouped comparison of the
frozen `v2-full` anchor, the same-data old-label control
`stride-control-v1`, the new-label `stride-quality-v1`, and four registered
feature ablations. The run completed normally, read neither test nor formal
OOD data, but did not pass the offline promotion gate.

## Registered cohort

- 600 independent decision states from 36 maps.
- 300 `official_adaptive` and 300 `v2-full` source states.
- 10,713 candidate neighborhoods and eight paired PP seeds per candidate.
- 166,402 oriented quality-label rows and 163,944 oriented control-label rows.
- Five validation folds grouped by complete map; every fold contains all four
  registered layout families.
- Full feature input: 124 frozen realized-dynamic features.

The outcome-blind fold manifest passed every coverage gate and records
`label_outcomes_read: false`.

## OOF comparison

All selection metrics below use the new Stage 4 quality score as a common
evaluation yardstick. Lower normalized regret is better.

| Model | Pairwise accuracy | Exact best | Top-3 | Mean quality regret | Normalized regret |
|---|---:|---:|---:|---:|---:|
| frozen `v2-full` | N/A | 0.371667 | 0.456667 | 0.113722 | 0.282464 |
| `stride-control-v1` | 0.710348 | 0.400000 | 0.495000 | 0.106639 | 0.260578 |
| `stride-quality-v1` | 0.707058 | 0.411667 | 0.503333 | 0.101116 | **0.254289** |

The quality model improved normalized regret over the same-data control by
`0.006289` absolute or `2.4134%` relative. The preregistered requirement was
`0.01` absolute or `5%` relative. This was the only failed full-model gate:
pairwise accuracy, frozen-anchor noninferiority, Exact-best/Top-3
noninferiority, three-of-five fold wins, and all registered subgroup gates
passed.

The same-data control improved normalized regret over the historical frozen
model by about `7.75%`. Consequently, expanded data and retraining explain most
of the observed offline improvement; the new label supplied a smaller
increment.

## Feature ablation

| Quality variant | Base features | Pairwise accuracy | Exact best | Top-3 | Normalized regret | Passed |
|---|---:|---:|---:|---:|---:|---:|
| full | 124 | 0.707058 | 0.411667 | 0.503333 | **0.254289** | no |
| no path geometry | 103 | 0.706738 | 0.398333 | 0.491667 | 0.259278 | no |
| no proposal context | 65 | 0.685851 | 0.378333 | 0.483333 | 0.263674 | no |
| no realized context | 82 | 0.657639 | 0.256667 | 0.351667 | 0.340638 | no |
| no state context | 101 | 0.705881 | 0.400000 | 0.485000 | 0.265605 | no |

Removing realized neighborhood measurements caused the largest degradation.
There is no evidence to replace or reduce the full 124-feature input before a
label and learnability diagnosis.

## Decision

`stride-quality-v1` is not eligible for formal Stage 5. No promoted or
Stage-5-eligible portable model was produced. The stored
`portable_equivalence_passed: false` means export was deliberately not run
after the offline gate failed; it is not an equivalence-test failure.

The six-stage sequence pauses for a registered Stage 4R diagnostic. Stage 4R
will distinguish weak label change, model learnability, and PP-seed noise. Any
runtime execution of the control or quality model before a new promotion gate
passes is explicitly diagnostic-only.

## TTF-first runtime objective

The user-selected final objective is lower end-to-end wall-clock time to first
feasibility (TTF). Offline regret is a screening proxy only. Runtime comparison
will use mean capped wall TTF as the primary value, with success-count
noninferiority as a hard constraint. Common-success paired TTF, repair rounds,
fixed-step wall-AUC, PP time, controller overhead, and failures remain required
explanatory metrics.

TTF starts immediately before `env.reset()` and ends at the first feasible
state. It includes reset/initial PP, candidate generation, feature extraction,
model inference, and PP+SIPPS repairs. An unsuccessful episode is charged the
same registered wall budget in capped TTF; actual consumed wall time is reported
separately.

## Reproducibility

- Stage 4 config SHA-256:
  `d54357ca8859229e7619cc7fba90a70294d7076279b77cc53ed50fddb5602808`
- Protocol report SHA-256:
  `29b5654cfbaef335bfa7b56f40ddd59d4d57aa9f45daac9253098e3ab4b8d8c6`
- Fold manifest SHA-256:
  `28eca15b6292126c862ac13ae43aa5a4f3ed59ecadd3d437ab9fe50a281e56e2`
- Candidate aggregates SHA-256:
  `e4976b2da43e617452bde0fa753ac01d8626d263f40495ea54773ae5847d994c`
- Dominance pairs SHA-256:
  `c2b028285c446555ec75c99a79452a62d31eebad331fa1edf01460a8193b62b7`
- Stage 4 report SHA-256:
  `536ab28dd9560843c4556e19a35de8f73e3df23d294a8e377ae247308ba815d1`
- OOF predictions SHA-256:
  `8584b33a4966a2712126a5fbeb9a1561b12b973e92cb6824dc9105c0a5e1a1d6`
