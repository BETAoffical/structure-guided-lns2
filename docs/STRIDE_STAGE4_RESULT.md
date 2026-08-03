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

## Stage 4R diagnostic result

Stage 4R confirmed that the main limitation is not the 124-feature input. The
new quality label is almost the same decision target as the old control label,
while the identity of the exact best candidate remains sensitive to the PP
seed.

- Quality/control exact-winner agreement: `0.980000`.
- Quality/mean-reduction exact-winner agreement: `0.981667`.
- Comparable-pair control/quality ordering disagreement: `0.005307`.
- Control-to-quality OOF action changed on only `5.8333%` of the 600 states.
- The quality model captured only `2.4134%` of the aggregate normalized Oracle
  opportunity left by the same-data control.
- Splitting the eight PP trials into two four-seed halves gave quality-winner
  agreement `0.663333`, mean rank correlation `0.847430`, and Top-3 overlap
  `0.759444`.

These results reject another feature ablation or exact-winner target as the
next primary change. A later label revision should learn robust good sets or
rankings, but runtime TTF evidence is collected first.

Both full-data diagnostic models were exported with the frozen V2 proposal
model and a replaced realized-dynamic ranker. Portable/native selection parity
was `600/600` with zero mismatches for each model. The bundles remain
`diagnostic_only`, are not default replacements, and do not authorize formal
OOD evaluation.

The first action-preserving shadow collection (`shadow-v1`) produced 14/14
interface errors because the optimized feature engine requested only the
executed V2 model's compact features. It is retained as failed evidence and
contains no model-quality result. After requesting the union of all shadow
model features, the preregistered fresh `shadow-v2` run passed:

- 14 registered tasks, 14 maps, solver seed 101, and a 20-second wall budget.
- 14 valid episodes, 13 V2 successes, and zero episode errors.
- 213 common shadow decisions.
- Zero invalid actions, action overrides, semantic mismatches, and score-range
  fallbacks.
- Control/quality action disagreement: `9.3897%`.
- Control/V2 and quality/V2 disagreement: `61.0329%` and `59.1549%`.
- Mean model inference: `1.0732 ms` and `1.0682 ms` per decision; complete
  two-model shadow accounting averaged `6.0291 ms` per decision.

The shadow is strictly action-preserving: V2 alone selected and executed every
neighborhood. Passing it permits a small paired in-distribution TTF Quick, not
promotion and not formal Stage 5.

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

## Stage 4R Quick result

The first `quick-v1` collection is retained only as failed engineering
evidence. Its controller/source-model audit compared unlike models and produced
execution errors, so it has no valid TTF winner. After splitting the audit into
active-model feature equivalence and matching-source action equivalence, the
fresh `quick-v2` collection completed all 33 scheduled episodes and passed every
registered integrity gate.

All three controllers solved all 11 paired tasks, so capped and common-success
TTF are identical for this diagnostic cohort:

| Controller | Success | Mean wall TTF (s) | Relative TTF vs V2 | Mean repairs | Mean PP replan (s) | Mean controller (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2-full` | 11/11 | **6.6322** | reference | **9.8182** | **1.1309** | **1.7624** |
| `stride-control-v1` | 11/11 | 7.7996 | `-17.6016%` | 12.6364 | 1.5719 | 2.4366 |
| `stride-quality-v1` | 11/11 | 7.3879 | `-11.3939%` | 11.3636 | 1.4195 | 2.1943 |

Here a negative improvement means slower TTF. Both STRIDE candidates preserve
success but require more repair rounds, more PP time, and more controller time
than frozen V2. `stride-quality-v1` is better than the same-data old-label
control, but it does not beat V2. The primary winner is therefore `v2-full`.
There were zero execution errors, invalid actions, fingerprint mismatches,
initial-conflict mismatches, or semantic mismatches; the registered TTF clock
was present for every episode, and no formal OOD or test outcome was read.

This is an 11-task, one-solver-seed, in-development diagnostic and is not a
formal speed claim. It rejects promotion of either Stage 4 controller and
supports the registered next decision:
`revise_quality_label_before_more_runtime_evaluation`. The next model iteration
must improve current-step neighborhood quality in a way that reduces realized
repair rounds and TTF; increasing training volume under the old label is not a
sufficient change on this evidence.

The post-run range audit found an engineering-only reporting defect in the
first diagnostic bundles. Their manifests registered all 124 realized feature
ranges although the compact control and quality rankers use only 74 and 73
base features. Dense runtime rows intentionally materialize only model inputs,
but the range diagnostic treated every omitted, pruned feature as zero and
could therefore report a false out-of-range value. This did not change scores,
rankings, actions, fallbacks, or the Quick TTF result. Recomputing against only
the actual compact inputs reduced mean selected-feature out-of-range fractions
from `0.1423` to `0.0358` for control and from `0.1828` to `0.0365` for quality;
neither controller had a corrected decision above `0.25`. The export and bundle
loader now require exact equality between registered range names and compact
model inputs. Fresh `models-v2` bundles preserve the original model payloads
and pass portable selection equivalence on all `600/600` development states.

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
- Stage 4R diagnostic report SHA-256:
  `8513f0c34e4d7082a892bd43fd741b5c2180365d0f5313609f51e72460e93f9c`
- Stage 4R diagnostic state table SHA-256:
  `6849189837cc9271f66eb5da85aa7285fba1bb0551196a64dc09f8172eeac0dd`
- Stage 4R model-export report SHA-256:
  `fcee4d17a386862a7665cfc7abdd49da6f2ad7ec5b901942c1070c6d2fc38b3b`
- Stage 4R shadow-v2 audit SHA-256:
  `643b3a28cb749a9e20a7ba408e06c38b0aea4d908ef57db59d3900cffabf254f`
- Stage 4R Quick-v2 report SHA-256:
  `db982f7a78ebb0a55cfc65c8086f20f9e1168a9b733ecf4f2392e3c3d1cec1a0`
- Stage 4R Quick-v2 execution schedule SHA-256:
  `a35ff001e9f920a03f1ab770ffbb3ca11d582a718c4295080fed1fcf8004e45b`
- Stage 4R Quick-v2 V2/control/quality manifest SHA-256 values:
  `29643131736e44c8c592b70b18f8d286baa2c26cd8387e3adeda3a6fac5cdd48`,
  `15e1856de766e0713efccbf7d19f7ead7fb8ef7ea1132f0a93dc27e2adae97b1`,
  and `c5b37f9ed7015d91831f6bb5bc89fd1e34200d372dea23819c622e1cb1d28983`.
- Stage 4R range-contract `models-v2` export report SHA-256:
  `044fa1e1770ba1bd751be052e63014b1c1999e508b097bed5e22745f25dd416f`
- Stage 4R range-contract control/quality manifest SHA-256 values:
  `23cf2c6888f26e9bb9b4bb157f648a11ef0eeb11bf711ae88016decb1ba75cb6`
  and `f6f005bffdf02ac6bf8c7af6e5bbfd743ae6df1f1ddf840a4be35c56f4b348fc`.
