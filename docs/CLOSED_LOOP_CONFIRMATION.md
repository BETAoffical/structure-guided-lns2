# InitLNS frozen realized-neighborhood closed-loop confirmation

## Purpose

The natural-distribution confirmation established that a frozen `realized_dynamic` ranker can choose a
better one-step explicit neighborhood on unseen maps. This stage tests whether that offline advantage
survives sequential control. It does not train or tune a model and does not evaluate static context.

The formal cohort uses master seed `20261219`, six new maps with two replicates from each registered
layout family, and all 24 balanced/bottleneck by 80/100-agent tasks. Tasks are never replaced based on
their conflict result. Qualification requires 24 valid resets, 18 nonzero states, four nonzero states per
layout and five active maps; failure is reported as inconclusive without drawing replacement seeds.

## Controller

At every nonterminal state, the learned policies generate Target, Collision and Random proposals at sizes
4, 8 and 16. Up to four conflict seeds, eight proposal seeds and two representative sets per family give
at most 288 proposal calls and 18 explicit candidates. Proposal generation must preserve the complete
state fingerprint.

The frozen pairwise GBDT scores every candidate against every other candidate. `proposal_dynamic` is the
representation ablation and `realized_dynamic` is the primary policy. Candidate ID breaks score ties.
The explicit repair seed depends on task, solver seed, state, decision and candidate, but not model name;
two policies selecting the same state and set therefore receive the same repair order. Invalid actions are
errors and never fall back to Adaptive.

The WSL system Python intentionally remains dependency-free. `scripts/export_closed_loop_models.py`
exports the frozen sklearn trees to the versioned
`artifacts/initlns-closed-loop-policy-v1/` bundle. It contains the trees, development feature ranges and
source provenance needed for collection, so inference does not depend on ignored `build/` files or an
WSL sklearn installation. Exported probabilities and candidate selections must match sklearn on the
complete development index before collection starts.

The 300-second end-to-end budget includes reset, proposals, feature extraction, inference and repair.
Native solver time and controller overhead are also reported separately. Episodes stop after 100 repair
steps, and each process has a 360-second hard timeout. Completed episodes support resume; partial JSONL
traces remain diagnostic only and are rerun from reset.

## Registered analysis

Official Adaptive, frozen `proposal_dynamic` and frozen `realized_dynamic` run from identical initial
fingerprints. Success is evaluated over all tasks. Policy-effect metrics use nonzero tasks and penalize
failure with 300 seconds and a 100-step conflict AUC padded with the final conflict count; successful
trajectories are padded with zero.

The primary policy passes only when its success count is not below Adaptive and either capped wall time or
fixed-budget conflict AUC improves by at least 5%. The qualifying metric must be no worse on at least four
of six maps and its 2,000-sample paired map bootstrap must not show significant degradation. All policy
episodes must be valid, with no action or fingerprint errors. `proposal_dynamic` is explanatory and does
not control the gate.

## Commands

```powershell
python scripts/export_closed_loop_models.py
python scripts/generate_dataset.py --config configs/closed_loop_confirmation_dataset.json
python scripts/collect_closed_loop_confirmation.py `
  --dataset build/initlns-closed-loop-confirmation-v1 `
  --output build/initlns-closed-loop-confirmation-v1-collection `
  --phase qualify
```

After qualification, run `official_adaptive`, `proposal_dynamic` and `realized_dynamic` with `--resume`,
then run:

```powershell
python scripts/analyze_closed_loop_confirmation.py `
  --collection build/initlns-closed-loop-confirmation-v1-collection `
  --output build/initlns-closed-loop-confirmation-v1-report `
  --strict
```

Before formal collection, the historical 619-conflict task is used for a two-step mechanism smoke with
`configs/closed_loop_confirmation_pilot.json`. Pilot outcomes are forbidden from formal analysis.

Passing permits policy-visited counterfactual collection and RL warm-start work. Failure keeps RL paused
and is diagnosed using candidate coverage, feature-range shift and trajectory divergence.

## Formal result

The unfiltered formal cohort passed qualification. All 24 resets were valid, three beltway tasks were
already feasible, and 21 tasks required repair. Every map contributed repair states. Initial conflicts
ranged from zero to 98, with 14 low, seven medium and three high-severity tasks.

All three policies solved 24/24 tasks with zero errors, timeouts, invalid actions or initial-fingerprint
mismatches. The primary `realized_dynamic` policy passed every registered gate:

- fixed-budget conflict AUC fell from 115.12 for Adaptive to 49.21, a 57.25% improvement;
- all six maps were no worse than Adaptive, and the map-bootstrap 95% improvement interval was
  `[44.66%, 61.07%]`;
- mean repair iterations fell from 12.71 to 7.14;
- mean low-level generated nodes fell by 19.37%, expanded nodes by 23.55%, reopened nodes by 30.07%, and
  low-level runs by 8.68% relative to Adaptive;
- `realized_dynamic` also improved AUC by 16.95% over the `proposal_dynamic` ablation.

The original registered implementation was not a wall-clock speedup. Mean end-to-end time on repairable
tasks rose from 0.42 seconds for Adaptive to 13.44 seconds for `realized_dynamic`; 13.11 seconds was
candidate-control overhead.

The subsequent hardening pass does not change that registered scientific result. It adds one native
proposal batch, a native portable-tree predictor, per-state topology/path caches, strict trace validation,
implementation fingerprints and an equivalence checker. A complete replay matched all 72 episodes and
602 transitions exactly, including selected neighborhoods, repair seeds, state fingerprints, conflict
trajectories and low-level counts. On that replay, mean `realized_dynamic` controller time on the 21
repairable tasks fell from 13.11 to 0.52 seconds (about 96%), and end-to-end time was 0.72 seconds versus
0.24 seconds for same-run Adaptive. The controller is therefore practical enough for larger confirmation,
but it is still about three times slower in wall time and no runtime superiority is claimed.

Reproduce the scientific comparison while excluding timing fields with:

```powershell
python scripts/verify_closed_loop_equivalence.py `
  --reference build/initlns-closed-loop-confirmation-v1-collection `
  --candidate build/initlns-closed-loop-hardening-final-v2
```

The registered decision is `advance_to_policy_visited_data_and_rl_warm_start`. The result establishes that
the frozen dynamic realized-neighborhood policy produces substantially better sequential conflict
trajectories on fresh maps. It does not establish a practical runtime gain, static-context transfer, OOD
generalization, or an RL result. The next stage must collect policy-visited states and reduce candidate
distribution shift with multiple solver seeds before RL warm-start work.
