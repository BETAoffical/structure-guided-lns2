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
exports the frozen sklearn trees to a numerical JSON representation whose SHA and source-pickle SHA are
registered in the collection config. Exported probabilities and candidate selections must match sklearn
on the complete development index before collection starts; no package is installed in WSL.

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
