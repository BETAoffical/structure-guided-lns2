# InitLNS natural-distribution confirmation

## Purpose

This stage preserves the natural difference between open and constrained layouts. Zero-conflict PP
initializations are successful observations in the end-to-end cohort. High-conflict initializations are
hard repair states rather than invalid samples. Only nonzero states receive one-step neighborhood-ranking
labels.

The earlier 12-map `v1b` set is now a design Pilot because its conflict distribution was inspected while
the qualification rule was revised. The formal confirmation uses master seed `20261117`, 12 entirely new
maps, and 48 balanced/bottleneck by 80/100-agent tasks without task replacement or conflict-based
resampling.

## Registered gates

All 48 resets must be valid. The unfiltered formal cohort must contain at least 30 nonzero states, at
least eight per layout family, and at least ten active maps. Failure is reported as an inconclusive sample;
the generator must not draw replacement task seeds.

Conflict density is `2E/(N(N-1))`. The Pilot fixes low severity at at most `0.001`, medium severity at
`(0.001, 0.01]`, and high severity above `0.01`. Severity and static context are explanatory analyses and
do not control the primary ranking gate.

Each explicit `(state, candidate, trial)` repair has a deterministic ID, a separate result file, an
independent reset and evaluation seed, and a 120-second process timeout. A timed-out trial can be resumed
without discarding completed trials from the same state. Formal ranking requires all eight trials for
every candidate; incomplete states are never silently dropped.

## Models and decision

The rankers remain frozen from the earlier 23-state, 412-candidate development index. Their source index,
report and pickle hashes are checked before formal labels are loaded. The primary comparison remains
`realized_dynamic` against `proposal_dynamic`; uniform random, internal conflict coverage and oracle are
reported alongside it. Static context cannot rescue a failed primary gate.

Passing requires at least five percentage points of Pareto top-1 gain, at least 5% lower remaining-conflict
regret, no significant map-bootstrap degradation, no worse results on at least two thirds of active maps,
improvement over both simple baselines, and no unsupported greater-than-80% neighborhood-size collapse.

## Commands

```powershell
python scripts/generate_dataset.py `
  --config configs/natural_distribution_confirmation_dataset.json
python scripts/run_natural_distribution_confirmation.py --mode freeze `
  --output build/initlns-natural-distribution-confirmation-v1-frozen-models
```

The Pilot smoke uses the prior `v1b` dataset and
`configs/natural_distribution_confirmation_pilot.json`. The formal collector uses
`configs/natural_distribution_confirmation_collection.json` and phases `qualify`, `baseline`, `propose`,
and `evaluate` with `--resume` after the first phase. All generated maps, traces, trials, models and
reports remain in ignored `build/` directories.

The registered Pilot smoke retained the inspected 619-conflict state, generated six proposals and three
explicit neighborhoods, and completed six isolated repair trials with zero errors and zero timeouts. The
three candidates had mean remaining-conflict values of 516, 557.5 and 595. Trial storage uses a short
state hash on disk to remain below Windows path limits while preserving the complete state ID in JSON.
These outcomes validate the mechanism only and are forbidden from formal model evaluation.

No RL training occurs in this stage. A pass permits a separate fresh-map closed-loop test; a failure keeps
RL paused and redirects work toward candidate construction or PP repair-order control.
