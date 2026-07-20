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
  --config research/configs/neighborhood/natural_distribution_confirmation_dataset.json
python research/scripts/neighborhood/run_natural_distribution_confirmation.py --mode freeze `
  --output build/initlns-frozen-policy-v1-sklearn
```

The Pilot smoke uses the prior `v1b` dataset and
`research/configs/neighborhood/natural_distribution_confirmation_pilot.json`. The formal collector uses
`research/configs/neighborhood/natural_distribution_confirmation_collection.json` and phases `qualify`, `baseline`, `propose`,
and `evaluate` with `--resume` after the first phase. All generated maps, traces, trials, models and
reports remain in ignored `build/` directories.

The registered Pilot smoke retained the inspected 619-conflict state, generated six proposals and three
explicit neighborhoods, and completed six isolated repair trials with zero errors and zero timeouts. The
three candidates had mean remaining-conflict values of 516, 557.5 and 595. Trial storage uses a short
state hash on disk to remain below Windows path limits while preserving the complete state ID in JSON.
These outcomes validate the mechanism only and are forbidden from formal model evaluation.

## Formal result

The unfiltered formal cohort passed qualification: all 48 resets were valid, seven PP initializations were
already feasible, and 41 required repair. All 12 maps contributed repair states, with 12, 15 and 14
nonzero states from `regular_beltway`, `compartmentalized` and `dead_end_aisles`, respectively. Initial
conflicts ranged from zero to 265. The natural severity distribution contained 23 low, 21 medium and four
high tasks, including zero-conflict tasks in the low end-to-end stratum.

Official Adaptive solved all 48 tasks. The mean time to feasibility was 0.273 seconds and the mean repair
count was 10.75 iterations. The conditional ranking cohort contained 41 states, 733 explicit neighborhoods
and 5,864 isolated trials. Collection completed with no errors, timeouts, missing trials or orphaned
outcomes; a resume audit recovered all 5,864 jobs without repeating repair work.

The frozen `realized_dynamic` ranker passed every registered primary gate against `proposal_dynamic`:

- Pareto top-1 increased from 19.51% to 43.90%, a gain of 24.39 percentage points.
- Mean remaining-conflict regret decreased from 0.5074 to 0.3357, a relative reduction of 33.83%.
- The map bootstrap intervals were positive for both top-1 gain `[0.0833, 0.3889]` and conflict-regret
  improvement `[0.0018, 0.3584]`.
- It was no worse on 10 of 12 maps, exceeded uniform random and internal-conflict coverage, and selected
  one neighborhood size in only 65.85% of states.

The improvement was strongest in medium and high conflict states. In the low stratum, top-1 changed from
31.25% to 37.50%; in medium it changed from 14.29% to 42.86%; in high it changed from 0% to 75%. The high
stratum contains only four states, so it is descriptive rather than stand-alone evidence.

Adding static layout, OD and density context did not add reliable value: `realized_context` reached 41.46%
top-1 versus 43.90% for `realized_dynamic`, while its conflict-regret reduction was only 0.39%. The formal
result therefore supports dynamic-state plus explicit-neighborhood ranking on unseen maps, but does not
restore the static transfer claim.

No RL training occurred in this stage. The registered decision is
`proceed_to_fresh_closed_loop_confirmation`: next, the frozen `realized_dynamic` ranker must choose an
explicit neighborhood at every repair step on another fresh map cohort and be compared end to end with
official Adaptive. RL remains paused until that closed-loop test succeeds.
