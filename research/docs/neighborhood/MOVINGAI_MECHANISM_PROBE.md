# MovingAI InitLNS mechanism probe

## Purpose

This is a stopping-rule experiment between the failed local-representation audit
and any expanded collection, supervised model, or RL training. It asks three
mechanism questions on standard maps:

1. Do different seed/rule/size actions produce different realized neighborhoods
   and immediate repair outcomes?
2. Is between-action variation larger than variation between two controlled trial
   seeds of the same action?
3. Do oracle Pareto-family preferences vary with the map or, within a map, agent
   density more than expected after episode-level permutation?

It does not test learned transfer and cannot establish an OOD claim. The MovingAI
`random-1` scenarios are static scenario prefixes, not release-time task streams.

## Registered design

`research/configs/neighborhood/movingai_mechanism_probe_dataset.json` selects two densities on each of:

- `random-32-32-20`
- `maze-32-32-2`
- `room-32-32-4`
- `warehouse-10-20-10-2-1`
- `warehouse-20-40-10-2-1`
- `den520d`

The resulting 12 instances use solver seeds 0 and 1. Each repairable Adaptive
episode contributes at most two evenly spaced states. A state evaluates at most
four conflicting seeds, Target/Collision/Random, sizes 4/8/16, and two trial
seeds at Horizon 1. The theoretical maximum is 3,456 outcomes.

The primary Pareto relation contains feasibility, remaining conflicts, and
conflict AUC. Generated nodes and runtime are recorded by the collector but are
not part of this probe's primary mechanism label.

## Run sequence

The fetched MovingAI files remain immutable. The adapter verifies their pinned
SHA256 values and creates a fingerprinted collection-compatible dataset under
ignored `build/` storage.

```bash
python3 research/scripts/neighborhood/prepare_movingai_probe.py \
  --dataset build/movingai-dev \
  --output build/movingai-mechanism-probe-dataset

PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/movingai-mechanism-probe-dataset \
  --config research/configs/neighborhood/movingai_mechanism_probe_collection.json \
  --output build/movingai-mechanism-probe-collection \
  --phase qualify --splits probe --workers 4
```

Run baseline and counterfactual phases only after inspecting qualification. Use
`--resume` so the dataset and configuration fingerprints must match.

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/movingai-mechanism-probe-dataset \
  --config research/configs/neighborhood/movingai_mechanism_probe_collection.json \
  --output build/movingai-mechanism-probe-collection \
  --phase baseline --splits probe --workers 4 --resume

PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/movingai-mechanism-probe-dataset \
  --config research/configs/neighborhood/movingai_mechanism_probe_collection.json \
  --output build/movingai-mechanism-probe-collection \
  --phase counterfactual --splits probe --workers 4 --resume

python3 research/scripts/neighborhood/analyze_movingai_probe.py \
  --collection build/movingai-mechanism-probe-collection \
  --output build/movingai-mechanism-probe-report
```

## Decision rule

- Insufficient repairable episodes: increase only the probe densities before
  collecting counterfactual labels.
- Little action or realized-neighborhood diversity: revise the high-level action
  space; do not train RL on aliases.
- Action variation is weaker than trial noise: increase trials before modeling.
- Stable action signal plus map/density permutation signal: next test contextual
  candidate ranking on independent maps.
- Stable action signal without map/density signal: retain a dynamic-state policy
  hypothesis and narrow the static-transfer claim.

Before analysis, states with identical solver fingerprints are merged and their
candidate trials are pooled. This is necessary when different solver seeds visit
the same Adaptive states. All permutations then use a task instance as the unit,
so duplicate solver-seed trajectories and two states from one task are not treated
as independent map samples.

## Current result

The formal run produced 24 valid qualification resets, 24 Adaptive baselines, 14
repairable episodes, 24 raw selected states, and 1,368 Horizon-1 outcomes with no
collection error, invalid action, replay mismatch, or trial-count mismatch.
`den520d` and `warehouse-20-40-10-2-1` were already feasible after initial SIPPS
at the registered densities and therefore produced no repair labels.

Solver seeds 0 and 1 visited identical states in every repairable task: the 24 raw
states contained only 12 unique fingerprints. The analysis consequently pooled
the two solver-seed branches as four action trials per unique state and used the
seven repairable task instances as permutation units.

After this correction, all 12 unique states still had more than one immediate
action outcome and more than one realized neighborhood. No fixed rule/size family
was the unique Pareto family in more than 16.7% of states. However, between-action
variation explained only 39.2% of conflict variation, below the registered 50%
threshold. The map permutation percentile was 90.0% and the within-map density
percentile was 0%, both below 95%.

The registered decision is therefore `increase_trials_before_modeling`. The next
experiment must add action-level random trials on unique states; changing only the
solver seed is not useful when it reproduces the same Adaptive trajectory. No
contextual ranker or RL training is authorized by this result.

## Post-probe quality audit

The follow-up audit found additional design limitations hidden by the outcome-row
count. Candidate rankings are unstable across the two duplicate-solver trial halves,
Horizon-1 AUC is algebraically redundant with remaining conflicts, and compute-aware
Pareto labels differ substantially from effectiveness-only labels. The original
within-map density statistic is also invariant to swapping a single low/high pair;
the quality audit replaces it with an exact directional alignment test.

See `research/docs/neighborhood/MOVINGAI_PROBE_QUALITY.md`. The revised next step combines more action
trials with independent scenarios. It remains a mechanism confirmation, not a
transfer-learning result.
