# InitLNS policy-visited natural-distribution development

## Study role

The `20270317` cohort is a development dataset. Its original strict qualification completed 216 valid
resets but failed the preregistered 75% per-layout repairable-rate gate. No source trajectory, candidate
label or aggregate model was produced by that run. This version retains all maps and zero-conflict
episodes, conditions ranking labels on states that actually require repair, and does not reinterpret the
development cohort as independent confirmation evidence.

The three core layout mechanisms remain: complete beltway redundancy, compartment gates and local
dead-end aisles. Historical OOD families are not used for training in this stage.

## Development qualification and labels

The natural-distribution development gate requires all 216 resets to be valid, distinct solver streams,
historical seed isolation, all 18 maps to provide repair data, at least 96/48 nonzero Train/Validation
episodes, and at least 24/12 per layout. The known qualification counts satisfy these execution gates;
that fact is explicitly post-qualification development information rather than a new confirmation result.

Frozen `realized_dynamic` trajectories contribute at most three early/middle/late states. Candidate pools
must exactly reproduce the source trace, and each explicit neighborhood receives four independent PP
repair-order trials. Zero-conflict episodes remain in end-to-end statistics and produce no ranking label.

The primary pairwise GBDT gives every trainable state total weight one. An inverse-layout-weighted model is
trained only as a sensitivity diagnostic and is excluded from the portable deployment bundle and all
acceptance decisions. Validation labels are never used for fitting.

## Independent confirmation

Independent confirmation is registered before its maps are generated. Master seed `20270421` produces
six new maps, two per core layout, and four balanced/bottleneck tasks per map. Solver seeds are `[1,2,3]`.
The cohort is never replaced based on conflict outcomes.

Qualification requires 72 valid resets, at least 36 repairable episodes, at least eight per layout and at
least five active maps. It runs only after the development offline gate passes. The frozen v2 model is
then compared with Adaptive and frozen v1 on the same task-seeds. Passing requires preserved success,
at least 5% conflict-AUC improvement over Adaptive, no more than 5% AUC degradation from v1, at least four
of six maps no worse than v1, and no invalid action, fingerprint mismatch or unexplained error.

## Commands

Development collection runs in WSL and writes to a new fingerprinted directory:

```bash
PYTHONPATH=build/linux/project python3 research/scripts/policy/collect_policy_visited_experience.py \
  --dataset build/initlns-policy-visited-v1 \
  --config research/configs/policy/policy_visited_natural_collection.json \
  --output build/initlns-policy-visited-natural-v2-collection \
  --phase all --workers 4
```

Windows trains and evaluates the fixed development learner:

```powershell
python research/scripts/policy/run_policy_visited_aggregation.py --phase train `
  --collection build/initlns-policy-visited-natural-v2-collection `
  --config research/configs/policy/policy_visited_natural_analysis.json `
  --output build/initlns-policy-visited-natural-v2-training
python research/scripts/policy/run_policy_visited_aggregation.py --phase offline `
  --collection build/initlns-policy-visited-natural-v2-collection `
  --training build/initlns-policy-visited-natural-v2-training `
  --config research/configs/policy/policy_visited_natural_analysis.json `
  --output build/initlns-policy-visited-natural-v2-offline
```

Only after the offline gate passes, generate the registered confirmation dataset and use the independent
confirmation runner. Intermediate maps, traces, indexes and models remain ignored under `build/`.

## Development result (2026-07-15)

Natural-distribution qualification passed as a development execution gate. All 216 resets were valid,
178 episodes had nonzero initial conflicts, and all 18 maps supplied at least one repair state. The
nonzero counts were 115 for Train and 63 for Validation. Per-layout counts were 48/38/29 for Train and
24/16/23 for Validation (`compartmentalized/dead_end_aisles/regular_beltway`). These differences are
retained as part of the natural conflict distribution rather than balanced by task replacement.

The frozen v1 source trajectories and the Validation Adaptive baseline produced 288 valid episodes.
Policy-state selection retained 442 early/middle/late states and 7,914 unique explicit neighborhoods.
All 31,656 expected PP repair-order trials completed, with zero collection errors, timeouts, replay
fingerprint mismatches or invalid labels. The resulting index contains 288 Train states and 154
Validation states; each candidate has exactly four trials.

The aggregate v2 learner was fit on the historical 23 states plus the 288 new Train states. Validation
labels were not used for fitting. Native sklearn and portable inference selected the same candidate on
all 311 training states; the maximum score difference was `3.55e-15`. The inverse-layout-weighted model
remained sensitivity-only and was not eligible for deployment or acceptance decisions.

The preregistered development offline gate failed. On 154 Validation states, v1 and v2
`realized_dynamic` both achieved a Pareto top-1 rate of `35.06%`. Mean remaining-conflict regret changed
from `0.381817` to `0.362729`, a relative improvement of `4.9993%`, just below the required 5%. More
importantly, the map-level 95% bootstrap interval for regret improvement was
`[-17.41%, 21.34%]`, so stability across the six Validation maps was not established. Size collapse did
not occur: the largest v2 selected-size share was `48.70%`, and the oracle supported multiple sizes in
63 of 154 states.

The registered decision is therefore to stop before independent confirmation. The `20270421` maps are
not generated, v2 is not promoted to a closed-loop policy, and RL pretraining remains paused. The near-
threshold point estimate is recorded as weak development evidence only, not as cross-map confirmation.

## Operational hardening

One resumed collection was interrupted when Windows briefly held the DrvFS destination of an atomic
progress-file replacement. Per-trial atomic outputs remained intact and resume completed the exact
registered run fingerprint. Atomic writes now retry bounded transient `PermissionError` failures.
Candidate ranking also treats score differences below 12 decimal places as ties before applying the
candidate-key rule; this prevents native sklearn and portable evaluators from choosing different
candidates because of a few floating-point ulps while preserving raw scores for diagnostics.
