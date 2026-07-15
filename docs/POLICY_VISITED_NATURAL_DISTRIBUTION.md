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
PYTHONPATH=build/linux/project python3 scripts/collect_policy_visited_experience.py \
  --dataset build/initlns-policy-visited-v1 \
  --config configs/policy_visited_natural_collection.json \
  --output build/initlns-policy-visited-natural-v2-collection \
  --phase all --workers 4
```

Windows trains and evaluates the fixed development learner:

```powershell
python scripts/run_policy_visited_aggregation.py --phase train `
  --collection build/initlns-policy-visited-natural-v2-collection `
  --config configs/policy_visited_natural_analysis.json `
  --output build/initlns-policy-visited-natural-v2-training
python scripts/run_policy_visited_aggregation.py --phase offline `
  --collection build/initlns-policy-visited-natural-v2-collection `
  --training build/initlns-policy-visited-natural-v2-training `
  --config configs/policy_visited_natural_analysis.json `
  --output build/initlns-policy-visited-natural-v2-offline
```

Only after the offline gate passes, generate the registered confirmation dataset and use the independent
confirmation runner. Intermediate maps, traces, indexes and models remain ignored under `build/`.
