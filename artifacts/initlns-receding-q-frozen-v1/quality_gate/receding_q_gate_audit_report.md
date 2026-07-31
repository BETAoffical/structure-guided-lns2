# Receding-Q quality-stability gate audit

Decision: `quality_stability_insufficient`

## Coverage

- States: 12
- Candidates: 215
- Rollouts: 430
- Paired trials: 2

## Stability

- Exact candidate-ID agreement: 16.667%
- Timing-only mismatch states: 4
- Exact winners not mutually optimal across seeds: 6
- True quality-instability states: 3
- Quality-winner-set overlap: 75.000%
- Mean quality-winner-set Jaccard: 56.539%
- Mean Top-3 overlap: 55.556%
- Paired-seed rank correlation: 0.531632
- Mean normalized-step-AUC regret: 0.068704
- Cross-seed feasibility misses: 2

## Checks

- cross_seed_feasibility_misses_at_most_1: `false`
- mean_normalized_step_auc_regret_at_most_0_02: `false`
- mean_top3_overlap_at_least_0_80: `false`
- paired_seed_rank_correlation_at_least_0_50: `true`
- quality_winner_set_overlap_at_least_0_80: `false`

## Timing tolerance sensitivity

- abs_0.000_rel_0.000: 16.667%
- abs_0.000_rel_0.010: 33.333%
- abs_0.000_rel_0.050: 66.667%
- abs_0.001_rel_0.000: 41.667%
- abs_0.005_rel_0.000: 66.667%
- abs_0.010_rel_0.000: 66.667%

## States requiring targeted paired seeds

- `s3__policy_train__policy_train_compartmentalized_double_horizontal_0001__task_0016__seed_0000__official_adaptive__decision_0000__0040`
- `s3__policy_train__policy_train_compartmentalized_double_horizontal_0002__task_0014__seed_0000__realized_dynamic__decision_0007__0017`
- `s3__policy_train__policy_train_regular_beltway_0000__task_0016__seed_0000__official_adaptive__decision_0004__0113`

This audit is diagnostic only and does not promote or replace `v2-full`.
