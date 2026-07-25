# Variable-horizon value-label stability follow-up

Decision: `label_stability_insufficient_for_value_model_pilot`

- Targeted states: 8; follow-up rollouts: 63.
- Uncensored fraction: 81.429% -> 90.000%.
- Extended censored branches resolved: 8/13.
- Minimum pairwise winner agreement across overheads: 52.778%.
- Pairwise winner agreement on targeted states: 31.250%.
- Target winner purity, median/minimum: 50.000%/50.000%.

## Checks

- all_censored_rollouts_extended: `true`
- followup_coverage_complete: `true`
- merged_uncensored_fraction_at_least_90pct: `true`
- target_pairwise_winner_agreement_at_least_60pct: `false`
- target_winner_purity_at_least_50pct: `true`

## Boundary

- Only previously unstable or censored states receive two additional PP seeds; stable states retain two seeds.
- Previously censored branches are extended to the new cap while completed branches reuse their original measurements.
- Continuation actions still use official Adaptive, so the labels estimate a teacher-conditioned Q rather than an on-policy v3 value.
- Strict all-seed agreement is diagnostic only because its probability decreases mechanically as the seed count grows; the promotion gate uses pairwise agreement.
- This is a label-stability gate and does not compare complete v2 or v3 episodes.
