# Receding-Q four-seed stability follow-up

Decision: `four_seed_receding_q_labels_insufficient`

- Target states: 7; additional trials: [2, 3].
- Mean pairwise rank correlation: 0.5005.
- Half-split exact winner: 28.571%; Top-3 overlap: 85.714%; operational stability: 28.571%.
- LOO vs v2: 17 wins / 9 losses / 2 ties; feasible delta +3.571%; step-AUC delta -0.0176.

## Checks

- followup_coverage_complete: `true`
- half_top3_overlap_on_at_least_80pct_states: `true`
- loo_feasible_rate_not_below_v2: `true`
- loo_normalized_step_auc_not_worse_than_v2_by_2pct: `true`
- loo_wins_not_below_losses: `true`
- mean_pairwise_rank_correlation_at_least_50pct: `true`
- operational_stability_on_at_least_70pct_states: `false`

## Scientific boundary

- Only the states whose trial-0 and trial-1 exact winners differed receive two additional PP seeds.
- Exact winner identity is diagnostic; the gate uses rank correlation, Top-3 overlap, cross-half outcome regret, and leave-one-seed-out behavior.
- The leave-one-seed-out policy uses measured outcomes from three seeds and is an optimistic label diagnostic, not a trained deployable controller.
- Continuation still uses the Adaptive teacher and the horizon remains three repairs; no complete episode is evaluated.
