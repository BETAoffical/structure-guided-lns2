# V3 value uncertainty audit

Decision: `distributional_value_signal_insufficient`

- Complete four-seed states: 8; independent maps: 4.
- Primary policy: `loo_expected_t0.05`; deviations: 24/32.
- Held-out wins/losses/ties versus v2: 13/11/8.
- Mean paired outcome and map-clustered 95% interval: 0.062 [-0.312, 0.688].
- Feasible-rate delta versus v2: +3.125%.
- Maps with positive/nonnegative net wins: 1/1 of 4.

## Method

- Training-fold aggregate order: `maximize feasible_rate; minimize mean_final_conflict_ratio, mean_normalized_conflict_auc_seconds, mean_total_seconds`.
- Held-out outcome order: `feasible beats censored; among feasible minimize total_seconds; among censored minimize final_conflict_ratio; then minimize normalized_conflict_auc_seconds and total_seconds`.
- Primary guard: `change v2 only for higher feasible_rate or at least 5% improvement in the first differing aggregate quality metric`.
- Uncertainty interval: `5000-sample bootstrap resampling four map clusters`.

## Checks

- at_least_eight_complete_states: `true`
- deviation_fraction_at_least_10pct: `true`
- feasible_rate_not_lower_than_v2: `true`
- map_clustered_ci_lower_not_below_minus_0_10: `false`
- nonnegative_net_outcome_on_at_least_3_of_4_maps: `false`
- win_rate_among_deviations_at_least_60pct: `false`
- wins_exceed_losses: `true`

## Boundary

- The audit uses only eight four-seed states from four maps; trials are paired outcomes, not independent maps.
- Three training seeds select an action for the held-out fourth seed, so this tests stochastic target stability without using state features.
- Continuation actions are official Adaptive and results are not complete v2 or v3 episodes.
- Retrospective Oracle-derived arms remain in the candidate set, making this an optimistic upper-bound diagnostic rather than deployment evidence.
- The 5% guard is fixed as the primary diagnostic; 0% and 10% are sensitivity checks and are not selected post hoc.
