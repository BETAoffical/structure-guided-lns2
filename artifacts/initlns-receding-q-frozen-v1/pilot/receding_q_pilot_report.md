# Fresh receding-Q label pilot

Decision: `fresh_receding_q_labels_insufficient`

- States: 12; maps: 5; actual candidates: 215; rollouts: 430.
- Horizon: 3; continuation teacher: `official_adaptive`.
- Action-sensitive states: 100.000%.
- Paired-seed candidate rank correlation: 0.5316; exact winner agreement: 16.667%.

## Checks

- action_sensitive_state_fraction_at_least_50pct: `true`
- all_actual_candidates_covered: `true`
- at_least_12_states: `true`
- at_least_4_maps: `true`
- coverage_complete: `true`
- feature_schema_exact: `true`
- paired_seed_rank_correlation_at_least_50pct: `true`
- winner_seed_agreement_at_least_50pct: `false`

## Scientific boundary

- The first action is every actual neighborhood generated from requested sizes 4/8/16; actual size can be smaller when the generator cannot fill the request.
- At each continuation state the complete actual candidate pool and features are regenerated to measure receding-controller selection cost, then official Adaptive supplies the independent teacher action.
- These labels estimate a fixed-horizon Q under an Adaptive continuation teacher, not an optimal or on-policy receding-Q controller.
- The fixed H=3 label avoids variable-horizon censoring but cannot establish complete-episode performance.
- Replay and prefix reconstruction time is excluded; root and continuation full-pool costs plus complete post-step orchestration are included. Root full-pool time is the source qualification measurement shared by every candidate in a state.
- This pilot is a label-stability gate and cannot promote or replace v2.
