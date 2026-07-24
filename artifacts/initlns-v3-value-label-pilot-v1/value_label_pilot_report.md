# Variable-horizon cost-to-go label pilot

Decision: `cost_to_go_labels_promising`

- States: 12; rollouts: 70.
- Action-sensitive state fraction: 100.000%.
- Uncensored branch fraction: 81.429%.

## Checks

- action_sensitive_state_fraction_at_least_50pct: `true`
- at_least_two_actions_per_state: `true`
- coverage_complete: `true`
- uncensored_branch_fraction_at_least_50pct: `true`
- winner_seed_agreement_at_least_50pct: `true`

## Boundary

- The continuation teacher is official Adaptive, so labels estimate Q under that teacher rather than an optimal or on-policy v3 continuation.
- Candidate arms were selected from existing v2, model-S3, and retrospective Oracle diagnostics; this pilot tests label signal and is not promotion evidence.
- Replay and prefix reconstruction time is excluded from cost-to-go labels.
- Censored branches require survival-aware handling before model training.
