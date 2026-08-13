# STRIDE StructShell Audit v1 execution amendment

The first complete analysis run was rejected by its integrity gate before acceptance. The registration used `legacy_opportunity_state_count = 41`, while the implementation counted states containing any of the 209 robust structural actions and observed 47.

Both numbers are present in the same immutable registered structural-coverage report:

- 41 is the number of states where the single highest-mean legacy structural action also robustly dominates the V2 best action;
- 47 is the number of states containing at least one of the 209 robust structural actions.

The audit's action-retention question necessarily uses the second population. The registration now freezes both identities with explicit names. No state or action was added, removed, or selected by result. The primary `structural_knee` rule, all comparator definitions, all readiness thresholds, the equal-four-size fallback, and every claim boundary remain unchanged. The rejected report is overwritten only after the corrected identity checks pass.
