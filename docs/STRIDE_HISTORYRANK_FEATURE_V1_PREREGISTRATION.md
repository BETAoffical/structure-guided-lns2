# STRIDE HistoryRank Feature V1 Preregistration

## Question

The exact-noop fallback rule improved mean one-step quality but was stable in only `21/45` first-repeat checkpoints. This audit asks whether a single interpretable, outcome-blind history interaction can identify a robust alternative more reliably than simply taking the next V2-ranked neighborhood.

## Frozen cohort and scopes

All 45 registered `first_repeat_stall` checkpoints are retained. The primary scope contains every runtime-retained alternative except the exact repeated candidate. A secondary diagnostic repeats the audit over every generated alternative. Only the primary scope controls readiness.

No checkpoint, candidate, map, or PP seed may be removed based on its result. The 16 strictly paired PP trials and their fixed `0-7` and `8-15` halves remain unchanged.

## Outcome-blind features

Before reading candidate outcomes, the registration freezes 13 deterministic directed features and their priority. They cover agent-set novelty, family novelty, normalized 124-dimensional feature distance, added/removed agents, conflict/component coverage, boundary exposure, path overlap, size change, and the frozen V2 score margin. Larger directed values are always preferred. Ties use the higher frozen V2 score when it exists and then ascending candidate ID.

For each feature and checkpoint, exactly one alternative is chosen. There is no fitted parameter, learned threshold, map-specific rule, or outcome-derived direction.

## Readiness gates

A feature passes only if its primary retained-candidate selections satisfy all of:

- stable one-step improvement on at least `60%` of checkpoints;
- mean paired seed improvement at least `0.02`;
- mean no-progress-rate delta at most `0`;
- both fixed seed halves positive on at least `60%` of checkpoints;
- strictly positive mean seed improvement on every map;
- zero integrity or join errors.

If multiple features pass, the first in the frozen priority order is the only selected target. If none pass, no simple single-feature rule is ready; the next step is to preregister a fresh-map cohort for a grouped history-aware ranker, not to tune these features on the same outcomes.

## Claim boundary

This is an offline checkpoint-local feature audit. It does not train a model, modify candidate generation, modify PP/SIPPS, run a solver controller, read TTF/runtime/future trajectories, or support a long-term, speed, or generalization claim.
