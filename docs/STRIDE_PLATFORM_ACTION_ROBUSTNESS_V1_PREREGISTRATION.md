# STRIDE Platform Action Robustness v1 preregistration

## Purpose

This is a zero-solver diagnostic over the already completed 45-case,
325-action, eight-seed Platform-entry Frontier collection.  It separates three
questions that the per-case/trial post-hoc pool oracle cannot answer:

1. does the frozen pool contain a concrete action whose platform benefit is
   stable across native PP seeds;
2. did either frozen deterministic rule identify such an action; and
3. when no stable action exists, are the observed alternatives merely
   seed/order-sensitive or absent from the tested pool.

No PP, continuation, training, runtime integration, or TTF run is authorized.
The output is mechanism diagnosis only.

## Frozen inputs and identity

- `build/stride-platformentry-frontier-v1-r2/platformentry_frontier_report.json`
- `build/stride-repairdependencypool-v1/platform_entry_witnesses.jsonl`
- exactly 45 registered cases, 325 unique concrete actions and 2,600 episodes;
- trial indices `0..7` for every concrete case/action;
- the historical action is the within-case paired reference;
- candidates with the same concrete `candidate_id` as the historical action
  are not counted as alternatives even if they carry another logical role.

The input report must already have passed all registered integrity checks.

## Frozen action-level rules

For every concrete alternative action, compute paired platform-entry deltas
against the historical action for the fixed halves `0..3` and `4..7`.

An action has **seed-half-stable platform improvement** only when the mean
paired platform delta is strictly negative in both halves.  With binary
outcomes this necessarily means at least one net avoided platform entry in
each half; a one-seed or one-half win is not stable evidence.

A stable action is **quality preserving** only when, over all eight paired
trials:

- success rate is not lower than the historical action;
- mean normalized fixed AUC is not higher; and
- mean repair decisions censored at 64 is not higher.

These conditions remain separate in the report.  They are not compressed into
a learned label and are not a promotion gate.

## Frozen case classification

Cases are assigned in this order:

1. `reference_no_platform`: the historical action entered no platform in all
   eight trials, so this cohort provides no platform-prevention contrast.
2. `deterministic_rule_success`: at least one quality-preserving stable
   alternative is the frozen deterministic compact-augment or same-size
   exchange action.
3. `selection_headroom`: a quality-preserving stable alternative exists, but
   neither deterministic rule selected it.
4. `platform_only_headroom`: a seed-half-stable platform alternative exists,
   but every such action violates at least one quality-preservation condition.
5. `seed_unstable_headroom`: no seed-half-stable alternative exists, but at
   least one concrete alternative has fewer total platform entries than the
   historical action or has mixed platform outcomes across the eight seeds.
6. `pool_or_repairer_gap`: none of the preceding conditions holds.

The existing causal annotation (`set`, `order`, `joint`, or `residual_pp`) is
joined only for a cross-tabulation.  It is outcome-enriched mechanism evidence
and is forbidden as an online feature or selector input.

## Interpretation and stopping rule

- `selection_headroom` supports investigating a better pre-entry selector,
  but not training one automatically.
- `platform_only_headroom` warns that platform avoidance would recreate the
  conservative-small-neighborhood failure mode.
- `seed_unstable_headroom` means a set-only deterministic selector cannot make
  the observed escape reliable under native PP realization.
- `pool_or_repairer_gap`, especially in `set`/`joint` cases, points to candidate
  composition or the PP repair mechanism rather than ranking.

No new data collection is allowed from this result.  A further implementation
branch requires a separate decision based on the complete, unscreened 45-case
cross-tabulation.
