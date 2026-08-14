# STRIDE HybridStructPool v1 preregistration

## Purpose

This milestone freezes a candidate-generation contract before any further
platform-repair experiment.  It does not train a selector and does not execute
native PP.  The pool is the exact-set union of:

1. the complete frozen V2 pool and its separate V2 anchor;
2. every exact-deduplicated StructPool family-size action at symmetric sizes
   `8`, `16`, `24`, and `32`; and
3. the frozen compact `stride-causalclosurepool-v2` actions.

The old StructPool topology families are retained, but their fixed per-family
preferred-size reducer is not.  CausalClosure is supplemental because it adds
compact opportunities but recalls only a small fraction of legacy structural
opportunity when used alone.

## Frozen inputs

- 78 difficult Maze state fingerprints;
- 1,367 complete V2 candidates;
- 1,135 legacy equal-four-size structural candidates;
- 930 compact CausalClosure candidates;
- the frozen 209-action robust legacy membership audit;
- the three CausalClosure-only opportunity states;
- the completed StructShell contract report.

Every input is registered by SHA-256.  Candidate outcomes may be used only to
audit membership of already frozen opportunity sets.  They cannot construct,
prune, order, or budget the union.

## Candidate contract

- exact agent-set deduplication with source provenance retained;
- V2 representation wins exact-set ties;
- no fixed family-size preference;
- no runtime candidate cap in this milestone;
- no result-based state or candidate filtering;
- no future trajectory, runtime, TTF, or PP-order field in the output.

## Readiness gates

The zero-solver audit must retain exactly 100% of:

- V2 candidates;
- equal-four-size legacy structural candidates;
- all 209 frozen robust legacy structural actions;
- compact CausalClosure candidates; and
- the best CausalClosure action in each of the three CausalClosure-only
  opportunity states.

All three maps must contain all three sources and all identity checks must pass.
Passing freezes the full union as the proposal-space contract.  It does not
validate a runtime budget reducer, selector, platform-prevention policy, TTF
improvement, or replacement of `v2-full`.

## Next boundary

Only after this contract passes may a separate outcome-blind budget audit test
budgets `6`, `8`, and `12`.  If a small budget discards registered membership,
the full pool remains the research contract and must be evaluated lazily rather
than silently reintroducing the old fixed-size rule.
