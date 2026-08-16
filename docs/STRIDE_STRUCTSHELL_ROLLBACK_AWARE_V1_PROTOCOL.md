# STRIDE StructShell Rollback-Aware v1 protocol

## Decision being tested

The bounded StructShell v3 confirmation did not promote the pool.  Its two
high-load Random failures repeatedly chose one size-32 candidate and returned
through native `conflict_bound_exceeded` rollback for hundreds of decisions.
Fresh episode-stream PP seeds therefore did not remove the attractor.  The
next intervention changes neither PP nor the frozen V2 ranker: it prevents the
same structural challenger from remaining the online action after three
native, exact, repair-equivalent rollbacks.

This is a post-selection safety layer.  It is not evidence that StructShell is
a replacement for V2 and it is not a new long-horizon label or ranker.

## Frozen runtime contract

- Pool ID: `stride-hybridstructpool-routed-v2`.
- Runtime ID: `stride-hybridstructpool-routed-runtime-v2`.
- Candidate sources: complete V2 pool plus StructShell only.
- Ranker and features: frozen V2 realized features and Copeland scoring.
- PP: native order and `episode_stream` seeds.
- Maximum PP calls per decision: one.
- Exact rollback predicate: `conflict_bound_exceeded`, `replan_success=false`,
  `pp_rolled_back=true`, and identical repair-structure fingerprints.
- Trigger: three consecutive exact rollbacks for the same repair fingerprint
  and the same pure StructShell challenger.
- Intervention: temporarily exclude that challenger and reuse only the
  repair-state candidate structure.  Recompute the current decision's 124
  realized features, V2 anchor and Copeland scores before choosing the next
  ranked unbanned action.
- Fallback: once no StructShell challenger remains, use the frozen V2 anchor
  once.  An exact anchor rollback invalidates the cache and reopens the
  bounded structural scan; V2 candidates are never frozen by this layer.
- Reset: any repair-structure change clears the streak, exclusions and cache.
- V2 candidates, including exact V2/StructShell duplicates, are never banned.

Every decision remains a separate trace record.  Cache reuse does not merge
decision indices or history identities.  `state.iteration` and cumulative
low-level work are V2 model inputs but are intentionally absent from the
repair-structure fingerprint, so feature rows and scores may never be reused;
only candidate proposal and StructShell generation are skipped after a proven
atomic rollback.

The frozen candidate pool is an explicit part of this rollback intervention,
not a claim of transparent engineering equivalence: ordinary V2 proposal seeds
include the full state fingerprint and decision index.  The diagnostic must
therefore attribute any effect to the combined bounded mechanism (freeze one
repair-state pool, refresh V2 scores, and ban a repeatedly rolled-back pure
StructShell challenger), not to candidate exclusion alone.

## Ordered validation

1. Pure contract tests verify the new ID, exact three-rollback threshold,
   V2-anchor fallback, state-change reset, non-exact no-op rejection and source
   hashing.
2. A development mechanism replay uses the known two high-load Random
   platforms.  It must show candidate changes after the third rollback, no
   second PP call in a decision, monotone cache hits, and fewer exact rollback
   decisions than frozen StructShell v1.  In both cases the original repair
   fingerprint must actually change, and the first-platform escape latency in
   repair decisions must be strictly below the paired v1 baseline; rotating
   StructShell/V2/anchor labels on the same fingerprint is not escape.  After
   that escape, no new repair fingerprint may form another three-rollback
   platform within the same episode.  This replay is diagnostic only.
3. Only if step 2 passes, preregister a bounded paired confirmation with fresh
   task and solver-seed keys.  Compare official Adaptive, V2 and exactly one
   rollback-aware challenger under the same 180-second right-censoring rule.

The challenger advances only if success is not below V2, mean restricted TTF
and normalized wall AUC improve, paired faster fraction is at least 50%, each
map regresses by at most five percentage points, and the 95% paired bootstrap
supports positive benefit.  PP, candidate generation, caching, scoring and
guard overhead are all charged to TTF.  Failure keeps V2 as the runtime
default; it does not authorize a new ranker or further pool tuning.
