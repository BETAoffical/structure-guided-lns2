# STRIDE Compact-Blocker Escape Durability v1 Protocol

## Purpose

This is a read-only, post-hoc mechanism audit of the 900 completed
compact-blocker continuation traces. The one-decision escape result is already
known. The audit freezes previously uncomputed horizon-3 and horizon-8 tests
before reading those outcomes.

After the metrics and gates were written, two baseline trial-0 traces were
opened solely to validate trace reconstruction. Their identities are recorded
in the registration; no metric, threshold or population rule was changed from
their outcomes. The audit remains explicitly post-hoc rather than a fresh
result-blind confirmation.

It asks whether compact plus blocker creates a persistent structural escape or
only a one-decision perturbation. It does not reopen the failed rescue-wall
gate and cannot authorize extension, runtime integration, training or TTF.

## Population and pairing

The source population is the complete 45-state, trial-0--3 collection. The
primary population contains every state registered as semantic-compaction
eligible and every state/trial where the paired shared first action exact-rolls
back with `conflict_bound_exceeded`. No state or trial may be selected using a
durability outcome.

The only contrast is:

- baseline: full set plus observed blockers and the shared fresh rescue seed;
- treatment: semantic compact set plus the same observed blockers and fresh
  rescue seed.

## Frozen metrics

Decision one is the single rescue decision. Horizons 1, 3 and 8 include that
decision and the following recorded repair decisions. Sustained escape at a
horizon requires all of the following:

1. decision one changes the repair fingerprint;
2. the original platform fingerprint is not revisited;
3. no three consecutive exact rollbacks form another platform.

Successful termination before a horizon is a complete platform-free
observation. A non-success termination before the horizon is right-censored and
masked from that horizon's paired contrast.

Secondary metrics are original conflict-edge retention AUC, normalized conflict
AUC, strict conflict improvement, original-platform re-entry, new-platform
formation and feasibility by horizon.

State-cluster paired bootstrap uses 10,000 deterministic replicates. A durable
signal requires positive lower 95% confidence limits for sustained escape at
both H=3 and H=8, a negative upper limit for H=8 original-edge-retention AUC,
and no map with more than five percentage points of H=8 worsening.

If H=1 improves but H=3 or H=8 does not, the result is classified as transient.
If H=1 is not stable or too few paired horizons are observable, it is
inconclusive.

## Execution boundary

The audit uses 16 read-only workers. It may reconstruct states from registered
trace deltas but may not call the solver or alter source artifacts. Outputs are
written atomically to a separate build directory.
