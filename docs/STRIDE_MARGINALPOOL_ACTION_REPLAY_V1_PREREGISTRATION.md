# STRIDE-MarginalPool Action Replay v1 Preregistration

## Question

Stage 1 established that all 45 repeated/stalled cases had alternative
neighborhoods available, while the V2 scorer repeatedly selected the same
structural neighborhood and the two preceding PP calls were exact path-level
no-ops.  It did not establish whether an already-generated alternative would
repair the current state more effectively.

This experiment answers that narrower question.  It replays every candidate
already present at the two frozen checkpoints under 16 strictly paired PP
seeds.  It does not add candidates, train a model, modify the solver, inspect a
future trajectory, or measure TTF.

## Frozen cohort

- 90 logical checkpoints from 45 frozen TailSwitch continuation contrasts.
- 78 unique repair-state fingerprints.
- 2,502 unique `(state_fingerprint, candidate_id)` actions.
- Both `first_structural_selection` and `first_repeat_stall` are retained.
- Duplicate physical states are executed once, while all logical checkpoint
  references and their actually selected candidates remain in the analysis.
- Classification (`adverse`, `beneficial`, `neutral`) is descriptive only and
  cannot change collection.

## State and action reconstruction

Each checkpoint state is loaded from its Stage 1 compressed state blob.  The
blob SHA-256 and full state fingerprint must match the frozen manifest.  The
corresponding TailSwitch `run_config.json` supplies the original dataset and
environment configuration.  Native `reset_paths` restores an independent PP
branch, and its repair-structure fingerprint must match before any action is
allowed to run.

Every candidate keeps its frozen agent set, provenance and 124-dimensional
`lns2.realized_features.v2` representation.  Candidate generation and ranking
are not rerun.

## Paired PP protocol

For each unique state/action, run trial indices 0 through 15.  The PP seed is a
deterministic hash of the repair-structure fingerprint and trial index.  Thus,
all candidates within the same state use exactly the same PP seed for a given
trial index, while the 16 indices remain distinct.  PP, SIPPS and native repair
semantics are unchanged.

Only the immediate repair result is stored:

```text
normalized reduction =
    (conflicts_before - conflicts_after) / max(1, conflicts_before)
```

Aggregates are the mean, population standard deviation, lower-half mean, two
fixed eight-seed half means, no-progress rate, replan-success rate, feasible
rate, minimum and maximum.  Runtime, TTF, remaining repair rounds, future
states and Cost-to-Go are forbidden.

## Stable dominance rule

An alternative `c` stably dominates the actually selected candidate `a` only
when all conditions hold:

```text
mean(c) - mean(a) >= 0.02
no_progress(c) <= no_progress(a)
first_half_mean(c)  - first_half_mean(a)  > 0
second_half_mean(c) - second_half_mean(a) > 0
```

Ties do not count as dominance.  This is a checkpoint-local current-step
criterion, not a long-term or TTF label.

## Execution order

1. Run a preflight on the first two unique state fingerprints in lexical
   order, using all their candidates but only trial indices 0 and 1.
2. Inspect only execution integrity: restoration, paired seeds, native action
   legality, schemas and forbidden fields.  Preflight outcomes cannot alter
   the cohort, thresholds or protocol.
3. If and only if preflight integrity passes, run all 78 unique states with all
   16 trials.  Artifacts are atomic and resumable per state.

## Decision after collection

- If an existing alternative stably dominates the selected action, the first
  actionable defect is ranking/history, and the next step is a history-aware
  ranker on the frozen pool.
- If the selected action is already the current-step best, current-step
  ranking is insufficient; the next experiment is a separately preregistered
  two-step repair-closure sequence diagnostic.
- If no existing candidate makes positive progress, the current pool or PP
  operator is inadequate; repair-closure candidate generation comes before
  ranker training.
- If the two seed halves reverse the winner/dominance direction, PP uncertainty
  must be modeled explicitly and no deterministic winner claim is allowed.

No result from this experiment may be presented as a TTF improvement, a
generalization result, or justification for replacing `v2-full`.
