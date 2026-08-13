# STRIDE CycleTransition Pilot v1 Preregistration

## Purpose

This pilot tests a narrower mechanism than MultiValue, SafeSlot, or a generic
long-tail classifier: whether forcing a candidate once creates the immediate
conditions for the selector to enter a repeatable state-action loop.

The audit is deliberately pre-action and quality constrained. It does not run
PP tentatively at deployment time, does not continue an episode to completion,
and does not treat lower risk as sufficient reason to select a weak or small
neighborhood.

## Frozen lessons

- MarginalPool observed 45/45 repeat checkpoints where two distinct PP seeds
  left the path-level state unchanged and the memoryless ranker selected the
  same structural action again.
- Existing alternatives often repair better, but one-step quality alone does
  not establish that an action avoids a later loop.
- ResidualHazard and SafeSlot did not provide a map-stable pre-action residual
  risk rule.
- MultiValue H=1/8/32/128 targets were unstable across seeds and continuation
  teachers, so no long-horizon scalar target is reused.
- HistoryRank showed that unrestricted novelty reduces no-progress but also
  discards useful conflict structure.

## Cohort and pool

The source is all 45 registered `first_structural_selection` occurrences from
the frozen MarginalPool root diagnostic. The pilot selects six unique state
fingerprints per Maze map using only a namespaced SHA-256 order. Tail class,
repair result, candidate score, and continuation outcome cannot enter cohort
selection.

For every selected state the candidate pool is regenerated and frozen before
any repair outcome is read:

- the complete original V2 base pool;
- the complete six-family StructPool grid at equal sizes 8, 16, 24, and 32;
- exact agent-set deduplication only;
- no SlotPool or six-candidate reduction;
- an exact base-only V2 anchor retained separately.

This is the full-pool fallback selected by the StructShell and ShellBudget
audits. It neither restores the old per-family preferred sizes nor substitutes
a single natural cut point.

## Measurement

Each candidate is forced once for each paired PP seed 0-7. No additional repair
action is executed. If the result remains active, the full pool is generated
once on the resulting state and the frozen V2 ranker performs one shadow
selection. The recorded vector separates:

- immediate normalized conflict reduction;
- exact path-state no-op;
- old-edge persistence, new-edge production, and edge-set Jaccard;
- exact and Jaccard neighborhood repetition at the next shadow selection;
- exact and soft policy-cycle indicators.

The future shadow selection is a diagnostic of the current loop mechanism. It
is not a proposed runtime rule and is not compressed into a generic tail-risk
label.

The actual source-controller choice at the first structural decision is
retained as the observed reference action. A structural candidate counts as an
escape opportunity only when it passes the exact V2-anchor quality constraint,
reduces the joint soft-policy-cycle rate relative to that source choice, and
does not increase its exact PP no-op rate. The full-grid V2 choice is reported
separately and is not substituted for the historical source choice.

## Preventing the V3 failure mode

Cycle evidence may only compare actions that satisfy the frozen immediate
quality constraint relative to the exact V2 anchor. Quality is a hard
admissibility condition, not a weighted term that a risk score can trade away.
The report also rejects a result in which eligible escape actions collapse to
one nominal size.

## Execution and stopping

- 16 workers for preparation and collection;
- one state-candidate-seed job per worker, so the final states still use all
  available CPU workers;
- 300-second hard job limit;
- atomic state-candidate-trial checkpoints;
- first error or timeout stops the run;
- no result-based state or candidate exclusion;
- no raw TTF timing is mixed with this parallel diagnostic.

Passing every readiness gate authorizes only a new preregistered,
whole-map-grouped pre-action predictability audit. It does not authorize model
training, runtime integration, Maze avoidance claims, or TTF experiments.
