# STRIDE-MarginalPool v1 root-diagnostic preregistration

## Purpose

This diagnostic separates two questions that the failed LoopGuard and
ProductivityGuard audits could not distinguish:

1. does the augmented selector repeat because the structural generator offers
   no sufficiently different structural action, or because the memoryless V2
   scorer keeps ranking the repeated action above available alternatives;
2. do the preceding PP calls truly leave the paths unchanged, or do they alter
   latent timing, waiting, route occupancy or bottleneck order while the
   conflict-pair set remains unchanged.

The diagnostic reads completed TailSwitch traces only. It does not run the
solver, rerun PP, train a model, modify a controller, or claim a TTF benefit.

## Frozen cohort and checkpoints

The cohort contains all 45 early-window contrasts triggered by the first
registered ProductivityGuard rule, `exact-triplet-zero-progress`. Selection is
based only on the observed pre-action repetition and the two already completed
repairs. The later TailSwitch outcome classification is retained only for
descriptive stratification and cannot include or exclude a case.

Each contrast contributes two pre-action checkpoints:

- the first structural continuation selection;
- the first proposal that completes the exact three-action repeated/stalled
  pattern.

The expected output is therefore 90 checkpoint records. Every record must
retain the complete candidate pool, exact candidate and state identities, and
the registered 124-dimensional feature representation.

## Frozen diagnoses

At the repeat/stall checkpoint, a structural generator collapse is recorded
when no different structural candidate has agent-set Jaccard at most `0.8`
relative to the repeated winner. A ranker lock is recorded when a sufficiently
different action exists in the combined pool but the repeated action is still
ranked first by the frozen score and tie rule. These are independent flags,
not a forced mutually exclusive narrative.

For the two completed repairs before the third proposal, the diagnostic
compares time-aligned paths, waits, path costs, delays and ordering at static
bottleneck cells. Any registered path-level change is latent path change; only
the absence of every registered change is a true PP no-op. The outcome of the
current third proposal and every later transition are forbidden inputs.

Persistent-edge full coverage, missing endpoints and partial conflict-component
coverage are reported as repair-closure evidence. They are not candidate
quality labels.

## Decision boundary

Passing integrity authorizes only the next preregistered experiment: restore
the two frozen checkpoint states and evaluate every candidate already present
in each pool under 16 strictly paired PP seeds. Stage 1 cannot decide which
candidate is best, cannot promote a controller, and cannot support a solver or
generalization claim.

