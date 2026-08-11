# STRIDE-HistoryRank Rule v1 Preregistration

Date: 2026-08-11

## Question

MarginalPool action replay found that an existing alternative stably dominates
the selected neighborhood at 42 of 45 first repeated/stalled checkpoints.  The
next narrow question is whether the root defect can be corrected without a
learned model: after two exact PP no-ops on the same neighborhood, exclude only
that exact candidate and use the highest-scoring remaining frozen V2 candidate.

This is selection-time history, not a wall-clock guard or a timeout fallback.
It activates before the third PP call and uses only two already-observed native
repair responses.

## Frozen cohort and rule

- retain all 45 registered `first_repeat_stall` checkpoints;
- require the two previous selections and the frozen current winner to share
  one candidate ID;
- require both previous PP calls to have `replan_success=false`, unchanged
  conflicts and zero changed paths;
- exclude that exact candidate ID only;
- select the remaining candidate with the highest frozen V2 score, breaking
  ties by ascending candidate ID;
- do not regenerate candidates, ban a family, use a Jaccard threshold, change
  PP/SIPPS, inspect runtime or read future states.

The candidate outcomes are the already-complete sixteen-seed paired immediate
repair aggregates.  Stable improvement uses the frozen 0.02 mean advantage,
non-increasing no-progress probability and positive improvement in both fixed
eight-seed halves.

## Gates

The simple rule is sufficient for a later runtime design only if:

- at least 60% of all 45 replacements stably improve on the repeated action;
- mean seed improvement is at least 0.02;
- mean no-progress-rate delta is at most zero;
- both seed halves improve in at least 60% of checkpoints;
- mean seed improvement is positive on every frozen map;
- all identities and action joins pass with zero errors.

Failure does not justify retaining repetition.  It means that merely excluding
the exact no-op action does not identify which alternative to select, so the
next step is a separately preregistered `stride-historyrank-v1` feature audit.

No result is a TTF, long-term return, cross-family generalization or default-
controller claim.
