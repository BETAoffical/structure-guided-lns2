# STRIDE StructShell rollback-aware TTF v1 protocol

## Decision being tested

This experiment asks one runtime question: with the rollback-aware StructShell
implementation frozen, does the controller reach a feasible solution faster
than frozen V2 and official Adaptive LNS2 under a common wall-clock budget?

The deployment objective is now eventual escape and time to feasibility.  A
successful episode may pass through several short, bounded repair platforms.
Those platforms remain visible in the trace and their complete cost is charged
to the episode, but their existence is not by itself a failure.  A platform
that persists until right-censoring remains a failure through the ordinary
success, restricted-TTF and tail metrics.

This change of decision target does not rewrite the earlier two-case mechanism
replay.  That replay remains a failed test of its registered
`no_post_escape_platform_transfer` gate.  Its two known Random cases, outcomes
and traces are development evidence only: they are not included in this TTF
cohort, do not set a new rollback threshold and are not pooled into any
estimate or confidence interval below.

## Frozen controllers and execution contract

The three arms are:

1. `official_adaptive`: official Adaptive LNS2;
2. `v2_only`: the frozen V2 runtime controller and primary internal baseline;
3. `structshell_rollback_aware_v2`: the frozen rollback-aware StructShell
   controller.

The challenger retains the registered three-exact-rollback trigger, frozen V2
features and Copeland ranker, native PP order, `episode_stream` seed policy and
at most one native PP call per decision.  No retry, blocker rescue, alternative
repair order, threshold adjustment, pool retuning or new ranker may be added.

All timed episodes use strict rotating three-arm serial execution with one
worker.  Qualification and non-timing checks may use 16 workers.  Every arm has
the same limits:

- 180 seconds of reset-inclusive episode wall time;
- 240 seconds for the episode process fuse;
- 300 seconds for the outer job fuse.

`wall_timeout` is a valid right-censored outcome at 180 seconds.  A process
timeout, execution error, invalid action or semantic/fingerprint mismatch is a
terminal experiment error.  Initial state fingerprints and initial conflicts
must match across all three arms for every paired key.

## Cohort and ordered collection

The complete schedule is frozen before the first timed episode.  It contains
10 registered map groups, two tasks per group and solver seeds 16--18: 60
paired keys and 180 three-arm episodes.  None of the keys may be replaced,
removed or reordered according to an observed controller result.

The map groups and task identities were used in the earlier StructShell
confirmation; only solver seeds 16--18 are fresh.  This is therefore a
fresh-seed confirmation on fixed tasks, not a new-map OOD or result-blind-map
claim.

Collection has two stages:

1. **Cost screen.** Run the first registered task from each of the 10 map
   groups with solver seed 16.  This is 10 paired keys and 30 episodes.
2. **Full confirmation.** If the cost screen passes, run every remaining
   registered key.  This adds 50 paired keys and produces the fixed total of
   60 paired keys and 180 episodes.

The cost screen is a preregistered futility and integrity check, not a smaller
confirmation experiment.  It produces no speed or generalization claim, has
no bootstrap significance requirement and cannot authorize controller
promotion.  Passing it only authorizes collection of the other 50 keys.

## Cost-screen gates

All of the following must hold on the 10 complete paired keys:

- complete three-arm coverage and strict initial-state pairing;
- zero execution errors, process timeouts, invalid actions and semantic or
  fingerprint mismatches;
- challenger success count is not below V2 success count;
- challenger mean 180-second restricted TTF is strictly below V2;
- challenger paired-faster fraction against V2 is at least 50%;
- challenger mean normalized wall conflict AUC is not above V2;
- challenger P95 restricted TTF and P95 repair decisions are not above V2;
- no map key has both a challenger success loss and a worse restricted TTF
  relative to V2.

Failure stops collection without tuning or substituting keys.  Because there
is only one screen key per map, these gates are deliberately interpreted only
as cost protection; their values must not be reported as confirmation
estimates.

## Final estimands and gates

Restricted TTF is
`min(reset-inclusive time to feasibility, 180 seconds)`, with every
right-censored episode assigned 180 seconds.  This is the primary timing
estimand because it retains discordant successes and failures.  Raw TTF is
reported on common-success pairs as a conditional diagnostic; it cannot
replace the restricted-TTF result or conceal a success loss.

Normalized wall conflict AUC integrates the observed conflict count through
the same 180-second horizon and divides by initial conflicts times 180
seconds.  Lower is better.  Initially feasible keys are excluded from this
normalized AUC only because the denominator is zero, and their exclusion must
be identical across arms.

The challenger passes against V2 only if all integrity conditions pass and all
of the following hold on the fixed 60 paired keys:

- success count is not below V2;
- mean restricted TTF is strictly below V2;
- paired-faster fraction is at least 50%;
- the lower endpoint of a deterministic 10,000-replicate paired bootstrap
  95% interval for relative restricted-TTF improvement is strictly above zero;
- mean normalized wall conflict AUC is not above V2;
- wall-time right-censoring count is not above V2;
- P95 and maximum repair decisions are not above V2;
- every registered map group's mean restricted-TTF regression is at most 5%;
- on common-success pairs, mean raw TTF is not above V2.

The bootstrap resamples registered map groups and retains all task/seed pairs
inside each sampled group.  This preserves paired execution and avoids
treating repeated solver seeds from one map as independent maps.

Official Adaptive is a formal external comparator, not a source of threshold
tuning.  A statement that the challenger is faster than official LNS2 is
allowed only if the same complete cohort also shows, against
`official_adaptive`:

- success count not lower;
- strictly lower mean restricted TTF;
- paired-faster fraction of at least 50%;
- a strictly positive lower endpoint of the corresponding 10,000-replicate
  map-cluster paired bootstrap interval;
- normalized wall conflict AUC not higher;
- wall-time right-censoring count not higher;
- no map-group restricted-TTF regression above 5%; and
- common-success mean raw TTF not higher.

Passing only the V2 contrast supports an internal improvement over V2; it does
not support a claim of outperforming official LNS2.  Passing neither contrast
keeps V2 as the runtime default.  No result from this study authorizes a new
ranker, another rollback threshold, an additional rescue mechanism or a TTF
claim outside the registered maps and loads.

## Tail and platform reporting

For every arm, the report includes success and censoring counts, mean/median/
P95 restricted TTF, common-success raw TTF, repair-decision P95 and maximum,
and normalized wall AUC.  It also reports platform-trigger frequency,
first-platform escape latency, longest identical repair-fingerprint rollback
streak, cumulative platform occupancy, later-platform count and terminal
platform count.

These platform measurements explain the tail; there is no requirement that a
successful challenger episode contain zero later platforms.  Later stalls are
acceptable only insofar as their elapsed time, repair work and residual
conflicts are already reflected in TTF, repair-tail and AUC outcomes.

## Timing accountability

Reset, state export and verification, V2 proposal generation, StructShell
generation, 124-dimensional feature construction, Copeland scoring, cache
handling, rollback-guard bookkeeping, native PP, trace writing and all Python
or native orchestration are charged to the reset-inclusive clock.  The report
must present mean and P95 time for candidate generation, feature construction,
scoring/selection, guard/cache work, PP and unaccounted residual time.

There is no separate post-hoc allowance for StructShell overhead: an expensive
controller can pass only by recovering that cost in end-to-end restricted
TTF.  Timing components are reported together with a 2% reconciliation alert.
The alert is diagnostic rather than a promotion gate because restricted TTF
stops at the feasibility/censoring boundary while trace finalization and
post-step fingerprinting finish afterward.  The trace must still confirm that
the challenger never performs a hidden second PP call in one decision.
