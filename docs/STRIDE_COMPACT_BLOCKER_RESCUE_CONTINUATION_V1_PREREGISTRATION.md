# STRIDE Compact-Blocker Rescue Continuation v1

## Question

Failure-informed blocker augmentation passed its bounded development mechanism
gate, while semantic compaction produced favorable but unstable one-step point
estimates. This experiment asks one interaction question:

> After the same exact native PP rollback, can unsupported original members be
> removed before adding observed external blockers, improving escape and PP
> cost without sacrificing continuation quality?

This is not an independent confirmation. It deliberately reuses all 45 frozen
outcome-enriched development states to decide whether the interaction deserves
a new result-blind cohort.

## One-call intervention

Every arm executes the same frozen first action with the same PP seed and
native order. After `conflict_bound_exceeded` with exact rollback, each rescue
arm gets the same fresh seed and exactly one next-decision PP call:

- same full set;
- semantically compact set;
- full set plus at most eight blockers observed in the failed PP;
- semantically compact set plus those same blockers.

The compact set retains all current conflict endpoints and all original
members with visible low-degree/articulation corridor overlap plus reverse
queue evidence. It removes only unsupported original members, requests no
target size and falls back to the exact full set when nothing is removable.
No arm explicitly controls PP order, rescues after native `time_limit`, or
tries a second rescue set after failure.

## Execution and gates

The initial phase is 45 states x four trials x five arms = 900 episodes with
16 workers. Episodes run to feasibility or 64 repair decisions / 180 seconds;
process and outer limits remain 240 and 300 seconds. All 45 states and arms may
extend uniformly to trials 4--7 only if the compact-blocker arm, on the paired
compact-eligible trigger subset:

1. lowers next-decision unresolved risk and rescue PP wall versus full blocker;
2. does not lower success or worsen normalized fixed AUC or restricted repair
   decisions;
3. worsens no map's unresolved point estimate by more than five points; and
4. the full blocker arm still improves directionally over the frozen control
   under the new producer identity.

The final gate requires both compact-blocker primary 95% bootstrap upper bounds
below zero, all safety metrics to pass, and the blocker-versus-control unresolved
95% upper bound below zero. Failure closes this interaction branch without
threshold tuning, state filtering or additional seeds.

Passing remains development evidence only. It cannot authorize runtime
integration, pool replacement, model training or raw-TTF testing. A separate
result-blind confirmation on new states and maps is still required.
