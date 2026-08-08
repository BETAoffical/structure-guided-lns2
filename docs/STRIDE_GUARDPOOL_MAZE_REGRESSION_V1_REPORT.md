# STRIDE GuardPool Known-Maze Regression v1 Result

## Decision

`stride-guardpool-v1` fails the frozen known-Maze regression and must not enter
the development TTF Quick.  It succeeds, but requires 61 repair iterations,
more than twice the preregistered maximum of 30 and more than four times the V2
result of 15.

This is a hard stop for the current GuardPool logic.  It is not a failure of
runtime integrity: all four episodes completed, initial state identity matched,
and there were zero execution errors, invalid actions, or fingerprint
mismatches.

## Paired result

| Controller | Success | Repairs | Raw wall TTF (s) | Repair time (s) | Controller time (s) | Maximum no-progress streak |
|---|---:|---:|---:|---:|---:|---:|
| `v2-full` | yes | 15 | 6.4229 | 3.9027 | 0.4191 | 2 |
| `v2-plus-structpool` | yes | 301 | 133.1809 | 109.5225 | 19.5504 | 140 |
| `v2-plus-slotpool` | yes | 61 | 10.5504 | 6.3632 | 1.9595 | 49 |
| `stride-guardpool-v1` | yes | 61 | 10.6594 | 6.4799 | 1.9166 | 49 |

The unguarded SlotPool and GuardPool conflict trajectories are identical.  Both
start with three structural choices and reduce conflicts quickly:

```text
66 -> 33 -> 19 -> 12
```

At 12 conflicts the high-stress activation gate already disables structural
candidates because its minimum is 16.  The controller is therefore already an
exact V2 fallback from decision 3 onward.  It later reaches one conflict and
stalls for 49 consecutive rounds.

The eight-round guard triggers at decision 19, but it has nothing left to
suppress: structural candidates have already been inactive for 16 decisions.
The same V2 fallback then needs 42 more repairs to remove the last conflict.
Consequently the guard cannot undo the harmful state reached by the first three
structural actions and cannot improve the SlotPool trajectory.

## Integrity audit

- four-controller coverage: passed;
- initial conflicts: 66 for every controller;
- initial fingerprint: identical;
- V2 result: reproduced at exactly 15 repairs;
- invalid actions: 0;
- fingerprint mismatches: 0;
- deterministic PP replay: passed for 61 groups sharing both decision index and
  before-state fingerprint;
- execution errors: 0.

The initial analysis incorrectly required PP seeds to remain equal after
controller states had diverged.  That diagnostic was corrected after the result
to use the actual deterministic contract: equal decision index plus equal
before-state fingerprint.  The correction changes only the integrity flag; the
61-repair GuardPool hard failure is unchanged.

## Interpretation

The frozen SlotPool model does what it was trained to do: its first structural
actions produce large immediate conflict reductions.  The regression exposes
the missing guarantee: strong one-step reduction can move the solver into a
low-conflict state that is much harder for subsequent PP repairs.  A late
no-progress fallback cannot repair this because it neither predicts the trap
before the structural action nor rolls the state back afterward.

The next design must act before the first structural deviation, for example by
an anchor-relative safety gate or a state-risk predictor trained to abstain from
structural challenges.  That is a new design and requires a new preregistration;
the current GuardPool parameters must not be tuned on this known regression.

## Artifacts

- report: `build/stride-guardpool-maze-regression-v2/guardpool_maze_regression_report.json`
- report SHA-256: `84b5ea9dd8c44f0abfd1cfbe49086ee7ed1d9def6e2e578b60648132790a3ebe`
- config SHA-256: `fdba12a54ad9503d5169d4a295a8d7ca2e028e20ddbc9e4efd6dbb048aaa228d`

The report is known-regression evidence only.  It is not independent
generalization evidence and supports no formal TTF improvement claim.
