# STRIDE RepairClosurePool v1 result

## Execution identity

- Frozen development cohort: 90 logical checkpoints / 78 unique states.
- Frozen V2 base pool: 1,367 existing candidates.
- Outcome-blind RepairClosurePool challengers: 922 candidates, with 16 strictly
  paired native PP trials per candidate (14,752 trials total).
- Runtime selector: not used.
- Repair order: not controlled.
- Workers: 16; hard per-job limit: 300 seconds; fail-fast on the first execution
  error or timeout.
- Report SHA-256:
  `e10b09b9a26bd277a699af3938f6bde797e1f538f8ae11ae17c3e9c40f4d7672`.

All 14,752 trials completed with zero errors and zero timeouts.  Native action,
state, candidate, seed and repair-fingerprint integrity passed.  No runtime,
TTF or future-trajectory field entered the labels.

## Frozen readiness result

The candidate-pool opportunity gate failed.

- A new candidate robustly beat the best frozen V2 base candidate in 5/78
  states (6.41%), below the preregistered 10% threshold.
- A new candidate entered the stable frontier in 20/78 states (25.64%), above
  the 10% threshold.
- The mean advantage of each state's best new candidate over its best V2 base
  candidate was -0.09240 normalized conflict reduction.
- New candidate sizes ranged from 6 to 64 agents; median size was 54 and mean
  size was 51.57.  The closure usually expanded close to the frozen 64-agent
  cap rather than identifying a compact dependency boundary.

By map, robust-best opportunity was 1/29 (3.45%) on `maze-128-128-1`, 1/21
(4.76%) on `maze-128-128-2`, and 3/28 (10.71%) on `maze-32-32-4`.  Stable
frontier additions occurred in 5/29, 5/21 and 10/28 states respectively.

## Interpretation and stop decision

The audit bypassed the known ranker issue, so this failure is attributable to
the proposed candidate generator rather than candidate selection.  The
evidence-front closure adds some useful actions, but its same-cell/path
dependency relation is too permissive: most candidates absorb many weakly
related agents, and even the best new action is usually worse than an existing
V2 action.

RepairClosurePool v1 is therefore not eligible for result-blind continuation,
model training, runtime integration or TTF testing.  No state may be removed
or selected based on this result.  The next safe development step is a new,
separately preregistered compact dependency generator that distinguishes
causal external blockers from generic long-path overlap and is audited against
this complete frozen result without claiming generalization.
