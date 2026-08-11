# MarginalPool partial-checkpoint continuation segment 2

Date: 2026-08-11

The first continuation segment completed seven of its eight imported partial
states.  The remaining state reached 38 of 42 candidates before its fourth
1800-second attempt expired.  It made checkpoint progress on every attempt;
the stop was cumulative segment-budget exhaustion, not an execution error and
not the two-attempt no-progress rule.

This second continuation segment imports all 77 complete states and the one
remaining partial state.  The source run, status, report, aggregate complete-
state digest, partial hash, and the 38-candidate/608-trial checkpoint are pinned
in `configs/stride_marginalpool_action_replay_continuation_v2.json`.  No state,
candidate, PP seed, feature, label, or result-based selection changes are
allowed.  The additional bounded execution segment is operational recovery
only and is not a TTF measurement.

The frozen execution policy remains four configured workers, a 1800-second
per-attempt fuse, at most four attempts, candidate-level atomic checkpoints,
and the two-consecutive-no-progress stop.  Only one state remains, so at most
one actual compute worker can be active.

The segment completed the remaining four candidates in its first attempt.
The final collection contains 78 states, 2,502 candidates and 40,032 trials,
with zero terminal errors or timeouts and all integrity gates passed.  The
scientific result is recorded in
`docs/STRIDE_MARGINALPOOL_ACTION_REPLAY_V1_REPORT.md`.

```text
wsl.exe -d Ubuntu-22.04 --cd "/mnt/c/Users/18448/Documents/lns2 2/structure-guided-lns2" -- /usr/bin/env PYTHONPATH=build/wsl-release /usr/bin/python3 scripts/run_stride_marginalpool_action_replay.py collect --config configs/stride_marginalpool_action_replay_v1_registration.json --output build/stride-marginalpool-action-replay-continuation-v2 --mode full --workers 4 --preflight-output build/stride-marginalpool-action-replay-preflight-v1 --recovery-source build/stride-marginalpool-action-replay-continuation-v1 --recovery-registration configs/stride_marginalpool_action_replay_continuation_v2.json --maximum-state-attempts 4 --per-state-attempt-timeout-seconds 1800 --resume
```
