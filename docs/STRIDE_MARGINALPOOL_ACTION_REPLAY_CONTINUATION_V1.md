# MarginalPool partial-checkpoint continuation v1

Date: 2026-08-11

The first registered recovery segment exhausted its four 1800-second attempts
with 70 of 78 states complete. All eight remaining states increased their
complete-candidate checkpoint count on the fourth attempt. The stopping event
was therefore a cumulative-budget exhaustion, not an execution error and not
the registered two-attempt no-progress condition.

This continuation is a separate run identity. It imports all 70 valid complete
states and all eight valid partial states from the failed recovery output. The
source run, status, report, aggregate complete-state hash, every partial-state
hash, and every partial candidate/trial count are pinned in
`configs/stride_marginalpool_action_replay_continuation_v1.json`.

The importer validates the old run fingerprint and state identity, rewrites
the artifacts to the new run fingerprint, and validates the imported partial
again. The worker then independently reproduces the candidates and 124D
features before accepting the checkpoint. No state, candidate, seed, label, or
outcome-dependent selection changes are allowed.

The continuation retains four workers, a 1800-second per-attempt fuse, four
attempts, candidate-level atomic checkpoints, and the two-consecutive-no-
progress stop. Its additional 7200-second allowance is explicitly a second
registered execution segment; it is not hidden inside the first segment's
metadata and is not a TTF measurement.

```text
wsl.exe -d Ubuntu-22.04 --cd "/mnt/c/Users/18448/Documents/lns2 2/structure-guided-lns2" -- /usr/bin/env PYTHONPATH=build/wsl-release /usr/bin/python3 scripts/run_stride_marginalpool_action_replay.py collect --config configs/stride_marginalpool_action_replay_v1_registration.json --output build/stride-marginalpool-action-replay-continuation-v1 --mode full --workers 4 --preflight-output build/stride-marginalpool-action-replay-preflight-v1 --recovery-source build/stride-marginalpool-action-replay-recovery-v1 --recovery-registration configs/stride_marginalpool_action_replay_continuation_v1.json --maximum-state-attempts 4 --per-state-attempt-timeout-seconds 1800 --resume
```
