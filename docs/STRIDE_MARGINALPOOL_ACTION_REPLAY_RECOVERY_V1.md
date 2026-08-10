# MarginalPool action replay recovery v1

Date: 2026-08-11

## Failure diagnosis

The first full collection stopped 7200.638 seconds after its four-worker
resume. This matched the registered 7200-second per-state timeout. The timed
out state was the maximum-workload cohort member: 600 agents, mean shortest
distance 664.377, 42 candidate actions, and 672 paired PP/SIPPS trials.

The timeout itself was valid. The visible `KeyError: 'row'` was a secondary
scheduler defect: the generic failure serializer expected dataset jobs with a
`row` field, whereas MarginalPool jobs use `job_id` and `state_record`.

## Frozen recovery policy

- Preserve the failed v1 output read-only.
- Import every valid completed state; do not inspect outcomes to choose states.
- Pin the source run, status, and all 22 imported state SHA-256 values in
  `configs/stride_marginalpool_action_replay_recovery_v1.json`.
- Keep all 78 states, 2502 candidates, trial indices 0-15, paired PP seeds,
  native PP/SIPPS semantics, 124D features, labels, and integrity gates.
- Use four workers.
- Limit each state attempt to 1800 seconds.
- Write an atomic partial checkpoint after all 16 trials for each candidate.
- Resume only from a validated complete-candidate checkpoint.
- Permit at most four attempts per state, for a 7200-second cumulative hard
  bound.
- Stop early after two consecutive timed-out attempts without an increase in
  completed candidate checkpoints.
- A terminal timeout remains an integrity failure; it is never silently
  omitted or treated as a successful label.

The 30-minute limit is an execution fuse, not a label and not a TTF metric.
Runtime values and future trajectories remain forbidden from candidate-quality
analysis.

## Recovery command

```text
wsl.exe -d Ubuntu-22.04 --cd "/mnt/c/Users/18448/Documents/lns2 2/structure-guided-lns2" -- /usr/bin/env PYTHONPATH=build/wsl-release /usr/bin/python3 scripts/run_stride_marginalpool_action_replay.py collect --config configs/stride_marginalpool_action_replay_v1_registration.json --output build/stride-marginalpool-action-replay-recovery-v1 --mode full --workers 4 --preflight-output build/stride-marginalpool-action-replay-preflight-v1 --recovery-source build/stride-marginalpool-action-replay-v1 --recovery-registration configs/stride_marginalpool_action_replay_recovery_v1.json --maximum-state-attempts 4 --per-state-attempt-timeout-seconds 1800 --resume
```

If an external terminal interruption occurs, rerun the identical command with
`--resume`. Complete states and complete-candidate partial checkpoints must be
validated against the same run fingerprint before reuse.
