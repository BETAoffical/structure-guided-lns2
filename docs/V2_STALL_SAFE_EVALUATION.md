# v2 stall-safe evaluation

`v2-stall-safe` is an experimental runtime guard around the frozen `v2-full` controller. It does not
change the model bundle, feature values, candidate scores, ranking, proposal pruner, or PP replanner.
The guard only changes the final admissible neighborhood after repeated PP failures leave paths,
conflicts, and SOC unchanged.

## Guard semantics

For a newly entered repair state, the first action is identical to `v2-full`. After two unchanged
failures at a size cap, the cap changes from 16 to 8 and then from 8 to 4. A candidate that has failed
twice in that state is temporarily blacklisted. Two further unchanged failures at cap 4 activate the
original `official_adaptive` selector until a repair changes the state. Any state change clears all
temporary failure history.

The guard is configured by `configs/v2_stall_guard_v1.json`. `state_revision` is not a stall signal:
the native environment increments it after failed attempts too. The trace therefore records both the
unmodified model winner and the final guarded choice, cap, blacklist, route, and guard overhead.

## Required staged evaluation

First run the same-state probe documented in the README. The guard should proceed only if a smaller
candidate or Adaptive improves PP success by at least 25 percentage points, or if rank 1 never reduces
conflicts while an alternative reduces them in at least two of eight paired trials.

If the probe passes, run the single target episode with an unlimited repair count and a 600-second wall
clock budget:

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick --evaluation-tracks wall-clock \
  --controllers official_adaptive,v2-full,v2-stall-safe \
  --wall-clock-seconds 600 --skip-wall-clock-sensitivity \
  --diagnostic-subset \
  --task-ids maze-128-128-1__random_04__agents_0600 \
  --solver-seeds 2 \
  --controller-runtime optimized --feature-backend native \
  --verification-profile deployment \
  --stall-guard-config configs/v2_stall_guard_v1.json \
  --output build/initlns-v2-stall-safe-targeted-v1
```

This subset cannot promote a controller. It must first show progress after the historical stall and end
below 8,821 conflicts without action, fingerprint, timing, or coverage errors. Only then run the three-way
quick wall-clock cohort, followed by one fixed-100-repair regression cohort.

## Long-horizon interpretation

Long-horizon runs use 1,800 seconds with no repair-count limit and report checkpoints at 300, 600, 1,200,
and 1,800 seconds. An unsolved task is eligible for a paired 3,600-second rerun only when its conflict
count still falls by at least 1% during the last 600 seconds. If improvement is below 1% while PP failure
exceeds 95%, it is marked as an algorithmic plateau; merely increasing the timeout is not treated as a
remedy.

The wall-clock cohort is the primary practical comparison. Fixed-100-repair AUC is retained only once as
a semantic/decision-quality regression check and is not interpreted as a deployment runtime result.
