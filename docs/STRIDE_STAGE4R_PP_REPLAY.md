# STRIDE Stage 4R same-state paired PP replay

## Question

The four-seed end-to-end diagnostic changed the initial PP solution and every
later PP random stream. It therefore showed seed sensitivity, but could not say
whether V2 and quality selected intrinsically different first neighborhoods or
whether PP randomness reversed the apparent winner.

This diagnostic fixes the incoming path state and compares the first
neighborhood selected by `v2-full` with the first neighborhood selected by
`stride-quality-v1`. Both actions are repaired under the same 16 deterministic
PP seeds. It evaluates one current decision only; it does not predict later LNS
steps and is not Cost-to-Go or Receding-Q.

## Registered scope

- All 16 decision-zero states from four tasks and solver seeds 1--4.
- Two recorded actions per state: V2 and quality.
- Sixteen paired PP seeds per action: 512 single-step repairs.
- Ten states have identical V2/quality neighborhoods. They are retained as an
  exact determinism control.
- Six states have different neighborhoods and form the primary selection
  comparison.
- State restoration uses the recorded paths and verifies the repair-structure
  fingerprint before every trial.
- Execution is serial and alternates action order within successive PP seeds.

The primary outcome is immediate conflict reduction. Repair time is secondary;
the user's objective is fewer repair rounds and lower end-to-end TTF rather
than optimizing the PP implementation itself.

## Pre-registered interpretation

For each different-action state, V2 and quality are compared under each paired
PP seed. The winner is the action with greater immediate conflict reduction.

- A state has a winner flip when at least one PP seed favors V2 and at least one
  favors quality.
- PP randomness is material when at least 25% of different-action states flip,
  or mean non-tie winner stability is below 70%.
- An action is robustly preferred only when it wins at least 10/16 seeds, has
  greater mean reduction, and does not increase no-progress probability.

If PP randomness is material, the next label/model must retain multi-seed
aggregation and expose action uncertainty or a robust margin. If rankings are
stable and V2 wins more states, the one-step quality label or its generalization
must be revised. If quality wins but end-to-end TTF remains worse, the next
target is residual-state hardness after the first repair.

This cohort is inherited from an outcome-informed diagnostic. No formal speed
claim or controller promotion is permitted.

## Commands

```bash
python3 scripts/run_stride_stage4r_pp_replay.py prepare
python3 scripts/run_stride_stage4r_pp_replay.py dry-run
python3 scripts/run_stride_stage4r_pp_replay.py run
python3 scripts/run_stride_stage4r_pp_replay.py analyze
```
