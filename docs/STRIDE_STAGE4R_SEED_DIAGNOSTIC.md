# STRIDE Stage 4R multi-seed tail diagnostic

## Purpose and evidence boundary

The first Stage 4R Quick used one solver seed on eleven map-disjoint tasks. It
showed that both STRIDE candidates were slower than `v2-full` in mean capped
wall time to feasible, with most of the loss concentrated on
`lt_gallowstemplar_n` and `maze-128-128-2`. A one-seed result cannot distinguish
a persistent selector weakness from an unlucky initial-PP and downstream-PP
random stream.

This follow-up is therefore an **outcome-informed diagnostic**, not a formal
benchmark. Its four tasks were chosen after reading the Quick-v2 seed-1
outcomes. Seeds 2--4 were registered before they were executed. The report must
set `formal_speed_claim` to false and must never promote a controller.

## Registered cohort

- Controllers: `v2-full`, `stride-control-v1`, `stride-quality-v1`.
- Solver seeds: 1, 2, 3, 4.
- Tasks: two generated contrast tasks and two 400-agent MovingAI problem tasks.
- Schedule: 4 tasks x 4 seeds x 3 controllers = 48 serial episodes in strict
  rotating Latin order.
- Budget: 120 seconds of reset-inclusive wall time per episode; 180-second
  external process timeout.
- Primary diagnostic metric: mean capped wall time to feasible (TTF).
- Supporting metrics: success count, common-success TTF, repair rounds,
  no-progress transitions, wall-conflict AUC, PP time, and controller overhead.

Changing `solver_seed` changes both the initial PP solution and the subsequent
PP random stream. This experiment measures end-to-end seed robustness; it is
not a same-state repair-replay experiment.

## Registered decision rule

For each of the two MovingAI problem tasks, a quality episode is adverse when
it is slower than its paired V2 episode or uses more repair rounds. If at least
three of four seeds are adverse on **both** problem tasks, the result is treated
as persistent tail weakness and the next action is to revise the one-step label
toward residual-state hardness. If signs are mixed across seeds, the next action
is a same-state paired PP-repair replay before changing the label.

The rule is diagnostic. The cohort is small and outcome-selected, so even a
positive aggregate TTF result only authorizes a larger, outcome-blind
development cohort.

## Commands

Run in WSL with the native module available:

```bash
python3 scripts/run_stride_stage4r_seed_diagnostic.py prepare
python3 scripts/run_stride_stage4r_seed_diagnostic.py dry-run
python3 scripts/run_stride_stage4r_seed_diagnostic.py run
python3 scripts/run_stride_stage4r_seed_diagnostic.py analyze
```

The default output is `build/stride-stage4r-seed-diagnostic-v1`. Long-running
execution is monitored at 30-minute intervals rather than polled continuously.
