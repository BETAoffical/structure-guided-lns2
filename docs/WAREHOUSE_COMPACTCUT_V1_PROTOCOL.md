# Warehouse Compact-Cut Q0/Q1 Protocol V1

## Scope

`stride-warehouse-compactcut-v1` is an independent benchmark-construction and
reset-qualification protocol. It may generate and audit registered maps/tasks
(Q0), then call native `reset()` only (Q1). It cannot run a controller step,
policy episode, formal schedule, or TTF collection. A positive result means only
that this frozen benchmark is ready for a separately preregistered controller
evaluation.

No task, reset, or result from the standard cross-aisle experiment or historical
compact Warehouse runs may be imported.

## Frozen cohort

- Eight generated 28x39 compact Warehouse maps: four `cross_four_gate` and four
  `double_horizontal` layouts.
- Map IDs and seeds are fixed in
  `configs/stride_warehouse_compactcut_v1.json`. The fourth
  `double_horizontal` seed is `2026081715`. During the result-blind static
  geometry review, candidate `2026081714` had a smaller cut partition of 96
  cells and failed the preregistered requirement of at least 120; the frozen
  rule selected the first subsequent integer seed passing the same gate, and
  `2026081715` passed with 288. No task, solver result, native reset, or
  controller output was observed in this replacement.
- Within each template, map indices `01` and `03` are preregistered as
  `development`; indices `02` and `04` are preregistered as
  `controller_held_out`. This role is part of every derived task, registry row,
  Q1 schedule key identity, and report row. Held-out maps cannot be used to
  choose or revise a later controller before that controller is frozen.
- Two task seeds: `521` and `557`.
- Exactly 120 agents per task.
- Two variants:
  `bidirectional_mandatory_cut_exchange` (structured) and
  `diagnostic_within_partition_exchange` (control).
- Four solver seeds: `71`, `72`, `73`, and `74`.
- Total: 32 task files and 128 reset keys (64 per variant).

The diagnostic control uses separately registered within-partition endpoints;
it is distance-near-matched only and is not claimed to be a paired causal or
fully difficulty-matched baseline. Its conflict statistics are reported but do
not gate benchmark readiness.

## Q0: geometry and byte registration

Q0 invokes no solver or controller. It must regenerate all eight maps and all 32
tasks deterministically, then independently audit:

- exact map ID, template, seed, dimensions, realised grid, and a real two-cell
  mandatory cut with no bypass;
- eight distinct realised map-grid SHA-256 values; any duplicate grid is a Q0
  hard failure even when its map ID or seed differs;
- unique starts, unique goals, zero fixed points, and reachability for all 120
  agents;
- every structured start-goal pair crosses the registered cut;
- for each structured task, each of the two travel directions sends exactly 30
  agents through each of the two registered cut cells;
- every diagnostic-control pair remains inside its registered partition;
- structured/control shortest-distance mean relative difference at most 5% and
  P95 relative difference at most 10%;
- a deterministic, collision-free prioritized joint MAPF feasibility witness for
  each task, checked by tick-level vertex, edge, and goal-occupancy simulation
  rather than by independent paths;
- MovingAI scenario row count, coordinate convention, dimensions, and exact
  four-neighbour shortest distances.

Q0 writes a proposed registry containing every map, metadata, scenario, task,
witness, and Q0-manifest hash. Q1 is mechanically blocked while the main config
says `pending_q0_task_hash_registration`. Formal WSL Q0 passed, and its proposal
is now copied byte-for-byte into
`configs/stride_warehouse_compactcut_tasks_v1.json`: registry SHA-256
`368ed483fec21875f34f0acbf888bb28562b0bce0f7a00fd4356813284aab7b9`,
with Q0-manifest SHA-256
`70d0c549e22a3d71b4ffb64e593b918e4d4182a445637d2625298c55bc3dffc7`.
The main config status is `frozen_q0_task_hash_registration`; Q1 may proceed
only against those registered bytes.

The task generator itself is frozen before Q0 by SHA-256. Formal Q0 generation
must use the final Q1 Ubuntu-22.04 WSL toolchain so the recorded generation
toolchain is identical to the one carried into Q1; cross-toolchain parity is
supporting evidence, not permission to mix Q0 artifacts from two toolchains.

The byte-reproduction record admits exactly two audited toolchains:

- Ubuntu-22.04 WSL: Python 3.10 (observed 3.10.12), NumPy 1.21.5, SciPy 1.8.0;
- Windows native: Python 3.12 (observed 3.12.3), NumPy 1.26.4, SciPy 1.13.1.

Both use `scipy.optimize.linear_sum_assignment`. For the registered eight-map,
two-task-seed cohort, all 16 full pair payloads had identical canonical SHA-256
under the two toolchains. This is cohort-level byte-parity evidence only, not a
general cross-version reproducibility claim. The registry records all 16 pair
hashes and, for every individual task, separate endpoint-payload, task-file,
scenario-file, and joint-witness hashes.

## Q1: reset-only qualification

Q1 re-generates and re-audits Q0 evidence byte-for-byte, validates the full
runtime config with `max_decisions=0`, and executes the exact 128 registered
reset keys. It uses 16 workers and a 240-second process timeout. The first
execution error or process timeout is terminal. An interruption may resume only
with the identical config, producer identity, schedule, Q0 registry, and runtime
preflight fingerprint, and only when no prior error or timeout exists.

Benchmark readiness requires:

- all 128 reset keys present exactly once;
- zero execution errors and zero process timeouts;
- every initial solution complete and internally consistent;
- all 64 structured resets have at least 16 conflict pairs, 32 active conflict
  agents, and a largest conflict component of at least 16, and are not initially
  feasible;
- all 64 diagnostic-control resets complete without error. Their structure
  thresholds are diagnostic and cannot fail or rescue the structured gate.

Any failed gate produces `stop_do_not_resample_replace_or_run_controller`.
Maps, tasks, variants, or seeds must not be replaced based on Q1 results.

## Commands and result boundary

The CLI exposes only `plan`, `prepare-q0`, `qualify`, and the `run` alias for
`qualify`. There is no `collect` or `analyze` command. Q0 may be run with a
pending registry; Q1 cannot. No output from this protocol supports a controller
performance, AUC, or TTF claim.
