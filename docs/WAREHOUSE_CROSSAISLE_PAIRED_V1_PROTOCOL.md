# Warehouse cross-aisle paired benchmark qualification v1

## Scope

This is an independent, reset-only qualification protocol for a future
Warehouse benchmark. It contains no controller arm, no policy episode, no
repair action, and no formal TTF measurement. Passing it means only that the
registered tasks are geometrically valid and that all four registered maps
have a preregistered common `q` whose two task variants satisfy the reset
severity gates. It does not promote V2, StructShell, or any other controller.

The four maps are the standard MovingAI Warehouse layouts
`warehouse-10-20-10-2-{1,2}` and `warehouse-20-40-10-2-{1,2}`. Their bytes are
checksum-pinned. No reset or controller artifact from the fixed16 screens is
imported.

## Two-stage freeze

Q0 is intentionally performed before Q1. The initial Q0-only configuration
may generate the deterministic 32-task dataset, but Q1 remains mechanically
blocked. Q0 emits a proposed task registry containing every scenario and task
JSON SHA-256 plus the Q0 manifest SHA-256. That registry and its SHA-256 are
then frozen in the tracked configuration. Before any native reset, the final
runner regenerates or re-audits every task byte and requires an exact match to
the frozen registry. A first Q1 run can therefore never dynamically accept an
unregistered task.

## Q0: geometry and pairing

The cohort fixes task seeds 419 and 463, `q` values 16 and 20, and two variants:

- `reciprocal_cross_aisle_exchange`, the structured challenge;
- `matched_shifted_exchange`, the secondary control.

For each map, the geometry detector must reproduce the registered internal
group count (19, 19, 39, or 39) and aisle width (1, 2, 1, or 2). Boundary groups
and the central group are excluded; exactly six mirrored groups above and six
below the centre are selected. Fewer than 12 eligible selected groups is a Q0
failure and stops the experiment before reset.

`q` is the load per physical row in each direction. The expected agent counts
are 384/480 on width-one layouts and 768/960 on width-two layouts for
`q=16/20`. Every task must have unique starts, unique goals, no start-goal
fixed point, and reachable endpoints. Within each paired task, the variants
must have the same start multiset and the same goal multiset. Their shortest-
path distance means may differ by at most 5%, and their P95 values by at most
10%.

The control is therefore reported only as a
“same-start and same-goal-multiset, distance-near-matched secondary control.”
Unless the complete distance multisets are identical, it must not be called a
fully difficulty-matched control.

## Q1: fresh reset-only qualification

Q1 fixes solver seeds 61–64. The complete cohort is 4 maps × 2 `q` values × 2
task seeds × 2 variants = 32 tasks and 128 native resets. Up to 16 reset workers
may run in parallel. The runner never calls `step()`.

Every reset must complete without an execution error or process timeout, have
a complete and internally consistent initial solution, carry a full state
fingerprint, and be initially infeasible. In addition:

- every structured reset requires at least 16 conflict pairs, at least 32
  active conflict agents, and a largest conflict component of at least 16;
- every matched-control reset requires at least one conflict pair.

For each map, `q=16` is selected only when all eight resets of each variant
pass. Otherwise `q=20` is tested under the same all-reset rule. The benchmark
is ready only if all four maps select their lowest common passing `q`. There is
no map, `q`, task, variant, or seed replacement.

## Identity, resume, and terminal failures

The Q1 execution schedule, final config, task registry, producer sources, and
native module identity are bound before the first reset. Results are appended
and hash-audited by registered task/seed key. A zero-error interruption may
resume only the missing schedule keys under the identical fingerprint.

The first worker execution error or process timeout stops Q1 and makes the
output terminal. A completed threshold failure is also final for this frozen
cohort: it may be inspected but not resampled. Neither case may be repaired by
changing a map, task, `q`, variant, or seed in place.

## Commands

`plan` is read-only. `prepare-q0` generates only the deterministic dataset and
the proposed hash registry. `qualify` requires the frozen tracked registry and
runs Q1 resets only; `--dry-run` stops before native execution. `run` is an
alias for the same Q0/Q1-only path. There are deliberately no `collect` or
`analyze` commands in this runner.
