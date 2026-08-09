# STRIDE Maze Tail Evidence v1 Protocol

## Purpose

The known Maze regression is not enough evidence to train a long-tail predictor.
It comes from one `maze-128-128-1` layout, two task seeds, and three solver
seeds.  The retrospective ClosurePool analyses also found that simple spatial
or temporal closure rules do not consistently separate harmful from beneficial
StructPool or SlotPool replacements.  This protocol therefore collects new,
independent Maze states before any predictor, gate, or runtime rule is designed.

This is a reset-only preflight.  It cannot establish tail incidence, TTF
improvement, fresh-map generalization, or justify training.

## Frozen candidate product

The task seeds `811` and `853`, solver seeds `17`, `29`, and `43`, load
ladders, task generator, and selection rule were registered before generating
the new dataset or running a reset.

| MovingAI map | Corridor character | Agent-count ladder |
| --- | --- | --- |
| `maze-32-32-4` | small, relatively open | 80, 120, 160, 200 |
| `maze-128-128-1` | narrow one-cell corridors | 60, 80, 100, 120 |
| `maze-128-128-2` | medium-width corridors | 300, 400, 500, 600 |
| `maze-128-128-10` | wide corridors, registered negative control | 400, 600, 800, 1000 |

Each map uses both `uniform_random` and `opposite_exchange` project-derived
static OD tasks.  These are not official MovingAI scenario files.  The full
product is 4 maps x 4 loads x 2 task seeds x 2 task variants = 64 tasks, and
64 tasks x 3 solver seeds = 192 reset jobs.

All four map archives and map members are checksum-pinned.  In particular,
`maze-128-128-10` is retained even if every reset remains conflict-free.  It
will not be replaced after observing controller outcomes.

## Outcome-blind selection

Selection may read only reset status, initial-completion and consistency flags,
initial conflicts and complexity, and the state fingerprint.  It may not read
candidate repair results, chosen actions, repair iterations, PP time, future
trajectories, or TTF.

For every map, task-variant family, and conflict band, at most one task is
selected:

- moderate: mean initial conflicts in `[16, 100)`, target 50;
- high: mean initial conflicts in `[100, 400]`, target 200.

Every solver seed for a selected task must be valid and complete and must have
at least 16 initial conflicting pairs.  The deterministic tie-break is target
distance, lower agent count, lower task seed, then task ID.

The preflight passes only when the complete 192-reset product has zero errors,
forbidden fields are absent, at least three maps and eight tasks are selected,
each task variant covers at least two maps, both task seeds occur, and both
conflict bands occur.  An unselected registered map remains in the report as a
negative control.

## Ordered next evidence

On a pass, the selected task IDs and artifact hashes are frozen in a new
configuration before any full solver episode is run.  The next block compares
`v2-full`, `v2-plus-structpool`, and `v2-plus-slotpool` under strict rotating
order, paired initial states, deterministic PP replay, and run-to-completion
raw TTF.  No key may be removed based on a controller result.

After full episodes, the first divergence state for each challenger is replayed
for the challenger action and its paired V2 action with 16 paired PP seeds.
Only then may action stability and independent harmful-tail coverage be
audited.  A predictor is designed only if those gates show enough independent
positive examples and stable labels.  Otherwise the complete evidence is
retained and a second task-seed block is preregistered without outcome-based
subselection.

## Registered files

- source: `configs/stride_maze_tail_evidence_source_v1.json`;
- preflight: `configs/stride_maze_tail_evidence_preflight_v1.json`;
- runtime: `configs/stride_maze_tail_evidence_preflight_runtime_v1.json`;
- implementation: `experiments/stride_maze_tail_evidence.py`;
- CLI: `scripts/run_stride_maze_tail_evidence.py`.
