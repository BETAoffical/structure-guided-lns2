# STRIDE Stage 4R high-load diagnostic protocol

## Question

Before changing the quality label or adding training maps, test whether the
current `stride-quality-v1` selector improves or harms high-load end-to-end
behavior relative to frozen `v2-full`. This is a development diagnosis, not a
formal Stage 5 result or a promotion test.

## Registered cohort

- `maze-128-128-2`, MovingAI random scenarios 11 and 17, 300 agents. This
  reduces the previous 400-agent load while retaining the 128x128 maze.
- `room-64-64-8`, MovingAI random scenarios 4 and 5, 500 agents. This lies
  between the previously studied 400- and 600-agent loads.
- Solver seeds 1, 2, 3, and 4.
- Frozen controllers `v2-full` and `stride-quality-v1` only.
- Strict serial execution with rotating controller order for every paired
  task/seed key.
- A 180-second wall budget per episode; unsuccessful episodes receive the cap
  in the primary metric.

The four tasks are fixed from map family, scenario, and agent load before any
current quality-controller outcome is read. They are not selected by relative
controller performance.

## Evidence boundary

`maze-128-128-2` is an existing development map. Reading current-policy
outcomes on `room-64-64-8` reclassifies that map as development evidence;
`room-64-64-16` remains unread for the current quality policy and is retained
as the later room holdout. Results may guide label and training-map changes,
so this run cannot support a formal OOD or publication speed claim.

## Decision rule

Mean capped wall TTF is primary and quality success count may not be below V2.
Common-success TTF, repair iterations, no-progress steps, wall-clock conflict
AUC, PP replan time, and controller time explain the result. Analysis is paired
by task and solver seed and must also verify equal initial fingerprints and
initial conflicts, zero execution errors, zero invalid actions, and complete
16-key coverage per controller.

Because 300- and 500-agent states are outside the main agent-count range of the
existing training collections, the report must also expose pruner fallback,
OOD fallback, and selected-feature outside-range fractions. A high-load result
is not attributed to learned ranking when the corresponding controller mostly
falls back to its safe route.

The result will decide whether the next change should primarily target:

1. high-load map/state coverage;
2. multi-PP-seed label uncertainty;
3. residual-state/stall hardness; or
4. a combination of these factors.
