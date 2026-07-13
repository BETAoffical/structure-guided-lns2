# Trace and Policy API

## Episode boundary

`lns2_repair` retains the official outer flow: try collision-free PP first, then invoke InitLNS when PP
fails, and stop when the first feasible solution is found. `LNS2RepairEnv` starts at the InitLNS phase and
uses SIPPS soft-conflict prioritized planning to construct its initial repair state.

An episode is terminal when conflicts reach zero. It is truncated when the time or repair-iteration limit
is reached before feasibility.

## Observation

Each observation includes:

- map rows, columns, and flattened obstacle mask;
- caller-provided map/task context;
- iteration, runtime, sum-of-costs, and colliding-pair count;
- conflict-graph edges;
- each agent's start, goal, path, path cost, shortest-path cost, delay, and conflict degree;
- cumulative low-level expanded, generated, reopened, and run counts.

## Actions

- `{"mode": "official"}` delegates to the configured official selector without consuming extra RNG.
- `{"mode": "seed", "heuristic": "collision", "seed_agent": 7,
  "neighborhood_size": 8}` fixes the seed and lets an official generator complete the neighborhood.
- `{"mode": "explicit_neighborhood", "agents": [2, 7, 9]}` requests a complete subset.

Any action can add `"random_seed": 123` to start a controlled per-action random stream. Omitting this
field consumes random numbers in the original official order and preserves upstream parity. The
requested seed is returned as `requested_random_seed` and written into JSONL actions.

Seed actions must reference a currently conflicting agent. Explicit sets must contain unique valid IDs
and touch the current conflict graph. Invalid external actions fall back to official selection and expose
`action_valid=false`.

## Step result

`step()` returns `observation`, `metrics`, `terminated`, and `truncated`. Metrics contain requested and
applied actions, the actual neighborhood, validity, repair success, conflict/cost deltas, and step time.
No reward is returned; experiments derive rewards from these raw values.

## JSONL

Trace schema version 1 uses three event types:

- `initial`: complete state immediately after InitLNS initial planning;
- `transition`: requested/applied action, before/after states, and raw outcomes;
- `finish`: final state and success flag.

Every line is standalone JSON and can be streamed without loading a complete run into memory.
