# Warehouse Cross-Aisle Paired V1 Result

## Decision

The reset-only benchmark qualification failed. No map has a registered `q`
for which both task variants pass every reset, so `benchmark_ready=false`,
`selected_map_count=0`, and the registered decision is
`stop_do_not_resample_or_replace`.

No controller action, policy episode, repair step, or TTF experiment was run.
This result does not compare V2, StructShell, Boundary16, or Official Adaptive.

## Registered identity and integrity

- Protocol source commit: `1a9ea72`.
- Output: `build/wh-cross-v1-r1`.
- Run fingerprint:
  `684e3c6152c61480a86e038e29c6573d3e49790a2059456f0a46bb25ccd046ae`.
- Runtime preflight fingerprint:
  `5525cdf83383a0a36dcb2e91b1dd4946619b73ae6460bb5a7a4b198d7dcd6e0c`.
- Q0 manifest SHA-256:
  `fa084691b492655a9e70ab7f2b7bbe6e049fa78c66986a43de755ff3022d0ee8`.
- Q1 manifest SHA-256:
  `592ddf6f5c2271cc62bf6d978c20a32b141c812bb36e6dd4e72bb4b8387790a5`.
- Qualification report SHA-256:
  `6db863449d0283cc14ac5c7a127c0165d868dcdfbbe1002d762fae2e4688b55b`.
- Benchmark selection SHA-256:
  `88eb31ce8bcddd0a3c2667151c2250c33d11687bee8303af2f8c676a641c7002`.

All 128 registered resets completed and were internally consistent. There
were zero execution errors, process timeouts, duplicate keys, unexpected
keys, or identity mismatches. The manifest and execution schedule contain the
same 128 unique task/seed keys. The four maps contribute 32 resets each; the
two variants and two `q` values contribute 64 each. Status records
`complete=true`, while `benchmark_ready=false`.

## Reset-only qualification result

Of 128 initial states, 84 were already feasible with zero conflicts and only
44 retained any conflict after InitLNS initialization.

For the intended structured variant, 48/64 resets had zero conflicts. The 16
nonzero resets were still small: the global maxima were 6 conflict pairs, 7
active conflict agents, and a largest conflict component of 7, all far below
the registered `16/32/16` gate. Every map-by-`q` structured slice contained at
least one zero-conflict reset.

For the distance-near-matched secondary control, 36/64 resets had zero
conflicts. Its global maxima were 8 conflict pairs, 12 active conflict agents,
and a largest component of 5. It was often harder than the intended reciprocal
flow, but still represented sparse, small-component repair states.

| Map | Agents | Variant | Nonzero resets | Mean conflicts | Max conflicts |
|---|---:|---|---:|---:|---:|
| warehouse-10-20-10-2-1 | 384 | matched shifted | 0/8 | 0.000 | 0 |
| warehouse-10-20-10-2-1 | 384 | reciprocal cross-aisle | 0/8 | 0.000 | 0 |
| warehouse-10-20-10-2-1 | 480 | matched shifted | 0/8 | 0.000 | 0 |
| warehouse-10-20-10-2-1 | 480 | reciprocal cross-aisle | 0/8 | 0.000 | 0 |
| warehouse-10-20-10-2-2 | 768 | matched shifted | 8/8 | 2.500 | 5 |
| warehouse-10-20-10-2-2 | 768 | reciprocal cross-aisle | 1/8 | 0.375 | 3 |
| warehouse-10-20-10-2-2 | 960 | matched shifted | 6/8 | 2.750 | 5 |
| warehouse-10-20-10-2-2 | 960 | reciprocal cross-aisle | 5/8 | 0.750 | 2 |
| warehouse-20-40-10-2-1 | 384 | matched shifted | 0/8 | 0.000 | 0 |
| warehouse-20-40-10-2-1 | 384 | reciprocal cross-aisle | 0/8 | 0.000 | 0 |
| warehouse-20-40-10-2-1 | 480 | matched shifted | 0/8 | 0.000 | 0 |
| warehouse-20-40-10-2-1 | 480 | reciprocal cross-aisle | 0/8 | 0.000 | 0 |
| warehouse-20-40-10-2-2 | 768 | matched shifted | 7/8 | 2.000 | 6 |
| warehouse-20-40-10-2-2 | 768 | reciprocal cross-aisle | 4/8 | 0.875 | 3 |
| warehouse-20-40-10-2-2 | 960 | matched shifted | 7/8 | 3.375 | 8 |
| warehouse-20-40-10-2-2 | 960 | reciprocal cross-aisle | 6/8 | 1.875 | 6 |

The width-two maps show more residual conflicts partly because the registered
same-`q` rule doubles their agent count. This result therefore cannot be used
to attribute a difference solely to aisle width.

## Why the geometric task did not create the intended repair state

The Q0 properties were real but insufficient. Reciprocal endpoints and a
shared direct cross-aisle certify static path overlap, not unavoidable
time-indexed conflicts. InitLNS randomizes the agent order and plans agents
sequentially. Its low-level planner can avoid already planned paths by waiting
or taking a longer route. Standard Warehouse maps provide many vertical
connections, loops, and open staging cells, so most apparent head-on traffic
was absorbed before the LNS repair phase began.

The design also spread demand over 12 parallel cross-aisle groups. This raised
the total agent count without concentrating enough agents into one conflict
component. Reciprocal two-cycles were easier for prioritized planning than the
cross-group shifted mapping; the latter generated more residual conflicts,
but not enough to satisfy the preregistered structural gate.

The qualification metrics were therefore measuring the right object: actual
post-initialization repair work, rather than static endpoint overlap. The
negative gate prevents controller timing from being run on tasks that mostly
require no LNS repair.

## Next step

Stop v1 exactly as registered. Do not increase `q`, lower the gate, select the
nonzero solver seeds, remove width-one maps, or relabel the matched control as
the structured task.

A future attempt requires a new experiment identity and must first establish,
without controller outcomes, whether each map contains a small mandatory cut
that can concentrate many OD paths into one component. Load should be defined
relative to cut capacity, not global agent count, and the reset-only gate must
remain. If standard static Warehouse initialization continues to absorb these
flows, the scientifically cleaner target is a separately labelled warm-start
Warehouse disruption/recovery benchmark with controller-independent frozen
conflict checkpoints, not another endpoint-only cold-start TTF claim.
