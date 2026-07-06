# Stage 1 Configuration Reference

The active example is `configs/stage1_example.json`.

Numeric configuration values are fixed when written as one number. A
two-number list is sampled uniformly for every map or task. Categorical values
are fixed strings, uniformly sampled string lists, or weighted mixtures.

## Dataset parameters

| Parameter | Meaning |
| --- | --- |
| `master_seed` | Root seed used to derive unique map and task seeds |
| `output_dir` | Default output directory |
| `tasks_per_map` | Number of independent task sets generated per map |
| `task_variants` | Ordered named task overrides; length must equal `tasks_per_map` |
| `task_scenarios` | Ordered scenario list; length must equal `tasks_per_map` |
| `layout_variants` | Named per-layout map overrides distributed evenly over the full dataset |
| `splits.<name>.maps` | Number of maps in a split |
| `splits.<name>.layout_counts` | Exact map count per layout; replaces random layout sampling |
| `splits.<name>.map` | Map-parameter overrides for one split |
| `splits.<name>.task` | Task-parameter overrides for one split |

When both `maps` and `layout_counts` are supplied, `maps` must equal the sum of
the layout counts. `task_variants` takes precedence over the compatibility
`task_scenarios` list. A layout's total map count must be divisible by its
number of configured variants.

## Map family and geometry

| Parameter | Meaning |
| --- | --- |
| `layout_mode` | One fixed family, a list sampled uniformly, or `mixed` |
| `layout_mixture` | Weights used when `layout_mode` is `mixed` |
| `rows`, `cols` | Grid dimensions |
| `shelf_block_height`, `shelf_block_width` | Base shelf-block dimensions |
| `horizontal_aisle_width`, `vertical_aisle_width` | Base cross/vertical aisle widths |
| `layout_jitter` | Probability of trimming one row/column from each shelf |
| `wall_clearance` | Free margin in addition to the beltway |

The shelf grid is centered inside the available warehouse interior. Any space
left after fitting complete shelf blocks is split between opposite sides
instead of accumulating at the right and bottom edges.

Supported layout families:

- `regular_beltway`: regular shelf grid with a complete outer route
- `partial_beltway`: continuous outer-route segments become narrow or blocked
- `wall_shelves`: selected boundary shelves extend to the map edge
- `dead_end_aisles`: selected horizontal and vertical aisles receive shelf-aligned caps
- `partial_cross_aisles`: shelf-aligned spans of cross aisles are closed
- `compartmentalized`: one of the configured two-wall/four-gate templates
- `mixed_width`: vertical aisle widths vary when the shelf grid is created
- `asymmetric`: shelf blocks on one side are systematically shorter
- `station_centric`: stations use clustered placement and queue regions

Only `regular_beltway`, `compartmentalized`, and `dead_end_aisles` are active
in the default feasibility configuration. The other families are retained for
compatibility and later experiments.

## Local structural complexity

| Parameter | Meaning |
| --- | --- |
| `combine_layout_features` | When true, probability-based local features may overlay the primary layout mode |
| `beltway_mode` | `full`, `partial`, or `none`; omitted values use the layout-family default |
| `outer_beltway_width` | Width of the semantic outer beltway |
| `beltway_narrow_segment_count` | Number of continuous reduced-width beltway segments |
| `beltway_blocked_segment_count` | Number of continuous fully blocked beltway segments |
| `beltway_segment_length` | Length of each affected beltway segment |
| `beltway_narrow_width` | Traversable width retained in a narrow segment |
| `beltway_affected_sides` | Eligible sides: top, bottom, left, right |
| `wall_shelf_extension_count` | Boundary shelf blocks extended to a wall in `wall_shelves` |
| `dead_end_aisle_count` | Number of shelf-aligned aisle caps in `dead_end_aisles` |
| `dead_end_orientation_counts` | Exact `vertical` and `horizontal` cap counts |
| `dead_end_depth` | Target depth recorded and used when positioning a cap |
| `cross_aisle_closure_count` | Number of cross-aisle closures |
| `cross_aisle_span_blocks` | Number of adjacent shelf blocks spanned by each closure |
| `variable_vertical_aisle_width` | Width range used by `mixed_width` while constructing the shelf grid |
| `gate_count`, `gate_width` | Gate count and width for a compartment divider |
| `divider_template` | `single`, `cross_four_gate`, `double_horizontal`, or `double_vertical` |
| `divider_orientation` | `vertical` or `horizontal` compartment divider |
| `divider_thickness` | Compartment divider thickness |
| `asymmetric_trim_width` | Columns removed from right-side shelf blocks |
| `buffer_zone_count` | Number of complete shelf blocks intentionally removed to create open buffers; the active example uses zero |

The legacy probability parameters remain available when
`combine_layout_features` is true. Every obstacle insertion is tentatively
applied and rolled back if it disconnects the traversable grid. A generated
map is rejected and deterministically resampled when its named primary feature
could not be created. Beltway narrowing blocks the wall-side lanes and keeps
the warehouse-side lane open. Dead-end caps must touch shelves on both sides,
and cross-aisle closures must connect shelf blocks above and below.

## Stations

| Parameter | Meaning |
| --- | --- |
| `station_count` | Number of station centers |
| `station_sides` | Candidate sides: left, right, top, bottom |
| `station_placement` | `distributed`, `clustered`, `single_side`, or `opposite_sides` |
| `station_cluster_count` | Number of station clusters |
| `station_clearance` | Minimum center spacing and local approach radius |
| `station_entrance_width` | Number of adjacent cells forming an entrance |
| `station_queue_depth`, `station_queue_width` | Shape of the station approach/queue area |
| `station_demand_distribution` | `uniform` or `zipf`; used when sampling station endpoints |

`opposite_sides` interleaves candidates from the configured sides. With two
stations and `["left", "right"]`, one station is guaranteed on each side.

## Topology constraints

`generation_attempts` limits deterministic resampling. Supported
`topology_constraints` keys:

- `minimum_articulation_count`
- `maximum_articulation_count`
- `minimum_dead_end_count`
- `maximum_dead_end_count`
- `minimum_average_degree`
- `maximum_average_degree`

Generated metadata records articulation cells, dead-end count, average degree,
and a route-redundancy proxy.

## Task volume and endpoint distance

| Parameter | Meaning |
| --- | --- |
| `agent_count` | Fixed or sampled number of agents |
| `agent_density` | Used only when `agent_count` is absent |
| `density_reference` | `free_cells`, `service_cells`, or `aisle_cells` |
| `minimum_shortest_distance` | Required endpoint distance |
| `maximum_shortest_distance` | Optional maximum endpoint distance |
| `max_sampling_attempts` | Attempts allowed per agent |

## Legacy per-agent flow

`flow_type` remains supported:

- `random`
- `storage_to_station`
- `station_to_storage`
- `one_way`
- `bidirectional`
- `hub_spoke`
- `mixed`, using `flow_mixture`

This mode is used when `scenario_type` is omitted.

## Scenario-level task flow

`scenario_type` is sampled once per task set. It can be one fixed value,
a uniformly sampled list, or `mixed` with `scenario_mixture`.

Supported scenarios:

- `uniform_random`
- `dominant_one_way`
- `balanced_bidirectional`
- `station_rush`
- `station_release`
- `cross_zone_exchange`
- `bottleneck_pressure`
- `dead_end_turnover`
- `intersection_crossing`
- `mixed_background`

The active feasibility configuration uses only `balanced_bidirectional` and
`uniform_random`. Balanced bidirectional alternates directions, producing
18/18, 30/30, or 24/24 assignments for the active 36-, 60-, and 48-Agent
variants.

| Parameter | Meaning |
| --- | --- |
| `dominant_flow_ratio` | Fraction following the scenario's principal flow |
| `background_flow` | Flow used by remaining agents |
| `primary_flow` | Principal flow for `mixed_background` |
| `opposing_flow_ratio` | Right-to-left share in opposing scenarios |
| `od_matrix` | Optional weighted `origin->destination` zone matrix; takes precedence over scenario flow |

OD zone names currently supported:

- `free`
- `storage`
- `station`
- `left`
- `center`
- `right`

## Hotspots and conflict pressure

| Parameter | Meaning |
| --- | --- |
| `origin_cluster_count`, `goal_cluster_count` | Number of endpoint hotspots |
| `cluster_radius` | Manhattan radius around hotspot centers |
| `hotspot_skew` | Probability of sampling from hotspot cells |
| `hotspot_distribution` | `uniform` or `zipf` |
| `required_bottleneck_crossing_ratio` | Fraction whose shortest path must pass a selected high-risk cell |
| `shared_corridor_ratio` | Additional shared-bottleneck pressure; currently enforced through the same bottleneck condition |
| `target_bottleneck_mode` | `articulation` or `highest_prior` |
| `swap_pair_ratio` | Fraction participating in goal-swap pairs |

## Fixed invariants

These are not configurable:

- obstacle cells are static;
- movement remains on an undirected four-neighbor grid;
- all traversable cells must be connected;
- starts are unique and goals are unique;
- each agent's start differs from its goal;
- endpoints must satisfy shortest-path bounds;
- semantic layers and structural priors align exactly with the grid.

True directed one-way roads, dynamic closures, release times, and lifelong task
arrival are intentionally not part of stage 1 because they require changes to
the instance format and low-level solver.
