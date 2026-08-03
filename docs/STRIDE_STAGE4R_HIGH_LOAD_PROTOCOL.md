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

## Completed result

The run completed all 32 registered schedule entries: 16 episodes per
controller, with zero execution errors. Every integrity gate passed, including
complete paired coverage, identical initial fingerprints and conflict counts,
the registered reset-inclusive TTF clock, zero invalid actions, and zero
semantic mismatches. Both controllers solved all 16 episodes.

| Controller | Success | Mean capped TTF (s) | Median TTF (s) | Mean repairs | Mean no-progress | Mean wall AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2-full` | 16/16 | 15.2026 | 15.1303 | 21.125 | 4.812 | 0.050147 |
| `stride-quality-v1` | 16/16 | 15.0755 | 14.8611 | 18.938 | 3.188 | 0.049812 |

The quality controller improved aggregate mean capped TTF by only `0.8366%`
(`0.1272` seconds). It used `2.1875` fewer repair rounds and `1.625` fewer
no-progress transitions on average, but was faster in exactly 8/16 pairs and
slower in 8/16. This mixed direction and small aggregate margin do not support
promotion or a general speed claim.

The map families behaved in opposite directions:

| Cohort | V2 mean TTF (s) | Quality mean TTF (s) | Quality TTF change | Faster pairs | Mean repair delta | Mean no-progress delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `maze300` | 14.4699 | 15.9352 | `-10.1265%` | 2/8 | +0.500 | +0.375 |
| `room500` | 15.9354 | 14.2157 | `+10.7914%` | 6/8 | -4.875 | -3.625 |

Here a negative change means that quality was slower. The room cohort is the
more relevant high-conflict result: its two tasks began with mean initial
conflict counts of `346` and `325.75`. Quality reduced mean repairs from
`37.5` to `32.625` and mean no-progress transitions from `8.375` to `4.75`.
This supports the user's intended mechanism--better neighborhoods can reduce
rounds and TTF without changing the PP implementation--on this development
room cohort.

The reduced 300-agent maze cohort did not reproduce a high-conflict regime.
Its two tasks began with only `6.5` and `12` mean initial conflicts and needed
about five repair rounds. It therefore tests large-map/long-path transfer more
than high-load repair. Quality was slower on this cohort, consistent with a
remaining topology-transfer weakness.

Neither controller triggered a pruner or OOD fallback. Mean selected-feature
outside-range fraction was `0.3238` for V2 and `0.0322` for quality overall.
For quality it was `0.1716` on maze and only `0.0098` on room. The opposing map
results therefore cannot be dismissed as safe-route fallback; they instead
motivate better coverage of maze-like states and other topology families.

The registered automated decision is
`retain_quality_candidate_and_expand_development_evidence`. This retains the
candidate only as diagnostic evidence. The next training design should still
use paired multi-PP-seed aggregate labels and action uncertainty, while adding
outcome-blind, topology-balanced high-conflict states. More maps alone are not
sufficient: the collection must cover the intended conflict-load and residual
state regimes. A later maze check should select an intermediate agent count or
scenario from initial-conflict preflight, without using relative controller
outcomes.

Reproducibility:

- Registration SHA-256:
  `0e6bdc1263e6362c951620291e0cfe3e0cbeda736f398a99ccb89c3bbe63f96a`
- Runtime config SHA-256:
  `68da53b05786431ea757a0ee5a0c7beaf7fe4b6e9258abef74d8b1884df5c8a1`
- Maze/room dataset manifest SHA-256 values:
  `7158660c2e206716bf4f3a1d44d00a41a754d96307e980dc51b1c48431d362b7`
  and `1f4532fc8195b7242e89c11d312e97ee669c4550ec55dfe443d28eab20c7e835`
- Execution schedule SHA-256:
  `9c849b6cf2c292386990587a0c5361a0ee5d8d17e98b533b2447355831eda730`
- Maze V2/quality manifest SHA-256 values:
  `3b1ed601d7bb7b3e3bc4c08a69245393d153793e4ad81448beac81c3c00bf4e7`
  and `0d41c4e15ee9863e4e59e9bc3029ed1de9019233c3c77979b05018b35d06a6c1`
- Room V2/quality manifest SHA-256 values:
  `b90fcfa96c4bdb6b16e0d2f13175370f2fa4ace558124574d4e79245f636fe29`
  and `29ff0e01493f718510e80bfa8fefe57fa2a6d39f4c20926f8b8dfbf3c3fb98ed`
- Analysis report SHA-256:
  `49a79c7bf21d5485fae85394a0816d61ab9414fde5fc4e6930cec54c23b7b8de`
