# STRIDE-LNS Stage 3 source-yield recovery

## Trigger

The registered ten-map Stage 3 source collection completed all 480 policy
episodes with zero errors and zero timeouts. Result-blind selection nevertheless
failed before any new PP label trials were run:

- only 43 of 240 task-seed resets had a nonzero initial conflict;
- the one-state-per-source-episode rule therefore allowed only 43 states per
  source policy instead of the registered 180;
- only two repairable resets were in the 100/200-agent band;
- the combined selection contained 86 states, with 82 high-agent and four
  low/mid-agent states.

This is a source-yield failure, not a selector, PP, or label result. The failed
selection remains under `build/stride-stage3-extension-selection-v1` and must
not be relabelled as a completed Stage 3 cohort. Its selection and report
SHA-256 values are respectively
`e91e15fbfeaa5d0ae19930ffdde6594c214884f9a13a7f182507931f37a50fac`
and `e0fb0842dd1046cd38522081470d94c4647a22b3e993866cbcd37c2c2039620f`.

## Frozen recovery design

The recovery adds a distinct compact-map source pool. It does not change the
124-dimensional state/candidate features, candidate generation, PP, the eight
paired PP seeds, or `lns2.stride.quality_label.v2`.

- master seed: `2026080204`;
- 12 new 40x48 maps: four regular beltway, four compartmentalized, and four
  dead-end-aisle layouts;
- eight tasks per map, emphasizing 100/200-agent congestion while retaining
  two 300-agent tasks;
- eight new solver seeds: 404, 505, 606, 707, 808, 909, 1001, and 1102;
- no map ID overlaps Stage 2, the first Stage 3 pool, or the 12 formal OOD maps.

Qualification is now an enforced development gate rather than the historical
"at least one conflict" check. Before policy episodes may run, the recovery
pool must contain at least 150 nonzero task-seed resets, including at least 60
low/mid-agent and 20 high-agent resets, at least 20 per layout family, at least
12 per solver seed, and all 12 maps active.

These thresholds provide a buffer over the exact deficit. Combined with the
first pool's 43 repairable resets, the result-blind selector must still produce
exactly 180 states per policy, cap every episode at one state, cover all
available maps, and satisfy both agent-band gates. Source qualification and
selection do not read PP repair outcomes.

## Executed qualification

The frozen recovery dataset generated 12 maps and 96 tasks successfully. The
dataset manifest SHA-256 is
`2302ca12a65f23af9a816da6bbfead731eef8e6aedc466893ffae8cb51a2efcf`;
its map IDs have zero overlap with the historical development maps and the 12
formal OOD maps.

All 768 task-seed resets completed with zero errors and zero timeouts. The
strict qualification passed every gate:

- 393 nonzero resets versus the required 150;
- 211 low/mid-agent and 182 high-agent resets versus the required 60 and 20;
- 209 compartmentalized, 101 dead-end-aisle, and 83 regular-beltway resets;
- 45--52 nonzero resets per solver seed versus the required 12;
- all 12 maps active, with distinct seed trajectories and consistent reset
  states.

The qualification manifest SHA-256 is
`0c2ee31c6972c37b8934a6cfa26f84b648b777d6564b42b866e1a117a1358502`;
the report SHA-256 is
`a184f354b9e9b9fb443a529346d411d6f258c95429894e6a31d8a9daa4ae691e`.
This authorizes recovery policy-episode collection, but it is not PP-label or
end-to-end solver evidence.

## Executed source collection and selection

Both recovery source policies completed all 768 registered episodes. Official
Adaptive solved 768/768; frozen V2 solved 764/768. The four unsolved frozen-V2
episodes are solver outcomes rather than collection errors: both policies had
zero errors and zero timeouts. The official and frozen-V2 collection-manifest
SHA-256 values are respectively
`485df61c6cd6cf1ce64adc348a6a1bf2907bfe5f19195f0f6fe18a941378762d`
and `3357861ef19de3adfa75f8e376d33b8292e1e739d41785030dae8365ffffd838`;
the joint collection-summary SHA-256 is
`6ef5b7a889ceb5933b4d8486b39a25aa74b6e7772b871235d6d9a37918b68e94`.

Result-blind selection then combined the first Stage 3 source pool with the
recovery pool and passed every registered extension gate:

- exactly 360 states, 180 per source policy;
- 360 distinct source episodes, preserving the one-state-per-episode cap;
- all 22 extension-source maps represented;
- 168 low/mid-agent and 192 high-agent states;
- exactly 60 low-, 60 medium-, and 60 high-conflict states per policy;
- official Adaptive contributed 90 low/mid and 90 high-agent states; frozen V2
  contributed 78 low/mid and 102 high-agent states.

The state-selection SHA-256 is
`6114d521b862b7c7af54f5b0c6da3cb6e2fbea52aec17455fa6a71eeae232be9`;
the selection-report SHA-256 is
`06ce74120e295e6e87e71dba9380022a338317d6dba4521ec98dea5270262ea6`.
Neither selection nor its gates read PP repair outcomes.

## Executed replay preflight

All 360 selected states passed three independent replay-and-candidate
preflights. Every state succeeded in all three repetitions, every state had one
candidate signature across repetitions, candidate counts remained between 16
and 18, and there were zero failed states. The preflight-report SHA-256 is
`ef5a889a0e41ce1d6f616a91eb12fedd7e3293fb37bd3c986902967b92a05e0d`.

This preflight establishes deterministic state reconstruction and candidate
generation. It does not establish label stability, selector quality, or
end-to-end runtime improvement. It authorizes the registered eight-seed PP
repair trials for the 360-state extension.

## Executed base PP repair trials

Trial indices 0 through 3 completed for all 360 extension states with zero
collection errors. The collection contains 6,442 state-candidate pairs and
25,768 trial rows: 334 states have 18 candidates, 14 have 17, and 12 have 16.
Every state-candidate pair has exactly one row for each registered index, and
each state uses one paired PP seed per index across all its candidates.

All 25,768 rows retain the frozen 124-dimensional feature schema and complete
post-repair structure. Native outcomes include 3,620 feasible repairs, 14,249
conflict-reduced repairs, 4,685 state changes without conflict reduction, one
accepted no-op, and 3,213 hard failures. Hard failures are valid candidate
outcomes used by the label; they are not collection errors.

The collection-report SHA-256 is
`30f22e34f7e6fa28b0f0f57317ef46bd3d5b88edd5d34d69d412d7e2e6eb4f91`;
the base repair-trial JSONL SHA-256 is
`4e408c7246e493142110a177619193b24184c99ff33938e5403e8d64c1dec7fe`.
This authorizes collection of trial indices 4 through 7; it is not yet a
complete `stride-quality-v2` labelled extension.

## Decision boundary

If qualification fails, policy episodes and PP label trials remain disabled and
the recovery design must be revised in a separately recorded protocol. If it
passes, the two source roots are combined with `select-states-v2 --split auto`.
The final Stage 3 gates remain exactly 600 states total, 300 per source policy,
complete trial indices 0 through 7, zero collection errors, and zero formal-OOD
map overlap.
