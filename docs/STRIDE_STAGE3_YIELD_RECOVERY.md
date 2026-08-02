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

## Decision boundary

If qualification fails, policy episodes and PP label trials remain disabled and
the recovery design must be revised in a separately recorded protocol. If it
passes, the two source roots are combined with `select-states-v2 --split auto`.
The final Stage 3 gates remain exactly 600 states total, 300 per source policy,
complete trial indices 0 through 7, zero collection errors, and zero formal-OOD
map overlap.
