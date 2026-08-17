# Warehouse Compact-Cut V1 Result

## Decision

The cold-start compact-cut Warehouse line stops at Q1. Q0 passed, and all 128
registered reset jobs executed successfully, but only 1 of 64 structured reset
states passed the preregistered `16 conflicts / 32 active conflict agents / 16
largest-component agents` state-supply gate.

The registered decision is:

`stop_do_not_resample_replace_or_run_controller`

No Fixed16 family screen, size study, router experiment, Official Adaptive
comparison, or TTF experiment was run. This result is not evidence that V2,
StructShell, or Official is faster or slower. V2 remains the default.

## Registered identity and integrity

- Source commit: `6411f87171b7`.
- Output root: `build/wh-compactcut-v1-r2`.
- Frozen config SHA-256:
  `c6680749f9acc06caa3b03d83178c1369781630c6a0c38cb3b5d1d9e6ed57621`.
- Run fingerprint:
  `771bfda8f72079809831e68eebefcb72875073cc09e598a12bd8c8739d18e1a4`.
- Runtime-preflight fingerprint:
  `46e462086bb92964019476c6fb512411280d8d509c585bd5d31ee7d87293bc81`.
- Schedule semantic fingerprint:
  `22627021ed044da79ba980bbaad38ee25d93765a3bb98136706d674279009000`.
- Q0 manifest SHA-256:
  `70d0c549e22a3d71b4ffb64e593b918e4d4182a445637d2625298c55bc3dffc7`.
- Q1 manifest SHA-256:
  `b62f8713a9614ea2d3ac72acb35d39614d795b34ca2505590aa14ca82bdc6cea`.
- Qualification report SHA-256:
  `a5a47d223591185b64ebfba8a853a8f08bcf3d7bd3e1343638623f1abd3a0306`.
- Benchmark-selection SHA-256:
  `54684a41d04e581e49d5dc55cefa12b9745f57e8f8bb5335d529d3431487d616`.

The 128 schedule keys and 128 manifest keys are unique and identical. All 128
rows have status `ok`, complete initial paths, and unique valid state
fingerprints. There were zero execution errors, process timeouts, unexpected
keys, task/hash mismatches, or native identity mismatches.

The runner used `max_decisions=0` and only called the native reset path.
`controller_step_invoked`, `policy_episode_invoked`, and `formal_ttf_invoked`
are all false.

## Q0 result

Q0 passed for all eight registered 28 x 39 maps and all 32 tasks. The map
grids are unique, every registered cut has capacity two and no bypass, and each
structured task sends exactly 30 agents in each direction through each gate.
All 16 task pairs passed the deterministic joint-feasibility witness and the
5% mean / 10% P95 shortest-distance matching gates.

This establishes geometric and task feasibility. It does not guarantee that
the cold-start InitLNS reset leaves a complex repair state.

## Q1 state-supply result

All 128 resets completed: 64 structured and 64 diagnostic-control resets.

Structured results:

- 55/64 were initially infeasible; 9/64 were already feasible.
- Initial conflicts: mean `8.453125`, median `6`, maximum `63`.
- Active conflict agents: mean `9.546875`, median `8`, maximum `37`.
- Largest conflict component: mean `5.171875`, median `4`, maximum `22`.
- `conflicts >= 16`: 10/64.
- `active conflict agents >= 32`: 2/64.
- `largest component >= 16`: 3/64.
- All three thresholds and initial infeasibility: 1/64.

The only passing key was
`whcc_cfg_04__bmcx__t0521__n0120`, solver seed `73`, with `63` conflicts,
`32` active conflict agents, and a largest component of `22`. Its state
fingerprint is
`81186f0ff126082946b44af1677f6dbe93191a13e971a17ef72fb32ecbc38bd3`.
It belongs to the controller-held-out split and must not be selected for
post-hoc development.

| Map | Registered role | Structured pass | Mean conflicts / active / component |
|---|---|---:|---:|
| `whcc_cfg_01` | development | 0/8 | 7.000 / 9.125 / 5.750 |
| `whcc_cfg_02` | controller-held-out | 0/8 | 8.500 / 8.875 / 4.000 |
| `whcc_cfg_03` | development | 0/8 | 10.875 / 12.250 / 6.000 |
| `whcc_cfg_04` | controller-held-out | 1/8 | 17.000 / 14.500 / 7.750 |
| `whcc_dh_01` | development | 0/8 | 5.625 / 7.625 / 4.500 |
| `whcc_dh_02` | controller-held-out | 0/8 | 4.875 / 5.250 / 4.125 |
| `whcc_dh_03` | development | 0/8 | 6.000 / 9.125 / 4.000 |
| `whcc_dh_04` | controller-held-out | 0/8 | 7.750 / 9.625 / 5.250 |

`cross_four_gate` passed 1/32; `double_horizontal` passed 0/32.
Development maps passed 0/32; controller-held-out maps passed 1/32.

The diagnostic control completed 64/64 and is report-only by protocol. Forty
of 64 control resets were already feasible; its mean conflict count was
`1.796875`. It cannot compensate for failure of the structured hard gate.

## Interpretation and stopping boundary

The failure is a state-supply failure, not a controller or runtime failure.
Even with mandatory capacity-two cuts and balanced bidirectional demand,
cold-start InitLNS usually resolves the intended coupling before a repair
controller can act. Running family or size comparisons on this cohort would
mostly benchmark simple or already feasible states rather than the registered
complex-repair question.

Compact-cut V1 must not be continued by increasing the agent count, replacing
maps or seeds, lowering thresholds, or selecting the single passing held-out
state. The development, size, router, and final-confirmation stages are
therefore intentionally not implemented or executed under this identity.

## Next research boundary

The next eligible line is a separate warm-start disruption/recovery checkpoint
experiment. It requires a new experiment ID, output root, task/solver seed
namespace, and protocol. Before any controller result is observed, each
checkpoint must freeze the full paths, task, conflict graph, state fingerprint,
and file hashes, and every structured checkpoint must independently pass the
same `16/32/16` qualification gate.

Any future timing claim must define TTF from checkpoint restoration and charge
all arms identically. Because restoring a native repair state resets ALNS
weights, random streams, counters, and time budgets, the claim boundary is
"recovery from a frozen disturbed path state", not exact continuation of the
original run and not a complete lifelong Warehouse system.

## Evidence

- `build/wh-compactcut-v1-r2/qualification/qualification_status.json`
- `build/wh-compactcut-v1-r2/qualification/qualification_manifest.jsonl`
- `build/wh-compactcut-v1-r2/qualification/qualification_report.json`
- `build/wh-compactcut-v1-r2/qualification/benchmark_selection.json`

The full Q0/Q1 output was archived as
`backups/wh-compactcut-v1-r2-q0-q1-20260817.tar.gz`, SHA-256
`2389ea3c233f4cf26e412722d81a27a4b3132bbf42ba492b731b32dee8b1a440`.
