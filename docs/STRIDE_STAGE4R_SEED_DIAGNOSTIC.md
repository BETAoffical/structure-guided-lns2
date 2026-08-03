# STRIDE Stage 4R multi-seed tail diagnostic

## Purpose and evidence boundary

The first Stage 4R Quick used one solver seed on eleven map-disjoint tasks. It
showed that both STRIDE candidates were slower than `v2-full` in mean capped
wall time to feasible, with most of the loss concentrated on
`lt_gallowstemplar_n` and `maze-128-128-2`. A one-seed result cannot distinguish
a persistent selector weakness from an unlucky initial-PP and downstream-PP
random stream.

This follow-up is therefore an **outcome-informed diagnostic**, not a formal
benchmark. Its four tasks were chosen after reading the Quick-v2 seed-1
outcomes. Seeds 2--4 were registered before they were executed. The report must
set `formal_speed_claim` to false and must never promote a controller.

## Registered cohort

- Controllers: `v2-full`, `stride-control-v1`, `stride-quality-v1`.
- Solver seeds: 1, 2, 3, 4.
- Tasks: two generated contrast tasks and two 400-agent MovingAI problem tasks.
- Schedule: 4 tasks x 4 seeds x 3 controllers = 48 serial episodes in strict
  rotating Latin order.
- Budget: 120 seconds of reset-inclusive wall time per episode; 180-second
  external process timeout.
- Primary diagnostic metric: mean capped wall time to feasible (TTF).
- Supporting metrics: success count, common-success TTF, repair rounds,
  no-progress transitions, wall-conflict AUC, PP time, and controller overhead.

Changing `solver_seed` changes both the initial PP solution and the subsequent
PP random stream. This experiment measures end-to-end seed robustness; it is
not a same-state repair-replay experiment.

## Registered decision rule

For each of the two MovingAI problem tasks, a quality episode is adverse when
it is slower than its paired V2 episode or uses more repair rounds. If at least
three of four seeds are adverse on **both** problem tasks, the result is treated
as persistent tail weakness and the next action is to revise the one-step label
toward residual-state hardness. If signs are mixed across seeds, the next action
is a same-state paired PP-repair replay before changing the label.

The rule is diagnostic. The cohort is small and outcome-selected, so even a
positive aggregate TTF result only authorizes a larger, outcome-blind
development cohort.

## Commands

Run in WSL with the native module available:

```bash
python3 scripts/run_stride_stage4r_seed_diagnostic.py prepare
python3 scripts/run_stride_stage4r_seed_diagnostic.py dry-run
python3 scripts/run_stride_stage4r_seed_diagnostic.py run
python3 scripts/run_stride_stage4r_seed_diagnostic.py analyze
```

The default output is `build/stride-stage4r-seed-diagnostic-v1`. Long-running
execution is monitored at 30-minute intervals rather than polled continuously.

## Completed result

The registered run completed all 48 schedule entries. Each controller produced
16/16 successful episodes with zero execution errors. All initial-state
fingerprints and conflict counts matched within each task/seed triple; the TTF
clock, action validity, and semantic-consistency gates also passed.

| Controller | Success | Mean capped TTF (s) | Median TTF (s) | TTF vs V2 | Mean repairs | Mean no-progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2-full` | 16/16 | **11.1493** | **3.3497** | reference | **11.000** | **2.938** |
| `stride-control-v1` | 16/16 | 13.1622 | 3.8572 | `-18.0540%` | 16.250 | 7.812 |
| `stride-quality-v1` | 16/16 | 13.0991 | 5.8473 | `-17.4884%` | 21.688 | 13.625 |

Negative improvement means slower TTF. Quality was faster than V2 on 4/16
paired episodes and slower on 12/16. Its mean repair count increased by 10.688
and its mean no-progress count increased by 10.688. The extra runtime is
therefore associated with more repair attempts and stalls, not a success-rate
tradeoff: every controller solved every episode.

The two registered MovingAI problem tasks had different seed behavior:

- `lt_gallowstemplar_n`: quality was slower on seeds 1--2 but faster on seeds
  3--4; only 2/4 seeds met the adverse rule. Mean TTF was 12.4781% slower.
- `maze-128-128-2`: quality was slower on three seeds and used more rounds on
  all four; 4/4 seeds met the adverse rule. Mean TTF was 14.1709% slower.

Because the persistent-tail rule required at least 3/4 adverse seeds on both
problem tasks, `intrinsic_tail_weakness` is false. The mixed signs on
`lt_gallowstemplar_n` make `seed_sensitive` true. This does **not** rescue the
quality model: V2 remains the observed best controller and neither STRIDE model
is promoted. It means that changing the label immediately would confound
selector weakness with PP randomness. The registered next step is a same-state,
same-neighborhood paired PP-seed replay before deciding how to revise the
one-step label.

This remains an outcome-informed four-task diagnostic and is not a formal speed
claim.

## Completed-run hashes

- Registration: `d88f5ffd42e8c836a09b345ba5827f3baf32ab87732bbd92aaf238b726374d0f`
- Dataset manifest: `484f343cb91ae063410ae8c7a3f26c632fd187d4a7e32543e94ef7753c666774`
- Runtime config: `63ed052702bb56de31fb136c7349c64b5830cca15280f4678ddaf30dd80d54f7`
- Execution schedule: `f689a5aa0187960958735af96e292d44356bc8ce91b1b36da12da7d6ee2ee523`
- Qualification report: `59f7c7ee02bd0bffe02f85ba0d92eab0f6fdbd70f13470505a44dd8f10c8d109`
- V2/control/quality manifests:
  `7bf441ebe234cc8df6ad25cb672eab868b237c46d8d306acb5ab9f4492256dde`,
  `831d7a0fac2f8b8faf5dfbea12fc34bbd9d3fc163781229bbaa552c32ee8b8ff`,
  and `590cc3ea40ee98e17307980ec56eb52e76621fcae87d9a3e6ffaad01594c5122`.
- Analysis report: `a42df7995a7b2ab7781118f0f2396ee37f0c8a08e08ddd90d2d8ace0b04968cc`
