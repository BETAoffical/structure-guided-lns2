# Frozen V1 MovingAI OOD Closed-Loop Confirmation

This preregistered stage tests whether the frozen `realized_dynamic` InitLNS controller transfers,
without retraining, from the structured development maps to standard MovingAI layouts. It does not add
static map, OD, or density context; it does not train a repair-order selector or RL policy.

## Frozen Inputs

The tracked portable v1 bundle, development index, native model, feature schema, and feature ranges keep
their previously registered SHA256 values. Online traces report feature-range violations, but OOD labels
cannot change the model or action. Official Adaptive and fixed Target, Collision, and Random use the same
PP+SIPPS solver and differ only in the InitLNS destroy strategy.

## Dataset

Twelve official MovingAI maps are fixed before qualification: three Random, three Maze, two Room, two
Warehouse, and two game maps. Each uses random scenarios 4 and 5, two registered agent counts, and solver
seeds `[1,2,3]`, producing 48 tasks and 144 initial episodes. Archive and extracted-file SHA256 values are
verified before preparation. These map IDs do not overlap the six MovingAI mechanism-probe maps.

Qualification requires 144 valid resets, at least 72 nonzero-conflict episodes, at least 8 active maps,
and at least one active map in each of the five layout families. Zero-conflict tasks remain in end-to-end
success statistics but do not enter conditional policy comparisons. A failed qualification is reported as
insufficient evidence; tasks, scenarios, and densities are not replaced.

## Registered Gate

Frozen v1 must preserve Adaptive's total success count, improve fixed 100-step conflict AUC by at least
5%, have a non-negative lower bound from 5,000 map-paired bootstrap samples, be no worse on at least 8/12
maps and 4/5 layout families, and produce no invalid action, initial-fingerprint mismatch, model error, or
unexplained episode failure. Wall-clock time, generated nodes, controller overhead, and feature-range
violations are reported but do not control acceptance.

Passing establishes dynamic realized-neighborhood generalization across these standard layouts, not the
original static-context transfer claim. Failure ends the cross-layout claim and triggers result
consolidation rather than another model or RL run.

## Commands

Three independent version labels are used and must not be mixed:

- `delta-gzip-v2` is the trace **storage** format.
- `feature-v2` is the fixed 124-feature training schema (82 proposal-only features). The deployed
  frozen ranker evaluates only the 86 canonical inputs referenced by its tree nodes.
- Controller bundle v3 packages the unchanged frozen v1 ranking semantics with feature-v2 execution,
  native feature extraction, promotion evidence, and an optional learned pruner.

The learned `proposal_pruner_v2` implementation and locked calibration report are retained, but the
pruner is not enabled: no threshold simultaneously preserved at least 99% of full-ranker winners and
reduced candidates by at least 15%. The promoted default is therefore `v2-full`: exact feature/model
acceleration with the complete candidate pool. This does not create a second scientific model.

```powershell
python scripts/fetch_movingai_devset.py --config configs/movingai_ood_devset.json `
  --output build/movingai-ood-dev
python scripts/prepare_movingai_probe.py --dataset build/movingai-ood-dev `
  --config configs/movingai_ood_dataset.json `
  --output build/initlns-movingai-ood-dataset-v1
```

Native collection runs in WSL with `PYTHONPATH=build/linux/project`:

```bash
python3 scripts/collect_closed_loop_confirmation.py \
  --dataset build/initlns-movingai-ood-dataset-v1 \
  --config configs/movingai_ood_collection.json \
  --output build/initlns-movingai-ood-collection-v2 --phase qualify \
  --trace-format delta-gzip-v2
```

`delta-gzip-v2` is the default. It stores each unique initial deterministic state once under
`state_blobs/`, records exact per-transition state deltas, and gzip-compresses each episode. The
validator reconstructs every state and recomputes the registered before/after/final fingerprints.
Use `--trace-format full-v1` only for storage diagnostics or legacy compatibility.

The existing 15 GiB full-v1 collection can be migrated without rerunning the solver or model. The
command also runs five-policy scientific equivalence and requires at least 90% storage reduction:

```bash
python3 scripts/convert_closed_loop_traces.py \
  --source build/initlns-movingai-ood-collection-v1 \
  --output build/initlns-movingai-ood-collection-v2-compact
```

Conversion never deletes the source. Conversion status is written to `conversion_progress.json`; the
final storage and equivalence records are `storage_conversion_report.json` and
`equivalence_report.json`. Only the legacy `episodes/` directory becomes eligible for a separate,
explicitly approved cleanup after the compact evidence and a fresh quick run pass. V1 models, manifests,
run metadata, and reports remain.

Policy execution is allowed only after qualification passes. Analyze the complete five-policy run with:

```powershell
python scripts/analyze_movingai_ood_confirmation.py `
  --collection build/initlns-movingai-ood-collection-v2-compact `
  --output build/initlns-movingai-ood-report-v2-compact
```

For a VS Code WSL terminal, the recommended end-to-end smoke command is:

```bash
cd "/mnt/c/Users/18448/Documents/lns2 2/structure-guided-lns2"
python3 scripts/run_final_model_evaluation.py --mode quick \
  --output build/initlns-final-model-quick-native-v2
```

It builds and tests the native module, verifies the frozen bundle and dataset, runs 75 episodes
(five registered tasks, three seeds, five policies), validates all compact traces, and emits CSV,
Markdown, and SVG results under `build/initlns-final-model-quick-native-v2/report`. The report includes
candidate counts, feature-stage timing, fallback/OOD rates, a paired fixed-suite feature speedup, and
clearly labelled controller/end-to-end timing projections. It is explicitly marked as non-formal.
For every learned quick decision, the same candidate state is also evaluated by the v1 reference
features and ranker; all scores (within `1e-12`), complete rankings, and selected candidates must match.
This shadow comparison does not run a second solver episode and therefore keeps the quick set at 75.

For diagnostics, force a controller or feature backend with:

```bash
python3 scripts/run_final_model_evaluation.py --mode quick \
  --controller v1-full --feature-backend python \
  --output build/initlns-final-model-quick-v1-diagnostic
python3 scripts/run_final_model_evaluation.py --mode quick \
  --controller v2-full --feature-backend native \
  --output build/initlns-final-model-quick-v2-native-diagnostic
```

`v2-cascade` is intentionally unavailable until a pruner passes locked validation. After the compact
equivalence audit and the default quick run both pass, run the registered 720-episode study with:

```bash
python3 scripts/run_final_model_evaluation.py --mode formal
```

Both modes update `run.log`, `status.json`, and `collection_progress.json` in the run directory. The
runner mirrors phase, completed-job, and error counts to the VS Code terminal and `run.log`; the JSON
files can also be monitored independently of the terminal. A non-empty output is immutable unless
`--resume` is explicit and both `runner_config.json` and the collection fingerprint match. If a run is
interrupted, invoke the identical command again with `--resume`; mismatched or legacy output is rejected
before its log or status is touched.

The registered native benchmark covers all five layout families plus maze400/600, room600, and
random600. Feature extraction was numerically equivalent within `1.14e-13`, reduced median feature
time by 90.88% overall (10.97x), and by 93.25% on maze600. Replacing the feature share in the recorded
v1 timing decomposition projects a 2.73x controller speedup and 1.43x controller-plus-repair speedup;
those two values are estimates until the quick/formal wall-clock run records observed evidence.

## Default LNS2/v2 dual-track bottleneck evaluation

The default evaluation now compares only original `official_adaptive` and `v2-full`. The historical
track preserves the registered 100-repair/300-second protocol. The wall-clock track disables both the
native and Python repair-count limits and runs each controller until it reaches a globally conflict-free
state or the 300-second deadline. Any task/seed that either controller does not solve is rerun for both
controllers from the same seed under an uncapped 600-second diagnostic budget.

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick \
  --evaluation-tracks historical,wall-clock \
  --controllers official_adaptive,v2-full \
  --wall-clock-seconds 300 \
  --wall-clock-sensitivity-seconds 600 \
  --feature-backend native \
  --controller-runtime optimized \
  --verification-profile deployment \
  --output build/initlns-lns2-bottleneck-quick-v2-exact
```

`audit` runs the compact proposal and dense feature paths with a full reference shadow comparison on
every repair. `deployment` validates the read-only solver revision on every repair and performs a full
state check every 20 repairs; final-state fingerprint validation remains mandatory. Both profiles keep
the same candidate pool, feature order, model scores, ranking, action, and PP implementation.

Before a long quick run, the exact runtime benchmark checks the frozen semantics and speed targets on
the registered representative tasks:

```bash
python3 scripts/benchmark_exact_runtime.py
```

The bottleneck report deliberately keeps two AUC definitions separate. The historical track reports
both raw fixed-100-repair AUC and initial-conflict-normalized fixed-100-repair AUC; these measure the
quality of the repair sequence. The wall-clock track reports normalized AUC integrated over real time;
this includes candidate selection, feature construction, PP, state export, and orchestration, and is the
metric for practical speed. A policy can improve fixed-step AUC while losing wall-clock AUC if its better
repair choices cost too much per iteration.

Each repair records candidate generation, feature stages, ranking, native neighborhood generation, PP,
repair bookkeeping, C++ state snapshots, Python export, orchestration, gzip trace writing, and complete
loop time. State-fingerprint time is also reported as a child of orchestration (and is not double-counted
in the additive total). Trace validation, atomic rename, hashing, and other episode-finalization costs are recorded
separately so they cannot be mistaken for model or PP time. The output report contains
`iteration_timings.csv`, `episode_timing_breakdown.csv`, `paired_bottleneck_decomposition.csv`,
`timing_summary.csv`, `neighborhood_pp_summary.csv`, `wall_clock_sensitivity.csv`, four SVG diagnostics, and
`v2_bottleneck_report.md`. It reports overall, map, layout-family, agent-count, and paired common-success
groups. Quick runs 8 tasks x 3 seeds x 2 controllers on each primary track. Timing is single-worker and
controller order is hash-rotated per task/seed.

Run formal only after the dual-track quick status and compact-storage audit pass:

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py --mode formal
```

`v1-full` and `v2-balanced` remain available solely through the archived protocol below; they are no
longer part of routine quick/formal runs or model training.

## Legacy four-way trade-off and balanced routing

The original solver is measured as the `official_adaptive` policy. `v1-full` is the old implementation:
at every repair it generates candidates, computes the old features, and uses the frozen ranker to select
the neighborhood. `v2-full` preserves that same every-repair model semantics while changing only the
feature schema, cache, compact model representation, and feature backend. LNS2/PP remains the low-level
path repairer in both full-model modes; neither mode hands neighborhood selection back to Adaptive after
the first repair. `v2-balanced` is a separate hybrid: low-conflict states take Adaptive without model
features, and the remaining states use the model. Storage, feature, bundle, and routing versions remain
independent.

Calibrate the routing threshold only from complete `policy_train` episodes, using four
layout-balanced map-group folds. The six registered conflict thresholds are run as separate collections
with the same 100-repair/300-second budget. The single selected threshold is then run once on the locked
`policy_validation` split. Horizon-4 state counterfactuals are not used for selection. The proposal
pruner remains disabled because its locked validation did not pass.

```bash
python3 scripts/calibrate_lns2_speed_quality.py
```

Calibration writes `balanced_controller.json`, `calibration_grid.csv`, and
`calibration_report.json` under `build/initlns-lns2-speed-quality-calibration`. It never changes the
bundle default. Because calibration now uses complete episodes, it schedules 1,152 training episodes
(144 each for LNS2, v2-full, and six thresholds) and 216 locked-validation episodes. It can be resumed
with `--resume`. Run the paired quick study only after that file exists:

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode quick --legacy-four-way --counterfactual-routes skipped \
  --timeout-sensitivity-seconds 600 \
  --output build/initlns-lns2-tradeoff-quick-native-v2
```

Quick uses five representative layout tasks plus maze600, room600, and random600, with three seeds. The
four complete collections are isolated under `collections/official_adaptive`, `collections/v1-full`,
`collections/v2-full`, and `collections/v2-balanced`. This is 96 complete quick episodes. For each
task/seed, their execution order is hash-rotated and the primary timing uses one worker. Every balanced
repair records the selected route, route switches, candidate counts, controller time, repair time, and
total decision time.

The primary comparison is always a complete episode. The report separately compares v2-full with
v1-full, v2-full with original LNS2, and v2-balanced with both. The v1/v2 traces are checked on their
entire common repair prefix: candidate pool, scores within `1e-12`, ranking, selected action, and random
seed must have zero mismatches, including the final budget-truncated decision. After-state fingerprints
and low-level counters are compared only when both sides completed that repair. An exclusion or length
difference is legal only when the shorter trace ends with a terminal `truncated=true` event, its episode
reports `external_timeout=true`, and elapsed wall time reaches the registered 300-second budget.

The 300-second result remains the only primary result. With
`--timeout-sensitivity-seconds 600`, every task-seed where any primary controller timed out is rerun for
all four controllers in `timeout-sensitivity-600/`. The repair limit remains 100, the native environment
and outer wall budget become 600 seconds, and the process guard becomes 660 seconds. This supplemental
report records new successes and ranking changes but never modifies primary metrics or promotion gates.

The supplemental counterfactual covers only states where v2-balanced actually chose Adaptive. It reuses
the LNS2 one-repair result already present in the main trajectory, replays the same prefix, and executes
the model exactly once. It does not continue for three more steps, does not run an extra LNS2 branch, and
does not cover states where the model was already used. Each state has an atomic resume checkpoint.

The report directory contains `paired_episodes.csv`, `controller_speed_comparison.csv`,
`v1_v2_semantic_equivalence.json`, `route_usage.csv`, `skipped_model_once.csv`,
`quality_speed_frontier.csv`, SVG plots, and `hybrid_necessity_report.md`. Quick is non-formal and cannot
change the default. Once quick, storage equivalence, common-prefix equivalence, and skipped-state replay
coverage all pass, run:

```bash
python3 scripts/run_lns2_tradeoff_evaluation.py \
  --mode formal --legacy-four-way --counterfactual-routes skipped
```

Formal covers 48 tasks, three seeds, and four controllers: 576 complete episodes. The default remains
`v2-full` unless v2 is semantically equivalent to v1 with significantly lower controller time and every
registered balanced quality, speed, SOC, integrity, and skipped-state coverage gate passes. The final
decision is one of `hybrid_supported`, `full_model_preferred`, `lns2_preferred`, or
`inconclusive_keep_v2_full`.

## Formal Result

The study was preregistered and pushed at commit `606374a` before qualification or policy outcomes were
read. All 12 archives and extracted maps/scenarios matched their registered SHA256 values. Qualification
passed with 144/144 valid resets, 74 nonzero-conflict episodes, nine active maps, and at least one active
map in every registered layout family. All five policies then completed 144 episodes with zero process
timeout, invalid action, initial-fingerprint mismatch, model error, or unexplained failure. Registered
300-second wall-budget truncations remain represented by capped failure metrics rather than process
errors.

| Policy | Successes | Mean fixed 100-step AUC |
| --- | ---: | ---: |
| Adaptive | 123/144 | 65,981.05 |
| Fixed Target | 122/144 | 66,027.09 |
| Fixed Collision | 126/144 | 67,155.55 |
| Fixed Random | 114/144 | 68,541.21 |
| Frozen `realized_dynamic` v1 | 131/144 | 63,272.41 |

Frozen v1 improved aggregate fixed-budget AUC by `4.105%`, just below the preregistered `5%` gate.
The map-paired bootstrap interval was positive at `[2.15%, 66.86%]`; all 9/9 active maps and all 5/5
layout families were no worse than Adaptive. Family improvements were 49.4% on Game, 2.3% on Maze,
51.5% on Random, 64.9% on Room, and 93.3% on Warehouse. The initial-conflict-normalized sensitivity
improved by 32.4%, and success increased by eight episodes. Mean capped wall time fell from 88.81s to
62.18s, but its bootstrap interval crossed zero and wall time remained diagnostic only. About 19.6% of
selected v1 features were outside the development range.

The strict decision is `stop_cross_layout_claim_and_consolidate_results` because every gate must pass and
the primary aggregate AUC improvement missed 5%. The result is therefore strong, broad OOD evidence but
not a confirmed preregistered cross-layout claim. It does not restore the static-context migration claim
and does not authorize RL or another tuned model. The confirmed headline remains same-family, multi-seed
generalization; the MovingAI result is reported as a near-threshold external-layout result.

The original full-v1 traces occupy about 15.13 GiB because every transition stores all paths and the
complete observation. They remain the immutable migration reference until compact conversion and
equivalence pass. The formal JSON SHA256 is
`e931721f0cdc08df6eaf9de75843e0a58c86e19f009195e675f3b358c156b46e`.

The verified delta-gzip-v2 migration contains all 720 episodes and 144 deduplicated initial-state
blobs. Traces plus state blobs occupy 50,329,794 bytes, a 99.6901% reduction from 16,242,547,808 bytes.
All five policy manifests, all 720 episode signatures, and every registered transition field match
full-v1 exactly. Reanalysis produces the same formal JSON SHA256 shown above.
