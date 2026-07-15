# Structure-Guided MAPF-LNS2

Research infrastructure for context-aware neighborhood control during the collision-repair phase of
MAPF-LNS2. The active solver is now the complete official MAPF-LNS2 codebase, not the former
dependency-free approximation.

## Research target

The active claim is deliberately narrow:

> Learn a map-, static-OD-, density-, and conflict-conditioned InitLNS high-level policy that jointly
> selects seed agent, Target/Collision/Random generation, and neighborhood size to improve
> time-to-feasible and cross-distribution generalization.

Primary metrics are time-to-feasible, conflict-pair AUC, and success under a fixed time budget. Final
sum-of-costs is secondary. The first learning action space chooses a seed agent, destroy heuristic, and
neighborhood size. Explicit agent subsets are supported for later autoregressive policies, but are not
the starting point.

## Solver provenance

The complete [Jiaoyang-Li/MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2) `init-LNS` source is
vendored at commit `1369823985a15944f9a339226d521f61605a6d17`. It includes `InitLNS`, MAPF-LNS,
SIPPS, path/constraint tables, CBS-family repair, and PIBT-family initial solvers. See
[`third_party/mapf_lns2/UPSTREAM.md`](third_party/mapf_lns2/UPSTREAM.md) and the USC Research License.

Project changes are limited to a step-wise repair API, policy/observer hooks, low-level counters, and a
repair-only switch. With the same 200-agent benchmark, seed, and parameters, the extended `official`
mode produces byte-identical paths to the untouched upstream commit.

Official [GPBS](https://github.com/shchan13/GPBS) is vendored independently at commit
`43f2a6fea50893871219b674535f83920175ae04` as an end-to-end feasibility baseline. It is not an
LNS2 destroy heuristic. See [`third_party/gpbs/UPSTREAM.md`](third_party/gpbs/UPSTREAM.md).

## Build on Ubuntu/WSL

Required packages are CMake, a C++14 compiler, Boost program-options/system/filesystem, Eigen3,
Python development headers, and pybind11 development headers. Inspect installed packages before adding
anything; the current Ubuntu 22.04 environment already contains every required dependency.

From an Ubuntu shell in this repository:

```bash
cmake -S . -B build/linux/project -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux/project --parallel 4
ctest --test-dir build/linux/project --output-on-failure
```

Build targets:

- `lns_official`: complete MAPF-LNS2 CLI, including optional anytime optimization.
- `lns2_repair`: PP + InitLNS repair, stopping at the first feasible solution.
- `lns2_env`: Python step environment for future contextual RL.
- `gpbs_official`: pinned independent GPBS feasibility solver.

## Run repair-only LNS2

```bash
build/linux/project/lns2_repair \
  --map third_party/mapf_lns2/random-32-32-20.map \
  --agents third_party/mapf_lns2/random-32-32-20-random-1.scen \
  --agentNum 200 \
  --cutoffTime 60 \
  --neighborSize 8 \
  --initDestroyStrategy Adaptive \
  --replanAlgo PP \
  --seed 0 \
  --trace build/linux/repair-trace.jsonl \
  --outputPaths build/linux/repair-paths.txt
```

`Adaptive`, `Target`, `Collision`, and `Random` are available for the InitLNS destroy strategy. The
official CLI also keeps `RandomWalk`, `Intersection`, `Random`, and `Adaptive` for feasible-solution
MAPF-LNS optimization.

## Python environment

```bash
PYTHONPATH=build/linux/project python3 - <<'PY'
import lns2_env

env = lns2_env.LNS2RepairEnv(
    "third_party/mapf_lns2/random-32-32-20.map",
    "third_party/mapf_lns2/random-32-32-20-random-1.scen",
    agent_count=200,
    context={"layout_mode": "random", "task_flow": "benchmark"},
)
state = env.reset(seed=0)
result = env.step({
    "mode": "seed",
    "heuristic": "collision",
    "seed_agent": state["conflict_edges"][0][0],
    "neighborhood_size": 8,
})
print(result["metrics"])
PY
```

Supported modes are `official`, `seed`, and `explicit_neighborhood`. Invalid external actions fall back
to the official selector and are marked with `action_valid=false`. The binding returns raw transition
metrics; reward design stays in Python experiment code.

External actions may include `random_seed` for deterministic counterfactual branches. Leaving it absent
preserves the original official random stream.

## Structured warehouse data

The retained generator creates controlled warehouse layouts and static OD task variants. Every map/task pair
now includes standard MovingAI `.map/.scen`, JSON metadata, text/SVG previews, and the legacy `.mapf`
file for archived experiments.

```powershell
python scripts/generate_dataset.py `
  --config configs/stage1_example.json `
  --output build/feasibility-dataset

python scripts/inspect_dataset.py --dataset build/feasibility-dataset
python scripts/generate_gallery.py
```

Manifests expose `map_file`, `scenario_file`, `map_metadata_file`, and `task_file`. Coordinates are
converted from `(row, col)` to MovingAI `(x, y)`, and scenario rows contain exact single-agent shortest
distances.

## Repair experience

The transfer pilot contains 102 instances with separate ID, unseen-layout, unseen-task, unseen-density,
and joint-OOD evaluation splits. The CPU-only collector records four official repair baselines and
replays controlled seed/heuristic/size actions from exact Adaptive states.

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/repair-transfer-pilot-v2 \
  --config configs/repair_collection_pilot.json \
  --output build/repair-experience-pilot \
  --phase all --workers 4
```

The expanded Train/Validation calibration uses
`configs/repair_collection_calibration.json` and is audited with
`scripts/analyze_repair_experience.py`. It can produce up to 11,664 controlled
outcomes without using Test/OOD data as labels.

See [`docs/REPAIR_COLLECTION.md`](docs/REPAIR_COLLECTION.md) for split definitions, smoke overrides,
resume behavior, calibration commands, quality gates, and the versioned output contract.

## Context gate and standard baselines

The 7,344-outcome context audit is implemented by `scripts/run_context_audit.py`. The current Pilot v2
audit failed its predeclared offline gate: full context gained 4.17 Pareto-hit percentage points over
the dynamic model but worsened AUC regret by 2.28%. Closed-loop evaluation, expanded collection, and RL
are intentionally paused. See [`docs/CONTEXT_AUDIT.md`](docs/CONTEXT_AUDIT.md).

`scripts/run_local_representation_audit.py` diagnoses whether that failure came from missing local
path/conflict structure or from hiding the actual neighborhood generated by LNS2. It rebuilds all
conflicts, creates pre-generation and realized-neighborhood indexes, and evaluates Horizon 1
effectiveness separately from compute and runtime sensitivity. See
[`docs/LOCAL_REPRESENTATION_AUDIT.md`](docs/LOCAL_REPRESENTATION_AUDIT.md).
The formal audit also failed its registered recovery gates; the next permitted experiment is a small
MovingAI mechanism probe, not expanded collection or RL training.

`scripts/prepare_movingai_probe.py` adapts the pinned standard maps into a fingerprinted probe split,
and `scripts/analyze_movingai_probe.py` separates immediate action effects from dual-trial noise and
tests map/density oracle heterogeneity at the task-instance level. See
[`docs/MOVINGAI_MECHANISM_PROBE.md`](docs/MOVINGAI_MECHANISM_PROBE.md).
The completed probe found 12 unique repair states and 1,368 outcomes. Action and neighborhood diversity
were real, but action identity explained only 39.2% of conflict variation after duplicate solver states
were pooled; map/density permutation gates also failed. The active decision is to increase action trials
before any contextual model or RL training.

`scripts/audit_movingai_probe_quality.py` performs the follow-up effective-sample and label-stability
audit. It replaces Monte Carlo context shuffles with exact assignments, reports state-normalized action
effects, duplicate-episode rank stability, realized-neighborhood Jaccard, and compute-aware Pareto
sensitivity. The audit shows that 1,368 rows represent only 12 independent states and four trials per
candidate; the data are not sufficient for a transfer claim. The corrected v2 adapter exposes three
pinned MovingAI task scenarios and uses one state-acquisition seed plus eight action trials. See
[`docs/MOVINGAI_PROBE_QUALITY.md`](docs/MOVINGAI_PROBE_QUALITY.md).

The v2 partial confirmation recovered 35 independent states and 7,776 outcomes. Eight trials improved
split-half rank Spearman to 0.684, but best-candidate overlap remained 0.376 and realized-neighborhood
Jaccard 0.391. Map-specific oracle heterogeneity was detectable, while density alignment was not; with
one map per layout family this is still not transfer evidence. The next collection must add independent
layout replicas rather than more repetitions of the same states.

`scripts/analyze_independent_layout_probe.py` implements that bounded confirmation without training a
model. Six newly seeded maps form a complete layout x OD x density design, qualification gates precede
all counterfactual work, and eight trials estimate each nominal action as a stochastic generator. The
analysis adds paired OD/density tests, independent-layout permutations, Holm correction, fixed-family
dominance, and a non-veto routing rule for realized-neighborhood Jaccard. See
[`docs/INDEPENDENT_LAYOUT_PROBE.md`](docs/INDEPENDENT_LAYOUT_PROBE.md).

The completed run produced 23 states and 6,480 outcomes with zero errors. Rank Spearman improved to
0.638, but action eta-squared (0.404) and Pareto-family Jaccard (0.432) failed their registered gates;
layout, OD, and density also had no Holm-corrected signal. The decision is to stop expansion and
redefine the action surface around generated-neighborhood ranking, not to begin RL.

The next bounded experiment is implemented by `scripts/collect_realized_neighborhood_probe.py`. It
deduplicates actual agent sets from the independent-layout proposals, replays them through the explicit
neighborhood API, and uses independent seeds to isolate PP repair-order variance. No model is trained;
the gate determines whether realized-neighborhood ranking is statistically well-defined. See
[`docs/REALIZED_NEIGHBORHOOD_PROBE.md`](docs/REALIZED_NEIGHBORHOOD_PROBE.md).

The completed explicit replay evaluated 412 concrete neighborhoods over 3,296 outcomes with zero
errors. Realized-action eta-squared increased from 0.404 to 0.595, rank Spearman reached 0.803, and
Pareto/best-candidate Jaccard reached 0.518/0.547. Every registered gate passed, so the next permitted
stage was a realized-neighborhood ranking audit, not RL or expanded static-context collection.

That audit is now implemented by `scripts/run_realized_neighborhood_ranking_audit.py`. It aggregates
all eight PP-order trials before labeling, trains fixed pairwise GBDTs in six leave-one-map-out folds,
and compares proposal metadata, explicit-neighborhood structure, and static context. The dynamic plus
realized profile improved Pareto top-1 from 13.0% to 43.5% and reduced conflict regret by 55.6%, with
no regression on any held-out map. Static context did not pass its incremental gate: its 4.35-point
top-1 gain and 92.0%/74.2% permutation percentiles were below the registered requirements. The active
direction is therefore dynamic-state plus concrete-neighborhood ranking; the static transfer claim and
RL remain paused. See
[`docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md`](docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md).

The next registered stage is implemented by `scripts/collect_realized_ranking_confirmation.py` and
`scripts/run_realized_ranking_confirmation.py`. A proposal-only native API generates candidate agent
sets without paying for discarded PP/SIPPS repairs. Fixed all-development rankers are then evaluated on
12 newly seeded maps with complete balanced/bottleneck and 80/100-agent pairing. Confirmation labels
cannot train or tune the frozen models, static context is exploratory only, and RL remains gated on the
independent result. See
[`docs/REALIZED_RANKING_CONFIRMATION.md`](docs/REALIZED_RANKING_CONFIRMATION.md).
The first two qualification-only generations stopped at 10/12 and 11/12 paired-map coverage,
respectively; no independent repair outcomes were collected. A proposed task-seed qualification pool was
rejected because it would select away the layout-dependent conflict distribution the project needs to
measure.

The replacement natural-distribution confirmation keeps zero-conflict tasks as successful PP outcomes and
keeps high-conflict tasks as repair states. The inspected 12-map set is a Pilot only; a new 12-map formal
set uses fixed frozen rankers and per-trial timeout/resume isolation. See
[`docs/NATURAL_DISTRIBUTION_CONFIRMATION.md`](docs/NATURAL_DISTRIBUTION_CONFIRMATION.md).
The formal run retained all 48 tasks, produced 41 repair states, 733 explicit neighborhoods and 5,864
isolated trials with no failures. The frozen realized-neighborhood ranker improved Pareto top-1 from 19.5%
to 43.9% and reduced remaining-conflict regret by 33.8%, passing every primary gate. Static context still
showed no incremental benefit, so the next gate is a fresh-map closed-loop test of the dynamic realized
ranker; RL remains paused.

The completed closed-loop stage compares official Adaptive with frozen proposal-only and
realized-neighborhood rankers on 24 tasks from six new maps. All policies solved 24/24 tasks. The realized
ranker reduced fixed-budget conflict AUC by 57.3%, repair iterations by 43.8%, and generated nodes by
19.4%, passing every registered gate on all six maps. Its exhaustive candidate controller was still much
slower in wall time (13.44s versus 0.42s), so this is evidence for better sequential decisions rather than
a deployed speedup. See
[`docs/CLOSED_LOOP_CONFIRMATION.md`](docs/CLOSED_LOOP_CONFIRMATION.md).

The hardened controller now ships its frozen trees and feature ranges in
`artifacts/initlns-closed-loop-policy-v1/`, without requiring sklearn in WSL. Native proposal batching,
native tree inference, feature caches and strict trace validation reduced mean controller overhead on
repairable tasks from 13.11s to 0.52s. A 72-episode, 602-transition replay was scientifically identical to
the registered run; same-run end-to-end time remains about three times Adaptive, so runtime superiority and
static-context transfer are still not claimed.

The registered multi-seed confirmation used 12 additional maps and solver seeds `[1,2,3]`. Both policies
solved 144/144 episodes; the frozen realized-neighborhood ranker reduced fixed-budget conflict AUC by
52.5%, was no worse on 11/12 maps, and improved AUC separately for all three solver seeds. Its mean wall
time remained slower (0.715s versus 0.273s), and static context was excluded. The result supports dynamic
realized-neighborhood control across same-family maps and solver randomness, not static-context transfer
or OOD generalization. The next stage collects policy-visited states before RL warm start. See
[`docs/CLOSED_LOOP_MULTISEED_CONFIRMATION.md`](docs/CLOSED_LOOP_MULTISEED_CONFIRMATION.md).

`scripts/fetch_movingai_devset.py` verifies and extracts six pinned MovingAI development maps.
`scripts/run_feasibility_benchmark.py` gives `lns2_repair` and `gpbs_official` identical map, scenario,
agent-count, time-limit, and seed inputs with common failure accounting. See
[`docs/MOVINGAI_BASELINES.md`](docs/MOVINGAI_BASELINES.md).

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

```bash
ctest --test-dir build/linux/project --output-on-failure
```

The Linux suite covers official C++ interfaces, repair-only execution, deterministic reset, seed and
explicit actions, fallback behavior, and the native Python module. The Windows Python suite covers map,
static OD semantics, metadata, MovingAI export, and split determinism.

## Documentation

- [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md): research stages and paper-code reuse.
- [`docs/TRACE_AND_POLICY_API.md`](docs/TRACE_AND_POLICY_API.md): observation, action, and JSONL schema.
- [`docs/REPAIR_COLLECTION.md`](docs/REPAIR_COLLECTION.md): qualification, baselines, and counterfactual data.
- [`docs/CONTEXT_AUDIT.md`](docs/CONTEXT_AUDIT.md): pairwise ablations, gates, and current negative result.
- [`docs/CONTEXT_SECONDARY_AUDIT.md`](docs/CONTEXT_SECONDARY_AUDIT.md): map-grouped diagnostics,
  context permutations, stopping rule, and the secondary negative result.
- [`docs/LOCAL_REPRESENTATION_AUDIT.md`](docs/LOCAL_REPRESENTATION_AUDIT.md): local path features,
  realized-neighborhood ranking, leakage boundary, and pre-registered decision rules.
- [`docs/MOVINGAI_BASELINES.md`](docs/MOVINGAI_BASELINES.md): pinned standard data and common GPBS/LNS2 runner.
- [`docs/INDEPENDENT_LAYOUT_PROBE.md`](docs/INDEPENDENT_LAYOUT_PROBE.md): independent layout/OD/density
  confirmation, staged collection gates, and registered interpretation.
- [`docs/REALIZED_NEIGHBORHOOD_PROBE.md`](docs/REALIZED_NEIGHBORHOOD_PROBE.md): explicit-neighborhood
  replay, proposal/evaluation random-seed separation, and ranking-stability gates.
- [`docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md`](docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md):
  leave-one-map-out explicit-set ranking, static-context ablation, and the current positive/negative result.
- [`docs/CLOSED_LOOP_CONFIRMATION.md`](docs/CLOSED_LOOP_CONFIRMATION.md): frozen pairwise policy,
  sequential candidate generation, failure accounting, and the final gate before RL warm-start work.
- [`docs/CLOSED_LOOP_MULTISEED_CONFIRMATION.md`](docs/CLOSED_LOOP_MULTISEED_CONFIRMATION.md): independent
  12-map, three-solver-seed confirmation with map-grouped inference.
- [`docs/ENVIRONMENT_AUDIT.md`](docs/ENVIRONMENT_AUDIT.md): WSL diagnosis and dependency inventory.
- [`docs/STAGE1.md`](docs/STAGE1.md): active warehouse dataset.
- [`archive/legacy_stage5/`](archive/legacy_stage5/): simplified solver and negative Stage 3-5 results.

## License

Project-owned code follows the repository license policy. Vendored MAPF-LNS2 is subject to the USC
Research Licenses in `third_party/mapf_lns2/license.txt` and `third_party/gpbs/license.md`; commercial
use requires separate permission.
