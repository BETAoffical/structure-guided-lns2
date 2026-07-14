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
- [`docs/ENVIRONMENT_AUDIT.md`](docs/ENVIRONMENT_AUDIT.md): WSL diagnosis and dependency inventory.
- [`docs/STAGE1.md`](docs/STAGE1.md): active warehouse dataset.
- [`archive/legacy_stage5/`](archive/legacy_stage5/): simplified solver and negative Stage 3-5 results.

## License

Project-owned code follows the repository license policy. Vendored MAPF-LNS2 is subject to the USC
Research Licenses in `third_party/mapf_lns2/license.txt` and `third_party/gpbs/license.md`; commercial
use requires separate permission.
