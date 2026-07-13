# Structure-Guided MAPF-LNS2

Research infrastructure for context-aware neighborhood control during the collision-repair phase of
MAPF-LNS2. The active solver is now the complete official MAPF-LNS2 codebase, not the former
dependency-free approximation.

## Research target

The first research question is deliberately narrow:

> Can map structure, task flow, agent density, the current conflict graph, and repair history reduce
> time-to-feasible on unseen MAPF domains without reducing success rate?

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

## Structured warehouse data

The retained generator creates controlled warehouse layouts and task-flow variants. Every map/task pair
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

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

```bash
ctest --test-dir build/linux/project --output-on-failure
```

The Linux suite covers official C++ interfaces, repair-only execution, deterministic reset, seed and
explicit actions, fallback behavior, and the native Python module. The Windows Python suite covers map,
task-flow, metadata, MovingAI export, and split determinism.

## Documentation

- [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md): research stages and paper-code reuse.
- [`docs/TRACE_AND_POLICY_API.md`](docs/TRACE_AND_POLICY_API.md): observation, action, and JSONL schema.
- [`docs/ENVIRONMENT_AUDIT.md`](docs/ENVIRONMENT_AUDIT.md): WSL diagnosis and dependency inventory.
- [`docs/STAGE1.md`](docs/STAGE1.md): active warehouse dataset.
- [`archive/legacy_stage5/`](archive/legacy_stage5/): simplified solver and negative Stage 3-5 results.

## License

Project-owned code follows the repository license policy. Vendored MAPF-LNS2 is subject to the USC
Research License in `third_party/mapf_lns2/license.txt`; commercial use requires separate permission.
