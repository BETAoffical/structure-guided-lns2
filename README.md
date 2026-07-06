# Structure-Guided LNS2

Independent C++17/Python project for testing whether warehouse structure and
past LNS repair outcomes can improve neighborhood selection. It does not
modify or import the previous `LNS2-RL` repository.

## Project status

| Stage | Status | Contents |
| --- | --- | --- |
| 0 | Complete | Dependency-free LNS2 baseline and path validation |
| 1 | Complete (MVP active) | Three controlled layouts and two task scenarios |
| 2 | Complete | Trace V2 and raw solver-run collection |
| 3 | Complete | Structured LNS repair cases and conflict heatmaps |
| 4 | Complete | Feature normalization, kNN retrieval, and offline guidance |
| 5 | Complete (negative result) | Closed-loop Repair guidance and paired evaluation |

No RL or learned neighborhood policy is used in the current feasibility
experiment.

## Stage 0: LNS2 baseline

The solver builds an initial solution with randomized prioritized planning,
detects vertex and edge conflicts, selects a neighborhood from the conflict
graph, and replans that neighborhood with space-time search. A candidate is
accepted when `(conflicting pairs, sum of costs)` does not worsen.

The `.mapf` format is:

```text
ROWS COLS
....@
.....
AGENT_COUNT
START_ROW START_COL GOAL_ROW GOAL_COL
...
```

### Native Windows build

Run these commands in a Visual Studio 2022 Developer PowerShell:

```powershell
cmake -S . -B build/windows -G "Visual Studio 17 2022" -A x64
cmake --build build/windows --config Release --parallel
ctest --test-dir build/windows -C Release --output-on-failure
```

Run one instance and save its raw LNS trace:

```powershell
build/windows/Release/lns2_cli.exe `
  --instance tests/data/warehouse_small.mapf `
  --seed 1234 `
  --neighborhood 6 `
  --iterations 500 `
  --time-limit-ms 3000 `
  --trace build/example-trace.jsonl
```

## Stage 1: feasibility dataset

The active dataset deliberately limits variation:

- layouts: `regular_beltway`, `compartmentalized`, `dead_end_aisles`;
- map size: `28 x 39`;
- compartment templates: cross, two horizontal walls, or two vertical walls;
- dead-end maps: two horizontal and two vertical shelf-connected caps;
- four tasks per map: 36-Agent baseline, 60-Agent dense, 48-Agent
  clustered, and 36-Agent random control;
- 18/6/12 train/validation/test maps, producing 144 instances.

Generate and inspect it:

```powershell
python scripts/generate_dataset.py `
  --config configs/stage1_example.json `
  --output build/feasibility-dataset

python scripts/inspect_dataset.py --dataset build/feasibility-dataset
python scripts/generate_gallery.py
python -m unittest tests.test_stage1_generators -v
```

See [Stage 1](docs/STAGE1.md) and the
[configuration reference](docs/CONFIGURATION.md).

## Stage 2: raw experience

Collect traces for the 72 training instances:

```powershell
python scripts/collect_experience.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --split train `
  --seeds 1,2,3 `
  --time-limit-ms 5000 `
  --output build/experience
```

Every task/solver-seed pair receives an independent JSONL file, producing 216
training runs. Each iteration records the seed conflict, selected agents,
before/after conflict locations, neighborhood paths, costs, acceptance, and
replanning time. See [Stage 2](docs/STAGE2.md).

## Stage 3: repair experience

Build the Train-only repair memory from Trace V2:

```powershell
python scripts/build_repair_experience.py `
  --dataset build/feasibility-dataset `
  --collection build/experience `
  --split train `
  --output build/repair-experience
```

The output contains iteration-level repair cases, run-level aggregates,
sparse conflict heatmaps, Agent descriptors, and successful/neutral/failed
neighborhood labels. See [Stage 3](docs/STAGE3.md).

## Stage 4: offline retrieval

Collect and convert Validation queries without adding them to memory:

```powershell
python scripts/collect_experience.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --split validation `
  --seeds 1,2,3 `
  --neighborhood 6 `
  --iterations 500 `
  --time-limit-ms 5000 `
  --output build/stage4-validation-collection

python scripts/build_query_experience.py `
  --dataset build/feasibility-dataset `
  --collection build/stage4-validation-collection `
  --split validation `
  --output build/stage4-validation-experience
```

Build the Train-only index and evaluate it:

```powershell
python scripts/build_retrieval_index.py `
  --memory build/repair-experience `
  --output build/stage4-index

python scripts/evaluate_retrieval.py `
  --index build/stage4-index `
  --queries build/stage4-validation-experience `
  --output build/stage4-evaluation
```

Stage 4 predicts sparse conflict heatmaps, repair effectiveness, and
transferable neighborhood role templates. It does not map those roles to
concrete Agents or demonstrate an improvement in solver performance. Those
experiments belong to Stage 5. See [Stage 4](docs/STAGE4.md).

## Stage 5: guided simplified LNS2

Stage 5 compares the existing simplified LNS2 baseline with a closed-loop
Repair-guided variant. It does not claim to reproduce or improve the complete
official MAPF-LNS2 solver.

Validation selects the confidence threshold:

```powershell
python scripts/run_stage5_experiment.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --index build/stage4-index `
  --evaluation build/stage4-evaluation `
  --split validation `
  --seeds 1,2,3 `
  --thresholds 0.8 `
  --output build/stage5-validation-paired
```

Test uses only the frozen Validation configuration:

```powershell
python scripts/run_stage5_experiment.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --index build/stage4-index `
  --evaluation build/stage4-evaluation `
  --split test `
  --seeds 1,2,3 `
  --config build/stage5-validation-paired/selected_config.json `
  --output build/stage5-test
```

On 144 paired Test runs, both strategies solved 128 runs and retained 77
conflicting pairs in total. Guided LNS2 recorded 6 paired wins, 4 losses, and
134 ties; the exact sign-test p-value was `0.754`. The current Repair guidance
therefore does not show a statistically significant overall improvement. See
[Stage 5](docs/STAGE5.md).

## Reproducibility

Map, task, and solver decisions are deterministic for a fixed implementation,
seed, and C++ standard library. Runtime fields vary. The C++ standard does not
guarantee identical `std::shuffle` sequences across different standard library
implementations, so Windows and Linux runs should be treated as separate
reproducibility environments.
