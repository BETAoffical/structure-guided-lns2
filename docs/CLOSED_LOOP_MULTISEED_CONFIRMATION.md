# InitLNS frozen policy multi-seed confirmation

## Question

The first closed-loop confirmation used six unseen maps and solver seed 0. It showed a large conflict-AUC
advantage for the frozen `realized_dynamic` ranker, but it cannot distinguish a stable policy effect from
one favorable initial PP/random-order realization. This confirmation freezes the same model and controller
and evaluates three solver seeds on a larger independent map cohort. It does not train or tune a model,
use static context, or produce RL data.

## Registered cohort

- Dataset master seed: `20270123`.
- Twelve new maps: four each from `regular_beltway`, `compartmentalized` and `dead_end_aisles`.
- Four unfiltered tasks per map: balanced/bottleneck crossed with 80/100 agents.
- Solver seeds: `1`, `2`, `3`.
- Policies: official Adaptive and the frozen `realized_dynamic` ranker.
- Total resets: 144; total policy episodes: 288.

All map and task seeds must be disjoint from every registered development and confirmation dataset. Zero
conflict resets remain successful PP outcomes; high-conflict resets remain in the cohort. Qualification
requires all 144 resets to be valid, at least 108 repairable task-seeds, at least 24 repairable task-seeds
per layout, at least 30 per solver seed, and at least 10 active maps. Failure is reported as insufficient
evidence without replacing task seeds.

Pre-outcome amendment: the initial qualification used `[0,1,2]` and revealed that glibc `srand(0)` and
`srand(1)` expose the same random stream. No policy episode had been run. The registered seeds were changed
to `[1,2,3]`, and qualification now rejects any pair of solver seeds whose complete vector of initial state
fingerprints is identical. This corrects duplicate treatment assignment; it does not replace any map or
task based on its outcome.

## Analysis

The primary outcomes remain success count, fixed 100-step conflict AUC, capped time to feasibility and
low-level search work. Adaptive and the frozen ranker must begin from identical fingerprints for every
`(task, solver_seed)` pair. The paired map bootstrap treats a map, with all of its tasks and solver seeds,
as one resampling unit; solver seeds are repeated measurements rather than independent maps.

The frozen ranker passes only if:

- success is not below Adaptive overall or for any individual solver seed;
- fixed conflict AUC or capped wall time improves by at least 5% overall;
- the qualifying metric is no worse on at least 8/12 maps and its 5,000-sample map bootstrap does not
  show significant degradation;
- fixed conflict AUC improves by at least 5% on at least two of three solver seeds;
- all traces are valid, with no invalid action or fingerprint mismatch.

The wall-time gate is retained but is not expected to pass because exhaustive candidate ranking remains
more expensive than Adaptive. Static map/OD/density context is excluded, so success would confirm dynamic
realized-neighborhood control across same-family maps and solver randomness, not the original static
transfer hypothesis or OOD generalization.

## Commands

```powershell
python scripts/generate_dataset.py --config configs/closed_loop_multiseed_dataset.json
```

Run collection in WSL with `PYTHONPATH=build/linux/project` using
`configs/closed_loop_multiseed_collection.json`, first with `--phase qualify`, then with `--phase all
--resume`. Analyze with:

```powershell
python scripts/analyze_closed_loop_confirmation.py `
  --collection build/initlns-closed-loop-multiseed-v1-collection `
  --config configs/closed_loop_multiseed_analysis.json `
  --output build/initlns-closed-loop-multiseed-v1-report `
  --strict
```
