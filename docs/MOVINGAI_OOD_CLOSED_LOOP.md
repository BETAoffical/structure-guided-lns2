# MovingAI OOD Closed-Loop Evaluation

This protocol tests whether the frozen dynamic explicit-neighborhood controller
generalizes to standard map structures that were absent from development.

## Data

The registered suite contains 12 MovingAI maps from Random, Maze, Room, Warehouse,
and Game families. Two fixed random scenarios, two fixed agent counts, and solver
seeds 1/2/3 produce 144 episodes. Map/scenario files and SHA256 values are fixed by
the dataset manifest; episodes are never replaced based on conflict count.

Zero-conflict episodes remain end-to-end successes and do not invoke the learned
controller. Conditional policy metrics use only nonzero-conflict episodes.

## Supported Policies

- Official Adaptive
- Official fixed Target
- Official fixed Collision
- Official fixed Random
- Frozen `v1-full`
- Exact accelerated `v2-full`
- Registered `v2-stall-safe` when explicitly requested

The active runtime does not expose balanced, cascade, or proposal-pruned variants.

## Execution

Inspect the collection interface:

```bash
python scripts/collect_closed_loop_confirmation.py --help
python scripts/analyze_movingai_ood_confirmation.py --help
```

Run the current dual-track quick/formal evaluation interface:

```bash
python scripts/run_lns2_tradeoff_evaluation.py --help
```

All strategies on a task must share the initial fingerprint. Runs use at most 100
repairs, a 300-second solver budget for the registered protocol, process-isolated
RNG, atomic episode output, and resume at episode granularity.

## Registered Result

Frozen v1 obtained 131/144 successes versus 123/144 for official Adaptive. Its
fixed-budget conflict AUC improved by 4.105%. This is a broad positive signal, but
the preregistered gate required at least 5%, so the strict result is **FAIL**.

This result does not prove incremental value from static map, OD, or density
features: v1 uses only the current dynamic conflict state and the realized candidate
agent set. It also does not establish an RL result.

The canonical numbers and source hashes are maintained by:

```bash
python scripts/consolidate_research_results.py \
  --config configs/result_consolidation.json --verify-build
```

## Runtime Interpretation

`v2-full` changes implementation cost, not the learned ranking. Feature vectors,
candidate IDs, pairwise scores, and selected candidates must match v1. Wall-clock
time includes candidate proposal generation, feature extraction, inference, PP +
SIPPS repair, process scheduling, and trace I/O; faster feature inference alone can
therefore produce only a small end-to-end improvement.

`v2-stall-safe` is a separately registered guard around v2. It does not retrain the
ranker or alter official repair logic.
