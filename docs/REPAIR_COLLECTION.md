# Repair Experience Collection

## Purpose

This stage collects trustworthy collision-repair experience from the complete MAPF-LNS2 kernel. It
does not train a supervised model or an RL policy. The output contains official baseline trajectories
and controlled counterfactual branches that later stages can turn into ranking labels or rewards.

## Transfer pilot

`configs/repair_transfer_pilot.json` generates 102 MovingAI instances across seven splits:

- `train`, `validation`, and `test_id` share layout and task families but use disjoint seeds;
- `test_ood_layout` holds out six layout families;
- `test_ood_task` holds out uniform, intersection, cross-zone, and swap-pair task flows;
- `test_ood_density` uses 60 and 120 agents outside the 80/100-agent training range;
- `test_joint_ood` combines unseen layouts, unseen tasks, and 120 agents.

OOD means out of distribution. Test and OOD splits are evaluation-only and never produce
counterfactual training labels.

```powershell
python scripts/generate_dataset.py `
  --config configs/repair_transfer_pilot.json `
  --output build/repair-transfer-pilot
```

## Collection

Run the native collector inside Ubuntu after building `lns2_env`:

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/repair-transfer-pilot \
  --config configs/repair_collection_pilot.json \
  --output build/repair-experience-pilot \
  --phase all \
  --workers 4
```

Phases are `qualify`, `baseline`, `counterfactual`, and `all`. `--resume` reuses completed
instance-seed qualifications, episode traces, and counterfactual episode files. A dataset or semantic
configuration mismatch is rejected instead of mixing incompatible runs.

The pilot defaults collect Adaptive, Target, Collision, and Random baselines with solver seeds 0 and 1.
Counterfactual states come only from train/validation Adaptive trajectories. Each candidate controls a
conflicting seed agent, Target/Collision/Random generator, neighborhood size, and branch random seed.

For a short end-to-end check:

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/repair-transfer-pilot \
  --config configs/repair_collection_pilot.json \
  --output build/repair-experience-smoke \
  --phase all --splits train --workers 4 --max-episodes 1 \
  --max-states 2 --max-seed-agents 4 --neighborhood-sizes 4,8 \
  --trials 1 --horizons 1,2
```

## Output contract

- `run_config.json` pins dataset and semantic configuration fingerprints.
- `qualification_manifest.jsonl` records whether every instance-seed reaches InitLNS repair.
- `collection_manifest.jsonl` indexes versioned per-policy episode traces.
- `counterfactual_manifest.jsonl` indexes state, outcome, and replay-error files by source episode.
- `summary.json` reports qualification, baseline, and counterfactual totals.

All manifest paths are relative to the collection root. A state fingerprint contains paths, conflicts,
costs, delays, map cells, iteration, and low-level counters while excluding wall-clock runtime and
caller context. A counterfactual branch is emitted only when replaying the requested-action prefix
reconstructs the exact fingerprint.

Collection is CPU-only. MAPF-LNS2 uses process-global `rand()`, so workers are separate spawned
processes rather than threads. Raw branch outcomes include conflict trajectories, conflict AUC,
sum-of-cost changes, low-level search deltas, and branch runtime; no reward is hard-coded.

## Acceptance snapshot

The 2026-07-14 local acceptance generated all 102 configured instances. On train, all 48 combinations
of 24 instances and solver seeds 0/1 entered collision repair, with 12.17 initial colliding pairs on
average. The bounded smoke run solved all four baseline episodes and emitted 36 outcomes from two
Adaptive states with no replay or collection errors. Raw data remains under ignored `build/` paths.
