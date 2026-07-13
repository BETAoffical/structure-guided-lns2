# Repair Experience Collection

## Purpose

This stage collects trustworthy collision-repair experience from the complete MAPF-LNS2 kernel. It
does not train a supervised model or an RL policy. The output contains official baseline trajectories
and controlled counterfactual branches that later stages can turn into ranking labels or rewards.

## Transfer pilot

`configs/repair_transfer_pilot.json` generates 102 MovingAI instances across seven splits:

- `train`, `validation`, and `test_id` share layout and task families but use disjoint seeds;
- `test_ood_layout` holds out six layout families;
- `test_ood_task` holds out uniform, four-way intersection, six-pair cross-zone, and swap-pair static OD modes;
- `test_ood_density` uses 60 and 120 agents outside the 80/100-agent training range;
- `test_joint_ood` combines unseen layouts, unseen tasks, and 120 agents.

The cross-zone variants keep exact six-pair OD quotas and apply a moderate
`hotspot_skew=0.5` within each origin/destination zone so that qualification
contains repair states on open layouts. Intersection variants instead use a
separate 60% shortest-path-through-intersection constraint.

OOD means out of distribution. Test and OOD splits are evaluation-only and never produce
counterfactual training labels.

```powershell
python scripts/generate_dataset.py `
  --config configs/repair_transfer_pilot.json `
  --output build/repair-transfer-pilot-v2
```

## Collection

Run the native collector inside Ubuntu after building `lns2_env`:

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/repair-transfer-pilot-v2 \
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
  --dataset build/repair-transfer-pilot-v2 \
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

The 2026-07-14 v2 acceptance generated all 102 instances and completed 204 qualification runs with no
errors. Train was 48/48 repairable; intersection-100/120 and cross-zone-100/120 were each 18/18. The
bounded train smoke completed all four official baselines and emitted 36 outcomes from two Adaptive
states. The OOD smoke wrote 24 baseline traces covering uniform, intersection, and cross-zone tasks,
and correctly emitted zero counterfactual labels.

The original v1 pilot remains under `build/repair-transfer-pilot` for audit only. Its intersection and
cross-zone names shared nearly identical endpoint logic and must not be used for formal OOD comparison.
Pilot v2 is written to a separate directory and uses task-semantics version 2; collection fingerprints
reject attempts to resume a v1 run with v2 data. Raw data remains under ignored `build/` paths.
