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

The collector holds a workspace-wide process lock and an output-specific lock. A second live
collector is rejected even when `--resume` is present. Every completed episode atomically updates
`collection_progress.json` and its phase manifest, so a detached WSL command remains observable and
does not need a duplicate resume process. A stale lock is archived only after its PID and Linux
process-start token no longer identify a live owner.

Inspect or stop the exact process recorded by an output lock from the same WSL distribution:

```bash
python3 scripts/manage_repair_collection.py status \
  --output build/repair-experience-pilot

python3 scripts/manage_repair_collection.py cancel \
  --output build/repair-experience-pilot
```

`SIGINT`, `SIGTERM`, and configured episode timeouts terminate owned child processes. Completed
episodes remain resumable; an interrupted or timed-out episode has no complete metadata marker and
is recomputed on the next compatible resume.

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

Use `--task-ids` to select exact task IDs. The sorted selection is included in the run fingerprint.
`--dry-run` writes no collection files and reports the qualification/baseline job counts and maximum
counterfactual reset count. Once compatible Adaptive baselines exist, it also reports the exact
selected state/branch count and a reset-only CPU-time lower bound.

The runner-hardening smoke uses one 100-agent and one 400-agent MovingAI task:

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/movingai-mechanism-probe-v2-dataset \
  --config configs/repair_collection_hardening_smoke.json \
  --output build/repair-collector-hardening-smoke \
  --phase all --workers 2 \
  --task-ids room-32-32-4__random_02__agents_0100,warehouse-10-20-10-2-1__random_01__agents_0400
```

The optional `counterfactual.episode_wall_time_limit_seconds` is part of the configuration
fingerprint. It bounds one source episode, not the complete collection. Runtime comparisons are
valid only when the workspace lock confirms that no second collector was competing for CPU.

The hardening acceptance selected one 100-agent room task and one 400-agent warehouse task. Dry-run
predicted 2 states and 48 resets with a 32.4-second reset-only CPU lower bound. The collection
finished in about 49 seconds with 48 outcomes and no error or timeout. Its manifest became visible
after the first 24-outcome episode, a live duplicate resume was rejected, exact cancel released both
locks, no worker remained, and all state/outcome hashes were unchanged by a compatible resume.

## Calibration collection

`configs/repair_collection_calibration.json` is the reproducible label-quality
calibration. It uses only the 24 Train and 12 Validation instances with solver
seeds 0 and 1, for 72 instance-seed sources and 288 four-policy baseline
episodes. Test and OOD splits are excluded from counterfactual collection.

Each Adaptive episode contributes up to three evenly spaced repair states. A
state evaluates up to six conflict seed agents, all Target/Collision/Random
generators, and neighborhood sizes 4, 8, and 16 at horizons 1 and 4. With one
trial per candidate, the theoretical maximum is 11,664 outcomes. One trial is
intended to audit coverage and action separation; stochastic stability remains
part of the later two-trial collection.

```bash
PYTHONPATH=build/linux/project python3 scripts/collect_repair_experience.py \
  --dataset build/repair-transfer-pilot-v2 \
  --config configs/repair_collection_calibration.json \
  --output build/repair-experience-calibration-v2 \
  --phase all --splits train,validation --workers 4
```

The quality analyzer checks collection integrity, coverage, horizon-specific
outcome diversity, Pareto sets, fixed action-family dominance, and early/middle/
late repair preferences. It does not synthesize a reward or treat the official
Adaptive action as a label.

```bash
python3 scripts/analyze_repair_experience.py \
  --collection build/repair-experience-calibration-v2
```

It writes `quality_report.json` and `quality_report.md` into the collection
directory and returns a nonzero exit status when an acceptance gate fails.
Future supervised training uses only Train outcomes. Validation remains isolated
for label/reward audits and model selection.

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

The expanded 2026-07-14 calibration completed 72/72 Train/Validation
qualifications and 288/288 baseline episodes with no runtime errors. The 72
Adaptive sources produced 200 repair states, 816 seed-agent selections, and
7,344 outcomes. Prefix replay mismatches, invalid actions, Test/OOD labels,
seed-coverage failures, and action-family coverage failures were all zero.
Every Horizon 4 state had more than one outcome, the most dominant fixed action
family was uniquely Pareto-optimal in 12.5% of states, and the mean Horizon 1/4
Pareto-set overlap was 50.6%. All quality gates passed.

The subsequent context-learnability gate did not pass. The planned 31,104-outcome
calibration expansion and 62,208-outcome semantic-v3 collection are therefore
paused; passing collection-quality checks alone is not evidence that static
context improves policy learning. See `research/docs/context/CONTEXT_AUDIT.md`.

The original v1 pilot remains under `build/repair-transfer-pilot` for audit only. Its intersection and
cross-zone names shared nearly identical endpoint logic and must not be used for formal OOD comparison.
Pilot v2 is written to a separate directory and uses task-semantics version 2; collection fingerprints
reject attempts to resume a v1 run with v2 data. Raw data remains under ignored `build/` paths.
