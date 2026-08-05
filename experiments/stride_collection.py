from __future__ import annotations

import concurrent.futures
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    STRIDE_TRIAL_SCHEMA,
    post_structure_metrics,
)
from experiments.trace_replay import replay_prefix, result_blind_decision_rows
from lns2_selector.runtime.artifact_validation import (
    candidate_records,
    repair_trial_semantics_valid,
    strict_integer,
    trial_product_matches,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


STRIDE_SELECTION_SCHEMA = "lns2.stride.state_selection.v1"
STRIDE_COLLECTION_SCHEMA = "lns2.stride.repair_collection.v1"
STRIDE_COLLECTION_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/artifact_validation.py",
    "lns2_selector/runtime/online_selection.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)
FULL_POOL_PROPOSAL = {
    "heuristics": ["target", "collision", "random"],
    "neighborhood_sizes": [4, 8, 16],
    "candidates_per_family": 2,
}
PP_TRIAL_INDICES = (0, 1, 2, 3)
STRIDE_PILOT_SPLIT = "stride_pilot"
STRIDE_SOURCE_POLICIES = {
    "official_adaptive": ("official_adaptive_manifest.jsonl", "official_adaptive"),
    "v2-full": ("realized_dynamic_manifest.jsonl", "v2-full"),
}


def _conflict_band(conflicts: int) -> str:
    if conflicts <= 10:
        return "low_1_10"
    if conflicts <= 100:
        return "medium_11_100"
    if conflicts <= 500:
        return "high_101_500"
    return "extreme_501_plus"


def _decision_stage(decision_index: int) -> str:
    if decision_index < 4:
        return "early"
    if decision_index < 8:
        return "middle"
    return "late"


def _agent_band(agent_count: int) -> str:
    return "low_mid" if agent_count <= 200 else "high"


def _selection_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Return the only fields permitted to influence STRIDE state sampling."""

    return {
        "source_policy": str(row["source_policy"]),
        "episode_id": str(row["episode_id"]),
        "map_id": str(row["map_id"]),
        "task_id": str(row["task_id"]),
        "solver_seed": int(row["solver_seed"]),
        "decision_index": int(row["decision_index"]),
        "before_fingerprint": str(row["before_fingerprint"]),
        "before_conflicts": int(row["before_conflicts"]),
        "agent_count": int(row["agent_count"]),
    }


def _balanced_result_blind_selection(
    pool: list[dict[str, Any]], *, target_per_policy: int, max_per_episode: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily balance pre-action strata with a hash-only tie break.

    No repair outcome, elapsed time, after-state field, or chosen source action is
    accepted by this helper.  The four registered conflict bands keep states
    above 500 rather than silently discarding them; they serve as a fallback for
    the registered 101--500 high-conflict target.
    """

    allowed = {
        "schema",
        "state_id",
        "map_id",
        "task_id",
        "split",
        "source_policy",
        "decision_stage",
        "conflict_band",
        "source_group",
        "layout_mode",
        "source_root",
        "episode_id",
        "before_fingerprint",
        "before_conflicts",
        "solver_seed",
        "decision_index",
        "agent_count",
        "agent_band",
        "prefix_actions",
    }
    forbidden = sorted({key for row in pool for key in row if key not in allowed})
    if forbidden:
        raise ValueError(f"result-blind STRIDE pool has forbidden fields: {forbidden}")
    selected: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_policy[str(row["source_policy"])].append(row)
    policy_reports: dict[str, Any] = {}
    for policy in STRIDE_SOURCE_POLICIES:
        candidates = list(by_policy.get(policy, []))
        episode_counts: Counter[str] = Counter()
        dimension_counts: dict[str, Counter[str]] = {
            name: Counter()
            for name in ("conflict_band", "decision_stage", "map_id", "source_group", "agent_band")
        }
        chosen: list[dict[str, Any]] = []
        while len(chosen) < target_per_policy:
            eligible = [
                row
                for row in candidates
                if episode_counts[str(row["episode_id"])] < max_per_episode
            ]
            if not eligible:
                break

            def score(row: dict[str, Any]) -> tuple[float, ...]:
                # Lower occupancy wins.  The ordering makes conflict/stage
                # coverage primary, followed by map/source/agent balance.
                high_band = (
                    "high_101_500"
                    if row["conflict_band"] == "extreme_501_plus"
                    else str(row["conflict_band"])
                )
                agent_target = math.ceil(0.30 * target_per_policy)
                conflict_target = math.ceil(target_per_policy / 3)
                stage_target = math.ceil(target_per_policy / 3)
                return (
                    float(
                        dimension_counts["agent_band"][str(row["agent_band"])]
                        >= agent_target
                    ),
                    float(dimension_counts["map_id"][str(row["map_id"])] > 0),
                    float(dimension_counts["conflict_band"][high_band] >= conflict_target),
                    float(
                        dimension_counts["decision_stage"][str(row["decision_stage"])]
                        >= stage_target
                    ),
                    float(dimension_counts["conflict_band"][high_band]),
                    float(dimension_counts["decision_stage"][str(row["decision_stage"])]),
                    float(dimension_counts["map_id"][str(row["map_id"])]),
                    float(dimension_counts["source_group"][str(row["source_group"])]),
                    float(dimension_counts["agent_band"][str(row["agent_band"])]),
                    float(episode_counts[str(row["episode_id"])]),
                    int(_fingerprint(_selection_identity(row))[:16], 16),
                )

            winner = min(eligible, key=score)
            chosen.append(winner)
            candidates.remove(winner)
            episode_counts[str(winner["episode_id"])] += 1
            for name, counter in dimension_counts.items():
                value = str(winner[name])
                if name == "conflict_band" and value == "extreme_501_plus":
                    value = "high_101_500"
                counter[value] += 1
        selected.extend(chosen)
        policy_reports[policy] = {
            "available_state_count": len(by_policy.get(policy, [])),
            "available_episode_count": len(
                {str(row["episode_id"]) for row in by_policy.get(policy, [])}
            ),
            "selected_state_count": len(chosen),
            "selected_episode_count": len(episode_counts),
            "max_states_in_episode": max(episode_counts.values(), default=0),
            "counts": {
                name: dict(sorted(counter.items()))
                for name, counter in dimension_counts.items()
            },
        }
    return selected, policy_reports


def build_stride_state_selection(
    *, source_roots: list[Path], output: Path, target_per_policy: int = 120,
    max_per_episode: int = 2,
) -> dict[str, Any]:
    """Build the preregistered Pilot cohort from pre-action trace state only."""

    if not source_roots:
        raise ValueError("STRIDE selection requires at least one source collection")
    if target_per_policy <= 0 or max_per_episode <= 0:
        raise ValueError("STRIDE selection limits must be positive")
    pool: list[dict[str, Any]] = []
    registered_maps: set[str] = set()
    roots = [root.resolve() for root in source_roots]
    for source_root in roots:
        run = _read_json(source_root / "run_config.json")
        dataset_root = Path(str(run["dataset"])).resolve()
        dataset = {
            str(row["task_id"]): row
            for row in _load_dataset_rows(dataset_root, [STRIDE_PILOT_SPLIT])
        }
        registered_maps.update(str(row["map_id"]) for row in dataset.values())
        for _, (manifest_name, source_policy) in STRIDE_SOURCE_POLICIES.items():
            for manifest in _read_jsonl(source_root / manifest_name):
                if manifest.get("status") != "ok":
                    continue
                task_id = str(manifest["task_id"])
                dataset_row = dataset.get(task_id)
                if dataset_row is None:
                    raise ValueError(f"STRIDE source task is absent from dataset: {task_id}")
                decisions, _ = result_blind_decision_rows(source_root, manifest)
                for decision in decisions:
                    before_conflicts = int(decision["before_conflicts"])
                    if before_conflicts <= 0:
                        continue
                    decision_index = int(decision["decision_index"])
                    identity = {
                        "source_policy": source_policy,
                        "episode_id": str(manifest["episode_id"]),
                        "map_id": str(manifest["map_id"]),
                        "task_id": task_id,
                        "solver_seed": int(manifest["solver_seed"]),
                        "decision_index": decision_index,
                        "before_fingerprint": str(decision["before_fingerprint"]),
                        "before_conflicts": before_conflicts,
                        "agent_count": int(manifest["agent_count"]),
                    }
                    pool.append(
                        {
                            "schema": STRIDE_SELECTION_SCHEMA,
                            "state_id": "stride-" + _fingerprint(identity)[:24],
                            "map_id": identity["map_id"],
                            "task_id": task_id,
                            "split": str(manifest["split"]),
                            "source_policy": source_policy,
                            "decision_stage": _decision_stage(decision_index),
                            "conflict_band": _conflict_band(before_conflicts),
                            "source_group": str(dataset_row.get("source_group", "unknown")),
                            "layout_mode": str(manifest.get("layout_mode", "unknown")),
                            "source_root": str(source_root),
                            "episode_id": str(manifest["episode_id"]),
                            "before_fingerprint": identity["before_fingerprint"],
                            "before_conflicts": before_conflicts,
                            "solver_seed": identity["solver_seed"],
                            "decision_index": decision_index,
                            "agent_count": identity["agent_count"],
                            "agent_band": _agent_band(identity["agent_count"]),
                            "prefix_actions": list(decision["prefix_actions"]),
                        }
                    )
    state_ids = [str(row["state_id"]) for row in pool]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("STRIDE source collections contain duplicate state identities")
    selected, policy_reports = _balanced_result_blind_selection(
        pool, target_per_policy=target_per_policy, max_per_episode=max_per_episode
    )
    selected.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "state_selection.jsonl", selected)
    selected_maps = {str(row["map_id"]) for row in selected}
    available_maps = {str(row["map_id"]) for row in pool}
    agent_counts = Counter(str(row["agent_band"]) for row in selected)
    total = len(selected)
    gates = {
        "target_per_policy": all(
            report["selected_state_count"] == target_per_policy
            for report in policy_reports.values()
        ),
        "episode_cap": all(
            report["max_states_in_episode"] <= max_per_episode
            for report in policy_reports.values()
        ),
        "all_available_maps": selected_maps == available_maps,
        "low_mid_agent_coverage": total > 0 and agent_counts["low_mid"] / total >= 0.30,
        "high_agent_coverage": total > 0 and agent_counts["high"] / total >= 0.30,
    }
    report = {
        "schema": STRIDE_SELECTION_SCHEMA,
        "schema_version": 1,
        "result_blind": True,
        "permitted_selection_inputs": sorted(_selection_identity(pool[0]).keys()) if pool else [],
        "forbidden_selection_inputs": [
            "actual_action", "actual_lns2", "after_fingerprint", "repair_seconds",
            "repair_state_changed", "replay_action",
        ],
        "source_roots": [str(root) for root in roots],
        "target_per_policy": target_per_policy,
        "max_per_episode": max_per_episode,
        "available_state_count": len(pool),
        "selected_state_count": total,
        "selected_map_count": len(selected_maps),
        "available_map_count": len(available_maps),
        "dataset_map_count": len(registered_maps),
        "unavailable_map_ids": sorted(registered_maps - available_maps),
        "selected_agent_band_counts": dict(sorted(agent_counts.items())),
        "policies": policy_reports,
        "gates": gates,
        "passed": all(gates.values()),
        "high_conflict_fallback": "states above 500 are retained and balance the 101-500 stratum",
    }
    _write_json(output / "state_selection_report.json", report)
    return report


def prepare_stride_pilot_dataset(
    *, generated: Path, movingai: Path, output: Path
) -> dict[str, Any]:
    """Merge the preregistered nine synthetic and six MovingAI dev maps."""

    sources = (
        ("generated", generated.resolve(), STRIDE_PILOT_SPLIT),
        ("movingai", movingai.resolve(), "balanced_wall_clock"),
    )
    manifest: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    output = output.resolve()
    for source_group, source_root, source_split in sources:
        source_manifest = source_root / source_split / "manifest.jsonl"
        source_hashes[source_group] = sha256_file(source_manifest)
        for raw in _read_jsonl(source_manifest):
            row = dict(raw)
            row["split"] = STRIDE_PILOT_SPLIT
            row["source_group"] = source_group
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source = source_root / source_split / relative
                if not source.is_file():
                    raise ValueError(f"STRIDE dataset source is missing: {source}")
                destination = output / STRIDE_PILOT_SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise ValueError(f"STRIDE dataset merge collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            manifest.append(row)

    task_ids = [str(row["task_id"]) for row in manifest]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("STRIDE Pilot task IDs are duplicated")
    by_source = Counter(str(row["source_group"]) for row in manifest)
    maps_by_source = {
        group: {str(row["map_id"]) for row in manifest if row["source_group"] == group}
        for group in ("generated", "movingai")
    }
    tasks_by_map = Counter(str(row["map_id"]) for row in manifest)
    if by_source != Counter({"generated": 72, "movingai": 24}):
        raise ValueError(f"STRIDE Pilot source task counts differ: {dict(by_source)}")
    if len(maps_by_source["generated"]) != 9 or len(maps_by_source["movingai"]) != 6:
        raise ValueError("STRIDE Pilot requires nine generated and six MovingAI maps")
    if any(
        tasks_by_map[map_id] != expected
        for group, expected in (("generated", 8), ("movingai", 4))
        for map_id in maps_by_source[group]
    ):
        raise ValueError("STRIDE Pilot tasks per map differ from registration")
    agent_counts = Counter(int(row["agent_count"]) for row in manifest)
    low_mid = sum(count for agents, count in agent_counts.items() if 80 <= agents <= 200)
    high = sum(count for agents, count in agent_counts.items() if 400 <= agents <= 600)
    if low_mid / len(manifest) < 0.30 or high / len(manifest) < 0.30:
        raise ValueError("STRIDE Pilot agent-band coverage is below 30 percent")
    formal_ids = {
        "den312d",
        "lak303d",
        "maze-128-128-1",
        "maze-128-128-10",
        "maze-32-32-4",
        "random-32-32-10",
        "random-64-64-10",
        "random-64-64-20",
        "room-64-64-16",
        "room-64-64-8",
        "warehouse-10-20-10-2-2",
        "warehouse-20-40-10-2-2",
    }
    current_ids = {str(row["map_id"]) for row in manifest}
    overlap = sorted(current_ids & formal_ids)
    if overlap:
        raise ValueError(f"STRIDE Pilot leaks formal MovingAI maps: {overlap}")

    manifest.sort(key=lambda row: str(row["task_id"]))
    split_root = output / STRIDE_PILOT_SPLIT
    split_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema": "lns2.stride.pilot_dataset.v1",
        "split": STRIDE_PILOT_SPLIT,
        "map_count": 15,
        "task_count": len(manifest),
        "source_task_counts": dict(sorted(by_source.items())),
        "source_map_counts": {
            key: len(value) for key, value in sorted(maps_by_source.items())
        },
        "agent_counts": {str(key): value for key, value in sorted(agent_counts.items())},
        "low_mid_agent_fraction": low_mid / len(manifest),
        "high_agent_fraction": high / len(manifest),
        "formal_map_overlap": overlap,
        "source_manifest_sha256": source_hashes,
    }
    _write_json(output / "dataset_summary.json", summary)
    return summary


def stride_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    if trial_index not in PP_TRIAL_INDICES:
        raise ValueError("STRIDE PP trial index must be 0, 1, 2, or 3")
    return int(
        _fingerprint(
            {
                "namespace": "stride-lns-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _selection_row_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_strings = (
        "state_id",
        "map_id",
        "task_id",
        "split",
        "source_policy",
        "decision_stage",
        "source_root",
        "before_fingerprint",
    )
    for name in required_strings:
        if not isinstance(row.get(name), str) or not str(row[name]):
            errors.append(f"{name} must be a non-empty string")
    for name in ("solver_seed", "decision_index", "agent_count"):
        value = row.get(name)
        if type(value) is not int or int(value) < 0:
            errors.append(f"{name} must be a nonnegative integer")
    if int(row.get("agent_count", 0)) <= 0:
        errors.append("agent_count must be positive")
    prefix = row.get("prefix_actions")
    if not isinstance(prefix, list) or any(not isinstance(item, dict) for item in prefix):
        errors.append("prefix_actions must be a list of action objects")
    return errors


def load_stride_selection(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("STRIDE state selection is empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema") not in {None, STRIDE_SELECTION_SCHEMA}:
            raise ValueError(f"selection row {index} has an unsupported schema")
        errors = _selection_row_errors(row)
        if errors:
            raise ValueError(f"selection row {index}: {'; '.join(errors)}")
        state_id = str(row["state_id"])
        if state_id in seen:
            raise ValueError(f"duplicate selected state: {state_id}")
        seen.add(state_id)
    return rows


def _replay_job(decision: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(str(decision["source_root"])).resolve()
    run = _read_json(source_root / "run_config.json")
    dataset_root = Path(str(run["dataset"])).resolve()
    matches = [
        row
        for row in _load_dataset_rows(dataset_root, [str(decision["split"])])
        if str(row["task_id"]) == str(decision["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected task must resolve exactly once: {decision['task_id']}"
        )
    configuration = dict(run["configuration"])
    environment = dict(configuration["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        len(decision["prefix_actions"]) + 1,
    )
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "proposal": dict(configuration["proposal"]),
        "solver_seed": int(decision["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }


def _paired_action(agents: list[int], seed: int) -> dict[str, Any]:
    if not agents:
        raise ValueError("STRIDE explicit neighborhood cannot be empty")
    return {
        "mode": "explicit_neighborhood",
        "agents": list(map(int, agents)),
        "random_seed": int(seed),
        "pp_random_seed": int(seed),
    }


def _validate_native_repair(
    result: dict[str, Any], *, expected_agents: list[int], expected_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = dict(result["observation"])
    metrics = dict(result["metrics"])
    if metrics.get("step_applied") is not True:
        raise RuntimeError("native deadline ended before STRIDE repair was applied")
    if not isinstance(metrics.get("replan_success"), bool):
        raise RuntimeError("native repair omitted strict replan_success")
    neighborhood = metrics.get("neighborhood")
    if not isinstance(neighborhood, list) or sorted(map(int, neighborhood)) != sorted(
        map(int, expected_agents)
    ):
        raise RuntimeError("native neighborhood differs from STRIDE candidate")
    requested = metrics.get("requested_pp_random_seed")
    if type(requested) is not int or int(requested) != int(expected_seed):
        raise RuntimeError("native requested PP seed differs from STRIDE seed")
    repair_order = metrics.get("repair_order")
    if not isinstance(repair_order, list):
        raise RuntimeError("native repair_order is missing")
    applied = metrics.get("applied_pp_random_seed")
    expected_applied = int(expected_seed) if repair_order else -1
    if type(applied) is not int or int(applied) != expected_applied:
        raise RuntimeError("native applied PP seed differs from STRIDE seed")
    terminated = result.get("terminated")
    truncated = result.get("truncated")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise RuntimeError("native repair omitted strict terminal flags")
    if bool(state.get("done")) != bool(terminated or truncated):
        raise RuntimeError("native done flag disagrees with terminal flags")
    if bool(state.get("feasible")) != bool(terminated):
        raise RuntimeError("native feasible flag disagrees with termination")
    return state, metrics


def _state_artifact_valid(
    payload: dict[str, Any], *, run_fingerprint: str, state_id: str,
    decision: dict[str, Any] | None = None,
) -> bool:
    if (
        payload.get("schema") != STRIDE_COLLECTION_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
        or payload.get("schema_version") != 1
    ):
        return False
    embedded_decision = payload.get("decision")
    if not isinstance(embedded_decision, dict):
        return False
    if decision is not None and embedded_decision != decision:
        return False
    expected_decision = decision if decision is not None else embedded_decision
    if expected_decision.get("state_id") != state_id:
        return False
    before_fingerprint = expected_decision.get("before_fingerprint")
    before_repair_fingerprint = payload.get("before_repair_fingerprint")
    before_conflicts = payload.get("before_conflicts")
    if (
        not isinstance(before_fingerprint, str)
        or not before_fingerprint
        or payload.get("before_fingerprint") != before_fingerprint
        or not isinstance(before_repair_fingerprint, str)
        or not before_repair_fingerprint
        or not strict_integer(before_conflicts, minimum=1)
        or expected_decision.get("before_conflicts") != before_conflicts
    ):
        return False
    candidates = candidate_records(payload.get("candidates"))
    trials = payload.get("trials")
    if candidates is None or not isinstance(trials, list):
        return False
    if any(
        any(agent >= int(expected_decision["agent_count"]) for agent in row["agents"])
        for row in candidates.values()
    ):
        return False
    if not trial_product_matches(
        trials,
        candidate_ids=tuple(candidates),
        trial_indices=PP_TRIAL_INDICES,
    ):
        return False
    metadata = {
        name: expected_decision[name]
        for name in (
            "map_id",
            "split",
            "source_policy",
            "decision_stage",
            "agent_count",
        )
    }
    features_by_candidate: dict[str, dict[str, Any]] = {}
    for row in trials:
        candidate_id = str(row["candidate_id"])
        trial_index = int(row["trial_index"])
        if not repair_trial_semantics_valid(
            row,
            schema=STRIDE_TRIAL_SCHEMA,
            state_id=state_id,
            candidate_id=candidate_id,
            trial_index=trial_index,
            pp_seed=stride_pp_seed(before_repair_fingerprint, trial_index),
            before_conflicts=before_conflicts,
            before_fingerprint=before_fingerprint,
            before_repair_fingerprint=before_repair_fingerprint,
            feature_schema_id=FROZEN_FEATURE_SCHEMA_ID,
            required_feature_names=PROFILE_FEATURE_NAMES["realized_dynamic"],
            expected_metadata=metadata,
        ):
            return False
        features = dict(row["features"])
        previous = features_by_candidate.setdefault(candidate_id, features)
        if previous != features:
            return False
    return True


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(
            existing,
            run_fingerprint=run_fingerprint,
            state_id=str(decision["state_id"]),
            decision=decision,
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "candidate_count": len(existing["candidates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"completed STRIDE state artifact is invalid: {output_path}")

    replay = _replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    initial_fingerprint = state_fingerprint(state)
    if initial_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"STRIDE replay mismatch for {decision['state_id']}: "
            f"expected {decision['before_fingerprint']}, got {initial_fingerprint}"
        )
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if before_conflicts <= 0 or bool(state.get("done")):
        raise ValueError("STRIDE selected state must be active and conflicting")
    state_summary = summarize_initial_state_complexity(state)
    proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=initial_fingerprint,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    feature_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    feature_rows, feature_metrics = feature_engine.realized_rows(
        candidates, state_hash=initial_fingerprint
    )
    features_by_candidate = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    if len(candidates) != len(features_by_candidate):
        raise RuntimeError("STRIDE candidate and feature counts differ")
    if any(len(values) != FROZEN_FEATURE_DIMENSION for values in features_by_candidate.values()):
        raise RuntimeError("STRIDE realized feature dimension differs from frozen V2")

    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        for trial_index in PP_TRIAL_INDICES:
            branch_environment, branch_state = replay_prefix(
                replay, decision["prefix_actions"]
            )
            branch_fingerprint = state_fingerprint(branch_state)
            if branch_fingerprint != initial_fingerprint:
                raise RuntimeError("STRIDE paired branch replay fingerprint changed")
            seed = stride_pp_seed(initial_repair_fingerprint, trial_index)
            action = _paired_action(agents, seed)
            result = _plain(branch_environment.step(action))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            repair_outcome = classify_repair_outcome(
                before_fingerprint=initial_repair_fingerprint,
                after_fingerprint=after_repair_fingerprint,
                replan_success=bool(metrics["replan_success"]),
                conflicts_before=before_conflicts,
                conflicts_after=conflicts_after,
                feasible=bool(after.get("feasible")),
            )
            trials.append(
                {
                    "schema": STRIDE_TRIAL_SCHEMA,
                    "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                    "state_id": str(decision["state_id"]),
                    "candidate_id": candidate_id,
                    "map_id": str(decision["map_id"]),
                    "split": str(decision["split"]),
                    "source_policy": str(decision["source_policy"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "agent_count": int(decision["agent_count"]),
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": initial_fingerprint,
                    "before_repair_fingerprint": initial_repair_fingerprint,
                    "features": features_by_candidate[candidate_id],
                    "trial_index": trial_index,
                    "pp_seed": seed,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": repair_outcome,
                    "conflicts_after": conflicts_after,
                    "after_fingerprint": state_fingerprint(after),
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "post_structure": post_structure_metrics(after),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
                }
            )

    payload = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": initial_fingerprint,
        "before_repair_fingerprint": initial_repair_fingerprint,
        "before_conflicts": before_conflicts,
        "state_summary": state_summary,
        "candidate_generation": generation,
        "proposal": proposal,
        "feature_metrics": feature_metrics,
        "candidates": candidates,
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "candidate_count": len(candidates),
        "trial_count": len(trials),
    }


def collect_stride_repairs(
    *,
    selection_path: Path,
    output: Path,
    workers: int,
    resume: bool,
    max_states: int | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("STRIDE workers must be positive")
    selected = load_stride_selection(selection_path)
    if max_states is not None:
        if max_states <= 0:
            raise ValueError("STRIDE max_states must be positive")
        selected = selected[:max_states]
    project_root = Path(__file__).resolve().parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=STRIDE_COLLECTION_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_configs = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "proposal": FULL_POOL_PROPOSAL,
        "source_run_config_sha256": dict(sorted(source_run_configs.items())),
        "pp_trial_indices": list(PP_TRIAL_INDICES),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = output.resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("STRIDE output belongs to another collection identity")
        if not resume:
            raise ValueError("STRIDE output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(
        output / "state_selection.jsonl",
        [{**row, "schema": STRIDE_SELECTION_SCHEMA} for row in selected],
    )

    jobs = []
    for row in selected:
        key = _fingerprint(
            {"state_id": row["state_id"], "before": row["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "decision": row,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_collect_state, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["decision"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output / "collection_status.json",
                {
                    "schema": STRIDE_COLLECTION_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "requested_state_count": len(jobs),
                    "completed_state_count": len(results),
                    "error_state_count": len(errors),
                    "errors": errors,
                    "status": "running",
                },
            )

    state_files = [Path(row["state_file"]) for row in results]
    all_trials: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    selected_by_id = {str(row["state_id"]): row for row in selected}
    for state_file in sorted(state_files):
        payload = _read_json(state_file)
        state_id = str(payload.get("state_id", ""))
        if not _state_artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            state_id=state_id,
            decision=selected_by_id.get(state_id),
        ):
            errors.append(
                {"state_id": str(payload.get("state_id", "")), "error": "invalid state artifact"}
            )
            continue
        candidate_counts[len(payload["candidates"])] += 1
        all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output / "repair_trials.jsonl", all_trials)
    report = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "selection_sha256": identity["selection_sha256"],
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "trial_count": len(all_trials),
        "expected_trial_count": sum(key * value * 4 for key, value in candidate_counts.items()),
        "complete": complete,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(
        output / "collection_status.json",
        {**report, "status": "complete" if complete else "error"},
    )
    return report
