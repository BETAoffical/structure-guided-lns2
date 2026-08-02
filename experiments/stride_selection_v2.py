from __future__ import annotations

import concurrent.futures
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    STRIDE_PILOT_SPLIT,
    STRIDE_SELECTION_SCHEMA,
    STRIDE_SOURCE_POLICIES,
    _agent_band,
    _balanced_result_blind_selection,
    _conflict_band,
    _decision_stage,
    _replay_job,
)
from experiments.trace_replay import replay_prefix, result_blind_decision_rows
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates


STRIDE_SELECTION_V2_SCHEMA = "lns2.stride.state_selection.v2"
STRIDE_PREFLIGHT_SCHEMA = "lns2.stride.selection_preflight.v1"
MAX_SOURCE_DECISION_INDEX = 11


def _excluded_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = _read_json(path)
    if isinstance(payload, list):
        values = payload
    else:
        values = payload.get("excluded_state_ids", payload.get("failed_state_ids", []))
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("STRIDE exclusion report must contain a string state-ID list")
    return set(values)


def build_stride_state_selection_v2(
    *,
    source_roots: list[Path],
    output: Path,
    exclusion_report: Path | None = None,
    target_per_policy: int = 120,
    max_per_episode: int = 2,
    split: str = STRIDE_PILOT_SPLIT,
) -> dict[str, Any]:
    """Build a result-blind cohort restricted to source decisions 0--11."""

    excluded = _excluded_ids(exclusion_report)
    pool: list[dict[str, Any]] = []
    registered_maps: set[str] = set()
    registered_splits: set[str] = set()
    roots = [root.resolve() for root in source_roots]
    for source_root in roots:
        run = _read_json(source_root / "run_config.json")
        dataset_root = Path(str(run["dataset"])).resolve()
        source_split = str(run["configuration"].get("split", ""))
        if not source_split:
            raise ValueError("STRIDE source omits its registered split")
        if split != "auto" and source_split != split:
            raise ValueError(
                f"STRIDE source split mismatch: expected {split}, "
                f"found {source_split}"
            )
        registered_splits.add(source_split)
        dataset = {
            str(row["task_id"]): row
            for row in _load_dataset_rows(dataset_root, [source_split])
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
                    decision_index = int(decision["decision_index"])
                    before_conflicts = int(decision["before_conflicts"])
                    if before_conflicts <= 0 or decision_index > MAX_SOURCE_DECISION_INDEX:
                        continue
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
                    state_id = "stride-" + _fingerprint(identity)[:24]
                    if state_id in excluded:
                        continue
                    pool.append(
                        {
                            "schema": STRIDE_SELECTION_SCHEMA,
                            "state_id": state_id,
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
    selected, policy_reports = _balanced_result_blind_selection(
        pool, target_per_policy=target_per_policy, max_per_episode=max_per_episode
    )
    selected.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
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
        "source_decision_cap": all(
            int(row["decision_index"]) <= MAX_SOURCE_DECISION_INDEX for row in selected
        ),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "state_selection.jsonl", selected)
    report = {
        "schema": STRIDE_SELECTION_V2_SCHEMA,
        "schema_version": 2,
        "result_blind": True,
        "registered_split": split,
        "registered_source_splits": sorted(registered_splits),
        "source_decision_index_range": [0, MAX_SOURCE_DECISION_INDEX],
        "excluded_state_ids": sorted(excluded),
        "source_roots": [str(root) for root in roots],
        "available_state_count": len(pool),
        "selected_state_count": total,
        "dataset_map_count": len(registered_maps),
        "available_map_count": len(available_maps),
        "selected_map_count": len(selected_maps),
        "unavailable_map_ids": sorted(registered_maps - available_maps),
        "selected_agent_band_counts": dict(sorted(agent_counts.items())),
        "policies": policy_reports,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output / "state_selection_report.json", report)
    return report


def _preflight_once(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    replay = _replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    observed = state_fingerprint(state)
    if observed != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"preflight fingerprint mismatch: expected {decision['before_fingerprint']}, got {observed}"
        )
    if bool(state.get("done")) or int(state["num_of_colliding_pairs"]) <= 0:
        raise RuntimeError("preflight target is terminal or conflict-free")
    repair_fingerprint = repair_structure_fingerprint(state)
    proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    candidates, _ = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=observed,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    signature = _fingerprint(
        [
            {"candidate_id": str(candidate["candidate_id"]), "agents": candidate["agents"]}
            for candidate in candidates
        ]
    )
    return {
        "state_id": str(decision["state_id"]),
        "repetition": int(job["repetition"]),
        "before_fingerprint": observed,
        "repair_fingerprint": repair_fingerprint,
        "candidate_count": len(candidates),
        "candidate_signature": signature,
    }


def preflight_stride_selection(
    *, selection_path: Path, output: Path, workers: int = 4, repetitions: int = 3
) -> dict[str, Any]:
    if workers <= 0 or repetitions < 2:
        raise ValueError("STRIDE preflight requires positive workers and at least two repetitions")
    selected = _read_jsonl(selection_path)
    jobs = [
        {"decision": row, "repetition": repetition}
        for row in selected
        for repetition in range(repetitions)
    ]
    results: dict[str, list[dict[str, Any]]] = {str(row["state_id"]): [] for row in selected}
    errors: dict[str, list[str]] = {str(row["state_id"]): [] for row in selected}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_preflight_once, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            state_id = str(job["decision"]["state_id"])
            try:
                results[state_id].append(future.result())
            except Exception as error:
                errors[state_id].append(f"{type(error).__name__}: {error}")
    failed: list[str] = []
    state_reports: list[dict[str, Any]] = []
    for state_id in sorted(results):
        rows = results[state_id]
        signatures = {
            (row["before_fingerprint"], row["repair_fingerprint"], row["candidate_signature"])
            for row in rows
        }
        passed = len(rows) == repetitions and not errors[state_id] and len(signatures) == 1
        if not passed:
            failed.append(state_id)
        state_reports.append(
            {
                "state_id": state_id,
                "passed": passed,
                "successful_repetitions": len(rows),
                "errors": errors[state_id],
                "signature_count": len(signatures),
                "candidate_counts": sorted({row["candidate_count"] for row in rows}),
            }
        )
    report = {
        "schema": STRIDE_PREFLIGHT_SCHEMA,
        "schema_version": 1,
        "selection": str(selection_path.resolve()),
        "selection_state_count": len(selected),
        "repetitions": repetitions,
        "passed_state_count": len(selected) - len(failed),
        "failed_state_count": len(failed),
        "failed_state_ids": failed,
        "passed": not failed,
        "states": state_reports,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "preflight_report.json", report)
    return report
