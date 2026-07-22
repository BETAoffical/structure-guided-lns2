from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.high_load_rescue import _state_job, is_no_progress_decision
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.trace_replay import decision_rows


V3_PILOT_COLLECTION_SCHEMA = "lns2.v3_pilot_collection.v1"
PILOT_NEIGHBORHOOD_SIZES = (4, 8, 16)
PILOT_STRATA = ("ordinary_progress", "high_cost_progress", "no_progress")
PILOT_SPLIT_QUOTAS = {"policy_train": 48, "policy_validation": 12}


def decision_stage(decision_index: int) -> str:
    index = int(decision_index)
    if index < 0:
        raise ValueError("decision index cannot be negative")
    if index < 10:
        return "early"
    if index < 20:
        return "middle"
    return "late"


def _stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{row['split']}|{row['map_id']}|{row['task_id']}|"
            f"{row['solver_seed']}|{row['decision_index']}|"
            f"{row['before_repair_fingerprint']}"
        ).encode("utf-8")
    ).hexdigest()


def _source_decisions(source_root: Path, split: str) -> list[dict[str, Any]]:
    run = _read_json(source_root / "run_config.json")
    if str(run.get("controller")) != "v2-full":
        raise ValueError("v3 pilot source must use v2-full")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in _read_jsonl(source_root / "realized_dynamic_manifest.jsonl"):
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        decisions, _events = decision_rows(source_root, manifest)
        for decision in decisions:
            fingerprint = str(decision["before_repair_fingerprint"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            outcome = dict(decision["actual_lns2"]["outcome"])
            result.append(
                {
                    **decision,
                    "split": split,
                    "source_root": str(source_root),
                    "source_run_fingerprint": str(run["run_fingerprint"]),
                    "map_id": str(manifest["map_id"]),
                    "layout_mode": str(manifest.get("layout_mode", "unknown")),
                    "agent_count": int(manifest["agent_count"]),
                    "task_id": str(manifest["task_id"]),
                    "solver_seed": int(manifest["solver_seed"]),
                    "episode_id": str(manifest["episode_id"]),
                    "decision_stage": decision_stage(int(decision["decision_index"])),
                    "source_controller_seconds": float(
                        outcome.get("controller_seconds", 0.0)
                    ),
                    "source_repair_seconds": float(outcome["repair_seconds"]),
                }
            )
    if not result:
        raise ValueError(f"v3 pilot source has no usable decisions: {split}")
    return result


def _assign_strata(
    rows: list[dict[str, Any]], *, minimum_high_cost_per_cell: int
) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["layout_mode"]), int(row["agent_count"]))].append(row)
    for cell_rows in grouped.values():
        progressing = []
        for row in cell_rows:
            if is_no_progress_decision(row):
                row["source_stratum"] = "no_progress"
            else:
                progressing.append(row)
        progressing.sort(
            key=lambda row: (
                -float(row["source_repair_seconds"]),
                _stable_key(row),
            )
        )
        high_count = min(
            len(progressing),
            max(minimum_high_cost_per_cell, math.ceil(0.25 * len(progressing))),
        )
        for index, row in enumerate(progressing):
            row["source_stratum"] = (
                "high_cost_progress" if index < high_count else "ordinary_progress"
            )


def _balanced_take(rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    if len(rows) < quota:
        return []
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    stage_counts: collections.Counter[str] = collections.Counter()
    map_counts: collections.Counter[str] = collections.Counter()
    cell_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    while remaining and len(selected) < quota:
        chosen = min(
            remaining,
            key=lambda row: (
                cell_counts[(str(row["layout_mode"]), int(row["agent_count"]))],
                stage_counts[str(row["decision_stage"])],
                map_counts[str(row["map_id"])],
                _stable_key(row),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        stage_counts[str(chosen["decision_stage"])] += 1
        map_counts[str(chosen["map_id"])] += 1
        cell_counts[
            (str(chosen["layout_mode"]), int(chosen["agent_count"]))
        ] += 1
    return selected


def select_v3_pilot_states(source: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(source).resolve()
    selected: list[dict[str, Any]] = []
    shortages = []
    source_counts = {}
    for split, quota in PILOT_SPLIT_QUOTAS.items():
        source_root = root / "sources" / split
        rows = _source_decisions(source_root, split)
        _assign_strata(rows, minimum_high_cost_per_cell=1)
        source_counts[split] = len(rows)
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            grouped[str(row["source_stratum"])].append(row)
        for stratum in PILOT_STRATA:
            available = grouped.get(stratum, [])
            values = _balanced_take(available, quota)
            if len(values) != quota:
                shortages.append(
                    {
                        "split": split,
                        "source_stratum": stratum,
                        "required": quota,
                        "available": len(available),
                    }
                )
                continue
            selected.extend(values)
    if shortages:
        raise ValueError(f"v3 pilot source lacks registered cell coverage: {shortages}")
    expected = sum(
        quota * len(PILOT_STRATA) for quota in PILOT_SPLIT_QUOTAS.values()
    )
    if len(selected) != expected:
        raise ValueError(
            f"v3 pilot selected {len(selected)} states instead of {expected}"
        )
    fingerprints = [str(row["before_repair_fingerprint"]) for row in selected]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("v3 pilot selected duplicate repair states")
    for index, row in enumerate(sorted(selected, key=_stable_key)):
        row["state_id"] = (
            f"v3__{row['split']}__{row['episode_id']}__"
            f"decision_{int(row['decision_index']):04d}__{index:04d}"
        )
    selected.sort(key=lambda row: (str(row["split"]), str(row["state_id"])))
    cell_counts = collections.Counter(
        (
            str(item["split"]),
            str(item["layout_mode"]),
            int(item["agent_count"]),
            str(item["source_stratum"]),
        )
        for item in selected
    )
    expected_cells = {
        (split, layout, agents)
        for split in PILOT_SPLIT_QUOTAS
        for layout in {str(row["layout_mode"]) for row in selected}
        for agents in {int(row["agent_count"]) for row in selected}
    }
    selected_cells = {
        (str(row["split"]), str(row["layout_mode"]), int(row["agent_count"]))
        for row in selected
    }
    if selected_cells != expected_cells:
        missing = sorted(expected_cells - selected_cells)
        raise ValueError(f"v3 pilot selection lacks layout/agent coverage: {missing}")
    train_maps = {
        str(row["map_id"]) for row in selected if row["split"] == "policy_train"
    }
    diagnostic_maps = {
        str(row["map_id"])
        for row in selected
        if row["split"] == "policy_validation"
    }
    map_overlap = sorted(train_maps & diagnostic_maps)
    if map_overlap:
        raise ValueError(f"v3 train and diagnostic maps overlap: {map_overlap}")
    report = {
        "schema": V3_PILOT_COLLECTION_SCHEMA,
        "source": str(root),
        "source_decision_count_by_split": source_counts,
        "selected_state_count": len(selected),
        "selected_state_count_by_split": dict(
            collections.Counter(str(row["split"]) for row in selected)
        ),
        "selected_state_count_by_cell": {
            "|".join(map(str, key)): value
            for key, value in sorted(cell_counts.items())
        },
        "selected_state_count_by_stage": dict(
            collections.Counter(str(row["decision_stage"]) for row in selected)
        ),
        "selected_state_count_by_stratum": dict(
            collections.Counter(str(row["source_stratum"]) for row in selected)
        ),
        "train_diagnostic_map_overlap": False,
        "selection_overhead_seconds": statistics.median(
            float(row["source_controller_seconds"])
            for row in selected
            if float(row["source_controller_seconds"]) >= 0.0
        ),
        "shortages": shortages,
        "passed": not shortages,
    }
    return selected, report


def _coverage_report(
    decisions: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    initial_trials: int,
) -> dict[str, Any]:
    features_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    trials_by_arm: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in feature_rows:
        features_by_state[str(row["state_id"])].append(row)
    for row in trial_rows:
        trials_by_arm[(str(row["state_id"]), str(row["candidate_id"]))] += 1
    errors = []
    for decision in decisions:
        state_id = str(decision["state_id"])
        candidates = features_by_state.get(state_id, [])
        represented_sizes = {
            int(str(family).rsplit(":", 1)[-1])
            for row in candidates
            if row["route"] == "model"
            for family in row.get("selection_families", ())
            if str(family).rsplit(":", 1)[-1].isdigit()
        }
        adaptive = [row for row in candidates if row["route"] == "official_adaptive"]
        if represented_sizes != set(PILOT_NEIGHBORHOOD_SIZES):
            errors.append(
                f"{state_id}: represented sizes={sorted(represented_sizes)}"
            )
        if len(adaptive) != 1:
            errors.append(f"{state_id}: adaptive_count={len(adaptive)}")
        if sum(bool(row.get("base_selected")) for row in candidates) != 1:
            errors.append(f"{state_id}: base selection coverage")
        for candidate in candidates:
            if trials_by_arm[(state_id, str(candidate["candidate_id"]))] < initial_trials:
                errors.append(
                    f"{state_id}/{candidate['candidate_id']}: insufficient paired trials"
                )
    return {
        "state_count": len(decisions),
        "feature_state_count": len(features_by_state),
        "candidate_count": len(feature_rows),
        "trial_count": len(trial_rows),
        "error_count": len(errors),
        "errors": errors,
        "passed": not errors and len(features_by_state) == len(decisions),
    }


def collect_v3_pilot_data(
    *,
    source: str | Path,
    output: str | Path,
    controller_bundle: str | Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    decisions, selection = select_v3_pilot_states(source_root)
    source_fingerprints = {
        split: _read_json(source_root / "sources" / split / "run_config.json")[
            "run_fingerprint"
        ]
        for split in PILOT_SPLIT_QUOTAS
    }
    identity = {
        "schema": V3_PILOT_COLLECTION_SCHEMA,
        "source": str(source_root),
        "source_fingerprints": source_fingerprints,
        "controller_bundle": str(Path(controller_bundle).resolve()),
        "selected_state_fingerprint": _fingerprint(
            [
                {
                    "state_id": row["state_id"],
                    "fingerprint": row["before_repair_fingerprint"],
                }
                for row in decisions
            ]
        ),
        "neighborhood_sizes": list(PILOT_NEIGHBORHOOD_SIZES),
        "initial_trials": 2,
        "maximum_trials": 4,
        "feature_backend": "native",
        "controller_runtime": "optimized",
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("v3 pilot collection output belongs to a different run")
        if not resume:
            raise ValueError("v3 pilot collection exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output_root / "state_selection.jsonl", decisions)
    _write_json(output_root / "state_selection_report.json", selection)

    jobs = []
    for decision in decisions:
        state_key = _fingerprint(
            {
                "split": decision["split"],
                "state_id": decision["state_id"],
                "fingerprint": decision["before_repair_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "decision": decision,
                "state_file": str(
                    output_root
                    / "states"
                    / str(decision["split"])
                    / f"{state_key}.json"
                ),
                "controller_bundle": str(Path(controller_bundle).resolve()),
                "neighborhood_sizes": list(PILOT_NEIGHBORHOOD_SIZES),
                "initial_trials": 2,
                "maximum_trials": 4,
                "feature_backend": "native",
                "controller_runtime": "optimized",
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    collection_started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_state_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["decision"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            elapsed_seconds = time.perf_counter() - collection_started
            finished_count = len(completed) + len(errors)
            states_per_minute = (
                60.0 * finished_count / elapsed_seconds
                if elapsed_seconds > 0.0
                else 0.0
            )
            remaining = len(jobs) - finished_count
            _write_json(
                output_root / "status.json",
                {
                    "schema": V3_PILOT_COLLECTION_SCHEMA,
                    "status": "running",
                    "completed_states": len(completed),
                    "total_states": len(jobs),
                    "error_states": len(errors),
                    "elapsed_seconds": elapsed_seconds,
                    "states_per_minute": states_per_minute,
                    "estimated_remaining_seconds": (
                        60.0 * remaining / states_per_minute
                        if states_per_minute > 0.0
                        else None
                    ),
                },
            )
    feature_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for row in completed:
        payload = _read_json(Path(row["state_file"]))
        feature_rows.extend(payload["candidates"])
        trial_rows.extend(payload["trials"])
    feature_rows.sort(
        key=lambda row: (str(row["split"]), str(row["state_id"]), str(row["candidate_id"]))
    )
    trial_rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["trial_index"]),
        )
    )
    _write_jsonl(output_root / "feature_index.jsonl", feature_rows)
    _write_jsonl(output_root / "trial_manifest.jsonl", trial_rows)
    coverage = _coverage_report(decisions, feature_rows, trial_rows, 2)
    _write_json(output_root / "coverage_report.json", coverage)
    report = {
        "schema": V3_PILOT_COLLECTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(completed),
        "error_state_count": len(errors),
        "errors": errors,
        "selection": selection,
        "coverage": coverage,
        "feature_index_sha256": sha256_file(output_root / "feature_index.jsonl"),
        "trial_manifest_sha256": sha256_file(output_root / "trial_manifest.jsonl"),
        "complete": not errors and len(completed) == len(jobs) and coverage["passed"],
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_PILOT_COLLECTION_SCHEMA,
            "status": "complete" if report["complete"] else "error",
            "completed_states": len(completed),
            "total_states": len(jobs),
            "error_states": len(errors),
        },
    )
    return report


__all__ = [
    "PILOT_NEIGHBORHOOD_SIZES",
    "V3_PILOT_COLLECTION_SCHEMA",
    "collect_v3_pilot_data",
    "decision_stage",
    "select_v3_pilot_states",
]
