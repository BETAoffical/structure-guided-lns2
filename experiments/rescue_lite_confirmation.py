from __future__ import annotations

import collections
import concurrent.futures
import csv
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.closed_loop_confirmation import (
    generate_online_candidates,
    run_closed_loop_collection,
    score_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.high_load_rescue import select_failure_decisions
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_aware import (
    adaptive_feature_row,
    classify_repair_outcome,
    load_repair_aware_bundle,
    repair_aware_order,
)
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.rescue_policy_audit import (
    _aggregate,
    _comparison,
    _simulate_sequence,
    enumerate_rescue_policies,
)
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import replay_prefix
from generators.config import load_json
from generators.dataset import generate_dataset


RESCUE_LITE_CONFIRMATION_SCHEMA = "lns2.rescue_lite_confirmation.v1"
CONFIRMATION_SPLIT = "policy_confirmation"
LAYOUTS = ("compartmentalized", "dead_end_aisles", "regular_beltway")
AGENT_COUNTS = (400, 600)
SIZES = (4, 8, 16)
FROZEN_POLICY_ID = "4>8>adaptive"
DEFAULT_STATE_QUOTA_PER_CELL = 5
DEFAULT_TRIALS = 4


def confirmation_seed(repair_fingerprint: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "rescue-lite-confirmation-paired-v1",
                "repair_fingerprint": str(repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _cell(layout: str, agent_count: int) -> str:
    if layout not in LAYOUTS or int(agent_count) not in AGENT_COUNTS:
        raise ValueError(f"unexpected confirmation cell: {layout}/{agent_count}")
    return f"{layout}__agents_{int(agent_count)}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty confirmation CSV: {path.name}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_status(root: Path, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {"schema": RESCUE_LITE_CONFIRMATION_SCHEMA, **values},
    )


def dataset_map_hashes(dataset: Path, split: str) -> dict[str, str]:
    result = {}
    for row in _read_jsonl(dataset / split / "manifest.jsonl"):
        map_id = str(row["map_id"])
        result.setdefault(map_id, sha256_file(dataset / split / str(row["map_file"])))
    return result


def validate_dataset_isolation(
    dataset: str | Path,
    references: Iterable[str | Path],
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    current = dataset_map_hashes(dataset_root, CONFIRMATION_SPLIT)
    overlaps = []
    reference_rows = []
    for value in references:
        root = Path(value).resolve()
        if not root.is_dir():
            continue
        summary = dict(read_json(root / "dataset_summary.json"))
        hashes = set()
        for split in dict(summary["splits"]):
            hashes.update(dataset_map_hashes(root, str(split)).values())
        shared = sorted(set(current.values()) & hashes)
        overlaps.extend(shared)
        reference_rows.append(
            {
                "dataset": str(root),
                "master_seed": int(summary["master_seed"]),
                "map_count": len(hashes),
                "shared_map_hash_count": len(shared),
            }
        )
    report = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "passed": not overlaps,
        "confirmation_map_count": len(current),
        "confirmation_map_hashes": current,
        "reference_datasets": reference_rows,
        "shared_map_hashes": sorted(set(overlaps)),
    }
    if overlaps:
        raise ValueError("confirmation maps overlap previous pilot maps")
    return report


def confirmation_task_waves(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    maps_by_layout: dict[str, list[str]] = collections.defaultdict(list)
    tasks_by_map: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        layout = str(row["layout_mode"])
        map_id = str(row["map_id"])
        if map_id not in maps_by_layout[layout]:
            maps_by_layout[layout].append(map_id)
        tasks_by_map[map_id].append(str(row["task_id"]))
    initial_maps = set()
    expansion_maps = set()
    for layout in LAYOUTS:
        maps = sorted(maps_by_layout.get(layout, []))
        if len(maps) != 4:
            raise ValueError(f"confirmation dataset requires four {layout} maps")
        initial_maps.update(maps[:2])
        expansion_maps.update(maps[2:])
    initial = sorted(
        task for map_id in initial_maps for task in tasks_by_map[map_id]
    )
    expansion = sorted(
        task for map_id in expansion_maps for task in tasks_by_map[map_id]
    )
    if len(initial) != 24 or len(expansion) != 24:
        raise ValueError("confirmation task waves must each contain 24 tasks")
    return initial, expansion


def build_confirmation_source_config(
    *,
    dataset: str | Path,
    output: str | Path,
    project_root: str | Path,
    split: str = CONFIRMATION_SPLIT,
    max_decisions: int = 30,
    wall_time_seconds: float = 120.0,
) -> Path:
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    root = Path(project_root).resolve()
    base = dict(read_json(root / "configs" / "closed_loop_multiseed_collection.json"))
    rows = _load_dataset_rows(dataset_root, [split])
    layouts = collections.Counter(str(row["layout_mode"]) for row in rows)
    map_ids = {str(row["map_id"]) for row in rows}
    tasks_per_map = {
        sum(str(row["map_id"]) == map_id for row in rows) for map_id in map_ids
    }
    if len(tasks_per_map) != 1:
        raise ValueError("confirmation source requires equal tasks per map")
    base.update(
        {
            "formal": False,
            "split": split,
            "solver_seeds": [0],
            # The closed-loop protocol registers both policies even though this
            # confirmation executes only the realized_dynamic source phase.
            "policies": ["official_adaptive", "realized_dynamic"],
            "dataset_design": {
                "map_count": len(map_ids),
                "tasks_per_map": next(iter(tasks_per_map)),
                "task_variants": sorted({str(row["task_variant"]) for row in rows}),
                "layout_counts": {
                    layout: len(
                        {str(row["map_id"]) for row in rows if row["layout_mode"] == layout}
                    )
                    for layout in LAYOUTS
                },
            },
            "environment": {
                "time_limit": float(wall_time_seconds),
                "max_repair_iterations": int(max_decisions),
                "neighborhood_size": 8,
                "replan_algorithm": "PP",
                "use_sipp": True,
            },
            "qualification": {
                "minimum_nonzero_states": 1,
                "minimum_nonzero_states_per_layout": 0,
                "minimum_active_maps": 1,
            },
            "max_decisions": int(max_decisions),
            "metric_iteration_budget": int(max_decisions),
            "wall_time_budget_seconds": float(wall_time_seconds),
            "episode_process_timeout_seconds": float(wall_time_seconds) + 60.0,
            "workers": 4,
            "reference_datasets": [],
        }
    )
    if sum(layouts.values()) != len(rows):
        raise ValueError("confirmation source has incomplete layout metadata")
    path = output_root / "protocol" / f"source_{split}.json"
    _write_json(path, base)
    return path


def _run_source_wave(
    *,
    dataset: Path,
    config: Path,
    output: Path,
    task_ids: list[str],
    project_root: Path,
    workers: int,
    resume: bool,
) -> None:
    common = {
        "workers": workers,
        "task_ids": task_ids,
        "controller": "v2-full",
        "feature_backend": "native",
        "controller_bundle": project_root
        / "artifacts"
        / "initlns-closed-loop-controller-v2",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "historical",
    }
    run_closed_loop_collection(
        dataset,
        config,
        output,
        phase="qualify",
        resume=resume or (output / "run_config.json").is_file(),
        **common,
    )
    run_closed_loop_collection(
        dataset,
        config,
        output,
        phase="realized_dynamic",
        resume=True,
        **common,
    )


def _learned_choice(
    candidates: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    scores: list[float],
    base_index: int,
    bundle: Any,
) -> tuple[int | None, float]:
    eligible = [index for index in range(len(candidates)) if index != base_index]
    started = time.perf_counter()
    predictions = bundle.predict([feature_rows[index] for index in eligible])
    ordered = repair_aware_order(
        [candidates[index] for index in eligible],
        predictions,
        [scores[index] for index in eligible],
    )
    if not ordered:
        return None, time.perf_counter() - started
    local_index = ordered[0]
    selected_index = eligible[local_index]
    adaptive_prediction = bundle.predict([adaptive_feature_row(feature_rows[0])])
    efficiency = float(predictions["efficiency"][local_index])
    adaptive_efficiency = float(adaptive_prediction["efficiency"][0])
    minimum = float(bundle.thresholds["minimum_predicted_efficiency"])
    margin = float(bundle.thresholds["adaptive_efficiency_margin"])
    elapsed = time.perf_counter() - started
    if efficiency + 1e-12 < max(minimum, adaptive_efficiency * (1.0 + margin)):
        return None, elapsed
    return selected_index, elapsed


def _prepare_state(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output"])
    if output.is_file():
        existing = _read_json(output)
        if (
            bool(existing.get("complete"))
            and str(existing.get("run_fingerprint")) == str(job["run_fingerprint"])
        ):
            return existing
    decision = dict(job["decision"])
    source_run = _read_json(Path(decision["source_root"]) / "run_config.json")
    dataset_root = Path(str(source_run["dataset"])).resolve()
    split = str(decision.get("split", CONFIRMATION_SPLIT))
    rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [split])
    }
    row = rows[str(decision["task_id"])]
    replay_job = {
        "dataset_root": str(dataset_root),
        "row": row,
        "environment": dict(source_run["configuration"]["environment"]),
        "solver_seed": int(decision["solver_seed"]),
    }
    environment, state = replay_prefix(replay_job, decision["prefix_actions"])
    before_full = state_fingerprint(state)
    before_repair = repair_structure_fingerprint(state)
    if before_full != str(decision["before_fingerprint"]):
        raise RuntimeError("confirmation replay full fingerprint mismatch")
    if before_repair != str(decision["before_repair_fingerprint"]):
        raise RuntimeError("confirmation replay repair fingerprint mismatch")

    main_bundle = load_controller_bundle(Path(job["controller_bundle"]))
    proposal = dict(source_run["configuration"]["proposal"])
    proposal["neighborhood_sizes"] = list(SIZES)
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=before_full,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    feature_rows, feature_metrics = engine.realized_rows(candidates, state_hash=before_full)
    _selected, scores, _margin = score_online_candidates(
        feature_rows, main_bundle.main_models["realized_dynamic"]
    )
    source_agents = tuple(sorted(map(int, decision["actual_action"].get("agents", ()))))
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if tuple(sorted(map(int, candidate["agents"]))) == source_agents
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"confirmation source action matches {len(matches)} regenerated candidates"
        )
    base_index = matches[0]
    fixed_started = time.perf_counter()
    top_by_size: dict[int, int] = {}
    for size in SIZES:
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if index != base_index and int(candidate["actual_size"]) == size
        ]
        if not eligible:
            payload = {
                "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
                "run_fingerprint": str(job["run_fingerprint"]),
                "complete": True,
                "valid": False,
                "invalid_reason": f"missing_unused_exact_size_{size}",
                "state": decision,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, payload)
            return payload
        top_by_size[size] = min(
            eligible,
            key=lambda index: (
                -round(float(scores[index]), 12),
                str(candidates[index]["candidate_id"]),
            ),
        )
    fixed_selector_seconds = time.perf_counter() - fixed_started
    learned_bundle = load_repair_aware_bundle(Path(job["repair_aware_bundle"]))
    learned_index, learned_selector_seconds = _learned_choice(
        candidates, feature_rows, scores, base_index, learned_bundle
    )

    selected_indices = set(top_by_size.values())
    if learned_index is not None:
        selected_indices.add(learned_index)
    arms = []
    for index in sorted(selected_indices, key=lambda value: str(candidates[value]["candidate_id"])):
        candidate = candidates[index]
        arms.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "route": "model",
                "agents": list(map(int, candidate["agents"])),
                "actual_size": int(candidate["actual_size"]),
                "main_score": float(scores[index]),
            }
        )
    arms.append(
        {
            "candidate_id": "official_adaptive",
            "route": "official_adaptive",
            "agents": [],
            "actual_size": 0,
            "main_score": -1e30,
        }
    )
    state_id = str(decision["before_repair_fingerprint"])
    metadata = {
        "state_id": state_id,
        "split": split,
        "map_id": str(decision["map_id"]),
        "layout_mode": str(decision["layout_mode"]),
        "task_id": str(decision["task_id"]),
        "task_variant": str(row["task_variant"]),
        "agent_count": int(row["agent_count"]),
        "cell": _cell(str(decision["layout_mode"]), int(row["agent_count"])),
        "solver_seed": int(decision["solver_seed"]),
        "decision_index": int(decision["decision_index"]),
        "source_root": str(decision["source_root"]),
        "prefix_actions": [dict(action) for action in decision["prefix_actions"]],
        "before_full_fingerprint": before_full,
        "before_repair_fingerprint": before_repair,
    }
    payload = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "valid": True,
        "state": metadata,
        "top_candidate_by_size": {
            str(size): str(candidates[index]["candidate_id"])
            for size, index in top_by_size.items()
        },
        "learned_candidate_id": (
            str(candidates[learned_index]["candidate_id"])
            if learned_index is not None
            else "official_adaptive"
        ),
        "base_candidate_id": str(candidates[base_index]["candidate_id"]),
        "arms": arms,
        "fixed_selector_seconds": fixed_selector_seconds,
        "learned_selector_seconds": learned_selector_seconds,
        "generation": generation,
        "feature_metrics": feature_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return payload


def _candidate_decisions(
    source_roots: Iterable[Path],
    dataset: Path,
    split: str = CONFIRMATION_SPLIT,
) -> list[dict[str, Any]]:
    task_rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset, [split])
    }
    decisions = []
    seen = set()
    for root in source_roots:
        if not (root / "realized_dynamic_manifest.jsonl").is_file():
            continue
        for decision in select_failure_decisions(root, maximum=100000):
            fingerprint = str(decision["before_repair_fingerprint"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            task = task_rows[str(decision["task_id"])]
            decision["agent_count"] = int(task["agent_count"])
            decision["task_variant"] = str(task["task_variant"])
            decision["cell"] = _cell(
                str(decision["layout_mode"]), int(task["agent_count"])
            )
            decisions.append(decision)
    return sorted(
        decisions,
        key=lambda row: (
            str(row["cell"]),
            str(row["map_id"]),
            str(row["task_id"]),
            int(row["decision_index"]),
        ),
    )


def _prepare_decisions(
    *,
    decisions: list[dict[str, Any]],
    output: Path,
    run_fingerprint: str,
    controller_bundle: Path,
    repair_aware_bundle: Path,
    workers: int,
    maximum_per_cell: int,
) -> list[dict[str, Any]]:
    limited = []
    counts: collections.Counter[str] = collections.Counter()
    for decision in decisions:
        cell = str(decision["cell"])
        if counts[cell] >= maximum_per_cell:
            continue
        counts[cell] += 1
        limited.append(decision)
    jobs = []
    for decision in limited:
        key = str(decision["before_repair_fingerprint"])
        jobs.append(
            {
                "decision": decision,
                "output": str(output / "prepared" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "controller_bundle": str(controller_bundle),
                "repair_aware_bundle": str(repair_aware_bundle),
            }
        )
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_prepare_state, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def select_confirmation_states(
    prepared: list[dict[str, Any]], *, quota_per_cell: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    valid = [row for row in prepared if bool(row.get("valid"))]
    by_cell: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in valid:
        by_cell[str(row["state"]["cell"])].append(row)
    selected = []
    counts = {}
    for layout in LAYOUTS:
        for agent_count in AGENT_COUNTS:
            cell = _cell(layout, agent_count)
            rows = sorted(
                by_cell.get(cell, []),
                key=lambda row: (
                    str(row["state"]["map_id"]),
                    str(row["state"]["task_id"]),
                    int(row["state"]["decision_index"]),
                ),
            )
            task_counts: collections.Counter[str] = collections.Counter()
            chosen = []
            by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in rows:
                by_map[str(row["state"]["map_id"])].append(row)
            while len(chosen) < quota_per_cell and any(by_map.values()):
                picked_in_round = False
                for map_id in sorted(by_map):
                    while by_map[map_id]:
                        row = by_map[map_id].pop(0)
                        task_id = str(row["state"]["task_id"])
                        if task_counts[task_id] >= 2:
                            continue
                        chosen.append(row)
                        task_counts[task_id] += 1
                        picked_in_round = True
                        break
                    if len(chosen) >= quota_per_cell:
                        break
                if not picked_in_round:
                    break
            selected.extend(chosen)
            counts[cell] = len(chosen)
    return selected, counts


def _branch_trial(
    *, preparation: dict[str, Any], arm: dict[str, Any], trial_index: int
) -> dict[str, Any]:
    state = dict(preparation["state"])
    source_run = _read_json(Path(state["source_root"]) / "run_config.json")
    dataset_root = Path(str(source_run["dataset"])).resolve()
    split = str(state.get("split", CONFIRMATION_SPLIT))
    rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [split])
    }
    replay_job = {
        "dataset_root": str(dataset_root),
        "row": rows[str(state["task_id"])],
        "environment": dict(source_run["configuration"]["environment"]),
        "solver_seed": int(state["solver_seed"]),
    }
    environment, before = replay_prefix(replay_job, state["prefix_actions"])
    before_full = state_fingerprint(before)
    before_repair = repair_structure_fingerprint(before)
    if before_full != str(state["before_full_fingerprint"]):
        raise RuntimeError("confirmation branch full fingerprint mismatch")
    if before_repair != str(state["before_repair_fingerprint"]):
        raise RuntimeError("confirmation branch repair fingerprint mismatch")
    random_seed = confirmation_seed(before_repair, trial_index)
    action = (
        {"mode": "official", "random_seed": random_seed}
        if str(arm["route"]) == "official_adaptive"
        else {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, arm["agents"])),
            "random_seed": random_seed,
        }
    )
    started = time.perf_counter()
    result = _plain(environment.step(action))
    repair_seconds = time.perf_counter() - started
    after = dict(result["observation"])
    metrics = dict(result["metrics"])
    after_full = state_fingerprint(after)
    after_repair = repair_structure_fingerprint(after)
    before_conflicts = int(before["num_of_colliding_pairs"])
    after_conflicts = int(after["num_of_colliding_pairs"])
    outcome_kind = classify_repair_outcome(
        before_fingerprint=before_repair,
        after_fingerprint=after_repair,
        replan_success=bool(metrics.get("replan_success")),
        conflicts_before=before_conflicts,
        conflicts_after=after_conflicts,
        feasible=bool(after.get("feasible")),
    )
    low_level = _low_level_delta(before, after)
    return {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "state_id": str(state["state_id"]),
        "split": split,
        "map_id": str(state["map_id"]),
        "layout_mode": str(state["layout_mode"]),
        "task_id": str(state["task_id"]),
        "agent_count": int(state["agent_count"]),
        "cell": str(state["cell"]),
        "candidate_id": str(arm["candidate_id"]),
        "route": str(arm["route"]),
        "actual_size": int(arm["actual_size"]),
        "trial_index": int(trial_index),
        "random_seed": int(random_seed),
        "complete": True,
        "status": "ok",
        "outcome": {
            "outcome_kind": outcome_kind,
            "before_fingerprint": before_repair,
            "after_fingerprint": after_repair,
            "before_full_fingerprint": before_full,
            "after_full_fingerprint": after_full,
            "state_changed": before_repair != after_repair,
            "conflicts_before": before_conflicts,
            "conflicts_after": after_conflicts,
            "signed_conflict_delta": before_conflicts - after_conflicts,
            "conflict_reduction": max(0, before_conflicts - after_conflicts),
            "replan_success": bool(metrics.get("replan_success")),
            "hard_failure": outcome_kind == "hard_failure",
            "feasible": bool(after.get("feasible")),
            "sum_of_costs_before": int(before["sum_of_costs"]),
            "sum_of_costs_after": int(after["sum_of_costs"]),
            "repair_seconds": repair_seconds,
            "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
            "generated": int(low_level.get("generated", 0)),
            "expanded": int(low_level.get("expanded", 0)),
            "reopened": int(low_level.get("reopened", 0)),
        },
    }


def _trial_state_job(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output"])
    if output.is_file():
        existing = _read_json(output)
        if (
            bool(existing.get("complete"))
            and str(existing.get("run_fingerprint")) == str(job["run_fingerprint"])
        ):
            return existing
    preparation = dict(job["preparation"])
    trials = []
    for arm in preparation["arms"]:
        for trial_index in range(int(job["trial_count"])):
            trials.append(
                _branch_trial(
                    preparation=preparation,
                    arm=dict(arm),
                    trial_index=trial_index,
                )
            )
    payload = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "state": preparation["state"],
        "top_candidate_by_size": preparation["top_candidate_by_size"],
        "learned_candidate_id": preparation["learned_candidate_id"],
        "fixed_selector_seconds": preparation["fixed_selector_seconds"],
        "learned_selector_seconds": preparation["learned_selector_seconds"],
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return payload


def _run_trials(
    *,
    selected: list[dict[str, Any]],
    output: Path,
    run_fingerprint: str,
    trial_count: int,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        {
            "preparation": row,
            "output": str(
                output / "states" / f"{row['state']['state_id']}.json"
            ),
            "run_fingerprint": run_fingerprint,
            "trial_count": trial_count,
        }
        for row in selected
    ]
    results = []
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_trial_state_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": job["preparation"]["state"]["state_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output / "collection_progress.json",
                {
                    "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
                    "completed_states": len(results),
                    "total_states": len(jobs),
                    "error_states": len(errors),
                },
            )
    if errors:
        raise RuntimeError(f"confirmation trial errors: {errors[:3]}")
    return results


def _gate(metrics: dict[str, Any], baseline: dict[str, Any], efficiency: float) -> bool:
    comparison = _comparison(metrics, baseline)
    return bool(
        float(comparison["state_escape_rate_delta"]) >= -1e-12
        and float(comparison["final_hard_failure_rate_delta"]) <= 1e-12
        and float(comparison["efficiency_ratio"]) + 1e-12 >= efficiency
        and float(comparison["mean_reduction_ratio"]) + 1e-12 >= 0.98
    )


def validate_trial_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    trial_count: int | None = None
    branch_count = 0
    for result in results:
        state = dict(result["state"])
        state_id = str(state["state_id"])
        expected_before = str(state["before_repair_fingerprint"])
        by_candidate: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        seeds: dict[int, int] = {}
        for trial in result["trials"]:
            by_candidate[str(trial["candidate_id"])].append(trial)
            index = int(trial["trial_index"])
            seed = int(trial["random_seed"])
            previous = seeds.setdefault(index, seed)
            if previous != seed:
                errors.append(f"{state_id}: paired seed mismatch at trial {index}")
            outcome = dict(trial["outcome"])
            before = str(outcome.get("before_fingerprint", ""))
            after = str(outcome.get("after_fingerprint", ""))
            if before != expected_before or len(after) != 64:
                errors.append(f"{state_id}: missing or mismatched repair fingerprint")
            changed = before != after
            if bool(outcome.get("state_changed")) != changed:
                errors.append(f"{state_id}: state_changed disagrees with fingerprints")
            if bool(outcome.get("hard_failure")) and changed:
                errors.append(f"{state_id}: hard failure changed repair state")
        index_sets = {
            tuple(sorted(int(row["trial_index"]) for row in rows))
            for rows in by_candidate.values()
        }
        if len(index_sets) != 1:
            errors.append(f"{state_id}: candidate trial coverage differs")
        elif index_sets:
            observed = len(next(iter(index_sets)))
            if trial_count is None:
                trial_count = observed
            elif trial_count != observed:
                errors.append(f"{state_id}: state trial count differs")
        branch_count += len(result["trials"])
    report = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "passed": not errors and bool(results),
        "state_count": len(results),
        "trial_count_per_arm": int(trial_count or 0),
        "branch_trial_count": branch_count,
        "error_count": len(errors),
        "errors": errors,
    }
    if not report["passed"]:
        raise ValueError(f"confirmation trial coverage failed: {errors[:3]}")
    return report


def analyze_confirmation(
    *, results: list[dict[str, Any]], output: str | Path
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    coverage = validate_trial_coverage(results)
    trial_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    preparations: dict[str, dict[str, Any]] = {}
    for result in results:
        state = dict(result["state"])
        state_id = str(state["state_id"])
        preparations[state_id] = result
        for trial in result["trials"]:
            key = (
                state_id,
                str(trial["candidate_id"]),
                int(trial["trial_index"]),
            )
            if key in trial_lookup:
                raise ValueError(f"duplicate confirmation trial: {key}")
            trial_lookup[key] = trial
    policies = enumerate_rescue_policies()
    simulated = []
    for state_id, preparation in sorted(preparations.items()):
        state = dict(preparation["state"])
        trials = sorted(
            {
                int(row["trial_index"])
                for row in preparation["trials"]
            }
        )
        top = {int(key): str(value) for key, value in preparation["top_candidate_by_size"].items()}
        for policy in policies:
            sequence = [top[size] for size in policy.size_order]
            for trial_index in trials:
                row = _simulate_sequence(
                    metadata={
                        "state_id": state_id,
                        "split": str(state.get("split", CONFIRMATION_SPLIT)),
                        "map_id": str(state["map_id"]),
                        "layout_mode": str(state["layout_mode"]),
                        "agent_count": int(state["agent_count"]),
                        "cell": str(state["cell"]),
                    },
                    trial_index=trial_index,
                    policy_id=policy.policy_id,
                    candidate_sequence=sequence,
                    adaptive_id="official_adaptive",
                    trials=trial_lookup,
                    assumption="exact-after-fingerprint",
                    reference_only=False,
                )
                row["schema"] = RESCUE_LITE_CONFIRMATION_SCHEMA
                simulated.append(row)
        learned = str(preparation["learned_candidate_id"])
        learned_sequence = [] if learned == "official_adaptive" else [learned]
        for trial_index in trials:
            row = _simulate_sequence(
                metadata={
                    "state_id": state_id,
                    "split": str(state.get("split", CONFIRMATION_SPLIT)),
                    "map_id": str(state["map_id"]),
                    "layout_mode": str(state["layout_mode"]),
                    "agent_count": int(state["agent_count"]),
                    "cell": str(state["cell"]),
                },
                trial_index=trial_index,
                policy_id="reference_learned_repair_aware",
                candidate_sequence=learned_sequence,
                adaptive_id="official_adaptive",
                trials=trial_lookup,
                assumption="exact-after-fingerprint",
                reference_only=True,
            )
            row["schema"] = RESCUE_LITE_CONFIRMATION_SCHEMA
            simulated.append(row)

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in simulated:
        grouped[str(row["policy_id"])].append(row)
    baseline = _aggregate(grouped["adaptive"])
    summary_rows = []
    for policy_id, rows in sorted(grouped.items()):
        metrics = _aggregate(rows)
        summary_rows.append(
            {
                "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
                "policy_id": policy_id,
                **metrics,
                **_comparison(metrics, baseline),
            }
        )
    summary = {str(row["policy_id"]): row for row in summary_rows}

    cell_rows = []
    for cell in sorted({str(row["cell"]) for row in simulated}):
        for policy_id in ("adaptive", FROZEN_POLICY_ID, "reference_learned_repair_aware"):
            rows = [
                row
                for row in grouped[policy_id]
                if str(row["cell"]) == cell
            ]
            metrics = _aggregate(rows)
            base_rows = [row for row in grouped["adaptive"] if str(row["cell"]) == cell]
            base_metrics = _aggregate(base_rows)
            cell_rows.append(
                {
                    "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
                    "cell": cell,
                    "policy_id": policy_id,
                    **metrics,
                    **_comparison(metrics, base_metrics),
                }
            )

    fixed = summary[FROZEN_POLICY_ID]
    learned = summary["reference_learned_repair_aware"]
    fixed_basic = _gate(fixed, baseline, 1.10)
    learned_basic = _gate(learned, baseline, 1.0)
    fixed_cells = [row for row in cell_rows if row["policy_id"] == FROZEN_POLICY_ID]
    noninferior_cells = sum(float(row["efficiency_ratio"]) + 1e-12 >= 1.0 for row in fixed_cells)
    worst_cell_ratio = min(float(row["efficiency_ratio"]) for row in fixed_cells)
    alternative_dominators = []
    for policy in policies:
        if policy.policy_id in {"adaptive", FROZEN_POLICY_ID}:
            continue
        other = summary[policy.policy_id]
        if (
            float(other["conflict_reduction_per_second"])
            >= 1.10 * float(fixed["conflict_reduction_per_second"])
            and float(other["state_escape_rate"]) + 1e-12
            >= float(fixed["state_escape_rate"])
            and float(other["final_hard_failure_rate"]) - 1e-12
            <= float(fixed["final_hard_failure_rate"])
        ):
            alternative_dominators.append(policy.policy_id)
    learned_dominates = bool(
        float(learned["conflict_reduction_per_second"])
        >= 1.10 * float(fixed["conflict_reduction_per_second"])
        and float(learned["state_escape_rate"]) + 1e-12
        >= float(fixed["state_escape_rate"])
        and float(learned["final_hard_failure_rate"]) - 1e-12
        <= float(fixed["final_hard_failure_rate"])
    )
    fixed_passed = bool(
        fixed_basic
        and noninferior_cells >= 5
        and worst_cell_ratio + 1e-12 >= 0.90
        and not alternative_dominators
        and not learned_dominates
    )
    if fixed_passed:
        decision = "rescue_lite_confirmed"
    elif learned_dominates and learned_basic:
        decision = "learned_rescue_reconsidered"
    elif not fixed_basic and not learned_basic:
        decision = "proceed_to_v3"
    else:
        decision = "inconclusive_collect_more"
    selector = {
        "fixed_mean_seconds": statistics.fmean(
            float(row["fixed_selector_seconds"]) for row in results
        ),
        "learned_mean_seconds": statistics.fmean(
            float(row["learned_selector_seconds"]) for row in results
        ),
    }
    report = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "decision": decision,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "state_count": len(results),
        "trial_count_per_state": len(simulated) // (len(results) * (len(policies) + 1)),
        "exact_repair_fingerprint_coverage": True,
        "coverage": coverage,
        "fixed_basic_gate_passed": fixed_basic,
        "cell_gate": {
            "noninferior_cell_count": noninferior_cells,
            "required_noninferior_cell_count": 5,
            "worst_efficiency_ratio": worst_cell_ratio,
            "passed": noninferior_cells >= 5 and worst_cell_ratio + 1e-12 >= 0.90,
        },
        "alternative_dominators": alternative_dominators,
        "learned_basic_gate_passed": learned_basic,
        "learned_dominates_fixed": learned_dominates,
        "fixed_metrics": fixed,
        "adaptive_metrics": baseline,
        "learned_metrics": learned,
        "selector_seconds": selector,
        "default_controller_changed": False,
        "long_jobs_started_after_confirmation": False,
    }
    _write_csv(output_root / "confirmation_trials.csv", simulated)
    _write_csv(output_root / "confirmation_summary.csv", summary_rows)
    _write_csv(output_root / "confirmation_cell_summary.csv", cell_rows)
    _write_json(output_root / "confirmation_report.json", report)
    markdown = [
        "# Rescue-lite independent confirmation",
        "",
        f"- Decision: `{decision}`",
        f"- Frozen policy: `{FROZEN_POLICY_ID}`",
        f"- States: {len(results)}; paired trials per state: {report['trial_count_per_state']}",
        "- Exact repair fingerprint coverage: true",
        "- Solver jobs launched after this report: none",
        "",
        "## Primary comparison",
        "",
        (
            f"`{FROZEN_POLICY_ID}` vs Adaptive: efficiency ratio "
            f"{float(fixed['efficiency_ratio']):.3f}, escape delta "
            f"{float(fixed['state_escape_rate_delta']):+.3f}, hard-failure delta "
            f"{float(fixed['final_hard_failure_rate_delta']):+.3f}, mean reduction "
            f"ratio {float(fixed['mean_reduction_ratio']):.3f}."
        ),
        "",
        "## Guardrails",
        "",
        f"- Non-inferior layout/agent cells: {noninferior_cells}/6",
        f"- Worst cell efficiency ratio: {worst_cell_ratio:.3f}",
        f"- Alternative fixed-order dominators: {alternative_dominators or 'none'}",
        f"- Learned selector dominates fixed rule: {str(learned_dominates).lower()}",
        "",
        "This confirmation does not register or deploy a new controller.",
        "",
    ]
    (output_root / "confirmation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report


def run_rescue_lite_confirmation(
    *,
    project_root: str | Path,
    output: str | Path,
    dataset_config: str | Path,
    controller_bundle: str | Path,
    repair_aware_bundle: str | Path,
    reference_datasets: Iterable[str | Path],
    workers: int = 4,
    resume: bool = False,
    quota_per_cell: int = DEFAULT_STATE_QUOTA_PER_CELL,
    trial_count: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    config_path = Path(dataset_config).resolve()
    controller_path = Path(controller_bundle).resolve()
    repair_aware_path = Path(repair_aware_bundle).resolve()
    if workers <= 0 or quota_per_cell <= 0 or trial_count <= 0:
        raise ValueError("workers, quota and trial count must be positive")
    identity = {
        "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
        "dataset_config_sha256": sha256_file(config_path),
        "controller_manifest_sha256": sha256_file(
            controller_path / "controller_manifest.json"
        ),
        "repair_aware_manifest_sha256": sha256_file(
            repair_aware_path / "repair_aware_manifest.json"
        ),
        "quota_per_cell": int(quota_per_cell),
        "trial_count": int(trial_count),
        "sizes": list(SIZES),
        "frozen_policy_id": FROZEN_POLICY_ID,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if output_root.is_dir() and any(output_root.iterdir()):
        if not resume:
            raise ValueError("confirmation output already exists; pass --resume")
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("confirmation resume fingerprint mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    dataset = output_root / "dataset"
    _write_status(output_root, phase="dataset", status="running")
    if not (dataset / "dataset_summary.json").is_file():
        generate_dataset(load_json(config_path), dataset)
    isolation = validate_dataset_isolation(dataset, reference_datasets)
    _write_json(output_root / "dataset_isolation.json", isolation)
    rows = _load_dataset_rows(dataset, [CONFIRMATION_SPLIT])
    initial_tasks, expansion_tasks = confirmation_task_waves(rows)
    source_config = build_confirmation_source_config(
        dataset=dataset, output=output_root, project_root=root
    )

    sources = [output_root / "sources" / "initial"]
    _write_status(output_root, phase="source-initial", status="running")
    _run_source_wave(
        dataset=dataset,
        config=source_config,
        output=sources[0],
        task_ids=initial_tasks,
        project_root=root,
        workers=workers,
        resume=resume,
    )
    decisions = _candidate_decisions(sources, dataset)
    prepared = _prepare_decisions(
        decisions=decisions,
        output=output_root,
        run_fingerprint=run_fingerprint,
        controller_bundle=controller_path,
        repair_aware_bundle=repair_aware_path,
        workers=workers,
        maximum_per_cell=max(quota_per_cell * 2, quota_per_cell + 3),
    )
    selected, counts = select_confirmation_states(
        prepared, quota_per_cell=quota_per_cell
    )
    expansion_used = any(counts.get(_cell(layout, agents), 0) < quota_per_cell for layout in LAYOUTS for agents in AGENT_COUNTS)
    if expansion_used:
        expansion = output_root / "sources" / "expansion"
        sources.append(expansion)
        _write_status(output_root, phase="source-expansion", status="running")
        _run_source_wave(
            dataset=dataset,
            config=source_config,
            output=expansion,
            task_ids=expansion_tasks,
            project_root=root,
            workers=workers,
            resume=resume,
        )
        decisions = _candidate_decisions(sources, dataset)
        prepared = _prepare_decisions(
            decisions=decisions,
            output=output_root,
            run_fingerprint=run_fingerprint,
            controller_bundle=controller_path,
            repair_aware_bundle=repair_aware_path,
            workers=workers,
            maximum_per_cell=max(quota_per_cell * 4, quota_per_cell + 8),
        )
        selected, counts = select_confirmation_states(
            prepared, quota_per_cell=quota_per_cell
        )
    required = {_cell(layout, agents): quota_per_cell for layout in LAYOUTS for agents in AGENT_COUNTS}
    if counts != required:
        report = {
            "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
            "decision": "insufficient_confirmation_states",
            "required_state_count_by_cell": required,
            "observed_state_count_by_cell": counts,
            "expansion_used": expansion_used,
            "long_jobs_started_after_confirmation": False,
        }
        _write_json(output_root / "confirmation_report.json", report)
        _write_status(output_root, phase="complete", status="insufficient")
        return report
    selection = [
        {
            "state": row["state"],
            "top_candidate_by_size": row["top_candidate_by_size"],
            "learned_candidate_id": row["learned_candidate_id"],
        }
        for row in selected
    ]
    _write_json(
        output_root / "selection_manifest.json",
        {
            "schema": RESCUE_LITE_CONFIRMATION_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "state_count_by_cell": counts,
            "states": selection,
        },
    )
    _write_status(output_root, phase="paired-trials", status="running")
    results = _run_trials(
        selected=selected,
        output=output_root,
        run_fingerprint=run_fingerprint,
        trial_count=trial_count,
        workers=workers,
    )
    trial_rows = [trial for result in results for trial in result["trials"]]
    _write_jsonl(output_root / "trial_manifest.jsonl", trial_rows)
    _write_status(output_root, phase="analysis", status="running")
    report = analyze_confirmation(results=results, output=output_root)
    report.update(
        {
            "run_fingerprint": run_fingerprint,
            "state_count_by_cell": counts,
            "expansion_used": expansion_used,
            "dataset_isolation": isolation,
            "branch_trial_count": len(trial_rows),
        }
    )
    _write_json(output_root / "confirmation_report.json", report)
    _write_status(output_root, phase="complete", status="complete", decision=report["decision"])
    return report


__all__ = [
    "AGENT_COUNTS",
    "CONFIRMATION_SPLIT",
    "DEFAULT_STATE_QUOTA_PER_CELL",
    "DEFAULT_TRIALS",
    "FROZEN_POLICY_ID",
    "LAYOUTS",
    "RESCUE_LITE_CONFIRMATION_SCHEMA",
    "analyze_confirmation",
    "confirmation_seed",
    "confirmation_task_waves",
    "dataset_map_hashes",
    "run_rescue_lite_confirmation",
    "select_confirmation_states",
    "validate_dataset_isolation",
    "validate_trial_coverage",
]
