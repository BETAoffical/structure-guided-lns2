from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state
from experiments.stride_topology_coverage import _mean
from experiments.trace_replay import replay_prefix


CONFIG_SCHEMA = "lns2.stride.guardrank_topology_relevance_config.v1"
ROW_SCHEMA = "lns2.stride.guardrank_topology_relevance_state.v1"
REPORT_SCHEMA = "lns2.stride.guardrank_topology_relevance_report.v1"


def _registered_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"registered relevance input SHA differs: {artifact['path']}")
    return path


def validate_guardrank_relevance_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected GuardRank topology-relevance config")
    if (
        config.get("scientific_status") != "input_only_development_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("candidate_generation_allowed"))
        or bool(config.get("candidate_repair_trials_allowed"))
        or bool(config.get("controller_actions_allowed"))
        or bool(config.get("controller_outcomes_allowed"))
        or bool(config.get("repair_labels_allowed"))
    ):
        raise ValueError("GuardRank topology relevance must remain input-only")
    if (
        config.get("diagnostic_id") != "stride-guardrank-relevance-v1"
        or config.get("predecessor_id") != "stride-guardrank-mapcoverage-v1"
        or int(config.get("expected_map_count", -1)) != 8
        or int(config.get("expected_task_count", -1)) != 48
        or int(config.get("expected_state_count", -1)) != 96
        or tuple(map(int, config.get("solver_seeds") or ())) != (1, 2)
    ):
        raise ValueError("GuardRank topology-relevance identity changed")
    if set(config.get("inputs") or {}) != {
        "failed_coverage_report",
        "failed_coverage_state_rows",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "source_config",
        "runtime_config",
    }:
        raise ValueError("GuardRank topology-relevance input registry changed")
    if dict(config.get("definitions") or {}) != {
        "articulation_event": "any_conflict_event_cell_is_grid_articulation",
        "low_degree_cell": "free_grid_degree_at_most_two",
        "low_degree_event": "any_conflict_event_cell_has_free_grid_degree_at_most_two",
        "relevant_state": "articulation_event_count_or_low_degree_event_count_is_positive",
    }:
        raise ValueError("GuardRank topology-relevance definitions changed")
    if dict(config.get("interpretation") or {}) != {
        "selected_state_any_relevance_fraction": 0.34375,
        "selected_high_mid_any_relevance_fraction": 0.4583333333333333,
        "minimum_full_minus_selected_high_mid_fraction_for_selection_cause": 0.10,
        "maximum_control_any_relevance_fraction_for_abstention": 0.25,
    }:
        raise ValueError("GuardRank topology-relevance interpretation changed")
    if dict(config.get("recommendation") or {}) != {
        "maximum_tasks_per_map": 4,
        "topology_group_order": (
            "relevant_seed_count_then_event_count_then_initial_conflicts"
        ),
        "control_group_order": (
            "zero_relevance_then_initial_conflicts_then_task_id"
        ),
        "expected_recommended_task_count": 32,
    }:
        raise ValueError("GuardRank topology-relevance recommendation changed")


def _event_counts(state: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_state(state)
    low_degree = {
        cell
        for cell in analysis.free_cells
        if int(analysis.degrees.get(cell, 0)) <= 2
    }
    articulation = [
        event
        for event in analysis.events
        if any(cell in analysis.articulation for cell in event.cells)
    ]
    low = [
        event
        for event in analysis.events
        if any(cell in low_degree for cell in event.cells)
    ]
    total = len(analysis.events)
    return {
        "conflict_event_count": total,
        "articulation_event_count": len(articulation),
        "low_degree_event_count": len(low),
        "articulation_event_ratio": len(articulation) / total if total else 0.0,
        "low_degree_event_ratio": len(low) / total if total else 0.0,
    }


def _task_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    result = []
    for task_id, task_rows in sorted(grouped.items()):
        result.append(
            {
                "task_id": task_id,
                "map_id": str(task_rows[0]["map_id"]),
                "layout_family": str(task_rows[0]["layout_family"]),
                "agent_count": int(task_rows[0]["agent_count"]),
                "state_count": len(task_rows),
                "relevant_seed_count": sum(bool(row["relevant"]) for row in task_rows),
                "articulation_relevant_seed_count": sum(
                    int(row["articulation_event_count"]) > 0 for row in task_rows
                ),
                "low_degree_relevant_seed_count": sum(
                    int(row["low_degree_event_count"]) > 0 for row in task_rows
                ),
                "topology_event_count": sum(
                    int(row["articulation_event_count"])
                    + int(row["low_degree_event_count"])
                    for row in task_rows
                ),
                "mean_initial_conflicts": _mean(
                    [float(row["initial_conflicts"]) for row in task_rows]
                ),
            }
        )
    return result


def _recommend_tasks(
    config: dict[str, Any], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    maximum = int(config["recommendation"]["maximum_tasks_per_map"])
    by_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_map[str(task["map_id"])].append(task)
    selected = []
    for map_id, map_tasks in sorted(by_map.items()):
        control = str(map_tasks[0]["layout_family"]).endswith("_control")
        if control:
            ordered = sorted(
                map_tasks,
                key=lambda row: (
                    int(row["relevant_seed_count"]) > 0,
                    -float(row["mean_initial_conflicts"]),
                    str(row["task_id"]),
                ),
            )
        else:
            ordered = sorted(
                map_tasks,
                key=lambda row: (
                    -int(row["relevant_seed_count"]),
                    -int(row["topology_event_count"]),
                    -float(row["mean_initial_conflicts"]),
                    str(row["task_id"]),
                ),
            )
        selected.extend({**row, "selection_rank": rank + 1} for rank, row in enumerate(ordered[:maximum]))
    return sorted(selected, key=lambda row: (str(row["map_id"]), int(row["selection_rank"])))


def collect_guardrank_relevance(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_guardrank_relevance_config(config)
    inputs = {
        name: _registered_path(project_root, dict(artifact))
        for name, artifact in config["inputs"].items()
    }
    failed_report = _read_json(inputs["failed_coverage_report"])
    if failed_report.get("passed") is not False:
        raise ValueError("relevance scan requires the registered failed coverage report")
    qualification_report = _read_json(inputs["qualification_report"])
    if qualification_report.get("errors") or qualification_report.get("valid_count") != 96:
        raise ValueError("relevance scan requires 96 valid qualification rows")
    dataset_rows = {
        str(row["task_id"]): row
        for row in _read_jsonl(inputs["dataset_manifest"])
    }
    qualification_rows = _read_jsonl(inputs["qualification_manifest"])
    runtime = _read_json(inputs["runtime_config"])
    dataset_root = (project_root / str(config["dataset_root"])).resolve()
    rows = []
    for qualification in sorted(
        qualification_rows,
        key=lambda row: (str(row["task_id"]), int(row["solver_seed"])),
    ):
        task_id = str(qualification["task_id"])
        dataset_row = dataset_rows.get(task_id)
        if dataset_row is None or qualification.get("status") != "ok":
            raise ValueError(f"invalid relevance input row: {task_id}")
        job = {
            "dataset_root": str(dataset_root),
            "row": dataset_row,
            "environment": runtime["environment"],
            "solver_seed": int(qualification["solver_seed"]),
            "replay_destroy_strategy": "Adaptive",
        }
        environment, state = replay_prefix(job, [])
        fingerprint = state_fingerprint(state)
        if (
            fingerprint != str(qualification["state_fingerprint"])
            or int(state["num_of_colliding_pairs"])
            != int(qualification["initial_conflicts"])
        ):
            raise RuntimeError(f"qualified relevance state did not replay: {task_id}")
        counts = _event_counts(state)
        preserved = state_fingerprint(_plain(environment.get_state())) == fingerprint
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "state_id": f"{task_id}::solver_seed_{int(qualification['solver_seed'])}",
                "task_id": task_id,
                "solver_seed": int(qualification["solver_seed"]),
                "map_id": str(dataset_row["benchmark_id"]),
                "layout_family": str(dataset_row["layout_family"]),
                "agent_count": int(dataset_row["agent_count"]),
                "initial_conflicts": int(qualification["initial_conflicts"]),
                "state_fingerprint": fingerprint,
                "state_fingerprint_preserved": preserved,
                **counts,
                "relevant": bool(
                    counts["articulation_event_count"]
                    or counts["low_degree_event_count"]
                ),
            }
        )
    failed_rows = _read_jsonl(inputs["failed_coverage_state_rows"])
    failed_by_id = {str(row["state_id"]): row for row in failed_rows}
    full_by_id = {str(row["state_id"]): row for row in rows}
    selected_reproduced = all(
        state_id in full_by_id
        and bool(failed["articulation_relevant"])
        == (int(full_by_id[state_id]["articulation_event_count"]) > 0)
        and bool(failed["low_degree_relevant"])
        == (int(full_by_id[state_id]["low_degree_event_count"]) > 0)
        for state_id, failed in failed_by_id.items()
    )
    high_mid = [
        row for row in rows if not str(row["layout_family"]).endswith("_control")
    ]
    controls = [
        row for row in rows if str(row["layout_family"]).endswith("_control")
    ]
    full_high_mid_fraction = _mean([float(bool(row["relevant"])) for row in high_mid])
    control_fraction = _mean([float(bool(row["relevant"])) for row in controls])
    interpretation = dict(config["interpretation"])
    relevance_delta = full_high_mid_fraction - float(
        interpretation["selected_high_mid_any_relevance_fraction"]
    )
    selection_cause = relevance_delta >= float(
        interpretation[
            "minimum_full_minus_selected_high_mid_fraction_for_selection_cause"
        ]
    )
    task_summaries = _task_summaries(rows)
    recommended = _recommend_tasks(config, task_summaries)
    maps = {str(row["map_id"]) for row in rows}
    tasks = {str(row["task_id"]) for row in rows}
    pairing = all(
        {int(row["solver_seed"]) for row in rows if row["task_id"] == task_id}
        == set(map(int, config["solver_seeds"]))
        for task_id in tasks
    )
    gates = {
        "expected_map_count": len(maps) == int(config["expected_map_count"]),
        "expected_task_count": len(tasks) == int(config["expected_task_count"]),
        "expected_state_count": len(rows) == int(config["expected_state_count"]),
        "complete_solver_seed_pairing": pairing,
        "all_state_fingerprints_preserved": all(
            bool(row["state_fingerprint_preserved"]) for row in rows
        ),
        "selected_state_relevance_reproduced": selected_reproduced,
        "recommended_task_count": len(recommended)
        == int(config["recommendation"]["expected_recommended_task_count"]),
        "control_abstention_fraction": control_fraction
        <= float(
            interpretation["maximum_control_any_relevance_fraction_for_abstention"]
        ),
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "topology_relevance_states.jsonl"
    tasks_path = output_root / "topology_relevance_tasks.jsonl"
    recommended_path = output_root / "recommended_tasks.jsonl"
    _write_jsonl(rows_path, rows)
    _write_jsonl(tasks_path, task_summaries)
    _write_jsonl(recommended_path, recommended)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "input_only_fresh_map_relevance_diagnostic",
        "candidate_generation_executed": False,
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "controller_outcomes_read": False,
        "repair_labels_created": False,
        "counts": {
            "map_count": len(maps),
            "task_count": len(tasks),
            "state_count": len(rows),
            "relevant_state_count": sum(bool(row["relevant"]) for row in rows),
            "recommended_task_count": len(recommended),
        },
        "fractions": {
            "all_state_any_relevance": _mean(
                [float(bool(row["relevant"])) for row in rows]
            ),
            "high_mid_any_relevance": full_high_mid_fraction,
            "control_any_relevance": control_fraction,
            "selected_high_mid_any_relevance": interpretation[
                "selected_high_mid_any_relevance_fraction"
            ],
            "full_minus_selected_high_mid": relevance_delta,
        },
        "selection_undercoverage_supported": selection_cause,
        "next_decision": (
            "register_topology_enriched_map_repair_cohort"
            if selection_cause
            else "keep_boundary_optional_and_prioritize_map_diverse_base_candidates"
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "artifacts": {
            "state_rows_sha256": sha256_file(rows_path),
            "task_rows_sha256": sha256_file(tasks_path),
            "recommended_tasks_sha256": sha256_file(recommended_path),
        },
        "inputs": {
            "config_sha256": sha256_file(config_path),
            **{
                f"{name}_sha256": sha256_file(path)
                for name, path in inputs.items()
            },
        },
    }
    _write_json(output_root / "topology_relevance_report.json", report)
    return report


__all__ = [
    "collect_guardrank_relevance",
    "validate_guardrank_relevance_config",
]
