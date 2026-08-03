from __future__ import annotations

import math
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
from experiments.stride_topology_probe import topology_interaction_features
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.online_selection import generate_online_candidates


CONFIG_SCHEMA = "lns2.stride.topology_coverage_config.v1"
ROW_SCHEMA = "lns2.stride.topology_coverage_candidate.v1"
REPORT_SCHEMA = "lns2.stride.topology_coverage_report.v1"


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _fraction(values: list[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / len(values) if values else 0.0


def _registered_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"registered input SHA differs: {artifact['path']}")
    return path


def validate_topology_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-coverage config")
    if (
        config.get("scientific_status") != "proposal_only_development_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("candidate_repair_trials_allowed"))
        or bool(config.get("controller_actions_allowed"))
        or bool(config.get("controller_outcomes_allowed"))
    ):
        raise ValueError("topology coverage must remain proposal-only")
    if (
        config.get("diagnostic_id") != "stride-topocoverage-v1"
        or config.get("predecessor_id") != "stride-topologydiag-v1"
        or int(config.get("expected_task_count", -1)) != 12
        or tuple(map(int, config.get("solver_seeds") or ())) != (1, 2)
        or int(config.get("expected_state_count", -1)) != 24
        or config.get("proposal_backend") != "optimized"
        or int(config.get("proposal_repetitions", -1)) != 2
        or tuple(map(int, config.get("expected_requested_sizes") or ())) != (4, 8, 16)
    ):
        raise ValueError("topology-coverage identity or cohort changed")
    if dict(config.get("diagnostic_gates") or {}) != {
        "minimum_candidate_count_per_state": 12,
        "minimum_articulation_relevant_state_count": 8,
        "minimum_low_degree_relevant_state_count": 12,
        "minimum_mean_max_incident_articulation_coverage": 0.60,
        "minimum_articulation_state_fraction_at_half_coverage": 0.60,
        "minimum_mean_max_incident_low_degree_coverage": 0.75,
        "minimum_low_degree_state_fraction_at_half_coverage": 0.75,
    }:
        raise ValueError("topology-coverage diagnostic gates changed")
    group_gates = dict(config.get("topology_group_gates") or {})
    if group_gates != {
        "required_groups": [
            "dao_ultra_bottleneck",
            "dao_articulated",
            "dao_low_articulation_control",
        ],
        "minimum_articulation_relevant_states_by_group": {
            "dao_ultra_bottleneck": 1,
            "dao_articulated": 1,
        },
        "minimum_low_degree_relevant_states_by_group": {
            "dao_ultra_bottleneck": 1,
            "dao_articulated": 1,
            "dao_low_articulation_control": 1,
        },
        "minimum_mean_max_incident_articulation_coverage_by_group": {
            "dao_ultra_bottleneck": 0.50,
            "dao_articulated": 0.50,
        },
    }:
        raise ValueError("topology-coverage group gates changed")
    if set(config.get("inputs") or {}) != {
        "preflight_report",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "source_config",
        "runtime_config",
    }:
        raise ValueError("topology-coverage input registry changed")


def _candidate_signature(candidates: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            str(candidate["candidate_id"]),
            tuple(map(int, candidate["agents"])),
            tuple(map(str, candidate["selection_families"])),
        )
        for candidate in candidates
    ]


def _coverage_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    relevant = [row for row in rows if bool(row[f"{kind}_relevant"])]
    values = [float(row[f"max_incident_{kind}_coverage"]) for row in relevant]
    return {
        "relevant_state_count": len(relevant),
        "mean_max_incident_coverage": _mean(values),
        "state_fraction_at_half_coverage": _fraction(values, 0.5),
        "mean_max_internal_coverage": _mean(
            [float(row[f"max_internal_{kind}_coverage"]) for row in relevant]
        ),
    }


def analyze_topology_coverage_rows(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_states = int(config["expected_state_count"])
    expected_tasks = int(config["expected_task_count"])
    expected_seeds = set(map(int, config["solver_seeds"]))
    expected_sizes = set(map(int, config["expected_requested_sizes"]))
    if len({str(row["state_id"]) for row in rows}) != len(rows):
        raise ValueError("topology coverage repeats a state")
    tasks = {str(row["task_id"]) for row in rows}
    complete_pairing = all(
        {int(row["solver_seed"]) for row in rows if row["task_id"] == task}
        == expected_seeds
        for task in tasks
    )
    articulation = _coverage_summary(rows, "articulation")
    low_degree = _coverage_summary(rows, "low_degree")
    required_groups = list(config["topology_group_gates"]["required_groups"])
    group_reports: dict[str, Any] = {}
    for group in required_groups:
        group_rows = [row for row in rows if row["layout_family"] == group]
        group_reports[group] = {
            "state_count": len(group_rows),
            "articulation": _coverage_summary(group_rows, "articulation"),
            "low_degree": _coverage_summary(group_rows, "low_degree"),
        }

    gates_config = dict(config["diagnostic_gates"])
    group_gates = dict(config["topology_group_gates"])
    gates = {
        "expected_state_count": len(rows) == expected_states,
        "expected_task_count": len(tasks) == expected_tasks,
        "complete_solver_seed_pairing": complete_pairing,
        "all_state_fingerprints_preserved": all(
            bool(row["state_fingerprint_preserved"]) for row in rows
        ),
        "all_proposal_repetitions_deterministic": all(
            bool(row["proposal_repetitions_deterministic"]) for row in rows
        ),
        "minimum_candidate_count_per_state": bool(rows)
        and min(int(row["candidate_count"]) for row in rows)
        >= int(gates_config["minimum_candidate_count_per_state"]),
        "all_requested_sizes_represented": all(
            set(map(int, row["requested_sizes_represented"])) == expected_sizes
            for row in rows
        ),
        "minimum_articulation_relevant_state_count": articulation[
            "relevant_state_count"
        ]
        >= int(gates_config["minimum_articulation_relevant_state_count"]),
        "minimum_low_degree_relevant_state_count": low_degree[
            "relevant_state_count"
        ]
        >= int(gates_config["minimum_low_degree_relevant_state_count"]),
        "minimum_mean_max_incident_articulation_coverage": articulation[
            "mean_max_incident_coverage"
        ]
        >= float(gates_config["minimum_mean_max_incident_articulation_coverage"]),
        "minimum_articulation_state_fraction_at_half_coverage": articulation[
            "state_fraction_at_half_coverage"
        ]
        >= float(gates_config["minimum_articulation_state_fraction_at_half_coverage"]),
        "minimum_mean_max_incident_low_degree_coverage": low_degree[
            "mean_max_incident_coverage"
        ]
        >= float(gates_config["minimum_mean_max_incident_low_degree_coverage"]),
        "minimum_low_degree_state_fraction_at_half_coverage": low_degree[
            "state_fraction_at_half_coverage"
        ]
        >= float(gates_config["minimum_low_degree_state_fraction_at_half_coverage"]),
        "all_topology_groups_represented": all(
            group_reports[group]["state_count"] > 0 for group in required_groups
        ),
    }
    for group, minimum in group_gates[
        "minimum_articulation_relevant_states_by_group"
    ].items():
        gates[f"{group}_minimum_articulation_relevant_states"] = (
            group_reports[group]["articulation"]["relevant_state_count"]
            >= int(minimum)
        )
    for group, minimum in group_gates[
        "minimum_low_degree_relevant_states_by_group"
    ].items():
        gates[f"{group}_minimum_low_degree_relevant_states"] = (
            group_reports[group]["low_degree"]["relevant_state_count"]
            >= int(minimum)
        )
    for group, minimum in group_gates[
        "minimum_mean_max_incident_articulation_coverage_by_group"
    ].items():
        gates[f"{group}_minimum_mean_max_incident_articulation_coverage"] = (
            group_reports[group]["articulation"]["mean_max_incident_coverage"]
            >= float(minimum)
        )
    passed = all(gates.values())
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    return {
        "schema": REPORT_SCHEMA,
        "scientific_status": "proposal_only_development_diagnostic",
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "state_count": len(rows),
        "task_count": len(tasks),
        "candidate_count": sum(candidate_counts),
        "candidate_count_per_state": {
            "minimum": min(candidate_counts, default=0),
            "mean": _mean(list(map(float, candidate_counts))),
            "maximum": max(candidate_counts, default=0),
        },
        "articulation": articulation,
        "low_degree": low_degree,
        "topology_groups": group_reports,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }


def _collect_topology_coverage(
    config_path: str | Path,
    output: str | Path,
    *,
    validator: Any,
    analyzer: Any,
    augmenter: Any = None,
    analyzer_uses_candidate_rows: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validator(config)
    inputs = {
        name: _registered_path(project_root, dict(artifact))
        for name, artifact in dict(config["inputs"]).items()
    }
    preflight = _read_json(inputs["preflight_report"])
    if preflight.get("passed") is not True:
        raise ValueError("topology coverage requires a passed preflight")
    recommended = list(preflight.get("recommended_tasks") or [])
    if len(recommended) != int(config["expected_task_count"]):
        raise ValueError("preflight recommended-task count differs")
    dataset_rows = {
        str(row["task_id"]): row for row in _read_jsonl(inputs["dataset_manifest"])
    }
    qualification_rows = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(inputs["qualification_manifest"])
    }
    runtime = _read_json(inputs["runtime_config"])
    dataset_root = (project_root / str(config["dataset_root"])).resolve()
    state_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for selected in sorted(recommended, key=lambda row: str(row["task_id"])):
        task_id = str(selected["task_id"])
        dataset_row = dataset_rows.get(task_id)
        if dataset_row is None:
            raise ValueError(f"recommended task is absent from dataset: {task_id}")
        for solver_seed in map(int, config["solver_seeds"]):
            qualification = qualification_rows.get((task_id, solver_seed))
            if qualification is None or qualification.get("status") != "ok":
                raise ValueError(f"qualification row is absent or invalid: {task_id}")
            job = {
                "dataset_root": str(dataset_root),
                "row": dataset_row,
                "environment": runtime["environment"],
                "solver_seed": solver_seed,
                "replay_destroy_strategy": "Adaptive",
            }
            environment, state = replay_prefix(job, [])
            fingerprint = state_fingerprint(state)
            if (
                fingerprint != str(qualification["state_fingerprint"])
                or int(state["num_of_colliding_pairs"])
                != int(qualification["initial_conflicts"])
            ):
                raise RuntimeError(f"qualified initial state did not replay: {task_id}")
            analysis = analyze_state(state)
            repetitions: list[list[dict[str, Any]]] = []
            generations = []
            for _ in range(int(config["proposal_repetitions"])):
                candidates, generation = generate_online_candidates(
                    environment,
                    state,
                    task_id=task_id,
                    solver_seed=solver_seed,
                    decision_index=0,
                    proposal_config=dict(runtime["proposal"]),
                    state_hash=fingerprint,
                    verify_full_state=True,
                    proposal_backend=str(config["proposal_backend"]),
                    shadow_validation=False,
                )
                if augmenter is not None:
                    candidates = augmenter(state, analysis, candidates, config)
                repetitions.append(candidates)
                generations.append(generation)
            signatures = [_candidate_signature(candidates) for candidates in repetitions]
            deterministic = all(signature == signatures[0] for signature in signatures[1:])
            after_fingerprint = state_fingerprint(_plain(environment.get_state()))
            preserved = after_fingerprint == fingerprint
            candidates = repetitions[0]
            feature_rows = []
            for candidate in candidates:
                features = topology_interaction_features(
                    state, analysis, list(map(int, candidate["agents"]))
                )
                row = {
                    "schema": ROW_SCHEMA,
                    "state_id": f"{task_id}::solver_seed_{solver_seed}",
                    "task_id": task_id,
                    "solver_seed": solver_seed,
                    "map_id": str(selected["map_id"]),
                    "layout_family": str(selected["layout_family"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "agents": list(map(int, candidate["agents"])),
                    "actual_size": int(candidate["actual_size"]),
                    "selection_families": list(map(str, candidate["selection_families"])),
                    "features": features,
                }
                if "proposal_audit" in candidate:
                    row["proposal_audit"] = _plain(candidate["proposal_audit"])
                candidate_rows.append(row)
                feature_rows.append(row)
            first_features = dict(feature_rows[0]["features"])
            art_relevant = first_features[
                "topology.state.articulation_event_ratio"
            ] > 0.0
            low_relevant = first_features[
                "topology.state.low_degree_event_ratio"
            ] > 0.0
            requested_sizes = sorted(
                {
                    int(family.rsplit(":", 1)[1])
                    for candidate in candidates
                    for family in candidate["selection_families"]
                }
            )
            state_row = {
                    "state_id": f"{task_id}::solver_seed_{solver_seed}",
                    "task_id": task_id,
                    "solver_seed": solver_seed,
                    "map_id": str(selected["map_id"]),
                    "layout_family": str(selected["layout_family"]),
                    "initial_conflicts": int(state["num_of_colliding_pairs"]),
                    "state_fingerprint": fingerprint,
                    "state_fingerprint_preserved": preserved,
                    "proposal_repetitions_deterministic": deterministic,
                    "candidate_count": len(candidates),
                    "proposal_count": int(generations[0]["proposal_count"]),
                    "unique_neighborhood_count": int(
                        generations[0]["unique_neighborhood_count"]
                    ),
                    "proposal_backend": str(generations[0]["backend"]),
                    "requested_sizes_represented": requested_sizes,
                    "articulation_relevant": art_relevant,
                    "low_degree_relevant": low_relevant,
                    "max_incident_articulation_coverage": max(
                        float(row["features"]["topology.realized.incident_articulation_event_coverage"])
                        for row in feature_rows
                    ),
                    "max_internal_articulation_coverage": max(
                        float(row["features"]["topology.realized.internal_articulation_event_coverage"])
                        for row in feature_rows
                    ),
                    "max_incident_low_degree_coverage": max(
                        float(row["features"]["topology.realized.incident_low_degree_event_coverage"])
                        for row in feature_rows
                    ),
                    "max_internal_low_degree_coverage": max(
                        float(row["features"]["topology.realized.internal_low_degree_event_coverage"])
                        for row in feature_rows
                    ),
                }
            if augmenter is not None:
                state_row["base_candidate_count"] = int(
                    generations[0]["candidate_count"]
                )
                state_row["added_candidate_count"] = len(candidates) - int(
                    generations[0]["candidate_count"]
                )
            state_rows.append(state_row)
    state_rows.sort(key=lambda row: str(row["state_id"]))
    candidate_rows.sort(
        key=lambda row: (str(row["state_id"]), str(row["candidate_id"]))
    )
    report = (
        analyzer(config, state_rows, candidate_rows)
        if analyzer_uses_candidate_rows
        else analyzer(config, state_rows)
    )
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "topology_coverage_states.jsonl"
    candidate_path = output_root / "topology_coverage_candidates.jsonl"
    _write_jsonl(state_path, state_rows)
    _write_jsonl(candidate_path, candidate_rows)
    report["inputs"] = {
        "config_sha256": sha256_file(config_path),
        **{
            f"{name}_sha256": sha256_file(path) for name, path in sorted(inputs.items())
        },
    }
    report["artifacts"] = {
        "state_rows_sha256": sha256_file(state_path),
        "candidate_rows_sha256": sha256_file(candidate_path),
    }
    _write_json(output_root / "topology_coverage_report.json", report)
    return report


def collect_topology_coverage(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    return _collect_topology_coverage(
        config_path,
        output,
        validator=validate_topology_coverage_config,
        analyzer=analyze_topology_coverage_rows,
    )


__all__ = [
    "_collect_topology_coverage",
    "analyze_topology_coverage_rows",
    "collect_topology_coverage",
    "validate_topology_coverage_config",
]
