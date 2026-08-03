from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import StateAnalysis, analyze_state
from experiments.stride_collection import _replay_job
from experiments.stride_robuststep import (
    _checked_input,
    _feature_probe_pairs,
    _feature_probe_record,
    _feature_probe_summary,
    _fit_feature_probe,
    _rebuild_stepgate_states,
    validate_robuststep_feature_probe_config,
)
from experiments.trace_replay import replay_prefix
from experiments.v2_factorial_audit import _select_model


CONFIG_SCHEMA = "lns2.stride.topology_probe_config.v1"
EXTRACTION_SCHEMA = "lns2.stride.topology_probe_extraction.v1"
FEATURE_ROW_SCHEMA = "lns2.stride.topology_interaction_features.v1"
REPORT_SCHEMA = "lns2.stride.topology_probe_report.v1"

STATE_FEATURE_NAMES = (
    "topology.state.articulation_event_ratio",
    "topology.state.low_degree_event_ratio",
)
CANDIDATE_FEATURE_NAMES = (
    "topology.realized.internal_articulation_event_coverage",
    "topology.realized.incident_articulation_event_coverage",
    "topology.realized.boundary_articulation_event_ratio",
    "topology.realized.internal_low_degree_event_coverage",
    "topology.realized.incident_low_degree_event_coverage",
    "topology.realized.boundary_low_degree_event_ratio",
    "topology.realized.articulation_agent_coverage",
    "topology.realized.low_degree_agent_coverage",
    "topology.realized.path_articulation_cell_coverage",
    "topology.realized.path_low_degree_cell_coverage",
    "topology.realized.articulation_visit_heat_coverage",
    "topology.realized.low_degree_visit_heat_coverage",
)
FEATURE_NAMES = (*STATE_FEATURE_NAMES, *CANDIDATE_FEATURE_NAMES)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def validate_topology_probe_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-probe config")
    if (
        config.get("scientific_status") != "consumed_topology_feature_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("diagnostic_result_may_promote_model"))
        or bool(config.get("candidate_repair_trials_allowed_during_extraction"))
    ):
        raise ValueError("topology probe must remain no-PP and diagnostic-only")
    if (
        config.get("diagnostic_controller_id") != "stride-topologydiag-v1"
        or config.get("baseline_id") != "stride-stepdiag-v1/exact-v2-86"
        or config.get("variant_id") != "stride-topologydiag-v1/exact-v2-86+interaction14"
        or config.get("feature_schema") != FEATURE_ROW_SCHEMA
        or tuple(map(str, config.get("feature_names") or ())) != FEATURE_NAMES
    ):
        raise ValueError("topology-probe identity or feature registry changed")
    folds = dict(config.get("fold_protocol") or {})
    maps = tuple(map(str, folds.get("held_out_maps") or ()))
    if (
        folds.get("mode") != "leave_one_whole_map_out"
        or int(folds.get("fold_count", -1)) != 6
        or len(maps) != 6
        or len(set(maps)) != 6
    ):
        raise ValueError("topology probe requires six whole-map folds")
    if dict(config.get("diagnostic_gates") or {}) != {
        "minimum_stable_state_count": 24,
        "minimum_overall_regret_improvement_vs_exact86": 0.01,
        "minimum_stable_regret_improvement_vs_exact86": 0.02,
        "minimum_stable_top3_delta_vs_exact86": 0.0,
        "minimum_map_regret_win_count_vs_exact86": 3,
        "maximum_worst_map_regret_degradation_vs_exact86": 0.05,
        "maximum_ost102d_regret_degradation_vs_exact86": 0.0,
    }:
        raise ValueError("topology-probe diagnostic gates changed")
    definitions = dict(config.get("definitions") or {})
    if definitions != {
        "low_degree_cell": "free_grid_degree_at_most_two",
        "articulation_event": "any_event_cell_is_static_grid_articulation",
        "low_degree_event": "any_event_cell_has_free_grid_degree_at_most_two",
        "internal_event": "both_conflicting_agents_are_selected",
        "incident_event": "at_least_one_conflicting_agent_is_selected",
        "boundary_event": "exactly_one_conflicting_agent_is_selected",
        "path_cell_coverage": "unique_selected_path_cells_over_unique_all_path_cells",
        "visit_heat_coverage": "global_visit_heat_on_unique_selected_cells_over_all_cells",
    }:
        raise ValueError("topology-probe feature definitions changed")


def topology_interaction_features(
    state: dict[str, Any], analysis: StateAnalysis, selected_agents: list[int]
) -> dict[str, float]:
    selected = set(map(int, selected_agents))
    paths = {
        int(agent["id"]): {int(cell) for cell in agent["path"]}
        for agent in state["agents"]
    }
    if not selected or not selected <= set(paths):
        raise ValueError("topology features require a non-empty known neighborhood")
    low_degree = {
        cell for cell in analysis.free_cells if int(analysis.degrees.get(cell, 0)) <= 2
    }
    all_path_cells = set().union(*paths.values())
    selected_path_cells = set().union(*(paths[agent] for agent in selected))
    articulation_path_cells = all_path_cells & analysis.articulation
    low_degree_path_cells = all_path_cells & low_degree
    selected_articulation_cells = selected_path_cells & analysis.articulation
    selected_low_degree_cells = selected_path_cells & low_degree
    articulation_agents = {
        agent for agent, path in paths.items() if path & analysis.articulation
    }
    low_degree_agents = {agent for agent, path in paths.items() if path & low_degree}

    articulation_events = [
        event
        for event in analysis.events
        if any(cell in analysis.articulation for cell in event.cells)
    ]
    low_degree_events = [
        event
        for event in analysis.events
        if any(cell in low_degree for cell in event.cells)
    ]

    def event_counts(events: list[Any]) -> tuple[int, int, int]:
        internal = sum(event.left in selected and event.right in selected for event in events)
        incident = sum(event.left in selected or event.right in selected for event in events)
        boundary = sum((event.left in selected) != (event.right in selected) for event in events)
        return internal, incident, boundary

    art_internal, art_incident, art_boundary = event_counts(articulation_events)
    low_internal, low_incident, low_boundary = event_counts(low_degree_events)

    def heat(cells: set[int]) -> float:
        return math.fsum(float(analysis.visit_heat.get(cell, 0)) for cell in cells)

    values = {
        "topology.state.articulation_event_ratio": _ratio(
            len(articulation_events), len(analysis.events)
        ),
        "topology.state.low_degree_event_ratio": _ratio(
            len(low_degree_events), len(analysis.events)
        ),
        "topology.realized.internal_articulation_event_coverage": _ratio(
            art_internal, len(articulation_events)
        ),
        "topology.realized.incident_articulation_event_coverage": _ratio(
            art_incident, len(articulation_events)
        ),
        "topology.realized.boundary_articulation_event_ratio": _ratio(
            art_boundary, art_incident
        ),
        "topology.realized.internal_low_degree_event_coverage": _ratio(
            low_internal, len(low_degree_events)
        ),
        "topology.realized.incident_low_degree_event_coverage": _ratio(
            low_incident, len(low_degree_events)
        ),
        "topology.realized.boundary_low_degree_event_ratio": _ratio(
            low_boundary, low_incident
        ),
        "topology.realized.articulation_agent_coverage": _ratio(
            len(selected & articulation_agents), len(articulation_agents)
        ),
        "topology.realized.low_degree_agent_coverage": _ratio(
            len(selected & low_degree_agents), len(low_degree_agents)
        ),
        "topology.realized.path_articulation_cell_coverage": _ratio(
            len(selected_articulation_cells), len(articulation_path_cells)
        ),
        "topology.realized.path_low_degree_cell_coverage": _ratio(
            len(selected_low_degree_cells), len(low_degree_path_cells)
        ),
        "topology.realized.articulation_visit_heat_coverage": _ratio(
            heat(selected_articulation_cells), heat(articulation_path_cells)
        ),
        "topology.realized.low_degree_visit_heat_coverage": _ratio(
            heat(selected_low_degree_cells), heat(low_degree_path_cells)
        ),
    }
    if tuple(values) != FEATURE_NAMES or any(
        not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("topology feature registry or finiteness differs")
    return values


def extract_topology_probe_features(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_topology_probe_config(config)
    feature_config_path = _checked_input(
        project_root, dict(config["feature_probe_config"])
    )
    feature_config = _read_json(feature_config_path)
    validate_robuststep_feature_probe_config(feature_config)
    collection = dict(config["base_collection"])
    collection_root = project_root / str(collection["root"])
    for name in ("collection_report", "run_config", "state_selection"):
        _checked_input(project_root, dict(collection[name]))
    state_files = sorted((collection_root / "states").glob("*.json"))
    if len(state_files) != int(config["expected_state_count"]):
        raise ValueError("topology extraction state-artifact count differs")

    rows = []
    replay_fingerprints = []
    map_counts: Counter[str] = Counter()
    state_hashes = {}
    for state_file in state_files:
        payload = _read_json(state_file)
        if (
            payload.get("schema") != "lns2.stride.repair_collection.v1"
            or payload.get("complete") is not True
        ):
            raise ValueError(f"invalid topology source state artifact: {state_file}")
        state_id = str(payload["state_id"])
        decision = dict(payload["decision"])
        replay = _replay_job(decision)
        _, state = replay_prefix(replay, decision["prefix_actions"])
        fingerprint = state_fingerprint(state)
        if (
            fingerprint != str(payload["before_fingerprint"])
            or fingerprint != str(decision["before_fingerprint"])
        ):
            raise RuntimeError(f"topology state replay mismatch: {state_id}")
        analysis = analyze_state(state)
        candidates = list(payload["candidates"])
        if not candidates:
            raise ValueError(f"topology source has no candidates: {state_id}")
        for candidate in candidates:
            rows.append(
                {
                    "schema": FEATURE_ROW_SCHEMA,
                    "state_id": state_id,
                    "candidate_id": str(candidate["candidate_id"]),
                    "map_id": str(decision["map_id"]),
                    "features": topology_interaction_features(
                        state, analysis, list(map(int, candidate["agents"]))
                    ),
                }
            )
        replay_fingerprints.append(fingerprint)
        map_counts[str(decision["map_id"])] += 1
        state_hashes[state_id] = sha256_file(state_file)

    rows.sort(key=lambda row: (str(row["state_id"]), str(row["candidate_id"])))
    if len(rows) != int(config["expected_candidate_count"]):
        raise ValueError("topology extraction candidate count differs")
    if len({(row["state_id"], row["candidate_id"]) for row in rows}) != len(rows):
        raise ValueError("topology extraction repeats a candidate")
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / "topology_features.jsonl"
    _write_jsonl(feature_path, rows)
    report = {
        "schema": EXTRACTION_SCHEMA,
        "scientific_status": "consumed_topology_feature_diagnostic",
        "historical_prefix_replay_used": True,
        "candidate_repair_trials_executed": False,
        "complete": True,
        "state_count": len(replay_fingerprints),
        "candidate_count": len(rows),
        "feature_dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "map_state_counts": dict(sorted(map_counts.items())),
        "unique_replay_fingerprint_count": len(set(replay_fingerprints)),
        "state_artifact_sha256": dict(sorted(state_hashes.items())),
        "topology_features_sha256": sha256_file(feature_path),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "feature_probe_config_sha256": sha256_file(feature_config_path),
            "collection_report_sha256": sha256_file(
                _checked_input(project_root, dict(collection["collection_report"]))
            ),
            "run_config_sha256": sha256_file(
                _checked_input(project_root, dict(collection["run_config"]))
            ),
            "state_selection_sha256": sha256_file(
                _checked_input(project_root, dict(collection["state_selection"]))
            ),
        },
    }
    _write_json(output_root / "topology_extraction_report.json", report)
    return report


def _model_summary(records: list[dict[str, Any]], maps: list[str]) -> dict[str, Any]:
    return {
        "overall": _feature_probe_summary(records),
        "stable_states": _feature_probe_summary(
            [row for row in records if row["oracle_half_winner_agreement"]]
        ),
        "unstable_states": _feature_probe_summary(
            [row for row in records if not row["oracle_half_winner_agreement"]]
        ),
        "by_map": {
            map_id: _feature_probe_summary(
                [row for row in records if row["map_id"] == map_id]
            )
            for map_id in maps
        },
    }


def analyze_topology_probe(
    config_path: str | Path, extraction: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_topology_probe_config(config)
    feature_config_path = _checked_input(
        project_root, dict(config["feature_probe_config"])
    )
    feature_report_path = _checked_input(
        project_root, dict(config["feature_probe_report"])
    )
    feature_config = _read_json(feature_config_path)
    validate_robuststep_feature_probe_config(feature_config)
    feature_report = _read_json(feature_report_path)
    extraction_root = Path(extraction).resolve()
    extraction_report_path = extraction_root / "topology_extraction_report.json"
    topology_path = extraction_root / "topology_features.jsonl"
    extraction_report = _read_json(extraction_report_path)
    if (
        extraction_report.get("schema") != EXTRACTION_SCHEMA
        or extraction_report.get("complete") is not True
        or extraction_report.get("historical_prefix_replay_used") is not True
        or extraction_report.get("candidate_repair_trials_executed") is not False
        or int(extraction_report.get("state_count", -1))
        != int(config["expected_state_count"])
        or int(extraction_report.get("candidate_count", -1))
        != int(config["expected_candidate_count"])
        or int(extraction_report.get("feature_dimension", -1))
        != len(FEATURE_NAMES)
        or str(extraction_report.get("topology_features_sha256"))
        != sha256_file(topology_path)
        or str(dict(extraction_report.get("inputs") or {}).get("config_sha256"))
        != sha256_file(config_path)
    ):
        raise ValueError("topology extraction report is invalid for this analysis")

    prepared = _rebuild_stepgate_states(project_root, feature_config)
    states = prepared["states"]
    state_ids = prepared["state_ids"]
    exact_specs = prepared["input_specs"]
    topology_rows = _read_jsonl(topology_path)
    topology = {
        (str(row["state_id"]), str(row["candidate_id"])): {
            str(name): float(value) for name, value in dict(row["features"]).items()
        }
        for row in topology_rows
    }
    expected_keys = {
        (state_id, str(candidate["candidate_id"]))
        for state_id, state in states.items()
        for candidate in state
    }
    if set(topology) != expected_keys or any(
        set(values) != set(FEATURE_NAMES) for values in topology.values()
    ):
        raise ValueError("topology feature/candidate coverage differs")
    augmented_states = {}
    for state_id, state in states.items():
        augmented_states[state_id] = [
            {
                **row,
                "features": {
                    **dict(row["features"]),
                    **topology[(state_id, str(row["candidate_id"]))],
                },
            }
            for row in state
        ]
    topology_specs = exact_specs + tuple(
        ("delta", name) for name in CANDIDATE_FEATURE_NAMES
    ) + tuple(("shared", name) for name in STATE_FEATURE_NAMES)
    maps = list(map(str, dict(config["fold_protocol"])["held_out_maps"]))
    parameters = dict(feature_config["model_parameters"])
    predictions = {}
    fold_rows = []
    for fold_index, held_out_map in enumerate(maps):
        train_ids = sorted(
            state_id
            for state_id, state in augmented_states.items()
            if str(state[0]["map_id"]) != held_out_map
        )
        test_ids = sorted(
            state_id
            for state_id, state in augmented_states.items()
            if str(state[0]["map_id"]) == held_out_map
        )
        train_pairs = _feature_probe_pairs(
            augmented_states, train_ids, topology_specs
        )
        test_pairs = _feature_probe_pairs(augmented_states, test_ids, topology_specs)
        estimator = _fit_feature_probe(train_pairs, parameters)
        probabilities = estimator.predict_proba(test_pairs["values"])[:, 1]
        correct = (probabilities >= 0.5) == (test_pairs["labels"] == 1)
        pairwise_accuracy = float(
            np.sum(test_pairs["weights"][correct]) / np.sum(test_pairs["weights"])
        )
        for state_id in test_ids:
            chosen = _select_model(
                augmented_states[state_id], estimator, topology_specs
            )
            predictions[state_id] = str(chosen["candidate_id"])
        fold_rows.append(
            {
                "fold_index": fold_index,
                "held_out_map": held_out_map,
                "training_state_count": len(train_ids),
                "test_state_count": len(test_ids),
                "training_directional_pair_count": train_pairs[
                    "directional_pair_count"
                ],
                "test_directional_pair_count": test_pairs["directional_pair_count"],
                "test_pairwise_accuracy": pairwise_accuracy,
            }
        )
    if set(predictions) != state_ids:
        raise ValueError("topology probe OOF predictions are incomplete")

    baseline_predictions = {
        str(row["state_id"]): str(row["selected_candidate_id"])
        for row in feature_report["records"][str(config["baseline_id"])]
    }
    if set(baseline_predictions) != state_ids:
        raise ValueError("topology baseline predictions are incomplete")
    variant_records = [
        _feature_probe_record(
            str(config["variant_id"]),
            state_id,
            predictions[state_id],
            augmented_states[state_id],
        )
        for state_id in sorted(state_ids)
    ]
    baseline_records = [
        _feature_probe_record(
            str(config["baseline_id"]),
            state_id,
            baseline_predictions[state_id],
            augmented_states[state_id],
        )
        for state_id in sorted(state_ids)
    ]
    variant = _model_summary(variant_records, maps)
    baseline = _model_summary(baseline_records, maps)
    overall_improvement = float(
        baseline["overall"]["mean_normalized_regret"]
    ) - float(variant["overall"]["mean_normalized_regret"])
    stable_improvement = float(
        baseline["stable_states"]["mean_normalized_regret"]
    ) - float(variant["stable_states"]["mean_normalized_regret"])
    stable_top3_delta = float(variant["stable_states"]["top3_hit_rate"]) - float(
        baseline["stable_states"]["top3_hit_rate"]
    )
    map_deltas = {
        map_id: float(variant["by_map"][map_id]["mean_normalized_regret"])
        - float(baseline["by_map"][map_id]["mean_normalized_regret"])
        for map_id in maps
    }
    gates_config = dict(config["diagnostic_gates"])
    gates = {
        "minimum_stable_state_count": int(variant["stable_states"]["state_count"])
        >= int(gates_config["minimum_stable_state_count"]),
        "minimum_overall_regret_improvement_vs_exact86": overall_improvement
        >= float(gates_config["minimum_overall_regret_improvement_vs_exact86"]),
        "minimum_stable_regret_improvement_vs_exact86": stable_improvement
        >= float(gates_config["minimum_stable_regret_improvement_vs_exact86"]),
        "minimum_stable_top3_delta_vs_exact86": stable_top3_delta
        >= float(gates_config["minimum_stable_top3_delta_vs_exact86"]),
        "minimum_map_regret_win_count_vs_exact86": sum(
            delta < -1e-12 for delta in map_deltas.values()
        )
        >= int(gates_config["minimum_map_regret_win_count_vs_exact86"]),
        "maximum_worst_map_regret_degradation_vs_exact86": max(map_deltas.values())
        <= float(gates_config["maximum_worst_map_regret_degradation_vs_exact86"]),
        "maximum_ost102d_regret_degradation_vs_exact86": map_deltas["ost102d"]
        <= float(gates_config["maximum_ost102d_regret_degradation_vs_exact86"]),
    }
    diagnostic_passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "consumed_topology_feature_diagnostic",
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "runtime_model_exported": False,
        "formal_ood_claim": False,
        "diagnostic_controller_id": str(config["diagnostic_controller_id"]),
        "feature_schema": FEATURE_ROW_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "input_dimension": len(topology_specs),
        "folds": fold_rows,
        "baseline": baseline,
        "variant": variant,
        "overall_regret_improvement_vs_exact86": overall_improvement,
        "stable_regret_improvement_vs_exact86": stable_improvement,
        "stable_top3_delta_vs_exact86": stable_top3_delta,
        "map_regret_deltas_vs_exact86": map_deltas,
        "map_regret_win_count_vs_exact86": sum(
            delta < -1e-12 for delta in map_deltas.values()
        ),
        "worst_map_regret_degradation_vs_exact86": max(map_deltas.values()),
        "gates": gates,
        "diagnostic_passed": diagnostic_passed,
        "diagnosis": (
            "topology_interaction_features_show_consumed_oof_signal"
            if diagnostic_passed
            else "topology_interaction_features_do_not_show_consumed_oof_signal"
        ),
        "next_decision": (
            "register_fresh_topology_balanced_feature_confirmation"
            if diagnostic_passed
            else "retain_v2_and_prioritize_broader_map_data_or_candidate_generation"
        ),
        "predictions": predictions,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "feature_probe_config_sha256": sha256_file(feature_config_path),
            "feature_probe_report_sha256": sha256_file(feature_report_path),
            "topology_extraction_report_sha256": sha256_file(
                extraction_report_path
            ),
            "topology_features_sha256": sha256_file(topology_path),
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "topology_probe_report.json", report)
    return report


__all__ = [
    "FEATURE_NAMES",
    "analyze_topology_probe",
    "extract_topology_probe_features",
    "topology_interaction_features",
    "validate_topology_probe_config",
]
