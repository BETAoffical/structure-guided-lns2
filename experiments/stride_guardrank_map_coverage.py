from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.stride_topology_boundary_coverage import (
    _augment,
    analyze_topology_boundary_coverage_rows,
)
from experiments.stride_topology_coverage import _collect_topology_coverage


CONFIG_SCHEMA = "lns2.stride.guardrank_map_coverage_config.v1"
REPORT_SCHEMA = "lns2.stride.guardrank_map_coverage_report.v1"


def validate_guardrank_map_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected GuardRank map-coverage config")
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
        raise ValueError("GuardRank map coverage must remain proposal-only")
    if (
        config.get("diagnostic_id") != "stride-guardrank-mapcoverage-v1"
        or config.get("candidate_generator_id") != "stride-topoboundary-v1"
        or config.get("predecessor_id")
        != "stride-guardrank-map-expansion-preflight-v1"
        or int(config.get("expected_task_count", -1)) != 16
        or tuple(map(int, config.get("solver_seeds") or ())) != (1, 2)
        or int(config.get("expected_state_count", -1)) != 32
        or config.get("proposal_backend") != "optimized"
        or int(config.get("proposal_repetitions", -1)) != 2
        or tuple(map(int, config.get("expected_requested_sizes") or ()))
        != (4, 8, 16)
    ):
        raise ValueError("GuardRank map-coverage identity or cohort changed")
    if set(config.get("inputs") or {}) != {
        "preflight_report",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "source_config",
        "runtime_config",
        "predecessor_boundary_report",
    }:
        raise ValueError("GuardRank map-coverage input registry changed")
    if dict(config.get("freshness") or {}) != {
        "map_id_overlap_with_training_cohort": 0,
        "map_id_overlap_with_formal_ood": 0,
        "quality_outcomes_consumed": False,
        "scope": "cross_map_train_extension_proposal_audit",
    }:
        raise ValueError("GuardRank map-coverage isolation changed")
    if dict(config.get("augmentation") or {}) != {
        "strategy": "one_endpoint_topology_core_then_global_boundary_fill",
        "kinds": ["articulation", "low_degree"],
        "neighborhood_size": 16,
        "topology_core_budget": 4,
        "maximum_added_candidates_per_state": 2,
        "maximum_total_candidates_per_state": 20,
        "selection_objective": (
            "new_incident_minus_internal_closure_then_component_diversity"
        ),
    }:
        raise ValueError("GuardRank map-coverage augmentation changed")
    if dict(config.get("diagnostic_gates") or {}) != {
        "minimum_candidate_count_per_state": 12,
        "maximum_candidate_count_per_state": 20,
        "maximum_added_candidates_per_state": 2,
        "minimum_articulation_relevant_state_count": 8,
        "minimum_low_degree_relevant_state_count": 16,
        "minimum_mean_max_incident_articulation_coverage": 0.60,
        "minimum_articulation_state_fraction_at_half_coverage": 0.60,
        "minimum_mean_max_incident_low_degree_coverage": 0.75,
        "minimum_low_degree_state_fraction_at_half_coverage": 0.75,
    }:
        raise ValueError("GuardRank map-coverage diagnostic gates changed")
    if dict(config.get("topology_group_gates") or {}) != {
        "required_groups": [
            "dao_compact_high_topology",
            "dao_compact_mid_topology",
            "dao_compact_low_topology_control",
        ],
        "minimum_articulation_relevant_states_by_group": {
            "dao_compact_high_topology": 1,
            "dao_compact_mid_topology": 1,
        },
        "minimum_low_degree_relevant_states_by_group": {
            "dao_compact_high_topology": 1,
            "dao_compact_mid_topology": 1,
            "dao_compact_low_topology_control": 1,
        },
        "minimum_mean_max_incident_articulation_coverage_by_group": {
            "dao_compact_high_topology": 0.50,
            "dao_compact_mid_topology": 0.50,
        },
    }:
        raise ValueError("GuardRank map-coverage topology-group gates changed")
    if dict(config.get("boundary_gates") or {}) != {
        "minimum_mean_incident_articulation_coverage": 0.70,
        "minimum_mean_incident_low_degree_coverage": 0.70,
        "minimum_mean_boundary_articulation_ratio": 0.65,
        "minimum_mean_boundary_low_degree_ratio": 0.65,
        "minimum_ultra_mean_boundary_articulation_ratio": 0.0,
        "minimum_ultra_mean_boundary_low_degree_ratio": 0.0,
        "minimum_mean_global_event_incident_delta_vs_base_frontier": -0.02,
        "minimum_mean_global_boundary_ratio_delta_vs_base_frontier": 0.0,
        "minimum_mean_component_reach_delta_vs_base_frontier": -0.05,
    }:
        raise ValueError("GuardRank map-coverage boundary gates changed")


def _analyze(
    config: dict[str, Any],
    state_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    report = analyze_topology_boundary_coverage_rows(
        config, state_rows, candidate_rows
    )
    report["summary"].pop("ultra_mean_boundary_articulation_ratio", None)
    report["summary"].pop("ultra_mean_boundary_low_degree_ratio", None)
    report["gates"].pop(
        "minimum_ultra_mean_boundary_articulation_ratio", None
    )
    report["gates"].pop("minimum_ultra_mean_boundary_low_degree_ratio", None)
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "scientific_status": "proposal_only_fresh_map_training_extension_audit",
            "diagnostic_id": str(config["diagnostic_id"]),
            "candidate_generator_id": str(config["candidate_generator_id"]),
            "freshness": dict(config["freshness"]),
        }
    )
    report["passed"] = all(bool(value) for value in report["gates"].values())
    report["next_decision"] = config[
        "next_decision_on_pass"
        if report["passed"]
        else "next_decision_on_failure"
    ]
    return report


def collect_guardrank_map_coverage(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    return _collect_topology_coverage(
        config_path,
        output,
        validator=validate_guardrank_map_coverage_config,
        analyzer=_analyze,
        augmenter=_augment,
        analyzer_uses_candidate_rows=True,
    )


__all__ = [
    "collect_guardrank_map_coverage",
    "validate_guardrank_map_coverage_config",
]
