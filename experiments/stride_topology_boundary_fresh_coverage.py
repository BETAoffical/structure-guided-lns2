from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.stride_topology_anchor_coverage import _base_config
from experiments.stride_topology_boundary_coverage import (
    _augment,
    analyze_topology_boundary_coverage_rows,
)
from experiments.stride_topology_coverage import (
    _collect_topology_coverage,
    validate_topology_coverage_config,
)


CONFIG_SCHEMA = "lns2.stride.topology_boundary_fresh_coverage_config.v1"


def validate_topology_boundary_fresh_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected fresh topology-boundary coverage config")
    validate_topology_coverage_config(_base_config(config))
    if (
        config.get("diagnostic_id") != "stride-topoboundary-fresh-coverage-v1"
        or config.get("candidate_generator_id") != "stride-topoboundary-v1"
        or config.get("predecessor_id") != "stride-topoboundary-fresh-preflight-v1"
    ):
        raise ValueError("fresh topology-boundary diagnostic identity changed")
    if dict(config.get("freshness") or {}) != {
        "task_id_overlap_with_consumed_cohort": 0,
        "quality_outcomes_consumed": False,
        "scope": "within_map_fresh_od_proposal_audit",
    }:
        raise ValueError("fresh topology-boundary isolation changed")
    if set(config.get("inputs") or {}) != {
        "preflight_report", "dataset_manifest", "qualification_manifest",
        "qualification_report", "source_config", "runtime_config",
        "predecessor_boundary_report",
    }:
        raise ValueError("fresh topology-boundary input registry changed")
    if dict(config.get("augmentation") or {}) != {
        "strategy": "one_endpoint_topology_core_then_global_boundary_fill",
        "kinds": ["articulation", "low_degree"],
        "neighborhood_size": 16,
        "topology_core_budget": 4,
        "maximum_added_candidates_per_state": 2,
        "maximum_total_candidates_per_state": 20,
        "selection_objective": "new_incident_minus_internal_closure_then_component_diversity",
    }:
        raise ValueError("fresh topology-boundary augmentation changed")
    if dict(config.get("boundary_gates") or {}) != {
        "minimum_mean_incident_articulation_coverage": 0.70,
        "minimum_mean_incident_low_degree_coverage": 0.70,
        "minimum_mean_boundary_articulation_ratio": 0.65,
        "minimum_mean_boundary_low_degree_ratio": 0.65,
        "minimum_ultra_mean_boundary_articulation_ratio": 0.65,
        "minimum_ultra_mean_boundary_low_degree_ratio": 0.65,
        "minimum_mean_global_event_incident_delta_vs_base_frontier": -0.02,
        "minimum_mean_global_boundary_ratio_delta_vs_base_frontier": 0.0,
        "minimum_mean_component_reach_delta_vs_base_frontier": -0.05,
    }:
        raise ValueError("fresh topology-boundary gates changed")


def _analyze(config: dict[str, Any], state_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = analyze_topology_boundary_coverage_rows(config, state_rows, candidate_rows)
    report["diagnostic_id"] = str(config["diagnostic_id"])
    report["candidate_generator_id"] = str(config["candidate_generator_id"])
    report["freshness"] = dict(config["freshness"])
    return report


def collect_topology_boundary_fresh_coverage(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    return _collect_topology_coverage(
        config_path, output,
        validator=validate_topology_boundary_fresh_coverage_config,
        analyzer=_analyze,
        augmenter=_augment,
        analyzer_uses_candidate_rows=True,
    )


__all__ = [
    "collect_topology_boundary_fresh_coverage",
    "validate_topology_boundary_fresh_coverage_config",
]
