from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.stride_topology_coverage import (
    _collect_topology_coverage,
    analyze_topology_coverage_rows,
    validate_topology_coverage_config,
)
from lns2_selector.runtime.topology_candidates import (
    generate_topology_anchor_candidates,
    merge_topology_anchor_candidates,
)


CONFIG_SCHEMA = "lns2.stride.topology_anchor_coverage_config.v1"


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    base_gate_names = {
        "minimum_candidate_count_per_state",
        "minimum_articulation_relevant_state_count",
        "minimum_low_degree_relevant_state_count",
        "minimum_mean_max_incident_articulation_coverage",
        "minimum_articulation_state_fraction_at_half_coverage",
        "minimum_mean_max_incident_low_degree_coverage",
        "minimum_low_degree_state_fraction_at_half_coverage",
    }
    base_input_names = {
        "preflight_report",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "source_config",
        "runtime_config",
    }
    return {
        **config,
        "schema": "lns2.stride.topology_coverage_config.v1",
        "diagnostic_id": "stride-topocoverage-v1",
        "predecessor_id": "stride-topologydiag-v1",
        "inputs": {
            name: artifact
            for name, artifact in config["inputs"].items()
            if name in base_input_names
        },
        "diagnostic_gates": {
            name: value
            for name, value in config["diagnostic_gates"].items()
            if name in base_gate_names
        },
    }


def validate_topology_anchor_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-anchor coverage config")
    validate_topology_coverage_config(_base_config(config))
    if (
        config.get("diagnostic_id") != "stride-topoanchor-coverage-v1"
        or config.get("candidate_generator_id") != "stride-topoanchor-v1"
        or config.get("predecessor_id") != "stride-topocoverage-v1"
    ):
        raise ValueError("topology-anchor diagnostic identity changed")
    if dict(config.get("augmentation") or {}) != {
        "strategy": "topology_relevant_conflict_pair_set_cover",
        "kinds": ["articulation", "low_degree"],
        "neighborhood_sizes": [4, 8, 16],
        "maximum_added_candidates_per_state": 6,
        "maximum_total_candidates_per_state": 24,
        "fill_order": "selected_conflict_adjacency_then_conflict_degree_then_event_weight",
    }:
        raise ValueError("topology-anchor augmentation protocol changed")
    if set(config["inputs"]) != {
        "preflight_report",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "source_config",
        "runtime_config",
        "predecessor_report",
        "predecessor_state_rows",
        "predecessor_candidate_rows",
    }:
        raise ValueError("topology-anchor input registry changed")
    if dict(config["diagnostic_gates"]).get("maximum_candidate_count_per_state") != 24:
        raise ValueError("topology-anchor candidate cap changed")
    if dict(config["diagnostic_gates"]).get("maximum_added_candidates_per_state") != 6:
        raise ValueError("topology-anchor addition cap changed")


def analyze_topology_anchor_coverage_rows(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    report = analyze_topology_coverage_rows(config, rows)
    maximum_total = int(config["diagnostic_gates"]["maximum_candidate_count_per_state"])
    maximum_added = int(config["diagnostic_gates"]["maximum_added_candidates_per_state"])
    report["gates"]["maximum_candidate_count_per_state"] = bool(rows) and max(
        int(row["candidate_count"]) for row in rows
    ) <= maximum_total
    report["gates"]["maximum_added_candidates_per_state"] = bool(rows) and max(
        int(row["added_candidate_count"]) for row in rows
    ) <= maximum_added
    report["passed"] = all(report["gates"].values())
    report["next_decision"] = config[
        "next_decision_on_pass" if report["passed"] else "next_decision_on_failure"
    ]
    report["diagnostic_id"] = str(config["diagnostic_id"])
    report["candidate_generator_id"] = str(config["candidate_generator_id"])
    report["augmentation"] = {
        "added_candidate_count": sum(int(row["added_candidate_count"]) for row in rows),
        "maximum_added_candidate_count_per_state": max(
            (int(row["added_candidate_count"]) for row in rows), default=0
        ),
        "mean_added_candidate_count_per_state": (
            sum(int(row["added_candidate_count"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }
    return report


def _augment(
    state: dict[str, Any], analysis: Any, candidates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    anchors = generate_topology_anchor_candidates(
        state, analysis, config["augmentation"]["neighborhood_sizes"]
    )
    return merge_topology_anchor_candidates(candidates, anchors)


def collect_topology_anchor_coverage(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    return _collect_topology_coverage(
        config_path,
        output,
        validator=validate_topology_anchor_coverage_config,
        analyzer=analyze_topology_anchor_coverage_rows,
        augmenter=_augment,
    )


__all__ = [
    "analyze_topology_anchor_coverage_rows",
    "collect_topology_anchor_coverage",
    "validate_topology_anchor_coverage_config",
]
