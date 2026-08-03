from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.stride_topology_anchor_quality import (
    _analyze_topology_quality_pilot,
    _collect_topology_quality_pilot,
)


CONFIG_SCHEMA = "lns2.stride.topology_boundary_quality_pilot_config.v1"
TRIAL_SCHEMA = "lns2.stride.topology_boundary_quality_trial.v1"
STATE_SCHEMA = "lns2.stride.topology_boundary_quality_state.v1"
COLLECTION_SCHEMA = "lns2.stride.topology_boundary_quality_collection.v1"
REPORT_SCHEMA = "lns2.stride.topology_boundary_quality_report.v1"


BOUNDARY_PROTOCOL = {
    "quality_name": "topology-boundary",
    "scientific_status": "paired_four_seed_immediate_quality_pilot",
    "trial_schema": TRIAL_SCHEMA,
    "state_schema": STATE_SCHEMA,
    "collection_schema": COLLECTION_SCHEMA,
    "report_schema": REPORT_SCHEMA,
    "augmented_prefix_key": "boundary_prefix",
    "augmented_label": "boundary",
    "expected_only_count_key": "expected_boundary_only_candidate_count",
    "minimum_top3_gate": "minimum_boundary_top3_state_rate",
    "maximum_regret_gate": "maximum_mean_boundary_best_normalized_regret",
    "report_filename": "topology_boundary_quality_report.json",
}


def validate_topology_boundary_quality_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-boundary quality config")
    if (
        config.get("scientific_status")
        != "paired_four_seed_immediate_quality_pilot"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("future_repair_rounds_used"))
        or bool(config.get("cost_to_go_used"))
        or bool(config.get("runtime_used_in_label"))
    ):
        raise ValueError("topology-boundary quality must remain immediate and non-promoting")
    if (
        config.get("pilot_id") != "stride-topoboundary-quality-pilot-v1"
        or config.get("candidate_generator_id") != "stride-topoboundary-v1"
        or config.get("baseline_pool_id") != "target-collision-random-v2-frozen"
        or int(config.get("expected_state_count", -1)) != 18
        or int(config.get("expected_candidate_count", -1)) != 347
        or int(config.get("expected_base_candidate_count", -1)) != 324
        or int(config.get("expected_boundary_only_candidate_count", -1)) != 23
        or int(config.get("expected_outcome_count", -1)) != 1388
        or tuple(map(int, config.get("trial_indices") or ())) != (0, 1, 2, 3)
        or tuple(map(int, config.get("first_half_indices") or ())) != (0, 1)
        or tuple(map(int, config.get("second_half_indices") or ())) != (2, 3)
        or int(config.get("workers", 0)) != 4
    ):
        raise ValueError("topology-boundary quality cohort or identity changed")
    if dict(config.get("expected_state_count_by_group") or {}) != {
        "dao_ultra_bottleneck": 6,
        "dao_articulated": 8,
        "dao_low_articulation_control": 4,
    }:
        raise ValueError("topology-boundary quality group registry changed")
    if dict(config.get("freshness") or {}) != {
        "scope": "within_map_fresh_od_immediate_quality",
        "task_id_overlap_with_consumed_cohort": 0,
        "prior_quality_outcomes_used_for_state_or_candidate_selection": False,
    }:
        raise ValueError("topology-boundary quality freshness registry changed")
    if dict(config.get("label") or {}) != {
        "id": "stride-topoboundary-mean-np100-v1",
        "mode": "mean_np100_current_step",
        "structure_weight": 0.02,
        "no_progress_penalty": 0.10,
        "primary_term": "normalized_current_conflict_reduction",
        "paired_seed_scope": "same_state_and_trial_index_for_every_candidate",
    }:
        raise ValueError("topology-boundary quality label changed")
    if dict(config.get("pilot_gates") or {}) != {
        "minimum_augmented_pool_strict_win_rate": 0.20,
        "minimum_mean_normalized_augmented_pool_gain": 0.01,
        "minimum_boundary_top3_state_rate": 0.35,
        "maximum_mean_boundary_best_normalized_regret": 0.15,
        "minimum_strict_wins_by_group": {
            "dao_ultra_bottleneck": 1,
            "dao_articulated": 1,
            "dao_low_articulation_control": 0,
        },
    }:
        raise ValueError("topology-boundary quality gates changed")
    if not bool(config.get("four_seed_uncertainty_is_diagnostic_only")):
        raise ValueError("four-seed uncertainty may not promote a model")
    if set(config.get("inputs") or {}) != {
        "coverage_report",
        "coverage_state_rows",
        "coverage_candidate_rows",
        "dataset_manifest",
        "qualification_manifest",
        "runtime_config",
    }:
        raise ValueError("topology-boundary quality input registry changed")


def collect_topology_boundary_quality_pilot(
    config_path: str | Path, output: str | Path, *, resume: bool = True
) -> dict[str, Any]:
    return _collect_topology_quality_pilot(
        config_path,
        output,
        resume=resume,
        validator=validate_topology_boundary_quality_config,
        protocol=BOUNDARY_PROTOCOL,
    )


def analyze_topology_boundary_quality_pilot(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    return _analyze_topology_quality_pilot(
        config_path,
        collection,
        output,
        validator=validate_topology_boundary_quality_config,
        protocol=BOUNDARY_PROTOCOL,
    )


__all__ = [
    "analyze_topology_boundary_quality_pilot",
    "collect_topology_boundary_quality_pilot",
    "validate_topology_boundary_quality_config",
]
