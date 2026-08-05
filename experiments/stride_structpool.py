from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DESIGN_SCHEMA = "lns2.stride.structpool_design.v1"
POOL_ID = "stride-structpool-v1"
RANKER_ID = "stride-robustaction-v1"

FORBIDDEN_OUTCOME_FIELDS = {
    "candidate_repair_outcome",
    "candidate_conflicts_after",
    "candidate_runtime",
    "controller_action",
    "controller_ttf",
    "future_trajectory",
    "repair_runtime",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structpool_design(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_structpool_design(config, project_root=path.parents[1])
    return config


def high_stress_gate(state_summary: dict[str, Any], config: dict[str, Any]) -> bool:
    """Evaluate the preregistered cheap gate without topology or outcome fields."""

    gate = dict(config["high_stress_gate"])
    forbidden_present = FORBIDDEN_OUTCOME_FIELDS & set(state_summary)
    if forbidden_present:
        raise ValueError(
            "high-stress gate received forbidden outcome fields: "
            + ", ".join(sorted(forbidden_present))
        )
    required = {
        "agent_count",
        "conflict_pair_count",
        "active_conflict_agent_count",
        "largest_conflict_component_size",
    }
    missing = required - set(state_summary)
    if missing:
        raise ValueError(
            "high-stress gate is missing state fields: " + ", ".join(sorted(missing))
        )
    if int(state_summary["conflict_pair_count"]) < int(
        gate["minimum_conflict_pair_count"]
    ):
        return False
    any_of = dict(gate["any_of"])
    return (
        int(state_summary["agent_count"]) >= int(any_of["minimum_agent_count"])
        or int(state_summary["active_conflict_agent_count"])
        >= int(any_of["minimum_active_conflict_agent_count"])
        or int(state_summary["largest_conflict_component_size"])
        >= int(any_of["minimum_largest_conflict_component_size"])
    )


def validate_structpool_design(
    config: dict[str, Any], *, project_root: str | Path | None = None
) -> None:
    if config.get("schema") != DESIGN_SCHEMA:
        raise ValueError("unexpected STRIDE StructPool design schema")
    if (
        config.get("scientific_status")
        != "preregistered_before_structpool_proposal_audit_or_repair_outcomes"
        or config.get("pool_id") != POOL_ID
        or config.get("planned_ranker_id") != RANKER_ID
        or config.get("pre_registration_parent_commit")
        != "429163b213ffb4ecc564a5f8d7c6362abcac6252"
    ):
        raise ValueError("STRIDE StructPool identity changed")

    scope = dict(config.get("research_scope") or {})
    if (
        scope.get("target") != "high_load_topology_constrained_mapf"
        or scope.get("ordinary_state_policy") != "exact_frozen_v2_full_fallback"
        or bool(scope.get("formal_speed_claim"))
        or list(scope.get("development_only_map_loads") or ())
        != ["maze-300", "room-500"]
        or bool(scope.get("development_evidence_may_train_or_calibrate"))
    ):
        raise ValueError("STRIDE StructPool research scope changed")

    candidate = dict(config.get("candidate_space") or {})
    base = dict(candidate.get("base_pool") or {})
    incumbent = dict(candidate.get("incumbent_additions") or {})
    expected_groups = [
        "bottleneck_crossing",
        "conflict_component",
        "topology_boundary",
        "spatiotemporal_hotspot",
        "path_overlap",
    ]
    if (
        base.get("generator") != "frozen_v2"
        or list(base.get("heuristics") or ()) != ["target", "collision", "random"]
        or list(base.get("neighborhood_sizes") or ()) != [4, 8, 16]
        or incumbent.get("generator") != "stride-topoboundary-v1"
        or list(incumbent.get("variants") or ()) != ["articulation", "low_degree"]
        or int(incumbent.get("neighborhood_size", -1)) != 16
        or int(incumbent.get("core_budget", -1)) != 4
        or list(candidate.get("novel_family_groups") or ()) != expected_groups
        or list(candidate.get("neighborhood_sizes") or ()) != [8, 16, 24, 32]
        or int(candidate.get("maximum_added_candidates", -1)) != 6
        or float(
            candidate.get("maximum_jaccard_similarity_between_novel_additions", -1.0)
        )
        != 0.8
        or bool(candidate.get("outcome_used_for_generation_or_reduction"))
    ):
        raise ValueError("STRIDE StructPool candidate-space contract changed")

    gate = dict(config.get("high_stress_gate") or {})
    if (
        gate.get("gate_id") != "stride-highstress-state-v1"
        or gate.get("evaluation_boundary")
        != "before_static_or_dynamic_topology_analysis"
        or int(gate.get("minimum_conflict_pair_count", -1)) != 16
        or dict(gate.get("any_of") or {})
        != {
            "minimum_agent_count": 96,
            "minimum_active_conflict_agent_count": 32,
            "minimum_largest_conflict_component_size": 16,
        }
        or set(map(str, gate.get("forbidden_inputs") or ()))
        != FORBIDDEN_OUTCOME_FIELDS
        or set(map(str, gate.get("allowed_inputs") or ()))
        != {
            "agent_count",
            "conflict_pair_count",
            "active_conflict_agent_count",
            "largest_conflict_component_size",
        }
        or gate.get("failure_action") != "exact_frozen_v2_full_fallback"
    ):
        raise ValueError("STRIDE StructPool high-stress gate changed")

    proposal = dict(config.get("proposal_only_stage") or {})
    if (
        bool(proposal.get("repair_or_controller_step_allowed"))
        or int(proposal.get("minimum_eligible_state_count", -1)) != 48
        or float(proposal.get("minimum_fraction_with_three_added_candidates", -1.0))
        != 0.9
        or int(proposal.get("minimum_aggregate_novel_family_coverage", -1)) != 5
        or int(proposal.get("minimum_aggregate_neighborhood_size_coverage", -1)) != 4
        or not all(
            bool(proposal.get(field))
            for field in (
                "require_deterministic_replay",
                "require_exact_base_preservation",
                "require_exact_incumbent_boundary_preservation",
                "require_candidate_cap",
            )
        )
    ):
        raise ValueError("STRIDE StructPool proposal-only gates changed")

    headroom = dict(config.get("headroom_pilot") or {})
    if (
        not bool(headroom.get("allowed_only_after_proposal_stage_passes"))
        or list(headroom.get("paired_pp_seed_indices") or ()) != list(range(16))
        or headroom.get("label")
        != "mean_current_step_normalized_conflict_reduction"
        or bool(headroom.get("timing_fields_used"))
        or bool(headroom.get("future_state_or_round_fields_used"))
        or float(headroom.get("minimum_state_fraction_with_novel_expected_gain", -1.0))
        != 0.2
        or float(
            headroom.get("minimum_mean_best_expected_gain_over_incumbent_pool", -1.0)
        )
        != 0.01
        or float(headroom.get("minimum_gain_for_state_opportunity", -1.0)) != 0.02
    ):
        raise ValueError("STRIDE StructPool headroom pilot changed")

    data = dict(config.get("data_stage_after_headroom") or {})
    if (
        not bool(data.get("allowed_only_after_headroom_pilot_passes"))
        or dict(data.get("map_group_target_counts") or {})
        != {
            "dao_high_topology": 12,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 2,
        }
        or int(data.get("minimum_independent_episode_count", -1)) != 160
        or int(data.get("maximum_selected_state_count", -1)) != 320
        or int(data.get("maximum_states_per_episode", -1)) != 2
        or int(data.get("paired_pp_trials_per_candidate", -1)) != 16
    ):
        raise ValueError("STRIDE StructPool data-stage contract changed")

    training = dict(config.get("training_stage") or {})
    if (
        training.get("ranker_id") != RANKER_ID
        or training.get("target")
        != "seed_aggregated_expected_current_step_normalized_conflict_reduction"
        or list(training.get("risk_inputs") or ())
        != ["seed_mean", "seed_standard_deviation", "lower_half_mean"]
        or training.get("anchor") != "same_pool_frozen_v2_ranking"
        or training.get("grouping") != "map_grouped_nested_cross_validation"
    ):
        raise ValueError("STRIDE StructPool training contract changed")

    runtime = dict(config.get("runtime_stage") or {})
    power = dict(config.get("power_state_policy") or {})
    if (
        not bool(runtime.get("allowed_only_after_user_reports_comparable_performance_restored"))
        or runtime.get("primary_metric") != "mean_run_to_completion_raw_wall_ttf"
        or bool(power.get("additional_performance_preflight"))
        or bool(power.get("timing_pooling_across_power_states"))
    ):
        raise ValueError("STRIDE StructPool timing boundary changed")

    if project_root is not None:
        root = Path(project_root).resolve()
        registered = {
            "incumbent": dict(config["incumbent"]["report"]),
            "training_design": {
                "path": config["evidence_isolation"]["training_design"],
                "sha256": config["evidence_isolation"]["training_design_sha256"],
            },
            "formal_ood": {
                "path": config["evidence_isolation"]["formal_ood_config"],
                "sha256": config["evidence_isolation"]["formal_ood_config_sha256"],
            },
            "fresh_evidence": {
                "path": config["evidence_isolation"]["fresh_evidence_config"],
                "sha256": config["evidence_isolation"]["fresh_evidence_config_sha256"],
            },
        }
        for name, spec in registered.items():
            path = root / str(spec["path"])
            if not path.is_file() or _sha256(path) != str(spec["sha256"]):
                raise ValueError(f"STRIDE StructPool registered {name} checksum changed")


__all__ = [
    "DESIGN_SCHEMA",
    "POOL_ID",
    "RANKER_ID",
    "high_stress_gate",
    "load_structpool_design",
    "validate_structpool_design",
]
