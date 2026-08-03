from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_topology_anchor_quality import (
    _aggregate_quality_scores,
    _analyze_topology_quality_pilot,
    _collect_topology_quality_pilot,
    _quality_inputs,
    _quality_ranking,
    _selected_quality_rows,
)
from experiments.stride_topology_boundary_quality import TRIAL_SCHEMA


CONFIG_SCHEMA = "lns2.stride.topology_boundary_quality_confirmation_config.v1"
STATE_SCHEMA = "lns2.stride.topology_boundary_quality_confirmation_state.v1"
COLLECTION_SCHEMA = "lns2.stride.topology_boundary_quality_confirmation_collection.v1"
REPORT_SCHEMA = "lns2.stride.topology_boundary_quality_confirmation_report.v1"


CONFIRMATION_PROTOCOL = {
    "quality_name": "topology-boundary-confirmation",
    "scientific_status": "paired_eight_seed_immediate_quality_confirmation",
    "trial_schema": TRIAL_SCHEMA,
    "state_schema": STATE_SCHEMA,
    "collection_schema": COLLECTION_SCHEMA,
    "report_schema": REPORT_SCHEMA,
    "augmented_prefix_key": "boundary_prefix",
    "augmented_label": "boundary",
    "expected_only_count_key": "expected_boundary_only_candidate_count",
    "minimum_top3_gate": "minimum_boundary_top3_state_rate",
    "maximum_regret_gate": "maximum_mean_boundary_best_normalized_regret",
    "report_filename": "topology_boundary_quality_confirmation_report.json",
}


QUALITY_GATES = {
    "minimum_augmented_pool_strict_win_rate": 0.20,
    "minimum_mean_normalized_augmented_pool_gain": 0.01,
    "minimum_boundary_top3_state_rate": 0.35,
    "maximum_mean_boundary_best_normalized_regret": 0.15,
    "minimum_strict_wins_by_group": {
        "dao_ultra_bottleneck": 1,
        "dao_articulated": 1,
        "dao_low_articulation_control": 0,
    },
}


def validate_topology_boundary_quality_confirmation_config(
    config: dict[str, Any],
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-boundary quality confirmation config")
    if (
        config.get("scientific_status")
        != "paired_eight_seed_immediate_quality_confirmation"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("future_repair_rounds_used"))
        or bool(config.get("cost_to_go_used"))
        or bool(config.get("runtime_used_in_label"))
    ):
        raise ValueError("topology-boundary confirmation must remain immediate and non-promoting")
    if (
        config.get("confirmation_id")
        != "stride-topoboundary-quality-confirmation-v1"
        or config.get("candidate_generator_id") != "stride-topoboundary-v1"
        or config.get("baseline_pool_id") != "target-collision-random-v2-frozen"
        or int(config.get("expected_state_count", -1)) != 18
        or int(config.get("expected_candidate_count", -1)) != 347
        or int(config.get("expected_base_candidate_count", -1)) != 324
        or int(config.get("expected_boundary_only_candidate_count", -1)) != 23
        or int(config.get("expected_collection_outcome_count", -1)) != 1388
        or int(config.get("expected_outcome_count", -1)) != 2776
        or tuple(map(int, config.get("collection_trial_indices") or ()))
        != (4, 5, 6, 7)
        or tuple(map(int, config.get("trial_indices") or ()))
        != (0, 1, 2, 3, 4, 5, 6, 7)
        or tuple(map(int, config.get("first_half_indices") or ())) != (0, 1, 2, 3)
        or tuple(map(int, config.get("second_half_indices") or ())) != (4, 5, 6, 7)
        or int(config.get("workers", 0)) != 4
    ):
        raise ValueError("topology-boundary confirmation cohort or identity changed")
    if dict(config.get("expected_state_count_by_group") or {}) != {
        "dao_ultra_bottleneck": 6,
        "dao_articulated": 8,
        "dao_low_articulation_control": 4,
    }:
        raise ValueError("topology-boundary confirmation group registry changed")
    if dict(config.get("freshness") or {}) != {
        "scope": "within_map_fresh_od_immediate_quality",
        "task_id_overlap_with_consumed_cohort": 0,
        "prior_quality_outcomes_used_for_state_or_candidate_selection": False,
        "new_seed_half_unseen_when_confirmation_registered": True,
    }:
        raise ValueError("topology-boundary confirmation freshness registry changed")
    if dict(config.get("label") or {}) != {
        "id": "stride-topoboundary-mean-np100-v1",
        "mode": "mean_np100_current_step",
        "structure_weight": 0.02,
        "no_progress_penalty": 0.10,
        "primary_term": "normalized_current_conflict_reduction",
        "paired_seed_scope": "same_state_and_trial_index_for_every_candidate",
    }:
        raise ValueError("topology-boundary confirmation label changed")
    if dict(config.get("pilot_gates") or {}) != QUALITY_GATES:
        raise ValueError("topology-boundary confirmation full-pool gates changed")
    if dict(config.get("independent_second_half_gates") or {}) != QUALITY_GATES:
        raise ValueError("topology-boundary confirmation independent gates changed")
    if dict(config.get("stability_gates") or {}) != {
        "minimum_half_pairwise_consistency": 0.75,
        "minimum_mean_half_top3_overlap": 0.75,
        "maximum_mean_cross_half_normalized_regret": 0.15,
    }:
        raise ValueError("topology-boundary confirmation stability gates changed")
    if not bool(config.get("eight_seed_uncertainty_is_promotion_gate")):
        raise ValueError("eight-seed uncertainty must remain a confirmation gate")
    if set(config.get("inputs") or {}) != {
        "coverage_report",
        "coverage_state_rows",
        "coverage_candidate_rows",
        "dataset_manifest",
        "qualification_manifest",
        "runtime_config",
        "base_collection_report",
        "base_trials",
        "base_analysis_report",
    }:
        raise ValueError("topology-boundary confirmation input registry changed")


def _validate_base_evidence(inputs: dict[str, Path]) -> None:
    collection = _read_json(inputs["base_collection_report"])
    analysis = _read_json(inputs["base_analysis_report"])
    rows = _read_jsonl(inputs["base_trials"])
    if (
        collection.get("complete") is not True
        or int(collection.get("error_state_count", -1)) != 0
        or int(collection.get("trial_count", -1)) != 1388
        or analysis.get("passed") is not True
        or len(rows) != 1388
        or {int(row["trial_index"]) for row in rows} != {0, 1, 2, 3}
        or {str(row.get("schema")) for row in rows} != {TRIAL_SCHEMA}
    ):
        raise ValueError("topology-boundary confirmation base evidence differs")


def collect_topology_boundary_quality_confirmation(
    config_path: str | Path, output: str | Path, *, resume: bool = True
) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = _read_json(resolved)
    validate_topology_boundary_quality_confirmation_config(config)
    _, inputs = _quality_inputs(resolved, config)
    _validate_base_evidence(inputs)
    return _collect_topology_quality_pilot(
        resolved,
        output,
        resume=resume,
        validator=validate_topology_boundary_quality_confirmation_config,
        protocol=CONFIRMATION_PROTOCOL,
    )


def _half_quality(
    *,
    rows: list[dict[str, Any]],
    candidate_rows: dict[str, list[dict[str, Any]]],
    state_metadata: dict[str, dict[str, Any]],
    indices: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = set(indices)
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["trial_index"]) in selected:
            by_state[str(row["state_id"])].append(row)
    state_reports: list[dict[str, Any]] = []
    score_config = {**config, "trial_indices": indices}
    for state_id in sorted(state_metadata):
        scores = _aggregate_quality_scores(by_state[state_id], score_config)
        kinds = {
            str(row["candidate_id"]): str(row["candidate_kind"])
            for row in candidate_rows[state_id]
        }
        base_ids = sorted(
            candidate_id
            for candidate_id, kind in kinds.items()
            if kind in {"base", "base_and_boundary"}
        )
        boundary_ids = sorted(
            candidate_id
            for candidate_id, kind in kinds.items()
            if kind == "boundary_only"
        )
        aggregated, ranking = _quality_ranking(
            scores, indices, float(config["label"]["no_progress_penalty"])
        )
        best_all = ranking[0]
        best_base = max(base_ids, key=lambda value: (aggregated[value], value))
        best_boundary = max(
            boundary_ids, key=lambda value: (aggregated[value], value)
        )
        span = max(aggregated.values()) - min(aggregated.values())
        gain = aggregated[best_all] - aggregated[best_base]
        regret = aggregated[best_all] - aggregated[best_boundary]
        state_reports.append(
            {
                "state_id": state_id,
                "layout_family": str(state_metadata[state_id]["layout_family"]),
                "augmented_pool_strict_win": (
                    best_all in boundary_ids and gain > 1e-12
                ),
                "normalized_augmented_pool_gain": gain / span if span > 1e-12 else 0.0,
                "boundary_in_top3": any(value in boundary_ids for value in ranking[:3]),
                "boundary_best_normalized_regret": (
                    regret / span if span > 1e-12 else 0.0
                ),
            }
        )
    strict_wins = sum(bool(row["augmented_pool_strict_win"]) for row in state_reports)
    groups = {}
    for group in config["expected_state_count_by_group"]:
        group_rows = [row for row in state_reports if row["layout_family"] == group]
        group_wins = sum(bool(row["augmented_pool_strict_win"]) for row in group_rows)
        groups[group] = {
            "state_count": len(group_rows),
            "strict_win_count": group_wins,
            "strict_win_rate": group_wins / len(group_rows) if group_rows else 0.0,
            "mean_normalized_augmented_pool_gain": statistics.fmean(
                float(row["normalized_augmented_pool_gain"]) for row in group_rows
            ) if group_rows else 0.0,
        }
    return {
        "trial_indices": indices,
        "state_count": len(state_reports),
        "summary": {
            "augmented_pool_strict_win_count": strict_wins,
            "augmented_pool_strict_win_rate": strict_wins / len(state_reports),
            "mean_normalized_augmented_pool_gain": statistics.fmean(
                float(row["normalized_augmented_pool_gain"]) for row in state_reports
            ),
            "boundary_top3_state_rate": statistics.fmean(
                float(row["boundary_in_top3"]) for row in state_reports
            ),
            "mean_boundary_best_normalized_regret": statistics.fmean(
                float(row["boundary_best_normalized_regret"]) for row in state_reports
            ),
        },
        "topology_groups": groups,
        "states": state_reports,
    }


def _quality_gate_results(
    half: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    summary = half["summary"]
    result = {
        "minimum_augmented_pool_strict_win_rate": float(
            summary["augmented_pool_strict_win_rate"]
        ) >= float(gates["minimum_augmented_pool_strict_win_rate"]),
        "minimum_mean_normalized_augmented_pool_gain": float(
            summary["mean_normalized_augmented_pool_gain"]
        ) >= float(gates["minimum_mean_normalized_augmented_pool_gain"]),
        "minimum_boundary_top3_state_rate": float(
            summary["boundary_top3_state_rate"]
        ) >= float(gates["minimum_boundary_top3_state_rate"]),
        "maximum_mean_boundary_best_normalized_regret": float(
            summary["mean_boundary_best_normalized_regret"]
        ) <= float(gates["maximum_mean_boundary_best_normalized_regret"]),
    }
    for group, minimum in gates["minimum_strict_wins_by_group"].items():
        result[f"{group}_minimum_strict_wins"] = (
            int(half["topology_groups"][group]["strict_win_count"])
            >= int(minimum)
        )
    return result


def analyze_topology_boundary_quality_confirmation(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = _read_json(resolved)
    validate_topology_boundary_quality_confirmation_config(config)
    _, inputs = _quality_inputs(resolved, config)
    _validate_base_evidence(inputs)
    collection_root = Path(collection).resolve()
    extension_report_path = collection_root / "collection_report.json"
    extension_trials_path = collection_root / "repair_trials.jsonl"
    extension_report = _read_json(extension_report_path)
    extension = _read_jsonl(extension_trials_path)
    if (
        extension_report.get("schema") != COLLECTION_SCHEMA
        or extension_report.get("complete") is not True
        or int(extension_report.get("error_state_count", -1)) != 0
        or int(extension_report.get("trial_count", -1))
        != int(config["expected_collection_outcome_count"])
        or len(extension) != int(config["expected_collection_outcome_count"])
        or {int(row["trial_index"]) for row in extension}
        != set(map(int, config["collection_trial_indices"]))
        or {str(row.get("schema")) for row in extension} != {TRIAL_SCHEMA}
    ):
        raise ValueError("topology-boundary confirmation extension differs")
    base = _read_jsonl(inputs["base_trials"])
    merged = sorted(
        [*base, *extension],
        key=lambda row: (
            str(row["state_id"]), int(row["trial_index"]), str(row["candidate_id"])
        ),
    )
    keys = {
        (str(row["state_id"]), str(row["candidate_id"]), int(row["trial_index"]))
        for row in merged
    }
    if len(merged) != int(config["expected_outcome_count"]) or len(keys) != len(merged):
        raise ValueError("topology-boundary confirmation combined product differs")
    output_root = Path(output).resolve()
    combined_root = output_root / "combined_collection"
    combined_root.mkdir(parents=True, exist_ok=True)
    combined_report = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "error_state_count": 0,
        "candidate_count": int(config["expected_candidate_count"]),
        "trial_count": len(merged),
        "base_trials_sha256": sha256_file(inputs["base_trials"]),
        "extension_trials_sha256": sha256_file(extension_trials_path),
    }
    _write_json(combined_root / "collection_report.json", combined_report)
    _write_jsonl(combined_root / "repair_trials.jsonl", merged)
    report = _analyze_topology_quality_pilot(
        resolved,
        combined_root,
        output_root,
        validator=validate_topology_boundary_quality_confirmation_config,
        protocol=CONFIRMATION_PROTOCOL,
    )
    state_rows, candidate_rows = _selected_quality_rows(
        config,
        inputs,
        augmented_prefix_key="boundary_prefix",
        augmented_label="boundary",
    )
    metadata = {str(row["state_id"]): row for row in state_rows}
    halves = {
        "registered_indices_0_3": _half_quality(
            rows=merged,
            candidate_rows=candidate_rows,
            state_metadata=metadata,
            indices=list(map(int, config["first_half_indices"])),
            config=config,
        ),
        "unseen_indices_4_7": _half_quality(
            rows=merged,
            candidate_rows=candidate_rows,
            state_metadata=metadata,
            indices=list(map(int, config["second_half_indices"])),
            config=config,
        ),
    }
    second_gates = _quality_gate_results(
        halves["unseen_indices_4_7"], config["independent_second_half_gates"]
    )
    report["independent_half_quality"] = halves
    report["action_uncertainty"]["diagnostic_only"] = False
    report["action_uncertainty"]["confirmation_gate"] = True
    report["gates"].update(
        {f"unseen_half_{name}": value for name, value in second_gates.items()}
    )
    stability = config["stability_gates"]
    report["gates"].update(
        {
            "minimum_half_pairwise_consistency": float(
                report["action_uncertainty"]["half_pairwise_consistency"]
            ) >= float(stability["minimum_half_pairwise_consistency"]),
            "minimum_mean_half_top3_overlap": float(
                report["action_uncertainty"]["mean_half_top3_overlap"]
            ) >= float(stability["minimum_mean_half_top3_overlap"]),
            "maximum_mean_cross_half_normalized_regret": float(
                report["action_uncertainty"]["mean_cross_half_normalized_regret"]
            ) <= float(stability["maximum_mean_cross_half_normalized_regret"]),
        }
    )
    report["passed"] = all(report["gates"].values())
    report["next_decision"] = config[
        "next_decision_on_pass" if report["passed"] else "next_decision_on_failure"
    ]
    report["inputs"].update(
        {
            "extension_collection_report_sha256": sha256_file(extension_report_path),
            "extension_trials_sha256": sha256_file(extension_trials_path),
        }
    )
    _write_json(
        output_root / CONFIRMATION_PROTOCOL["report_filename"], report
    )
    return report


__all__ = [
    "analyze_topology_boundary_quality_confirmation",
    "collect_topology_boundary_quality_confirmation",
    "validate_topology_boundary_quality_confirmation_config",
]
