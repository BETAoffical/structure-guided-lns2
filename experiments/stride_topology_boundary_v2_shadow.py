from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_topology_anchor_quality import (
    _aggregate_quality_scores,
    _quality_inputs,
    _quality_ranking,
    _selected_quality_rows,
)
from experiments.stride_topology_boundary_quality_confirmation import (
    validate_topology_boundary_quality_confirmation_config,
)
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.online_selection import (
    online_candidate_rows,
    score_online_candidates,
)


CONFIG_SCHEMA = "lns2.stride.topology_boundary_v2_shadow_config.v1"
SELECTION_SCHEMA = "lns2.stride.topology_boundary_v2_shadow_selection.v1"
REPORT_SCHEMA = "lns2.stride.topology_boundary_v2_shadow_report.v1"


def _registered_path(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"boundary V2 Shadow input SHA differs: {specification['path']}")
    return path


def validate_topology_boundary_v2_shadow_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-boundary V2 Shadow config")
    if (
        config.get("scientific_status")
        != "exploratory_non_promoting_after_confirmation_failure"
        or config.get("experiment_id") != "stride-boundary-explore-v1"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("ttf_claim_allowed"))
        or bool(config.get("future_repair_rounds_used"))
        or bool(config.get("cost_to_go_used"))
        or bool(config.get("runtime_used_in_label"))
    ):
        raise ValueError("topology-boundary V2 Shadow must remain exploratory")
    if (
        config.get("frozen_controller_id") != "v2-full"
        or config.get("baseline_pool_id") != "target-collision-random-v2-frozen"
        or config.get("augmented_pool_id")
        != "target-collision-random-v2-plus-stride-topoboundary-v1"
        or int(config.get("expected_state_count", -1)) != 18
        or int(config.get("expected_candidate_count", -1)) != 347
        or int(config.get("expected_base_candidate_count", -1)) != 324
        or int(config.get("expected_boundary_only_candidate_count", -1)) != 23
        or int(config.get("expected_outcome_count", -1)) != 2776
        or config.get("expected_feature_schema_id") != "lns2.realized_features.v2"
        or int(config.get("expected_feature_dimension", -1)) != 124
        or tuple(map(int, config.get("trial_indices") or ())) != tuple(range(8))
        or tuple(map(int, config.get("first_half_indices") or ())) != (0, 1, 2, 3)
        or tuple(map(int, config.get("second_half_indices") or ())) != (4, 5, 6, 7)
    ):
        raise ValueError("topology-boundary V2 Shadow cohort or identity changed")
    if dict(config.get("selection_protocol") or {}) != {
        "outcome_blind_before_join": True,
        "baseline_and_augmented_use_same_frozen_model": True,
        "baseline_and_augmented_scored_as_separate_pools": True,
        "quality_yardstick": "same_eight_paired_pp_seed_mean_np100_current_step",
        "forbidden_selection_fields": [
            "conflicts_after",
            "repair_outcome",
            "replan_success",
            "post_structure",
            "pp_replan_seconds",
            "native_step_seconds",
        ],
    }:
        raise ValueError("topology-boundary V2 Shadow selection protocol changed")
    if dict(config.get("exploratory_quick_gates") or {}) != {
        "minimum_boundary_selected_state_count": 1,
        "minimum_improved_state_count": 1,
        "minimum_mean_normalized_selected_gain": 0.01,
        "minimum_unseen_half_mean_normalized_selected_gain": 0.0,
        "maximum_worsened_state_rate": 0.35,
    }:
        raise ValueError("topology-boundary V2 Shadow exploratory gates changed")
    if set(config.get("inputs") or {}) != {
        "confirmation_config",
        "confirmation_report",
        "combined_trials",
        "controller_manifest",
    }:
        raise ValueError("topology-boundary V2 Shadow input registry changed")
    if config.get("next_decision_on_pass") != "register_small_paired_boundary_v2_ttf_quick":
        raise ValueError("topology-boundary V2 Shadow pass decision changed")
    if (
        config.get("next_decision_on_failure")
        != "retain_formal_failure_and_stop_boundary_runtime_route"
    ):
        raise ValueError("topology-boundary V2 Shadow failure decision changed")


def _normalized_regret(scores: dict[str, float], selected_id: str) -> float:
    span = max(scores.values()) - min(scores.values())
    return (
        (max(scores.values()) - float(scores[selected_id])) / span
        if span > 1e-12
        else 0.0
    )


def _normalized_gain(
    scores: dict[str, float], baseline_id: str, augmented_id: str
) -> float:
    span = max(scores.values()) - min(scores.values())
    return (
        (float(scores[augmented_id]) - float(scores[baseline_id])) / span
        if span > 1e-12
        else 0.0
    )


def _shadow_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty topology-boundary V2 Shadow")
    mean = lambda key: statistics.fmean(float(row[key]) for row in records)
    return {
        "state_count": len(records),
        "action_changed_count": sum(bool(row["action_changed"]) for row in records),
        "action_changed_rate": mean("action_changed"),
        "boundary_selected_state_count": sum(
            bool(row["augmented_selected_boundary_only"]) for row in records
        ),
        "boundary_selected_state_rate": mean("augmented_selected_boundary_only"),
        "improved_state_count": sum(bool(row["selected_quality_improved"]) for row in records),
        "improved_state_rate": mean("selected_quality_improved"),
        "worsened_state_count": sum(bool(row["selected_quality_worsened"]) for row in records),
        "worsened_state_rate": mean("selected_quality_worsened"),
        "mean_normalized_selected_gain": mean("normalized_selected_gain"),
        "first_half_mean_normalized_selected_gain": mean(
            "first_half_normalized_selected_gain"
        ),
        "unseen_half_mean_normalized_selected_gain": mean(
            "unseen_half_normalized_selected_gain"
        ),
        "baseline_mean_normalized_regret": mean("baseline_normalized_regret"),
        "augmented_mean_normalized_regret": mean("augmented_normalized_regret"),
        "baseline_exact_best_rate": mean("baseline_exact_best"),
        "augmented_exact_best_rate": mean("augmented_exact_best"),
        "baseline_in_quality_top3_rate": mean("baseline_in_quality_top3"),
        "augmented_in_quality_top3_rate": mean("augmented_in_quality_top3"),
        "mean_baseline_v2_margin": mean("baseline_v2_margin"),
        "mean_augmented_v2_margin": mean("augmented_v2_margin"),
        "mean_augmented_selected_feature_outside_fraction": mean(
            "augmented_selected_feature_outside_fraction"
        ),
    }


def _shadow_gate_results(
    summary: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    return {
        "minimum_boundary_selected_state_count": int(
            summary["boundary_selected_state_count"]
        )
        >= int(gates["minimum_boundary_selected_state_count"]),
        "minimum_improved_state_count": int(summary["improved_state_count"])
        >= int(gates["minimum_improved_state_count"]),
        "minimum_mean_normalized_selected_gain": float(
            summary["mean_normalized_selected_gain"]
        )
        >= float(gates["minimum_mean_normalized_selected_gain"]),
        "minimum_unseen_half_mean_normalized_selected_gain": float(
            summary["unseen_half_mean_normalized_selected_gain"]
        )
        >= float(gates["minimum_unseen_half_mean_normalized_selected_gain"]),
        "maximum_worsened_state_rate": float(summary["worsened_state_rate"])
        <= float(gates["maximum_worsened_state_rate"]),
    }


def _score_frozen_v2_pools(
    *,
    project_root: Path,
    confirmation_config_path: Path,
    confirmation_config: dict[str, Any],
    controller_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    _, quality_inputs = _quality_inputs(confirmation_config_path, confirmation_config)
    state_rows, candidates_by_state = _selected_quality_rows(
        confirmation_config,
        quality_inputs,
        augmented_prefix_key="boundary_prefix",
        augmented_label="boundary",
    )
    dataset_rows = {
        str(row["task_id"]): row
        for row in _read_jsonl(quality_inputs["dataset_manifest"])
    }
    runtime = _read_json(quality_inputs["runtime_config"])
    bundle = load_controller_bundle(controller_manifest_path.parent)
    if (
        bundle.manifest.get("default_controller") != "v2-full"
        or bundle.manifest.get("feature_schema_id") != "lns2.realized_features.v2"
    ):
        raise ValueError("boundary V2 Shadow bundle is not frozen v2-full")
    model = bundle.main_models["realized_dynamic"]
    feature_names = tuple(map(str, model.feature_names))
    ranges = dict(bundle.main_ranges["realized_dynamic"])
    selections: list[dict[str, Any]] = []
    feature_rows_by_state: dict[str, list[dict[str, Any]]] = {}
    for state_row in state_rows:
        state_id = str(state_row["state_id"])
        task_id = str(state_row["task_id"])
        replay = {
            "dataset_root": str(
                (project_root / str(confirmation_config["dataset_root"])).resolve()
            ),
            "row": dict(dataset_rows[task_id]),
            "environment": dict(runtime["environment"]),
            "solver_seed": int(state_row["solver_seed"]),
            "replay_destroy_strategy": "Adaptive",
        }
        _, state = replay_prefix(replay, [])
        if state_fingerprint(state) != str(state_row["state_fingerprint"]):
            raise RuntimeError(f"boundary V2 Shadow replay mismatch: {state_id}")
        candidates = candidates_by_state[state_id]
        online_rows = online_candidate_rows(state, candidates)
        kinds = {
            str(row["candidate_id"]): str(row["candidate_kind"])
            for row in candidates
        }
        base_rows = [
            row
            for row in online_rows
            if kinds[str(row["candidate_id"])] in {"base", "base_and_boundary"}
        ]
        if not base_rows or len(online_rows) <= len(base_rows):
            raise ValueError(f"boundary V2 Shadow pool is not augmented: {state_id}")
        for row in online_rows:
            features = dict(row["features"]["realized_dynamic"])
            if set(features) != set(feature_names):
                raise ValueError(f"boundary V2 Shadow feature schema differs: {state_id}")
        base_index, base_scores, base_margin = score_online_candidates(base_rows, model)
        all_index, all_scores, all_margin = score_online_candidates(online_rows, model)
        base_selected = str(base_rows[base_index]["candidate_id"])
        all_selected = str(online_rows[all_index]["candidate_id"])
        selected_features = dict(online_rows[all_index]["features"]["realized_dynamic"])
        outside = sum(
            float(selected_features[name]) < float(bounds[0])
            or float(selected_features[name]) > float(bounds[1])
            for name, bounds in ranges.items()
        )
        selections.append(
            {
                "schema": SELECTION_SCHEMA,
                "state_id": state_id,
                "task_id": task_id,
                "map_id": str(state_row["map_id"]),
                "layout_family": str(state_row["layout_family"]),
                "solver_seed": int(state_row["solver_seed"]),
                "state_fingerprint": str(state_row["state_fingerprint"]),
                "candidate_count": len(online_rows),
                "base_candidate_count": len(base_rows),
                "boundary_only_candidate_count": sum(
                    kind == "boundary_only" for kind in kinds.values()
                ),
                "baseline_selected_candidate_id": base_selected,
                "augmented_selected_candidate_id": all_selected,
                "augmented_selected_candidate_kind": kinds[all_selected],
                "action_changed": base_selected != all_selected,
                "augmented_selected_boundary_only": kinds[all_selected]
                == "boundary_only",
                "baseline_v2_margin": float(base_margin),
                "augmented_v2_margin": float(all_margin),
                "baseline_selected_v2_score": float(base_scores[base_index]),
                "augmented_selected_v2_score": float(all_scores[all_index]),
                "feature_dimension": len(feature_names),
                "augmented_selected_feature_outside_fraction": outside
                / len(ranges)
                if ranges
                else 0.0,
            }
        )
        feature_rows_by_state[state_id] = online_rows
    integrity = {
        "state_count": len(selections),
        "candidate_count": sum(row["candidate_count"] for row in selections),
        "base_candidate_count": sum(row["base_candidate_count"] for row in selections),
        "boundary_only_candidate_count": sum(
            row["boundary_only_candidate_count"] for row in selections
        ),
        "feature_dimension_set": sorted(
            {int(row["feature_dimension"]) for row in selections}
        ),
    }
    return selections, feature_rows_by_state, integrity


def run_topology_boundary_v2_shadow(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_topology_boundary_v2_shadow_config(config)
    registered = {
        name: _registered_path(project_root, dict(specification))
        for name, specification in config["inputs"].items()
    }
    confirmation_config = _read_json(registered["confirmation_config"])
    validate_topology_boundary_quality_confirmation_config(confirmation_config)
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Proposal/state data and frozen V2 are scored first. Outcome content is
    # not parsed until the outcome-blind selections have been made durable.
    selections, feature_rows, integrity = _score_frozen_v2_pools(
        project_root=project_root,
        confirmation_config_path=registered["confirmation_config"],
        confirmation_config=confirmation_config,
        controller_manifest_path=registered["controller_manifest"],
    )
    selection_path = output_root / "outcome_blind_selections.jsonl"
    _write_jsonl(selection_path, selections)

    confirmation_report = _read_json(registered["confirmation_report"])
    if confirmation_report.get("passed") is not False:
        raise ValueError("boundary V2 Shadow requires the failed formal confirmation")
    trials = _read_jsonl(registered["combined_trials"])
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        by_state[str(row["state_id"])].append(row)
    selection_by_state = {str(row["state_id"]): row for row in selections}
    if set(by_state) != set(selection_by_state):
        raise ValueError("boundary V2 Shadow outcome state set differs")

    records: list[dict[str, Any]] = []
    for state_id in sorted(selection_by_state):
        selection = selection_by_state[state_id]
        scores = _aggregate_quality_scores(by_state[state_id], confirmation_config)
        full_scores, full_rank = _quality_ranking(
            scores,
            list(map(int, config["trial_indices"])),
            float(confirmation_config["label"]["no_progress_penalty"]),
        )
        first_scores, _ = _quality_ranking(
            scores,
            list(map(int, config["first_half_indices"])),
            float(confirmation_config["label"]["no_progress_penalty"]),
        )
        second_scores, _ = _quality_ranking(
            scores,
            list(map(int, config["second_half_indices"])),
            float(confirmation_config["label"]["no_progress_penalty"]),
        )
        baseline_id = str(selection["baseline_selected_candidate_id"])
        augmented_id = str(selection["augmented_selected_candidate_id"])
        if set(full_scores) != {
            str(row["candidate_id"]) for row in feature_rows[state_id]
        }:
            raise ValueError(f"boundary V2 Shadow outcome candidate set differs: {state_id}")
        gain = _normalized_gain(full_scores, baseline_id, augmented_id)
        records.append(
            {
                **selection,
                "best_quality_candidate_id": full_rank[0],
                "normalized_selected_gain": gain,
                "first_half_normalized_selected_gain": _normalized_gain(
                    first_scores, baseline_id, augmented_id
                ),
                "unseen_half_normalized_selected_gain": _normalized_gain(
                    second_scores, baseline_id, augmented_id
                ),
                "selected_quality_improved": gain > 1e-12,
                "selected_quality_worsened": gain < -1e-12,
                "baseline_normalized_regret": _normalized_regret(
                    full_scores, baseline_id
                ),
                "augmented_normalized_regret": _normalized_regret(
                    full_scores, augmented_id
                ),
                "baseline_exact_best": full_scores[baseline_id]
                >= full_scores[full_rank[0]] - 1e-12,
                "augmented_exact_best": full_scores[augmented_id]
                >= full_scores[full_rank[0]] - 1e-12,
                "baseline_in_quality_top3": baseline_id in set(full_rank[:3]),
                "augmented_in_quality_top3": augmented_id in set(full_rank[:3]),
            }
        )
    summary = _shadow_summary(records)
    subgroup_summaries = {}
    for group in sorted({str(row["layout_family"]) for row in records}):
        subgroup_summaries[group] = _shadow_summary(
            [row for row in records if str(row["layout_family"]) == group]
        )
    integrity.update(
        {
            "outcome_count": len(trials),
            "trial_index_set": sorted({int(row["trial_index"]) for row in trials}),
            "selection_forbidden_fields_observed": sorted(
                set(config["selection_protocol"]["forbidden_selection_fields"])
                & {name for row in selections for name in row}
            ),
        }
    )
    integrity_gates = {
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(config["expected_candidate_count"]),
        "base_candidate_count": integrity["base_candidate_count"]
        == int(config["expected_base_candidate_count"]),
        "boundary_only_candidate_count": integrity["boundary_only_candidate_count"]
        == int(config["expected_boundary_only_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(config["expected_outcome_count"]),
        "trial_indices": integrity["trial_index_set"]
        == list(map(int, config["trial_indices"])),
        "feature_dimension": integrity["feature_dimension_set"]
        == [int(config["expected_feature_dimension"])],
        "selection_outcome_blind": not integrity["selection_forbidden_fields_observed"],
    }
    exploratory_gates = _shadow_gate_results(
        summary, dict(config["exploratory_quick_gates"])
    )
    passed_for_quick = all(integrity_gates.values()) and all(
        exploratory_gates.values()
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": config["scientific_status"],
        "experiment_id": config["experiment_id"],
        "formal_confirmation_remains_failed": True,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "ttf_measured": False,
        "frozen_controller_id": "v2-full",
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "summary": summary,
        "subgroups": subgroup_summaries,
        "exploratory_quick_gates": exploratory_gates,
        "passed_for_exploratory_quick": passed_for_quick,
        "next_decision": config[
            "next_decision_on_pass" if passed_for_quick else "next_decision_on_failure"
        ],
        "records": records,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "confirmation_config_sha256": sha256_file(
                registered["confirmation_config"]
            ),
            "confirmation_report_sha256": sha256_file(
                registered["confirmation_report"]
            ),
            "combined_trials_sha256": sha256_file(registered["combined_trials"]),
            "controller_manifest_sha256": sha256_file(
                registered["controller_manifest"]
            ),
            "outcome_blind_selections_sha256": sha256_file(selection_path),
        },
    }
    _write_json(output_root / "topology_boundary_v2_shadow_report.json", report)
    return report


__all__ = [
    "run_topology_boundary_v2_shadow",
    "validate_topology_boundary_v2_shadow_config",
]
