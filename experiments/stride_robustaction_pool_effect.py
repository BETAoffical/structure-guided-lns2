from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_label_collection import (
    state_artifact_tree_sha256,
    validate_robustaction_label_collection_config,
)
from experiments.stride_robustaction_opportunity import (
    _registered,
    _trial_scores_by_action,
    _validate_aggregate,
)
from lns2_selector.runtime.online_selection import score_online_candidates


CONFIG_SCHEMA = "lns2.stride.robustaction_pool_effect_config.v1"
SELECTION_SCHEMA = "lns2.stride.robustaction_pool_effect_selection.v1"
EFFECT_SCHEMA = "lns2.stride.robustaction_pool_effect_record.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_pool_effect_report.v1"


def validate_robustaction_pool_effect_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RobustAction pool-effect schema changed")
    if (
        config.get("scientific_status")
        != "preregistered_candidate_pool_effect_before_base_selection_outcome_join"
        or config.get("experiment_id")
        != "stride-robustaction-structpool-pool-effect-v1"
        or config.get("candidate_pool_id") != "v2-plus-stride-structpool-v1"
        or config.get("pre_registration_git_commit")
        != "4a1dcf9a350dffa6cd94713d496b5728cd968459"
    ):
        raise ValueError("RobustAction pool-effect identity changed")

    comparison = dict(config.get("comparison") or {})
    if comparison != {
        "baseline": "frozen_v2_ranking_over_exact_base_candidates_only",
        "augmented": (
            "same_frozen_v2_ranking_over_exact_base_plus_structpool_pool"
        ),
        "selection_must_be_persisted_before_outcomes": True,
        "feature_profile": "realized_dynamic",
        "feature_dimension": 124,
        "score_tie_round_digits": 12,
        "tie_break": "ascending_candidate_id",
    }:
        raise ValueError("RobustAction pool-effect comparison changed")

    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "all_state_count": 320,
        "active_structpool_state_count": 98,
        "inactive_fallback_state_count": 222,
        "candidate_count": 6285,
        "base_candidate_count": 5701,
        "structpool_candidate_count": 584,
        "trial_count": 100560,
        "map_count": 28,
        "active_state_definition": (
            "candidate_pool_contains_at_least_one_structpool_candidate"
        ),
        "primary_summary_scope": "98_active_structpool_states",
        "inactive_state_role": "exact_selection_and_zero_gain_integrity_control",
    }:
        raise ValueError("RobustAction pool-effect cohort changed")

    quality = dict(config.get("paired_quality") or {})
    if (
        tuple(map(int, quality.get("trial_indices") or ())) != tuple(range(16))
        or tuple(map(int, quality.get("first_fixed_half_indices") or ()))
        != tuple(range(8))
        or tuple(map(int, quality.get("second_fixed_half_indices") or ()))
        != tuple(range(8, 16))
        or quality.get("target") != "normalized_current_step_conflict_reduction"
        or quality.get("raw_selected_gain")
        != "augmented_seed_mean_minus_baseline_seed_mean"
        or quality.get("normalized_selected_gain")
        != "raw_selected_gain_divided_by_full_pool_seed_mean_span"
        or quality.get("paired_win_rule")
        != "augmented_score_gt_baseline_score_plus_tie_epsilon"
        or float(quality.get("tie_epsilon", -1.0)) != 1e-12
        or dict(quality.get("robust_improvement") or {})
        != {
            "minimum_paired_win_fraction": 0.75,
            "minimum_mean_effect": 0.02,
            "require_positive_effect_in_both_fixed_halves": True,
        }
        or dict(quality.get("robust_regression") or {})
        != {
            "minimum_paired_loss_fraction": 0.75,
            "minimum_mean_loss": 0.02,
            "require_negative_effect_in_both_fixed_halves": True,
        }
    ):
        raise ValueError("RobustAction pool-effect paired quality changed")

    gates = dict(config.get("paired_ttf_quick_gates") or {})
    if gates != {
        "scope": "active_structpool_states_only",
        "source": (
            "unchanged_stride_topology_boundary_v2_shadow_exploratory_quick_gates"
        ),
        "minimum_structpool_selected_state_count": 1,
        "minimum_improved_state_count": 1,
        "minimum_mean_normalized_selected_gain": 0.01,
        "minimum_second_half_mean_normalized_selected_gain": 0.0,
        "maximum_worsened_state_rate": 0.35,
    }:
        raise ValueError("RobustAction pool-effect TTF gates changed")

    expected_inputs = {
        "label_collection_config",
        "candidate_aggregates",
        "repair_trials",
        "state_manifest",
        "preflight_rows",
        "opportunity_audit_config",
        "augmented_anchor_selections",
        "opportunity_report",
        "frozen_v2_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("RobustAction pool-effect inputs changed")
    if (
        config.get("preflight_state_artifact_root")
        != "build/stride-robustaction-label-preflight-v1/states"
        or config.get("preflight_state_artifact_tree_sha256")
        != "6bc5f8a4d0edcb2881eee4361fc5c5c4362b32f3cbc2953ef8a8a75998c07bc2"
    ):
        raise ValueError("RobustAction pool-effect preflight tree changed")
    if dict(config.get("outputs") or {}) != {
        "outcome_blind_pool_selections": "outcome_blind_pool_selections.jsonl",
        "paired_pool_effects": "paired_pool_effects.jsonl",
        "report": "pool_effect_report.json",
    }:
        raise ValueError("RobustAction pool-effect outputs changed")

    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "new_ranker_trained": False,
        "training_allowed": False,
        "runtime_read": False,
        "ttf_read": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "pass_authorizes_only_paired_closed_loop_ttf_quick": True,
    }:
        raise ValueError("RobustAction pool-effect claim boundary changed")
    if (
        config.get("next_decision_on_pass")
        != "preregister_original_v2_vs_v2_plus_structpool_paired_closed_loop_raw_ttf_quick"
        or config.get("next_decision_on_failure")
        != "revise_outcome_blind_structpool_generation_or_activation_without_training_a_new_ranker"
    ):
        raise ValueError("RobustAction pool-effect next decision changed")

    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        observed_tree = state_artifact_tree_sha256(
            project_root / str(config["preflight_state_artifact_root"])
        )
        if observed_tree != str(config["preflight_state_artifact_tree_sha256"]):
            raise ValueError("RobustAction pool-effect preflight tree hash differs")


def _online_row(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    return {
        "candidate_id": candidate_id,
        "candidate_key": candidate_id,
        "features": {"realized_dynamic": dict(candidate["features"])},
    }


def _score_pool_selections(
    *,
    project_root: Path,
    config: dict[str, Any],
    preflight_rows_path: Path,
    augmented_anchor_path: Path,
    controller_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metadata_rows = _read_jsonl(preflight_rows_path)
    metadata = {str(row["state_id"]): row for row in metadata_rows}
    augmented_rows = _read_jsonl(augmented_anchor_path)
    augmented = {str(row["state_id"]): row for row in augmented_rows}
    if (
        len(metadata) != len(metadata_rows)
        or len(augmented) != len(augmented_rows)
        or set(metadata) != set(augmented)
    ):
        raise ValueError("RobustAction pool-effect preflight metadata differs")

    bundle = load_controller_bundle(controller_manifest_path.parent)
    if bundle.manifest.get("default_controller") != "v2-full":
        raise ValueError("RobustAction pool-effect bundle is not frozen v2-full")
    model = bundle.main_models["realized_dynamic"]
    expected_dimension = int(config["comparison"]["feature_dimension"])
    state_root = project_root / str(config["preflight_state_artifact_root"])
    selections: list[dict[str, Any]] = []
    candidates_by_state: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(state_root.glob("*.json")):
        payload = _read_json(path)
        state_id = str(payload["state_id"])
        if state_id not in metadata or state_id in candidates_by_state:
            raise ValueError(f"RobustAction pool-effect state identity differs: {state_id}")
        candidates = list(payload["candidates"])
        base_candidates = [
            row for row in candidates if str(row["candidate_kind"]) == "base"
        ]
        structpool_candidates = [
            row for row in candidates if str(row["candidate_kind"]) == "structpool"
        ]
        if not base_candidates:
            raise ValueError(f"RobustAction pool-effect lacks base candidates: {state_id}")
        for candidate in candidates:
            features = dict(candidate["features"])
            if len(features) != expected_dimension or not all(
                math.isfinite(float(value)) for value in features.values()
            ):
                raise ValueError(f"RobustAction pool-effect feature differs: {state_id}")
        base_rows = [_online_row(row) for row in base_candidates]
        full_rows = [_online_row(row) for row in candidates]
        base_index, base_scores, base_margin = score_online_candidates(base_rows, model)
        full_index, full_scores, full_margin = score_online_candidates(full_rows, model)
        base_selected = base_candidates[base_index]
        full_selected = candidates[full_index]
        registered_augmented = str(augmented[state_id]["anchor_candidate_id"])
        if str(full_selected["candidate_id"]) != registered_augmented:
            raise ValueError(f"RobustAction augmented V2 selection changed: {state_id}")
        active = bool(structpool_candidates)
        selections.append(
            {
                "schema": SELECTION_SCHEMA,
                "state_id": state_id,
                "map_id": str(metadata[state_id]["map_id"]),
                "task_id": str(metadata[state_id]["task_id"]),
                "source_policy": str(metadata[state_id]["source_policy"]),
                "layout_mode": str(metadata[state_id]["layout_mode"]),
                "before_conflicts": int(metadata[state_id]["before_conflicts"]),
                "candidate_count": len(candidates),
                "base_candidate_count": len(base_candidates),
                "structpool_candidate_count": len(structpool_candidates),
                "structpool_active": active,
                "baseline_candidate_id": str(base_selected["candidate_id"]),
                "augmented_candidate_id": str(full_selected["candidate_id"]),
                "augmented_candidate_kind": str(full_selected["candidate_kind"]),
                "action_changed": str(base_selected["candidate_id"])
                != str(full_selected["candidate_id"]),
                "baseline_v2_score": float(base_scores[base_index]),
                "augmented_v2_score": float(full_scores[full_index]),
                "baseline_v2_margin": float(base_margin),
                "augmented_v2_margin": float(full_margin),
                "candidate_repair_outcomes_read": False,
            }
        )
        candidates_by_state[state_id] = candidates
    if set(candidates_by_state) != set(metadata):
        raise ValueError("RobustAction pool-effect artifact state set differs")
    return sorted(selections, key=lambda row: str(row["state_id"])), candidates_by_state


def _span(values: dict[str, float]) -> float:
    return max(values.values()) - min(values.values()) if values else 0.0


def _normalized_gain(
    values: dict[str, float], baseline_id: str, augmented_id: str, epsilon: float
) -> float:
    span = _span(values)
    return (
        (float(values[augmented_id]) - float(values[baseline_id])) / span
        if span > epsilon
        else 0.0
    )


def _normalized_regret(
    values: dict[str, float], selected_id: str, epsilon: float
) -> float:
    span = _span(values)
    return (
        (max(values.values()) - float(values[selected_id])) / span
        if span > epsilon
        else 0.0
    )


def build_pool_effect_record(
    *,
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    scores_by_candidate: dict[str, list[float]],
    epsilon: float,
    robust_win_fraction: float,
    robust_mean_effect: float,
) -> dict[str, Any]:
    baseline_id = str(selection["baseline_candidate_id"])
    augmented_id = str(selection["augmented_candidate_id"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    if set(by_id) != set(scores_by_candidate):
        raise ValueError("RobustAction pool-effect candidate outcome set differs")
    baseline = scores_by_candidate[baseline_id]
    augmented = scores_by_candidate[augmented_id]
    if len(baseline) != 16 or len(augmented) != 16:
        raise ValueError("RobustAction pool-effect requires 16 paired outcomes")
    deltas = [right - left for left, right in zip(baseline, augmented)]
    raw_gain = statistics.fmean(deltas)
    first_gain = statistics.fmean(deltas[:8])
    second_gain = statistics.fmean(deltas[8:])
    win_count = sum(delta > epsilon for delta in deltas)
    loss_count = sum(delta < -epsilon for delta in deltas)
    tie_count = 16 - win_count - loss_count
    candidate_means = {
        candidate_id: statistics.fmean(values)
        for candidate_id, values in scores_by_candidate.items()
    }
    first_means = {
        candidate_id: statistics.fmean(values[:8])
        for candidate_id, values in scores_by_candidate.items()
    }
    second_means = {
        candidate_id: statistics.fmean(values[8:])
        for candidate_id, values in scores_by_candidate.items()
    }
    full_rank = sorted(candidate_means, key=lambda value: (-candidate_means[value], value))
    robust_improved = bool(
        win_count / 16.0 >= robust_win_fraction
        and raw_gain >= robust_mean_effect
        and first_gain > epsilon
        and second_gain > epsilon
    )
    robust_worsened = bool(
        loss_count / 16.0 >= robust_win_fraction
        and raw_gain <= -robust_mean_effect
        and first_gain < -epsilon
        and second_gain < -epsilon
    )
    return {
        "schema": EFFECT_SCHEMA,
        **selection,
        "baseline_candidate_kind": str(by_id[baseline_id]["candidate_kind"]),
        "raw_selected_gain": raw_gain,
        "first_half_raw_selected_gain": first_gain,
        "second_half_raw_selected_gain": second_gain,
        "normalized_selected_gain": _normalized_gain(
            candidate_means, baseline_id, augmented_id, epsilon
        ),
        "first_half_normalized_selected_gain": _normalized_gain(
            first_means, baseline_id, augmented_id, epsilon
        ),
        "second_half_normalized_selected_gain": _normalized_gain(
            second_means, baseline_id, augmented_id, epsilon
        ),
        "paired_win_count": win_count,
        "paired_loss_count": loss_count,
        "paired_tie_count": tie_count,
        "paired_win_fraction": win_count / 16.0,
        "paired_loss_fraction": loss_count / 16.0,
        "selected_quality_improved": raw_gain > epsilon,
        "selected_quality_worsened": raw_gain < -epsilon,
        "selected_quality_tied": abs(raw_gain) <= epsilon,
        "robust_improved": robust_improved,
        "robust_worsened": robust_worsened,
        "baseline_normalized_regret": _normalized_regret(
            candidate_means, baseline_id, epsilon
        ),
        "augmented_normalized_regret": _normalized_regret(
            candidate_means, augmented_id, epsilon
        ),
        "baseline_exact_best": candidate_means[baseline_id]
        >= candidate_means[full_rank[0]] - epsilon,
        "augmented_exact_best": candidate_means[augmented_id]
        >= candidate_means[full_rank[0]] - epsilon,
        "baseline_in_quality_top3": baseline_id in set(full_rank[:3]),
        "augmented_in_quality_top3": augmented_id in set(full_rank[:3]),
        "best_quality_candidate_id": full_rank[0],
    }


def summarize_pool_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty RobustAction pool effect")

    def mean(name: str) -> float:
        return statistics.fmean(float(row[name]) for row in records)

    return {
        "state_count": len(records),
        "action_changed_count": sum(bool(row["action_changed"]) for row in records),
        "action_changed_rate": mean("action_changed"),
        "structpool_selected_state_count": sum(
            row["augmented_candidate_kind"] == "structpool" for row in records
        ),
        "structpool_selected_state_rate": statistics.fmean(
            row["augmented_candidate_kind"] == "structpool" for row in records
        ),
        "improved_state_count": sum(
            bool(row["selected_quality_improved"]) for row in records
        ),
        "improved_state_rate": mean("selected_quality_improved"),
        "worsened_state_count": sum(
            bool(row["selected_quality_worsened"]) for row in records
        ),
        "worsened_state_rate": mean("selected_quality_worsened"),
        "tied_state_count": sum(bool(row["selected_quality_tied"]) for row in records),
        "tied_state_rate": mean("selected_quality_tied"),
        "robust_improved_state_count": sum(
            bool(row["robust_improved"]) for row in records
        ),
        "robust_improved_state_rate": mean("robust_improved"),
        "robust_worsened_state_count": sum(
            bool(row["robust_worsened"]) for row in records
        ),
        "robust_worsened_state_rate": mean("robust_worsened"),
        "mean_raw_selected_gain": mean("raw_selected_gain"),
        "first_half_mean_raw_selected_gain": mean("first_half_raw_selected_gain"),
        "second_half_mean_raw_selected_gain": mean("second_half_raw_selected_gain"),
        "mean_normalized_selected_gain": mean("normalized_selected_gain"),
        "first_half_mean_normalized_selected_gain": mean(
            "first_half_normalized_selected_gain"
        ),
        "second_half_mean_normalized_selected_gain": mean(
            "second_half_normalized_selected_gain"
        ),
        "pooled_paired_win_fraction": sum(int(row["paired_win_count"]) for row in records)
        / (16.0 * len(records)),
        "pooled_paired_loss_fraction": sum(
            int(row["paired_loss_count"]) for row in records
        )
        / (16.0 * len(records)),
        "baseline_mean_normalized_regret": mean("baseline_normalized_regret"),
        "augmented_mean_normalized_regret": mean("augmented_normalized_regret"),
        "baseline_exact_best_rate": mean("baseline_exact_best"),
        "augmented_exact_best_rate": mean("augmented_exact_best"),
        "baseline_in_quality_top3_rate": mean("baseline_in_quality_top3"),
        "augmented_in_quality_top3_rate": mean("augmented_in_quality_top3"),
    }


def _ttf_gate_results(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    return {
        "minimum_structpool_selected_state_count": int(
            summary["structpool_selected_state_count"]
        )
        >= int(gates["minimum_structpool_selected_state_count"]),
        "minimum_improved_state_count": int(summary["improved_state_count"])
        >= int(gates["minimum_improved_state_count"]),
        "minimum_mean_normalized_selected_gain": float(
            summary["mean_normalized_selected_gain"]
        )
        >= float(gates["minimum_mean_normalized_selected_gain"]),
        "minimum_second_half_mean_normalized_selected_gain": float(
            summary["second_half_mean_normalized_selected_gain"]
        )
        >= float(gates["minimum_second_half_mean_normalized_selected_gain"]),
        "maximum_worsened_state_rate": float(summary["worsened_state_rate"])
        <= float(gates["maximum_worsened_state_rate"]),
    }


def audit_robustaction_pool_effect(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robustaction_pool_effect_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    label_config = _read_json(inputs["label_collection_config"])
    validate_robustaction_label_collection_config(label_config, project_root=project_root)
    opportunity_report = _read_json(inputs["opportunity_report"])
    if opportunity_report.get("integrity_passed") is not True:
        raise ValueError("RobustAction opportunity product lacks integrity")

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    selections, candidates_by_state = _score_pool_selections(
        project_root=project_root,
        config=config,
        preflight_rows_path=inputs["preflight_rows"],
        augmented_anchor_path=inputs["augmented_anchor_selections"],
        controller_manifest_path=inputs["frozen_v2_manifest"],
    )
    selection_path = output_root / str(outputs["outcome_blind_pool_selections"])
    _write_jsonl(selection_path, selections)

    # Outcome parsing begins only after the base-only and augmented V2 choices
    # have been written to the durable selection artifact above.
    aggregates = _read_jsonl(inputs["candidate_aggregates"])
    trials = _read_jsonl(inputs["repair_trials"])
    manifests = _read_jsonl(inputs["state_manifest"])
    cohort = dict(config["cohort"])
    if (
        len(selections) != int(cohort["all_state_count"])
        or len(aggregates) != int(cohort["candidate_count"])
        or len(trials) != int(cohort["trial_count"])
        or len(manifests) != int(cohort["all_state_count"])
    ):
        raise ValueError("RobustAction pool-effect product count differs")

    aggregate_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_keys: set[tuple[str, str]] = set()
    for row in aggregates:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        if key in aggregate_keys:
            raise ValueError(f"duplicate RobustAction pool-effect aggregate: {key}")
        aggregate_keys.add(key)
        aggregate_by_state[key[0]].append(row)
    epsilon = float(config["paired_quality"]["tie_epsilon"])
    score_by_action = _trial_scores_by_action(trials, epsilon=epsilon)
    if set(score_by_action) != aggregate_keys:
        raise ValueError("RobustAction pool-effect trial identity differs")

    selection_by_state = {str(row["state_id"]): row for row in selections}
    if set(selection_by_state) != set(candidates_by_state) or set(
        selection_by_state
    ) != set(aggregate_by_state):
        raise ValueError("RobustAction pool-effect state identity differs")
    effects: list[dict[str, Any]] = []
    robust = dict(config["paired_quality"]["robust_improvement"])
    for state_id in sorted(selection_by_state):
        candidates = aggregate_by_state[state_id]
        scores = {
            str(row["candidate_id"]): score_by_action[
                (state_id, str(row["candidate_id"]))
            ]
            for row in candidates
        }
        for row in candidates:
            _validate_aggregate(
                row, scores[str(row["candidate_id"])], epsilon=epsilon
            )
        effects.append(
            build_pool_effect_record(
                selection=selection_by_state[state_id],
                candidates=candidates,
                scores_by_candidate=scores,
                epsilon=epsilon,
                robust_win_fraction=float(robust["minimum_paired_win_fraction"]),
                robust_mean_effect=float(robust["minimum_mean_effect"]),
            )
        )
    effect_path = output_root / str(outputs["paired_pool_effects"])
    _write_jsonl(effect_path, effects)

    active = [row for row in effects if bool(row["structpool_active"])]
    inactive = [row for row in effects if not bool(row["structpool_active"])]
    all_summary = summarize_pool_effect(effects)
    active_summary = summarize_pool_effect(active)
    inactive_summary = summarize_pool_effect(inactive)
    gates = _ttf_gate_results(
        active_summary, dict(config["paired_ttf_quick_gates"])
    )
    base_count = sum(int(row["base_candidate_count"]) for row in selections)
    structpool_count = sum(
        int(row["structpool_candidate_count"]) for row in selections
    )
    integrity = {
        "all_state_count": len(effects) == int(cohort["all_state_count"]),
        "active_state_count": len(active)
        == int(cohort["active_structpool_state_count"]),
        "inactive_state_count": len(inactive)
        == int(cohort["inactive_fallback_state_count"]),
        "candidate_count": len(aggregates) == int(cohort["candidate_count"]),
        "base_candidate_count": base_count == int(cohort["base_candidate_count"]),
        "structpool_candidate_count": structpool_count
        == int(cohort["structpool_candidate_count"]),
        "trial_count": len(trials) == int(cohort["trial_count"]),
        "map_count": len({str(row["map_id"]) for row in effects})
        == int(cohort["map_count"]),
        "selection_persisted_before_outcomes": True,
        "augmented_selection_matches_registered_anchor": True,
        "inactive_exact_fallback": all(
            not row["action_changed"]
            and row["baseline_candidate_id"] == row["augmented_candidate_id"]
            and abs(float(row["raw_selected_gain"])) <= epsilon
            for row in inactive
        ),
        "exact_trial_and_aggregate_identity": set(score_by_action) == aggregate_keys,
    }
    integrity_passed = all(integrity.values())
    passed_for_ttf_quick = integrity_passed and all(gates.values())

    groups = sorted({str(row["layout_mode"]) for row in active})
    policies = sorted({str(row["source_policy"]) for row in active})
    maps = sorted({str(row["map_id"]) for row in active})
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_candidate_pool_effect_no_runtime_no_training",
        "experiment_id": str(config["experiment_id"]),
        "candidate_pool_id": str(config["candidate_pool_id"]),
        "config_sha256": sha256_file(config_path),
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "all_states": all_summary,
        "active_structpool_states": active_summary,
        "inactive_fallback_states": inactive_summary,
        "active_by_topology_group": {
            group: summarize_pool_effect(
                [row for row in active if row["layout_mode"] == group]
            )
            for group in groups
        },
        "active_by_source_policy": {
            policy: summarize_pool_effect(
                [row for row in active if row["source_policy"] == policy]
            )
            for policy in policies
        },
        "active_by_map": {
            map_id: summarize_pool_effect(
                [row for row in active if row["map_id"] == map_id]
            )
            for map_id in maps
        },
        "paired_ttf_quick_gates": gates,
        "passed_for_paired_ttf_quick": passed_for_ttf_quick,
        "new_ranker_trained": False,
        "training_allowed": False,
        "runtime_read": False,
        "ttf_read": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "next_decision": config[
            "next_decision_on_pass"
            if passed_for_ttf_quick
            else "next_decision_on_failure"
        ],
        "artifacts": {
            "outcome_blind_pool_selections": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            "paired_pool_effects": {
                "path": str(effect_path),
                "sha256": sha256_file(effect_path),
            },
        },
    }
    _write_json(output_root / str(outputs["report"]), report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EFFECT_SCHEMA",
    "REPORT_SCHEMA",
    "SELECTION_SCHEMA",
    "audit_robustaction_pool_effect",
    "build_pool_effect_record",
    "summarize_pool_effect",
    "validate_robustaction_pool_effect_config",
]
