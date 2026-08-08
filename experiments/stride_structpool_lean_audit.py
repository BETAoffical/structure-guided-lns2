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
from experiments.stride_robustaction_pool_effect import (
    _normalized_regret,
    _online_row,
    _span,
)
from lns2_selector.runtime.online_selection import score_online_candidates


CONFIG_SCHEMA = "lns2.stride.structpool_lean_audit_config.v1"
SELECTION_SCHEMA = "lns2.stride.structpool_lean_selection.v1"
EFFECT_SCHEMA = "lns2.stride.structpool_lean_effect.v1"
REPORT_SCHEMA = "lns2.stride.structpool_lean_audit_report.v1"


def validate_structpool_lean_audit_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("StructPool Lean audit schema changed")
    if (
        config.get("scientific_status")
        != "preregistered_lean_pool_filter_before_outcome_join"
        or config.get("experiment_id") != "stride-structpool-lean-audit-v1"
        or config.get("candidate_pool_id") != "stride-structpool-lean-v1"
        or config.get("pre_registration_parent_commit")
        != "6e7fd018c9778a279e8d568e85852609e4982681"
    ):
        raise ValueError("StructPool Lean audit identity changed")

    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_ranking_over_exact_base_plus_full_structpool",
        "challenger": "same_frozen_v2_ranking_after_registered_lean_filter",
        "selection_must_be_persisted_before_outcomes": True,
        "feature_profile": "realized_dynamic",
        "feature_dimension": 124,
        "score_tie_round_digits": 12,
        "tie_break": "ascending_candidate_id",
    }:
        raise ValueError("StructPool Lean comparison changed")
    if dict(config.get("lean_filter") or {}) != {
        "candidate_kind": "structpool",
        "remove_only_if_family_groups_exact": ["bottleneck_crossing"],
        "preserve_mixed_family_candidates": True,
        "replacement_candidate_added": False,
        "runtime_strategy": "generate_full_structpool_then_filter_before_features",
    }:
        raise ValueError("StructPool Lean filter changed")
    if dict(config.get("cohort") or {}) != {
        "all_state_count": 320,
        "active_structpool_state_count": 98,
        "inactive_fallback_state_count": 222,
        "full_candidate_count": 6285,
        "full_base_candidate_count": 5701,
        "full_structpool_candidate_count": 584,
        "expected_removed_candidate_count": 49,
        "expected_affected_state_count": 49,
        "expected_lean_candidate_count": 6236,
        "expected_lean_structpool_candidate_count": 535,
        "trial_count": 100560,
        "map_count": 28,
        "primary_summary_scope": "98_active_structpool_states",
    }:
        raise ValueError("StructPool Lean cohort changed")

    quality = dict(config.get("paired_quality") or {})
    if (
        tuple(map(int, quality.get("trial_indices") or ())) != tuple(range(16))
        or tuple(map(int, quality.get("first_fixed_half_indices") or ()))
        != tuple(range(8))
        or tuple(map(int, quality.get("second_fixed_half_indices") or ()))
        != tuple(range(8, 16))
        or quality.get("target") != "normalized_current_step_conflict_reduction"
        or quality.get("comparison") != "lean_selected_minus_full_selected"
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
        raise ValueError("StructPool Lean paired quality changed")
    if dict(config.get("runtime_quick_gates") or {}) != {
        "scope": "98_active_structpool_states",
        "maximum_action_changed_count": 2,
        "maximum_selected_quality_worsened_count": 0,
        "maximum_robust_worsened_count": 0,
        "minimum_mean_normalized_selected_gain": 0.0,
        "maximum_mean_normalized_regret_delta": 0.0,
        "minimum_exact_best_rate_delta": 0.0,
        "minimum_quality_top3_rate_delta": 0.0,
    }:
        raise ValueError("StructPool Lean runtime gates changed")

    expected_inputs = {
        "label_collection_config",
        "candidate_aggregates",
        "repair_trials",
        "state_manifest",
        "preflight_rows",
        "full_pool_selections",
        "full_pool_report",
        "frozen_v2_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool Lean inputs changed")
    if (
        config.get("preflight_state_artifact_root")
        != "build/stride-robustaction-label-preflight-v1/states"
        or config.get("preflight_state_artifact_tree_sha256")
        != "6bc5f8a4d0edcb2881eee4361fc5c5c4362b32f3cbc2953ef8a8a75998c07bc2"
    ):
        raise ValueError("StructPool Lean preflight tree changed")
    if dict(config.get("outputs") or {}) != {
        "outcome_blind_selections": "outcome_blind_lean_selections.jsonl",
        "paired_effects": "paired_lean_effects.jsonl",
        "report": "lean_audit_report.json",
    }:
        raise ValueError("StructPool Lean outputs changed")
    if dict(config.get("claim_boundary") or {}) != {
        "new_ranker_trained": False,
        "training_allowed": False,
        "runtime_read": False,
        "ttf_read": False,
        "future_trajectory_read": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "pass_authorizes_only_runtime_implementation_and_paired_quick": True,
    }:
        raise ValueError("StructPool Lean claim boundary changed")
    if (
        config.get("next_decision_on_pass")
        != "implement_full_then_filter_lean_v1_and_preregister_four_controller_raw_ttf_quick"
        or config.get("next_decision_on_failure")
        != "retain_speed2_full_structpool_and_reject_lean_v1"
    ):
        raise ValueError("StructPool Lean next decision changed")

    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        observed_tree = state_artifact_tree_sha256(
            project_root / str(config["preflight_state_artifact_root"])
        )
        if observed_tree != str(config["preflight_state_artifact_tree_sha256"]):
            raise ValueError("StructPool Lean preflight tree hash differs")


def is_removed_by_lean_filter(candidate: dict[str, Any]) -> bool:
    return bool(
        str(candidate.get("candidate_kind")) == "structpool"
        and tuple(map(str, candidate.get("structpool_family_groups") or ()))
        == ("bottleneck_crossing",)
    )


def build_lean_candidate_pool(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if not is_removed_by_lean_filter(candidate)]


def _score_selections(
    *,
    project_root: Path,
    config: dict[str, Any],
    preflight_rows_path: Path,
    full_pool_selections_path: Path,
    controller_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metadata_rows = _read_jsonl(preflight_rows_path)
    metadata = {str(row["state_id"]): row for row in metadata_rows}
    registered_rows = _read_jsonl(full_pool_selections_path)
    registered = {str(row["state_id"]): row for row in registered_rows}
    if (
        len(metadata) != len(metadata_rows)
        or len(registered) != len(registered_rows)
        or set(metadata) != set(registered)
    ):
        raise ValueError("StructPool Lean metadata identity differs")

    bundle = load_controller_bundle(controller_manifest_path.parent)
    if bundle.manifest.get("default_controller") != "v2-full":
        raise ValueError("StructPool Lean bundle is not frozen v2-full")
    model = bundle.main_models["realized_dynamic"]
    expected_dimension = int(config["comparison"]["feature_dimension"])
    state_root = project_root / str(config["preflight_state_artifact_root"])
    selections = []
    candidates_by_state: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(state_root.glob("*.json")):
        payload = _read_json(path)
        state_id = str(payload["state_id"])
        if state_id not in metadata or state_id in candidates_by_state:
            raise ValueError(f"StructPool Lean state identity differs: {state_id}")
        candidates = list(payload["candidates"])
        lean_candidates = build_lean_candidate_pool(candidates)
        removed = [row for row in candidates if is_removed_by_lean_filter(row)]
        if not lean_candidates or any(
            str(row["candidate_kind"]) == "base" and row not in lean_candidates
            for row in candidates
        ):
            raise ValueError(f"StructPool Lean changed the base pool: {state_id}")
        for candidate in candidates:
            features = dict(candidate["features"])
            if len(features) != expected_dimension or not all(
                math.isfinite(float(value)) for value in features.values()
            ):
                raise ValueError(f"StructPool Lean feature differs: {state_id}")
        full_index, full_scores, full_margin = score_online_candidates(
            [_online_row(row) for row in candidates], model
        )
        lean_index, lean_scores, lean_margin = score_online_candidates(
            [_online_row(row) for row in lean_candidates], model
        )
        full_selected = candidates[full_index]
        lean_selected = lean_candidates[lean_index]
        if str(full_selected["candidate_id"]) != str(
            registered[state_id]["augmented_candidate_id"]
        ):
            raise ValueError(f"StructPool full V2 selection changed: {state_id}")
        structpool_count = sum(
            str(row["candidate_kind"]) == "structpool" for row in candidates
        )
        lean_structpool_count = sum(
            str(row["candidate_kind"]) == "structpool" for row in lean_candidates
        )
        selections.append(
            {
                "schema": SELECTION_SCHEMA,
                "state_id": state_id,
                "map_id": str(metadata[state_id]["map_id"]),
                "task_id": str(metadata[state_id]["task_id"]),
                "source_policy": str(metadata[state_id]["source_policy"]),
                "layout_mode": str(metadata[state_id]["layout_mode"]),
                "before_conflicts": int(metadata[state_id]["before_conflicts"]),
                "full_candidate_count": len(candidates),
                "lean_candidate_count": len(lean_candidates),
                "full_structpool_candidate_count": structpool_count,
                "lean_structpool_candidate_count": lean_structpool_count,
                "removed_candidate_count": len(removed),
                "removed_candidate_ids": sorted(str(row["candidate_id"]) for row in removed),
                "structpool_active": structpool_count > 0,
                "full_candidate_id": str(full_selected["candidate_id"]),
                "lean_candidate_id": str(lean_selected["candidate_id"]),
                "full_candidate_kind": str(full_selected["candidate_kind"]),
                "lean_candidate_kind": str(lean_selected["candidate_kind"]),
                "action_changed": str(full_selected["candidate_id"])
                != str(lean_selected["candidate_id"]),
                "full_v2_score": float(full_scores[full_index]),
                "lean_v2_score": float(lean_scores[lean_index]),
                "full_v2_margin": float(full_margin),
                "lean_v2_margin": float(lean_margin),
                "removed_candidate_was_full_selection": any(
                    str(row["candidate_id"]) == str(full_selected["candidate_id"])
                    for row in removed
                ),
                "candidate_repair_outcomes_read": False,
            }
        )
        candidates_by_state[state_id] = candidates
    if set(candidates_by_state) != set(metadata):
        raise ValueError("StructPool Lean state artifact set differs")
    return sorted(selections, key=lambda row: str(row["state_id"])), candidates_by_state


def build_lean_effect_record(
    *,
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
    scores_by_candidate: dict[str, list[float]],
    epsilon: float,
    robust_win_fraction: float,
    robust_mean_effect: float,
) -> dict[str, Any]:
    full_id = str(selection["full_candidate_id"])
    lean_id = str(selection["lean_candidate_id"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    if set(by_id) != set(scores_by_candidate):
        raise ValueError("StructPool Lean candidate outcome set differs")
    full_scores = scores_by_candidate[full_id]
    lean_scores = scores_by_candidate[lean_id]
    if len(full_scores) != 16 or len(lean_scores) != 16:
        raise ValueError("StructPool Lean requires 16 paired outcomes")
    deltas = [right - left for left, right in zip(full_scores, lean_scores)]
    raw_gain = statistics.fmean(deltas)
    first_gain = statistics.fmean(deltas[:8])
    second_gain = statistics.fmean(deltas[8:])
    win_count = sum(delta > epsilon for delta in deltas)
    loss_count = sum(delta < -epsilon for delta in deltas)
    tie_count = len(deltas) - win_count - loss_count
    means = {
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
    full_rank = sorted(means, key=lambda value: (-means[value], value))

    def normalized_gain(values: dict[str, float]) -> float:
        span = _span(values)
        return (
            (float(values[lean_id]) - float(values[full_id])) / span
            if span > epsilon
            else 0.0
        )

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
        "outcomes_joined_after_selection": True,
        "raw_selected_gain": raw_gain,
        "first_half_raw_selected_gain": first_gain,
        "second_half_raw_selected_gain": second_gain,
        "normalized_selected_gain": normalized_gain(means),
        "first_half_normalized_selected_gain": normalized_gain(first_means),
        "second_half_normalized_selected_gain": normalized_gain(second_means),
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
        "full_normalized_regret": _normalized_regret(means, full_id, epsilon),
        "lean_normalized_regret": _normalized_regret(means, lean_id, epsilon),
        "full_exact_best": means[full_id] >= means[full_rank[0]] - epsilon,
        "lean_exact_best": means[lean_id] >= means[full_rank[0]] - epsilon,
        "full_in_quality_top3": full_id in set(full_rank[:3]),
        "lean_in_quality_top3": lean_id in set(full_rank[:3]),
        "best_quality_candidate_id": full_rank[0],
    }


def summarize_lean_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty StructPool Lean effect")

    def mean(name: str) -> float:
        return statistics.fmean(float(row[name]) for row in records)

    full_regret = mean("full_normalized_regret")
    lean_regret = mean("lean_normalized_regret")
    full_exact = mean("full_exact_best")
    lean_exact = mean("lean_exact_best")
    full_top3 = mean("full_in_quality_top3")
    lean_top3 = mean("lean_in_quality_top3")
    return {
        "state_count": len(records),
        "removed_candidate_count": sum(int(row["removed_candidate_count"]) for row in records),
        "affected_state_count": sum(int(row["removed_candidate_count"]) > 0 for row in records),
        "action_changed_count": sum(bool(row["action_changed"]) for row in records),
        "action_changed_rate": mean("action_changed"),
        "lean_structpool_selected_state_count": sum(
            str(row["lean_candidate_kind"]) == "structpool" for row in records
        ),
        "selected_quality_improved_count": sum(
            bool(row["selected_quality_improved"]) for row in records
        ),
        "selected_quality_worsened_count": sum(
            bool(row["selected_quality_worsened"]) for row in records
        ),
        "selected_quality_tied_count": sum(
            bool(row["selected_quality_tied"]) for row in records
        ),
        "robust_improved_count": sum(bool(row["robust_improved"]) for row in records),
        "robust_worsened_count": sum(bool(row["robust_worsened"]) for row in records),
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
        "pooled_paired_loss_fraction": sum(int(row["paired_loss_count"]) for row in records)
        / (16.0 * len(records)),
        "full_mean_normalized_regret": full_regret,
        "lean_mean_normalized_regret": lean_regret,
        "mean_normalized_regret_delta": lean_regret - full_regret,
        "full_exact_best_rate": full_exact,
        "lean_exact_best_rate": lean_exact,
        "exact_best_rate_delta": lean_exact - full_exact,
        "full_in_quality_top3_rate": full_top3,
        "lean_in_quality_top3_rate": lean_top3,
        "quality_top3_rate_delta": lean_top3 - full_top3,
    }


def _gate_results(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    return {
        "maximum_action_changed_count": int(summary["action_changed_count"])
        <= int(gates["maximum_action_changed_count"]),
        "maximum_selected_quality_worsened_count": int(
            summary["selected_quality_worsened_count"]
        )
        <= int(gates["maximum_selected_quality_worsened_count"]),
        "maximum_robust_worsened_count": int(summary["robust_worsened_count"])
        <= int(gates["maximum_robust_worsened_count"]),
        "minimum_mean_normalized_selected_gain": float(
            summary["mean_normalized_selected_gain"]
        )
        >= float(gates["minimum_mean_normalized_selected_gain"]),
        "maximum_mean_normalized_regret_delta": float(
            summary["mean_normalized_regret_delta"]
        )
        <= float(gates["maximum_mean_normalized_regret_delta"]),
        "minimum_exact_best_rate_delta": float(summary["exact_best_rate_delta"])
        >= float(gates["minimum_exact_best_rate_delta"]),
        "minimum_quality_top3_rate_delta": float(summary["quality_top3_rate_delta"])
        >= float(gates["minimum_quality_top3_rate_delta"]),
    }


def audit_structpool_lean(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_structpool_lean_audit_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    label_config = _read_json(inputs["label_collection_config"])
    validate_robustaction_label_collection_config(label_config, project_root=project_root)
    full_report = _read_json(inputs["full_pool_report"])
    if (
        full_report.get("integrity_passed") is not True
        or full_report.get("runtime_read") is not False
        or full_report.get("ttf_read") is not False
    ):
        raise ValueError("StructPool Lean full-pool source lacks integrity")

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    selections, candidates_by_state = _score_selections(
        project_root=project_root,
        config=config,
        preflight_rows_path=inputs["preflight_rows"],
        full_pool_selections_path=inputs["full_pool_selections"],
        controller_manifest_path=inputs["frozen_v2_manifest"],
    )
    selection_path = output_root / str(outputs["outcome_blind_selections"])
    _write_jsonl(selection_path, selections)

    aggregates = _read_jsonl(inputs["candidate_aggregates"])
    trials = _read_jsonl(inputs["repair_trials"])
    manifests = _read_jsonl(inputs["state_manifest"])
    cohort = dict(config["cohort"])
    if (
        len(selections) != int(cohort["all_state_count"])
        or len(aggregates) != int(cohort["full_candidate_count"])
        or len(trials) != int(cohort["trial_count"])
        or len(manifests) != int(cohort["all_state_count"])
    ):
        raise ValueError("StructPool Lean product count differs")

    aggregate_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_keys: set[tuple[str, str]] = set()
    for row in aggregates:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        if key in aggregate_keys:
            raise ValueError(f"duplicate StructPool Lean aggregate: {key}")
        aggregate_keys.add(key)
        aggregate_by_state[key[0]].append(row)
    epsilon = float(config["paired_quality"]["tie_epsilon"])
    score_by_action = _trial_scores_by_action(trials, epsilon=epsilon)
    if set(score_by_action) != aggregate_keys:
        raise ValueError("StructPool Lean trial identity differs")

    selection_by_state = {str(row["state_id"]): row for row in selections}
    if set(selection_by_state) != set(candidates_by_state) or set(
        selection_by_state
    ) != set(aggregate_by_state):
        raise ValueError("StructPool Lean state identity differs")
    robust = dict(config["paired_quality"]["robust_improvement"])
    effects = []
    for state_id in sorted(selection_by_state):
        candidates = aggregate_by_state[state_id]
        scores = {
            str(row["candidate_id"]): score_by_action[
                (state_id, str(row["candidate_id"]))
            ]
            for row in candidates
        }
        for row in candidates:
            _validate_aggregate(row, scores[str(row["candidate_id"])], epsilon=epsilon)
        effects.append(
            build_lean_effect_record(
                selection=selection_by_state[state_id],
                candidates=candidates,
                scores_by_candidate=scores,
                epsilon=epsilon,
                robust_win_fraction=float(robust["minimum_paired_win_fraction"]),
                robust_mean_effect=float(robust["minimum_mean_effect"]),
            )
        )
    effect_path = output_root / str(outputs["paired_effects"])
    _write_jsonl(effect_path, effects)

    active = [row for row in effects if bool(row["structpool_active"])]
    inactive = [row for row in effects if not bool(row["structpool_active"])]
    affected = [row for row in active if int(row["removed_candidate_count"]) > 0]
    all_summary = summarize_lean_effect(effects)
    active_summary = summarize_lean_effect(active)
    inactive_summary = summarize_lean_effect(inactive)
    affected_summary = summarize_lean_effect(affected)
    gates = _gate_results(active_summary, dict(config["runtime_quick_gates"]))
    full_count = sum(int(row["full_candidate_count"]) for row in selections)
    lean_count = sum(int(row["lean_candidate_count"]) for row in selections)
    full_structpool_count = sum(
        int(row["full_structpool_candidate_count"]) for row in selections
    )
    lean_structpool_count = sum(
        int(row["lean_structpool_candidate_count"]) for row in selections
    )
    removed_count = sum(int(row["removed_candidate_count"]) for row in selections)
    integrity = {
        "all_state_count": len(effects) == int(cohort["all_state_count"]),
        "active_state_count": len(active)
        == int(cohort["active_structpool_state_count"]),
        "inactive_state_count": len(inactive)
        == int(cohort["inactive_fallback_state_count"]),
        "full_candidate_count": full_count == int(cohort["full_candidate_count"]),
        "lean_candidate_count": lean_count == int(cohort["expected_lean_candidate_count"]),
        "full_base_candidate_count": full_count - full_structpool_count
        == int(cohort["full_base_candidate_count"]),
        "full_structpool_candidate_count": full_structpool_count
        == int(cohort["full_structpool_candidate_count"]),
        "lean_structpool_candidate_count": lean_structpool_count
        == int(cohort["expected_lean_structpool_candidate_count"]),
        "removed_candidate_count": removed_count
        == int(cohort["expected_removed_candidate_count"]),
        "affected_state_count": len(affected)
        == int(cohort["expected_affected_state_count"]),
        "trial_count": len(trials) == int(cohort["trial_count"]),
        "map_count": len({str(row["map_id"]) for row in effects})
        == int(cohort["map_count"]),
        "selection_persisted_before_outcomes": True,
        "full_selection_matches_registered_pool_effect": True,
        "removed_candidate_never_full_selection": all(
            not bool(row["removed_candidate_was_full_selection"]) for row in selections
        ),
        "inactive_exact_fallback": all(
            not bool(row["action_changed"])
            and int(row["removed_candidate_count"]) == 0
            for row in inactive
        ),
        "exact_trial_and_aggregate_identity": set(score_by_action) == aggregate_keys,
    }
    integrity_passed = all(integrity.values())
    passed = integrity_passed and all(gates.values())
    topology_groups = sorted({str(row["layout_mode"]) for row in active})
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_lean_pool_filter_no_runtime_no_training",
        "experiment_id": str(config["experiment_id"]),
        "candidate_pool_id": str(config["candidate_pool_id"]),
        "config_sha256": sha256_file(config_path),
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "all_states": all_summary,
        "active_structpool_states": active_summary,
        "affected_active_states": affected_summary,
        "inactive_fallback_states": inactive_summary,
        "active_by_topology_group": {
            group: summarize_lean_effect(
                [row for row in active if str(row["layout_mode"]) == group]
            )
            for group in topology_groups
        },
        "runtime_quick_gates": gates,
        "passed_for_runtime_quick": passed,
        "new_ranker_trained": False,
        "training_allowed": False,
        "runtime_read": False,
        "ttf_read": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
        "artifacts": {
            "outcome_blind_selections": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            "paired_effects": {
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
    "audit_structpool_lean",
    "build_lean_candidate_pool",
    "build_lean_effect_record",
    "is_removed_by_lean_filter",
    "summarize_lean_effect",
    "validate_structpool_lean_audit_config",
]
