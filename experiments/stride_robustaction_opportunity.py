from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_label_collection import (
    AGGREGATE_SCHEMA,
    REPORT_SCHEMA as COLLECTION_REPORT_SCHEMA,
    state_artifact_tree_sha256,
    validate_robustaction_label_collection_config,
)
from lns2_selector.runtime.online_selection import score_online_candidates


CONFIG_SCHEMA = "lns2.stride.robustaction_opportunity_audit_config.v1"
ANCHOR_SCHEMA = "lns2.stride.robustaction_outcome_blind_v2_anchor.v1"
ACTION_SCHEMA = "lns2.stride.robustaction_action_comparison.v1"
STATE_SCHEMA = "lns2.stride.robustaction_state_opportunity.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_opportunity_audit_report.v1"
TOPOLOGY_GROUPS = (
    "dao_high_topology",
    "dao_mid_topology",
    "dao_low_topology_control",
)


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered RobustAction opportunity input is missing: {path}")
    observed = sha256_file(path)
    expected = str(specification["sha256"])
    if observed != expected:
        raise ValueError(
            f"registered RobustAction opportunity input changed: {path}: "
            f"expected {expected}, got {observed}"
        )
    return path


def validate_robustaction_opportunity_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RobustAction opportunity audit schema changed")
    if (
        config.get("scientific_status")
        != "preregistered_post_collection_audit_before_outcome_scan"
        or config.get("audit_id")
        != "stride-robustaction-structpool-opportunity-audit-v1"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v2"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("candidate_pool_id") != "v2-plus-stride-structpool-v1"
        or config.get("pre_registration_git_commit")
        != "843e3e64e3187250d5ad2cd971b7840491fc5baa"
    ):
        raise ValueError("RobustAction opportunity audit identity changed")

    expected_inputs = {
        "label_collection_config",
        "collection_report",
        "collection_status",
        "candidate_aggregates",
        "repair_trials",
        "state_manifest",
        "frozen_v2_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("RobustAction opportunity input registry changed")

    anchor = dict(config.get("outcome_blind_anchor_contract") or {})
    if anchor != {
        "selector": "v2-full",
        "feature_profile": "realized_dynamic",
        "feature_schema": "lns2.realized_features.v2",
        "feature_dimension": 124,
        "pool": "exact_preflight_candidate_pool",
        "preflight_state_artifact_root": (
            "build/stride-robustaction-label-preflight-v1/states"
        ),
        "preflight_state_artifact_tree_sha256": (
            "6bc5f8a4d0edcb2881eee4361fc5c5c4362b32f3cbc2953ef8a8a75998c07bc2"
        ),
        "score_tie_round_digits": 12,
        "tie_break": "ascending_candidate_id",
        "persist_anchor_selection_before_parsing_outcomes": True,
    }:
        raise ValueError("RobustAction outcome-blind anchor contract changed")

    robust = dict(config.get("robust_action_contract") or {})
    if (
        tuple(map(int, robust.get("trial_indices") or ())) != tuple(range(16))
        or tuple(map(int, robust.get("first_fixed_half_indices") or ()))
        != tuple(range(8))
        or tuple(map(int, robust.get("second_fixed_half_indices") or ()))
        != tuple(range(8, 16))
        or robust.get("per_seed_target")
        != "normalized_current_step_conflict_reduction"
        or robust.get("paired_win_rule")
        != "challenger_score_gt_anchor_score_plus_tie_epsilon"
        or float(robust.get("minimum_paired_win_fraction", -1.0)) != 0.75
        or float(robust.get("minimum_absolute_mean_effect", -1.0)) != 0.02
        or robust.get("fixed_half_rule")
        != "challenger_minus_anchor_gt_tie_epsilon_in_each_fixed_half"
        or float(robust.get("tie_epsilon", -1.0)) != 1e-12
        or robust.get("state_opportunity_rule")
        != "at_least_one_robust_positive_nonanchor_action"
        or robust.get("action_fraction_denominator")
        != "all_nonanchor_actions_in_all_320_states"
    ):
        raise ValueError("RobustAction robust-action contract changed")

    uncertainty = dict(config.get("action_uncertainty_contract") or {})
    if uncertainty != {
        "fixed_half_winner": (
            "maximum_fixed_half_mean_then_ascending_candidate_id"
        ),
        "fixed_half_top3_overlap": "intersection_size_divided_by_3",
        "pairwise_direction": "sign_of_left_minus_right_with_tie_epsilon",
        "pairwise_direction_agreement": (
            "same_direction_in_both_fixed_halves_over_all_unordered_candidate_pairs"
        ),
        "candidate_seed_standard_deviation": (
            "population_standard_deviation_over_16_paired_scores"
        ),
    }:
        raise ValueError("RobustAction uncertainty contract changed")

    product = dict(config.get("expected_product") or {})
    if product != {
        "state_count": 320,
        "candidate_count": 6285,
        "trial_count": 100560,
        "trial_count_per_candidate": 16,
        "map_count": 28,
        "topology_groups": list(TOPOLOGY_GROUPS),
    }:
        raise ValueError("RobustAction opportunity product contract changed")

    gates = dict(config.get("opportunity_gates") or {})
    if gates != {
        "minimum_states_with_nonanchor_robust_win_fraction": 0.25,
        "minimum_robust_positive_action_fraction": 0.04,
        "minimum_positive_opportunity_maps_by_topology_group": {
            "dao_high_topology": 3,
            "dao_mid_topology": 3,
            "dao_low_topology_control": 2,
        },
    }:
        raise ValueError("RobustAction opportunity gates changed")

    outputs = dict(config.get("outputs") or {})
    if outputs != {
        "outcome_blind_anchor_selections": "outcome_blind_anchor_selections.jsonl",
        "action_comparisons": "action_comparisons.jsonl",
        "state_opportunities": "state_opportunities.jsonl",
        "report": "opportunity_report.json",
    }:
        raise ValueError("RobustAction opportunity outputs changed")

    boundary = dict(config.get("claim_boundary") or {})
    if set(boundary) != {
        "training_allowed_only_if_all_opportunity_gates_pass",
        "ttf_read",
        "runtime_read",
        "future_trajectory_read",
        "formal_speed_claim",
        "default_replacement_allowed",
        "outcome_filtering_allowed",
    } or boundary.get("training_allowed_only_if_all_opportunity_gates_pass") is not True or any(
        boundary.get(name) is not False
        for name in (
            "ttf_read",
            "runtime_read",
            "future_trajectory_read",
            "formal_speed_claim",
            "default_replacement_allowed",
            "outcome_filtering_allowed",
        )
    ):
        raise ValueError("RobustAction opportunity claim boundary changed")

    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        state_root = (
            project_root / str(anchor["preflight_state_artifact_root"])
        ).resolve()
        if state_artifact_tree_sha256(state_root) != str(
            anchor["preflight_state_artifact_tree_sha256"]
        ):
            raise ValueError("RobustAction opportunity preflight tree changed")


def _direction(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _rank_by(candidates: Iterable[dict[str, Any]], field: str) -> list[str]:
    return [
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (-float(row[field]), str(row["candidate_id"])),
        )
    ]


def fixed_half_uncertainty(
    candidates: list[dict[str, Any]], *, epsilon: float
) -> dict[str, Any]:
    if len(candidates) < 3:
        raise ValueError("RobustAction uncertainty requires at least three candidates")
    first_rank = _rank_by(candidates, "first_fixed_half_mean")
    second_rank = _rank_by(candidates, "second_fixed_half_mean")
    by_id = {str(row["candidate_id"]): row for row in candidates}
    pair_count = 0
    pair_agreement_count = 0
    for left_id, right_id in combinations(sorted(by_id), 2):
        left = by_id[left_id]
        right = by_id[right_id]
        first_direction = _direction(
            float(left["first_fixed_half_mean"])
            - float(right["first_fixed_half_mean"]),
            epsilon,
        )
        second_direction = _direction(
            float(left["second_fixed_half_mean"])
            - float(right["second_fixed_half_mean"]),
            epsilon,
        )
        pair_count += 1
        pair_agreement_count += int(first_direction == second_direction)
    return {
        "first_fixed_half_winner_candidate_id": first_rank[0],
        "second_fixed_half_winner_candidate_id": second_rank[0],
        "fixed_half_exact_winner_agreement": first_rank[0] == second_rank[0],
        "fixed_half_top3_overlap": len(set(first_rank[:3]) & set(second_rank[:3]))
        / 3.0,
        "fixed_half_pairwise_direction_pair_count": pair_count,
        "fixed_half_pairwise_direction_agreement_count": pair_agreement_count,
        "fixed_half_pairwise_direction_agreement": (
            pair_agreement_count / pair_count if pair_count else 1.0
        ),
    }


def robust_action_comparison(
    *,
    challenger: dict[str, Any],
    anchor: dict[str, Any],
    challenger_scores: list[float],
    anchor_scores: list[float],
    minimum_paired_win_fraction: float,
    minimum_absolute_mean_effect: float,
    epsilon: float,
) -> dict[str, Any]:
    if len(challenger_scores) != 16 or len(anchor_scores) != 16:
        raise ValueError("RobustAction comparison requires 16 paired scores")
    paired_win_count = sum(
        challenger_score > anchor_score + epsilon
        for challenger_score, anchor_score in zip(challenger_scores, anchor_scores)
    )
    paired_win_fraction = paired_win_count / 16.0
    mean_effect = statistics.fmean(challenger_scores) - statistics.fmean(anchor_scores)
    first_effect = statistics.fmean(challenger_scores[:8]) - statistics.fmean(
        anchor_scores[:8]
    )
    second_effect = statistics.fmean(challenger_scores[8:]) - statistics.fmean(
        anchor_scores[8:]
    )
    robust_positive = bool(
        paired_win_fraction >= minimum_paired_win_fraction
        and mean_effect >= minimum_absolute_mean_effect
        and first_effect > epsilon
        and second_effect > epsilon
    )
    return {
        "schema": ACTION_SCHEMA,
        "candidate_id": str(challenger["candidate_id"]),
        "candidate_kind": str(challenger["candidate_kind"]),
        "anchor_candidate_id": str(anchor["candidate_id"]),
        "paired_win_count": paired_win_count,
        "paired_win_fraction": paired_win_fraction,
        "mean_effect_over_anchor": mean_effect,
        "first_fixed_half_effect_over_anchor": first_effect,
        "second_fixed_half_effect_over_anchor": second_effect,
        "robust_positive": robust_positive,
    }


def _score_outcome_blind_anchors(
    *,
    project_root: Path,
    config: dict[str, Any],
    controller_manifest_path: Path,
) -> list[dict[str, Any]]:
    anchor_contract = dict(config["outcome_blind_anchor_contract"])
    state_root = (
        project_root / str(anchor_contract["preflight_state_artifact_root"])
    ).resolve()
    bundle = load_controller_bundle(controller_manifest_path.parent)
    if bundle.manifest.get("default_controller") != "v2-full":
        raise ValueError("RobustAction anchor bundle is not frozen v2-full")
    model = bundle.main_models["realized_dynamic"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state_path in sorted(state_root.glob("*.json")):
        payload = _read_json(state_path)
        state_id = str(payload["state_id"])
        if state_id in seen:
            raise ValueError(f"duplicate RobustAction preflight state: {state_id}")
        seen.add(state_id)
        candidates = list(payload["candidates"])
        online_rows = []
        for candidate in candidates:
            features = dict(candidate["features"])
            if len(features) != int(anchor_contract["feature_dimension"]):
                raise ValueError(f"RobustAction anchor feature dimension differs: {state_id}")
            if not all(math.isfinite(float(value)) for value in features.values()):
                raise ValueError(f"RobustAction anchor feature is non-finite: {state_id}")
            candidate_id = str(candidate["candidate_id"])
            online_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "features": {"realized_dynamic": features},
                }
            )
        selected_index, scores, margin = score_online_candidates(online_rows, model)
        selected = candidates[selected_index]
        rows.append(
            {
                "schema": ANCHOR_SCHEMA,
                "state_id": state_id,
                "map_id": str(payload["map_id"]),
                "task_id": str(payload["task_id"]),
                "source_policy": str(payload["source_policy"]),
                "layout_mode": str(payload["layout_mode"]),
                "before_conflicts": int(payload["before_conflicts"]),
                "candidate_count": len(candidates),
                "candidate_signature": str(payload["candidate_signature"]),
                "feature_signature": str(payload["feature_signature"]),
                "anchor_candidate_id": str(selected["candidate_id"]),
                "anchor_candidate_kind": str(selected["candidate_kind"]),
                "anchor_v2_score": float(scores[selected_index]),
                "anchor_v2_margin": float(margin),
                "candidate_repair_outcomes_read": False,
            }
        )
    return sorted(rows, key=lambda row: str(row["state_id"]))


def _trial_scores_by_action(
    trials: list[dict[str, Any]], *, epsilon: float
) -> dict[tuple[str, str], list[float]]:
    grouped: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in trials:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        index = int(row["trial_index"])
        if index in grouped[key]:
            raise ValueError(f"duplicate RobustAction trial: {key}/{index}")
        expected = (
            int(row["before_conflicts"]) - int(row["conflicts_after"])
        ) / max(1, int(row["before_conflicts"]))
        observed = float(row["normalized_conflict_reduction"])
        if abs(expected - observed) > epsilon:
            raise ValueError(f"RobustAction normalized target differs: {key}/{index}")
        grouped[key][index] = observed
    result: dict[tuple[str, str], list[float]] = {}
    for key, values in grouped.items():
        if set(values) != set(range(16)):
            raise ValueError(f"incomplete RobustAction paired trials: {key}")
        result[key] = [values[index] for index in range(16)]
    return result


def _validate_aggregate(
    row: dict[str, Any], scores: list[float], *, epsilon: float
) -> None:
    expected = {
        "seed_mean": statistics.fmean(scores),
        "seed_standard_deviation": statistics.pstdev(scores),
        "first_fixed_half_mean": statistics.fmean(scores[:8]),
        "second_fixed_half_mean": statistics.fmean(scores[8:]),
    }
    for field, value in expected.items():
        if abs(float(row[field]) - value) > epsilon:
            raise ValueError(
                f"RobustAction aggregate differs: {row['state_id']}/"
                f"{row['candidate_id']}/{field}"
            )
    if row.get("schema") != AGGREGATE_SCHEMA or int(row.get("trial_count", -1)) != 16:
        raise ValueError("RobustAction aggregate schema or trial count differs")


def _group_summary(rows: list[dict[str, Any]], robust_field: str) -> dict[str, Any]:
    count = len(rows)
    positive = sum(bool(row[robust_field]) for row in rows)
    return {
        "count": count,
        "positive_count": positive,
        "positive_fraction": positive / count if count else 0.0,
    }


def _nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def audit_robustaction_opportunity(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robustaction_opportunity_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    label_config = _read_json(inputs["label_collection_config"])
    validate_robustaction_label_collection_config(label_config, project_root=project_root)

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])

    # This durable file is constructed only from the frozen preflight features
    # and V2 model. Outcome JSONL files are not parsed until after it is written.
    anchor_rows = _score_outcome_blind_anchors(
        project_root=project_root,
        config=config,
        controller_manifest_path=inputs["frozen_v2_manifest"],
    )
    anchor_path = output_root / str(outputs["outcome_blind_anchor_selections"])
    _write_jsonl(anchor_path, anchor_rows)

    collection_report = _read_json(inputs["collection_report"])
    collection_status = _read_json(inputs["collection_status"])
    if (
        collection_report.get("schema") != COLLECTION_REPORT_SCHEMA
        or collection_report.get("integrity_passed") is not True
        or collection_status.get("status") != "complete"
        or collection_status.get("integrity_passed") is not True
    ):
        raise ValueError("RobustAction label collection is not integrity-complete")

    aggregates = _read_jsonl(inputs["candidate_aggregates"])
    trials = _read_jsonl(inputs["repair_trials"])
    manifests = _read_jsonl(inputs["state_manifest"])
    product = dict(config["expected_product"])
    if (
        len(anchor_rows) != int(product["state_count"])
        or len(manifests) != int(product["state_count"])
        or len(aggregates) != int(product["candidate_count"])
        or len(trials) != int(product["trial_count"])
    ):
        raise ValueError("RobustAction opportunity product count differs")

    anchors = {str(row["state_id"]): row for row in anchor_rows}
    manifest_by_state = {str(row["state_id"]): row for row in manifests}
    if len(anchors) != len(anchor_rows) or len(manifest_by_state) != len(manifests):
        raise ValueError("RobustAction opportunity state identity is duplicated")
    if set(anchors) != set(manifest_by_state):
        raise ValueError("RobustAction anchor and outcome state sets differ")

    aggregates_by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregate_keys: set[tuple[str, str]] = set()
    for row in aggregates:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        if key in aggregate_keys:
            raise ValueError(f"duplicate RobustAction aggregate: {key}")
        aggregate_keys.add(key)
        aggregates_by_state[key[0]].append(row)
    score_by_action = _trial_scores_by_action(
        trials, epsilon=float(config["robust_action_contract"]["tie_epsilon"])
    )
    if set(score_by_action) != aggregate_keys:
        raise ValueError("RobustAction trial and aggregate action sets differ")

    robust = dict(config["robust_action_contract"])
    epsilon = float(robust["tie_epsilon"])
    action_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for state_id in sorted(anchors):
        anchor_selection = anchors[state_id]
        candidates = sorted(
            aggregates_by_state[state_id], key=lambda row: str(row["candidate_id"])
        )
        if len(candidates) != int(manifest_by_state[state_id]["candidate_count"]):
            raise ValueError(f"RobustAction state candidate count differs: {state_id}")
        by_id = {str(row["candidate_id"]): row for row in candidates}
        anchor_id = str(anchor_selection["anchor_candidate_id"])
        if anchor_id not in by_id:
            raise ValueError(f"RobustAction V2 anchor is absent from outcomes: {state_id}")
        for candidate_id, candidate in by_id.items():
            _validate_aggregate(
                candidate, score_by_action[(state_id, candidate_id)], epsilon=epsilon
            )
        anchor = by_id[anchor_id]
        anchor_scores = score_by_action[(state_id, anchor_id)]
        state_actions = []
        for candidate_id, candidate in sorted(by_id.items()):
            if candidate_id == anchor_id:
                continue
            row = robust_action_comparison(
                challenger=candidate,
                anchor=anchor,
                challenger_scores=score_by_action[(state_id, candidate_id)],
                anchor_scores=anchor_scores,
                minimum_paired_win_fraction=float(
                    robust["minimum_paired_win_fraction"]
                ),
                minimum_absolute_mean_effect=float(
                    robust["minimum_absolute_mean_effect"]
                ),
                epsilon=epsilon,
            )
            row.update(
                {
                    "state_id": state_id,
                    "map_id": str(candidate["map_id"]),
                    "task_id": str(candidate["task_id"]),
                    "source_policy": str(candidate["source_policy"]),
                    "layout_mode": str(candidate["layout_mode"]),
                    "before_conflicts": int(candidate["before_conflicts"]),
                    "selection_families": list(candidate["selection_families"]),
                    "structpool_family_groups": list(
                        candidate["structpool_family_groups"]
                    ),
                }
            )
            state_actions.append(row)
            action_rows.append(row)

        uncertainty = fixed_half_uncertainty(candidates, epsilon=epsilon)
        robust_actions = [row for row in state_actions if row["robust_positive"]]
        best = (
            sorted(
                robust_actions,
                key=lambda row: (
                    -float(row["mean_effect_over_anchor"]),
                    str(row["candidate_id"]),
                ),
            )[0]
            if robust_actions
            else None
        )
        state_rows.append(
            {
                "schema": STATE_SCHEMA,
                "state_id": state_id,
                "map_id": str(anchor_selection["map_id"]),
                "task_id": str(anchor_selection["task_id"]),
                "source_policy": str(anchor_selection["source_policy"]),
                "layout_mode": str(anchor_selection["layout_mode"]),
                "before_conflicts": int(anchor_selection["before_conflicts"]),
                "candidate_count": len(candidates),
                "nonanchor_action_count": len(state_actions),
                "anchor_candidate_id": anchor_id,
                "anchor_candidate_kind": str(anchor["candidate_kind"]),
                "robust_positive_action_count": len(robust_actions),
                "has_robust_nonanchor_opportunity": bool(robust_actions),
                "best_robust_candidate_id": (
                    str(best["candidate_id"]) if best is not None else None
                ),
                "best_robust_mean_effect_over_anchor": (
                    float(best["mean_effect_over_anchor"]) if best is not None else None
                ),
                **uncertainty,
            }
        )

    expected_nonanchor_count = int(product["candidate_count"]) - int(
        product["state_count"]
    )
    if len(action_rows) != expected_nonanchor_count or len(state_rows) != int(
        product["state_count"]
    ):
        raise ValueError("RobustAction comparison accounting differs")

    action_path = output_root / str(outputs["action_comparisons"])
    state_path = output_root / str(outputs["state_opportunities"])
    _write_jsonl(action_path, action_rows)
    _write_jsonl(state_path, state_rows)

    positive_maps = {
        group: sorted(
            {
                str(row["map_id"])
                for row in state_rows
                if row["layout_mode"] == group
                and row["has_robust_nonanchor_opportunity"]
            }
        )
        for group in TOPOLOGY_GROUPS
    }
    gates = dict(config["opportunity_gates"])
    state_summary = _group_summary(state_rows, "has_robust_nonanchor_opportunity")
    action_summary = _group_summary(action_rows, "robust_positive")
    gate_results = {
        "minimum_states_with_nonanchor_robust_win_fraction": (
            float(state_summary["positive_fraction"])
            >= float(gates["minimum_states_with_nonanchor_robust_win_fraction"])
        ),
        "minimum_robust_positive_action_fraction": (
            float(action_summary["positive_fraction"])
            >= float(gates["minimum_robust_positive_action_fraction"])
        ),
        "minimum_positive_opportunity_maps_by_topology_group": all(
            len(positive_maps[group])
            >= int(gates["minimum_positive_opportunity_maps_by_topology_group"][group])
            for group in TOPOLOGY_GROUPS
        ),
    }

    pair_count = sum(
        int(row["fixed_half_pairwise_direction_pair_count"]) for row in state_rows
    )
    pair_agreement = sum(
        int(row["fixed_half_pairwise_direction_agreement_count"])
        for row in state_rows
    )
    standard_deviations = [
        float(row["seed_standard_deviation"]) for row in aggregates
    ]
    maps = {str(row["map_id"]) for row in state_rows}
    integrity = {
        "state_count": len(state_rows) == int(product["state_count"]),
        "candidate_count": len(aggregates) == int(product["candidate_count"]),
        "trial_count": len(trials) == int(product["trial_count"]),
        "nonanchor_action_count": len(action_rows) == expected_nonanchor_count,
        "map_count": len(maps) == int(product["map_count"]),
        "complete_collection": True,
        "outcome_blind_anchor_persisted_before_outcome_parse": True,
        "exact_anchor_and_outcome_state_identity": set(anchors)
        == set(aggregates_by_state),
        "exact_trial_and_aggregate_action_identity": set(score_by_action)
        == aggregate_keys,
    }
    integrity_passed = all(integrity.values())
    opportunity_passed = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_current_step_opportunity_audit_no_training",
        "audit_id": str(config["audit_id"]),
        "data_line_id": str(config["data_line_id"]),
        "planned_model_id": str(config["planned_model_id"]),
        "candidate_pool_id": str(config["candidate_pool_id"]),
        "config_sha256": sha256_file(config_path),
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "opportunity_gates": gate_results,
        "opportunity_passed": opportunity_passed,
        "passed": integrity_passed and opportunity_passed,
        "state_opportunity": state_summary,
        "action_opportunity": action_summary,
        "positive_opportunity_maps_by_topology_group": positive_maps,
        "positive_opportunity_map_counts_by_topology_group": {
            group: len(values) for group, values in positive_maps.items()
        },
        "state_opportunity_by_source_policy": {
            policy: _group_summary(
                [row for row in state_rows if row["source_policy"] == policy],
                "has_robust_nonanchor_opportunity",
            )
            for policy in sorted({str(row["source_policy"]) for row in state_rows})
        },
        "state_opportunity_by_topology_group": {
            group: _group_summary(
                [row for row in state_rows if row["layout_mode"] == group],
                "has_robust_nonanchor_opportunity",
            )
            for group in TOPOLOGY_GROUPS
        },
        "robust_positive_actions_by_candidate_kind": dict(
            sorted(
                Counter(
                    str(row["candidate_kind"])
                    for row in action_rows
                    if row["robust_positive"]
                ).items()
            )
        ),
        "action_uncertainty": {
            "fixed_half_exact_winner_agreement_rate": statistics.fmean(
                float(row["fixed_half_exact_winner_agreement"])
                for row in state_rows
            ),
            "mean_fixed_half_top3_overlap": statistics.fmean(
                float(row["fixed_half_top3_overlap"]) for row in state_rows
            ),
            "pooled_fixed_half_pairwise_direction_agreement": (
                pair_agreement / pair_count if pair_count else 1.0
            ),
            "pair_count": pair_count,
            "candidate_seed_standard_deviation": {
                "mean": statistics.fmean(standard_deviations),
                "median": statistics.median(standard_deviations),
                "p90_nearest_rank": _nearest_rank(standard_deviations, 0.90),
                "maximum": max(standard_deviations),
            },
        },
        "artifacts": {
            "outcome_blind_anchor_selections": {
                "path": str(anchor_path),
                "sha256": sha256_file(anchor_path),
            },
            "action_comparisons": {
                "path": str(action_path),
                "sha256": sha256_file(action_path),
            },
            "state_opportunities": {
                "path": str(state_path),
                "sha256": sha256_file(state_path),
            },
        },
        "training_allowed": integrity_passed and opportunity_passed,
        "ttf_read": False,
        "runtime_read": False,
        "future_trajectory_read": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "next_decision": (
            "build_map_grouped_robustaction_training_product"
            if integrity_passed and opportunity_passed
            else "stop_before_training_and_report_exact_failed_opportunity_gates"
        ),
    }
    report_path = output_root / str(outputs["report"])
    _write_json(report_path, report)
    return report


__all__ = [
    "ACTION_SCHEMA",
    "ANCHOR_SCHEMA",
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "STATE_SCHEMA",
    "audit_robustaction_opportunity",
    "fixed_half_uncertainty",
    "robust_action_comparison",
    "validate_robustaction_opportunity_config",
]
