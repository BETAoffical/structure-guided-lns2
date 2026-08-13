from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_plateau_escape_witness import _spearman
from experiments.stride_repairability_opportunity import _candidate_key
from experiments.stride_tailswitch import _mean


CONFIG_SCHEMA = "lns2.stride.repairability_stability_registration.v1"
REPORT_SCHEMA = "lns2.stride.repairability_stability_report.v1"
ROW_SCHEMA = "lns2.stride.repairability_stability_checkpoint.v1"
STATUS_SCHEMA = "lns2.stride.repairability_stability_status.v1"
EXPERIMENT_ID = "stride-repairability-stability-v1"
EXPECTED_PARENT = "65166112073f33ad8d71109f1ef2705eb1be6724"


def _half_metrics(
    rows: list[dict[str, Any]], indices: tuple[int, ...]
) -> dict[str, float]:
    by_index = {int(row["trial_index"]): row for row in rows}
    selected = [by_index[index] for index in indices]
    return {
        "replan_success_rate": _mean(
            float(bool(row["replan_success"])) for row in selected
        ),
        "seed_mean": _mean(
            float(row["normalized_conflict_reduction"]) for row in selected
        ),
        "no_progress_rate": _mean(
            float(bool(row["no_progress"])) for row in selected
        ),
    }


def _quality_noninferior(
    candidate: dict[str, Any], anchor: dict[str, Any], epsilon: float
) -> bool:
    return bool(
        float(candidate["seed_mean"]) + epsilon >= float(anchor["seed_mean"])
        and float(candidate["lower_half_mean"]) + epsilon
        >= float(anchor["lower_half_mean"])
        and float(candidate["first_fixed_half_mean"]) + epsilon
        >= float(anchor["first_fixed_half_mean"])
        and float(candidate["second_fixed_half_mean"]) + epsilon
        >= float(anchor["second_fixed_half_mean"])
        and float(candidate["no_progress_rate"])
        <= float(anchor["no_progress_rate"]) + epsilon
    )


def _direction(value: float, epsilon: float) -> int:
    return int(value > epsilon) - int(value < -epsilon)


def _tie_aware_rank_correlation(
    left: list[float], right: list[float]
) -> tuple[float, str]:
    left_constant = len(set(left)) <= 1
    right_constant = len(set(right)) <= 1
    if left_constant and right_constant:
        return 1.0, "both_halves_constant_decision_equivalent"
    if left_constant or right_constant:
        return 0.0, "one_half_constant"
    return _spearman(left, right), "informative"


def checkpoint_stability(
    *,
    logical: dict[str, Any],
    checkpoint: dict[str, Any],
    aggregates: dict[tuple[str, str], dict[str, Any]],
    trials: dict[tuple[str, str], list[dict[str, Any]]],
    halves: tuple[tuple[int, ...], tuple[int, ...]],
    epsilon: float,
) -> dict[str, Any]:
    state = str(logical["state_fingerprint"])
    anchor_id = str(
        checkpoint["candidate_pool_diagnostic"]["original_pool_anchor_candidate_id"]
    )
    candidate_ids = list(map(str, logical["candidate_ids"]))
    if anchor_id not in candidate_ids:
        raise ValueError(f"V2 anchor left checkpoint pool: {logical['case_id']}")
    anchor = aggregates[_candidate_key(state, anchor_id)]
    robust_ids = [anchor_id]
    for candidate_id in candidate_ids:
        if candidate_id == anchor_id:
            continue
        if _quality_noninferior(
            aggregates[_candidate_key(state, candidate_id)], anchor, epsilon
        ):
            robust_ids.append(candidate_id)
    half_by_candidate = {
        candidate_id: [
            _half_metrics(trials[_candidate_key(state, candidate_id)], half)
            for half in halves
        ]
        for candidate_id in robust_ids
    }
    anchor_halves = half_by_candidate[anchor_id]
    rates_0 = [
        half_by_candidate[candidate_id][0]["replan_success_rate"]
        for candidate_id in robust_ids
    ]
    rates_1 = [
        half_by_candidate[candidate_id][1]["replan_success_rate"]
        for candidate_id in robust_ids
    ]
    rank_correlation, rank_correlation_status = _tie_aware_rank_correlation(
        rates_0, rates_1
    )

    orderings = []
    for half_index in (0, 1):
        orderings.append(
            sorted(
                robust_ids,
                key=lambda candidate_id: (
                    -float(
                        half_by_candidate[candidate_id][half_index][
                            "replan_success_rate"
                        ]
                    ),
                    -float(
                        half_by_candidate[candidate_id][half_index]["seed_mean"]
                    ),
                    float(
                        half_by_candidate[candidate_id][half_index][
                            "no_progress_rate"
                        ]
                    ),
                    int(
                        aggregates[_candidate_key(state, candidate_id)][
                            "actual_size"
                        ]
                    ),
                    candidate_id,
                ),
            )
        )
    top_count = min(3, len(robust_ids))
    top_0 = orderings[0][:top_count]
    top_1 = orderings[1][:top_count]
    top3_overlap = (
        len(set(top_0) & set(top_1)) / top_count if top_count else 1.0
    )
    cross_regrets = []
    for train_half, validation_half in ((0, 1), (1, 0)):
        selected_id = orderings[train_half][0]
        best_validation = max(
            float(
                half_by_candidate[candidate_id][validation_half][
                    "replan_success_rate"
                ]
            )
            for candidate_id in robust_ids
        )
        selected_validation = float(
            half_by_candidate[selected_id][validation_half]["replan_success_rate"]
        )
        cross_regrets.append(best_validation - selected_validation)

    direction_rows = []
    for candidate_id in robust_ids:
        if candidate_id == anchor_id:
            continue
        directions = []
        deltas = []
        for half_index in (0, 1):
            delta = float(
                half_by_candidate[candidate_id][half_index]["replan_success_rate"]
            ) - float(anchor_halves[half_index]["replan_success_rate"])
            deltas.append(delta)
            directions.append(_direction(delta, epsilon))
        direction_rows.append(
            {
                "candidate_id": candidate_id,
                "half_deltas": deltas,
                "half_directions": directions,
                "direction_agrees": directions[0] == directions[1],
                "strict_nonzero_direction_agrees": bool(
                    directions[0] != 0
                    and directions[1] != 0
                    and directions[0] == directions[1]
                ),
            }
        )
    return {
        "schema": ROW_SCHEMA,
        "logical_checkpoint_id": str(logical["logical_checkpoint_id"]),
        "case_id": str(logical["case_id"]),
        "state_fingerprint": state,
        "map_id": str(logical["map_id"]),
        "checkpoint_kind": str(logical["checkpoint_kind"]),
        "classification": str(logical["classification"]),
        "challenger": str(logical["challenger"]),
        "robust_quality_candidate_count": len(robust_ids),
        "half_rank_correlation": rank_correlation,
        "half_rank_correlation_status": rank_correlation_status,
        "top3_overlap": top3_overlap,
        "top3_half_0": top_0,
        "top3_half_1": top_1,
        "cross_half_regret_0_to_1": cross_regrets[0],
        "cross_half_regret_1_to_0": cross_regrets[1],
        "mean_cross_half_regret": _mean(cross_regrets),
        "direction_pair_count": len(direction_rows),
        "direction_agreement_count": sum(
            row["direction_agrees"] for row in direction_rows
        ),
        "strict_nonzero_direction_agreement_count": sum(
            row["strict_nonzero_direction_agrees"] for row in direction_rows
        ),
        "anchor_direction_agreement": (
            sum(row["direction_agrees"] for row in direction_rows)
            / len(direction_rows)
            if direction_rows
            else 1.0
        ),
        "direction_rows": direction_rows,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direction_count = sum(int(row["direction_pair_count"]) for row in rows)
    direction_agree = sum(int(row["direction_agreement_count"]) for row in rows)
    correlations = [float(row["half_rank_correlation"]) for row in rows]
    correlation_status_counts = collections.Counter(
        str(row["half_rank_correlation_status"]) for row in rows
    )
    return {
        "checkpoint_count": len(rows),
        "mean_robust_quality_candidate_count": _mean(
            float(row["robust_quality_candidate_count"]) for row in rows
        ),
        "mean_top3_overlap": _mean(float(row["top3_overlap"]) for row in rows),
        "mean_rank_correlation": _mean(correlations),
        "median_rank_correlation": (
            float(statistics.median(correlations)) if correlations else 0.0
        ),
        "rank_correlation_status_counts": dict(
            sorted(correlation_status_counts.items())
        ),
        "mean_cross_half_regret": _mean(
            float(row["mean_cross_half_regret"]) for row in rows
        ),
        "anchor_direction_pair_count": direction_count,
        "anchor_direction_agreement_count": direction_agree,
        "anchor_direction_agreement": (
            direction_agree / direction_count if direction_count else 1.0
        ),
    }


def _summaries(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        output.append({**dict(zip(fields, key)), **_group_summary(groups[key])})
    return output


def _apply_gates(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    return {
        "top3_overlap": float(summary["mean_top3_overlap"])
        >= float(gates["minimum_mean_top3_overlap"]),
        "rank_correlation": float(summary["median_rank_correlation"])
        >= float(gates["minimum_median_rank_correlation"]),
        "anchor_direction_agreement": float(
            summary["anchor_direction_agreement"]
        )
        >= float(gates["minimum_anchor_direction_agreement"]),
        "cross_half_regret": float(summary["mean_cross_half_regret"])
        <= float(gates["maximum_mean_cross_half_regret"]),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    return "\n".join(
        [
            "# STRIDE Repairability Stability V1",
            "",
            "This existing-data audit compares fixed PP-seed halves. It runs no solver and trains no model.",
            "",
            f"- Mean Top-3 overlap: `{overall['mean_top3_overlap']:.4f}`",
            f"- Median candidate rank correlation: `{overall['median_rank_correlation']:.4f}`",
            f"- Anchor-relative direction agreement: `{overall['anchor_direction_agreement']:.4f}`",
            f"- Mean cross-half regret: `{overall['mean_cross_half_regret']:.4f}`",
            f"- Overall gates passed: `{str(report['overall_gates_passed']).lower()}`",
            f"- Every map/checkpoint group passed: `{str(report['every_map_checkpoint_group_passed']).lower()}`",
            f"- Stability passed: `{str(report['stability_passed']).lower()}`",
            f"- Next action: `{report['next_action']}`",
            "",
            "No training, runtime integration, TTF, prevention, or generalization claim is authorized by this report.",
            "",
        ]
    )


def load_registration(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_existing_paired_seed_label_stability_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != EXPECTED_PARENT
    ):
        raise ValueError("repairability-stability registration changed")
    expected_boundary = {
        "label_stability_only": True,
        "new_solver_runs_allowed": False,
        "model_training_allowed": False,
        "future_trajectory_read": False,
        "runtime_or_ttf_read": False,
        "runtime_controller_change_allowed": False,
        "longtail_prevention_claim": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "no_result_based_exclusion": True,
    }
    if dict(config["claim_boundary"]) != expected_boundary:
        raise ValueError("repairability-stability claim boundary changed")
    return path, config, {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }


def analyze_repairability_stability(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, config, inputs = load_registration(config_path)
    opportunity_report = _read_json(inputs["opportunity_report"])
    opportunity_status = _read_json(inputs["opportunity_status"])
    if not (
        opportunity_report.get("integrity_passed") is True
        and opportunity_status.get("complete") is True
    ):
        raise ValueError("repairability opportunity source changed")
    aggregate_rows = _read_jsonl(inputs["candidate_aggregates"])
    trial_rows = _read_jsonl(inputs["repair_trials"])
    logical_rows = _read_jsonl(inputs["logical_checkpoint_results"])
    checkpoint_rows = _read_jsonl(inputs["root_diagnostic_checkpoints"])
    aggregates = {
        _candidate_key(row["state_fingerprint"], row["candidate_id"]): row
        for row in aggregate_rows
    }
    trials: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        trials[
            _candidate_key(row["state_fingerprint"], row["candidate_id"])
        ].append(row)
    checkpoints = {
        (str(row["case_id"]), str(row["checkpoint_kind"])): row
        for row in checkpoint_rows
    }
    halves = tuple(
        tuple(map(int, half)) for half in config["cohort"]["fixed_halves"]
    )
    if halves != (tuple(range(8)), tuple(range(8, 16))):
        raise ValueError("fixed seed halves changed")
    epsilon = float(config["robust_quality_pool"]["epsilon"])
    rows = [
        checkpoint_stability(
            logical=logical,
            checkpoint=checkpoints[
                (str(logical["case_id"]), str(logical["checkpoint_kind"]))
            ],
            aggregates=aggregates,
            trials=trials,
            halves=halves,
            epsilon=epsilon,
        )
        for logical in logical_rows
    ]
    rows.sort(key=lambda row: str(row["logical_checkpoint_id"]))
    overall = _group_summary(rows)
    by_map_checkpoint = _summaries(rows, ("map_id", "checkpoint_kind"))
    gates = dict(config["stability_gates"])
    overall_gate_results = _apply_gates(overall, gates)
    map_gate_results = [
        {
            "map_id": row["map_id"],
            "checkpoint_kind": row["checkpoint_kind"],
            "gates": _apply_gates(row, gates),
            "passed": all(_apply_gates(row, gates).values()),
        }
        for row in by_map_checkpoint
    ]
    every_map = all(row["passed"] for row in map_gate_results)
    stability_passed = all(overall_gate_results.values()) and every_map
    integrity_gates = {
        "logical_checkpoint_count_matches": len(rows)
        == int(config["cohort"]["logical_checkpoint_count"]),
        "distinct_state_count_matches": len({row["state_fingerprint"] for row in rows})
        == int(config["cohort"]["distinct_state_count"]),
        "candidate_count_matches": len(aggregate_rows)
        == int(config["cohort"]["candidate_count"]),
        "trial_count_matches": len(trial_rows)
        == int(config["cohort"]["trial_count"]),
        "all_candidates_have_16_trials": all(len(value) == 16 for value in trials.values()),
        "no_result_based_exclusion": len(rows) == len(logical_rows),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": all(integrity_gates.values()),
        "integrity_gates": integrity_gates,
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "overall": overall,
        "by_checkpoint_kind": _summaries(rows, ("checkpoint_kind",)),
        "by_classification_checkpoint": _summaries(
            rows, ("classification", "checkpoint_kind")
        ),
        "by_map_checkpoint": by_map_checkpoint,
        "overall_gate_results": overall_gate_results,
        "overall_gates_passed": all(overall_gate_results.values()),
        "map_checkpoint_gate_results": map_gate_results,
        "every_map_checkpoint_group_passed": every_map,
        "stability_passed": stability_passed,
        "next_action": (
            config["decision_rule"]["all_gates_pass"]
            if stability_passed
            else config["decision_rule"]["any_gate_fails"]
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "inputs": {
            "registration_sha256": sha256_file(path),
            "registered_file_sha256": {
                name: sha256_file(value) for name, value in sorted(inputs.items())
            },
        },
    }
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    row_path = output_path / "repairability_stability_checkpoints.jsonl"
    _write_jsonl(row_path, rows)
    report["checkpoint_rows_sha256"] = sha256_file(row_path)
    report_path = output_path / "repairability_stability_report.json"
    _write_json(report_path, report)
    markdown_path = output_path / "repairability_stability_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        output_path / "repairability_stability_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": report["integrity_passed"],
            "stability_passed": stability_passed,
            "next_action": report["next_action"],
            "report_sha256": sha256_file(report_path),
            "checkpoint_rows_sha256": report["checkpoint_rows_sha256"],
            "markdown_sha256": sha256_file(markdown_path),
        },
    )
    return report


__all__ = [
    "analyze_repairability_stability",
    "checkpoint_stability",
    "load_registration",
]
