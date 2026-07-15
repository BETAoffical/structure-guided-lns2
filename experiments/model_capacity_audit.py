from __future__ import annotations

import collections
import math
import pickle
from pathlib import Path
from typing import Any

from experiments.closed_loop_confirmation import _sha256, score_online_candidates
from experiments.policy_visited_aggregation_analysis import (
    _portable_model,
    _portable_payload,
    train_equal_state_pairwise_model,
)
from experiments.ranking_objective_audit import (
    PROFILE,
    _evaluate_model,
    _map_bootstrap,
    _maps_no_worse,
    leave_one_train_map_out,
    objective_acceptance,
)
from experiments.realized_neighborhood_ranking_audit import (
    _grouped,
    dominance_pairs,
    pairwise_accuracy,
    summarize_records,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


SCHEMA = "lns2.model_capacity_audit.v1"
CAPACITY_ORDER = ("small", "current", "large", "very_large")
ELIGIBLE_CAPACITIES = ("large", "very_large")


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported model-capacity audit config")
    if str(config.get("feature_profile")) != PROFILE:
        raise ValueError("capacity audit must use realized_dynamic")
    if tuple(map(str, config.get("capacity_order", []))) != CAPACITY_ORDER:
        raise ValueError("capacity order differs from the preregistration")
    expected = {
        "small": (50, 7, 40),
        "current": (100, 15, 20),
        "large": (300, 31, 10),
        "very_large": (500, 63, 5),
    }
    capacities = dict(config.get("capacities", {}))
    if set(capacities) != set(CAPACITY_ORDER):
        raise ValueError("capacity model set changed")
    for name, (iterations, leaves, minimum_leaf) in expected.items():
        parameters = dict(capacities[name])
        if (
            int(parameters.get("max_iter", -1)) != iterations
            or int(parameters.get("max_leaf_nodes", -1)) != leaves
            or int(parameters.get("min_samples_leaf", -1)) != minimum_leaf
            or not math.isclose(float(parameters.get("learning_rate", -1.0)), 0.05)
            or not math.isclose(float(parameters.get("l2_regularization", -1.0)), 0.1)
            or int(parameters.get("random_state", -1)) != 20260714
        ):
            raise ValueError(f"capacity parameters changed: {name}")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("capacity audit requires 5,000 map bootstrap samples")


def _load_inputs(
    training_root: Path,
    offline_root: Path,
    objective_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = dict(config["registered_inputs"])
    checks = {
        "aggregate_train_index_sha256": _sha256(
            training_root / "aggregate_train_index.jsonl"
        ),
        "validation_index_sha256": _sha256(training_root / "validation_index.jsonl"),
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "offline_report_sha256": _sha256(offline_root / "offline_report.json"),
        "objective_audit_report_sha256": _sha256(
            objective_root / "audit_report.json"
        ),
    }
    if checks != expected:
        raise ValueError(f"registered capacity-audit inputs changed: {checks}")
    objective = _read_json(objective_root / "audit_report.json")
    if (
        str(objective.get("decision")) != "stop_objective_alignment"
        or bool(objective.get("validation_evaluated"))
        or bool(objective.get("confirmation_generation_allowed"))
    ):
        raise ValueError("capacity audit requires the registered failed objective audit")
    aggregate = _read_jsonl(training_root / "aggregate_train_index.jsonl")
    train = [row for row in aggregate if str(row.get("split")) == "policy_train"]
    anchors = [row for row in aggregate if str(row.get("split")) != "policy_train"]
    validation = _read_jsonl(training_root / "validation_index.jsonl")
    if len(_grouped(train)) != 288 or len(_grouped(anchors)) != 23:
        raise ValueError("capacity audit Train or anchor state count changed")
    if len(_grouped(validation)) != 154:
        raise ValueError("capacity audit Validation state count changed")
    return train, anchors, validation


def _point_improves(
    baseline: dict[str, Any], challenger: dict[str, Any], thresholds: dict[str, Any]
) -> bool:
    top_delta = float(challenger["pareto_top1_hit_rate"]) - float(
        baseline["pareto_top1_hit_rate"]
    )
    baseline_regret = float(baseline["mean_conflict_regret"])
    challenger_regret = float(challenger["mean_conflict_regret"])
    regret_improvement = (
        (baseline_regret - challenger_regret) / baseline_regret
        if baseline_regret
        else 0.0
    )
    qualifies = (
        top_delta >= float(thresholds["minimum_top1_improvement"])
        or regret_improvement
        >= float(thresholds["minimum_conflict_regret_improvement"])
    )
    other_not_degraded = (
        float(challenger["pareto_top1_hit_rate"])
        >= float(baseline["pareto_top1_hit_rate"])
        - float(thresholds["maximum_top1_degradation"])
        and regret_improvement
        >= -float(thresholds["maximum_conflict_regret_degradation"])
    )
    return qualifies and other_not_degraded


def _weighted_pairwise_accuracy(
    rows_by_fold: list[tuple[list[dict[str, Any]], Any]]
) -> float:
    weighted = 0.0
    count = 0
    for rows, model in rows_by_fold:
        pairs = len(dominance_pairs(rows))
        weighted += pairwise_accuracy(rows, model) * pairs
        count += pairs
    return weighted / count if count else 0.0


def _cross_validate(
    train: list[dict[str, Any]], anchors: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    folds = leave_one_train_map_out(train, anchors)
    records = {name: {} for name in CAPACITY_ORDER}
    accuracy_inputs: dict[str, list[tuple[list[dict[str, Any]], Any]]] = {
        name: [] for name in CAPACITY_ORDER
    }
    fold_reports = []
    for fold_number, split in enumerate(folds):
        fit_rows = list(split["fit_rows"])
        held_rows = list(split["held_rows"])
        diagnostics = {}
        for name in CAPACITY_ORDER:
            model, diagnostic = train_equal_state_pairwise_model(
                fit_rows, PROFILE, dict(config["capacities"][name])
            )
            selected = _evaluate_model(held_rows, model, name)
            if set(selected) & set(records[name]):
                raise ValueError("capacity LOMO evaluated a state more than once")
            records[name].update(selected)
            accuracy_inputs[name].append((held_rows, model))
            diagnostics[name] = diagnostic
        fold_reports.append(
            {
                "fold": fold_number,
                "validation_map": split["validation_map"],
                "validation_state_count": len(_grouped(held_rows)),
                "anchor_state_count": split["anchor_state_count"],
                "training_diagnostics": diagnostics,
            }
        )
    summaries = {
        name: summarize_records(
            records[name], _weighted_pairwise_accuracy(accuracy_inputs[name])
        )
        for name in CAPACITY_ORDER
    }
    registered = dict(config["expected_current_lomo"])
    tolerance = float(registered["absolute_tolerance"])
    baseline_reproduced = (
        math.isclose(
            float(summaries["current"]["pareto_top1_hit_rate"]),
            float(registered["pareto_top1_hit_rate"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        and math.isclose(
            float(summaries["current"]["mean_conflict_regret"]),
            float(registered["mean_conflict_regret"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )
    if not baseline_reproduced:
        raise ValueError("current-capacity LOMO baseline did not reproduce")
    comparisons = {}
    for name in ("small",) + ELIGIBLE_CAPACITIES:
        no_worse, map_details = _maps_no_worse(records["current"], records[name])
        bootstrap = _map_bootstrap(
            records["current"], records[name], int(config["bootstrap_samples"])
        )
        acceptance = objective_acceptance(
            summaries["current"],
            summaries[name],
            bootstrap,
            no_worse,
            12,
            dict(config["thresholds"]),
        )
        comparisons[name] = {
            "eligible": name in ELIGIBLE_CAPACITIES,
            "bootstrap": bootstrap,
            "map_details": map_details,
            "acceptance": acceptance,
        }
    eligible = [
        name
        for name in ELIGIBLE_CAPACITIES
        if comparisons[name]["acceptance"]["passed"]
    ]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda name: (
                -float(summaries[name]["pareto_top1_hit_rate"]),
                float(summaries[name]["mean_conflict_regret"]),
                CAPACITY_ORDER.index(name),
            ),
        )[0]
    return {
        "map_count": 12,
        "state_count": len(_grouped(train)),
        "anchor_state_count": len(_grouped(anchors)),
        "validation_labels_used_for_capacity_selection": False,
        "baseline_reproduced": baseline_reproduced,
        "folds": fold_reports,
        "summaries": summaries,
        "comparisons": comparisons,
        "eligible_capacities": eligible,
        "winner": winner,
        "passed": winner is not None,
    }, records


def _fit_full_models(
    train: list[dict[str, Any]], anchors: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    rows = anchors + train
    models = {}
    diagnostics = {}
    records = {}
    summaries = {}
    for name in CAPACITY_ORDER:
        model, diagnostic = train_equal_state_pairwise_model(
            rows, PROFILE, dict(config["capacities"][name])
        )
        selected = _evaluate_model(train, model, name)
        models[name] = model
        diagnostics[name] = diagnostic
        records[name] = selected
        summaries[name] = summarize_records(selected, pairwise_accuracy(train, model))
    return models, {"summaries": summaries, "training_diagnostics": diagnostics}, records


def _diagnosis(
    cross_validation: dict[str, Any], in_sample: dict[str, Any], config: dict[str, Any]
) -> str:
    if cross_validation["passed"]:
        return "capacity_limited"
    baseline = in_sample["summaries"]["current"]
    if any(
        _point_improves(
            baseline,
            in_sample["summaries"][name],
            dict(config["thresholds"]),
        )
        for name in ELIGIBLE_CAPACITIES
    ):
        return "overfit"
    return "representation_limited"


def export_capacity_model(
    capacity: str,
    model: Any,
    train_rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    payload = _portable_payload(model, "capacity-audit-frozen-source")
    model_path = output / "pairwise_realized_dynamic.json"
    _write_json(model_path, payload)
    portable = _portable_model(payload)
    mismatch = 0
    maximum_delta = 0.0
    for candidates in _grouped(train_rows).values():
        native_index, native_scores, _ = score_online_candidates(candidates, model)
        portable_index, portable_scores, _ = score_online_candidates(candidates, portable)
        mismatch += native_index != portable_index
        maximum_delta = max(
            maximum_delta,
            max(abs(a - b) for a, b in zip(native_scores, portable_scores)),
        )
    equivalence = {
        "state_count": len(_grouped(train_rows)),
        "selection_mismatch_count": mismatch,
        "maximum_score_delta": maximum_delta,
        "passed": mismatch == 0 and maximum_delta <= 1e-12,
    }
    if not equivalence["passed"]:
        raise ValueError("capacity model portable inference mismatch")
    manifest = {
        "schema": "lns2.capacity_pairwise_bundle.v1",
        "schema_version": 1,
        "selector_type": "pairwise",
        "capacity": capacity,
        "feature_profile": PROFILE,
        "feature_names": list(model.feature_names),
        "model_file": model_path.name,
        "model_sha256": _sha256(model_path),
        "equivalence": equivalence,
        "confirmation_labels_seen": False,
    }
    _write_json(output / "portable_manifest.json", manifest)
    return manifest


def run_model_capacity_audit(
    training: str | Path,
    offline: str | Path,
    objective_audit: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    if phase not in {"cross_validate", "validate", "all"}:
        raise ValueError("phase must be cross_validate, validate, or all")
    training_root = Path(training).resolve()
    offline_root = Path(offline).resolve()
    objective_root = Path(objective_audit).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    train, anchors, validation = _load_inputs(
        training_root, offline_root, objective_root, config
    )
    run_fingerprint = _fingerprint(
        {
            "schema": SCHEMA,
            "configuration": config,
            "implementation": _sha256(Path(__file__)),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "run_config.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "configuration": config,
        },
    )
    cross_validation, cv_records = _cross_validate(train, anchors, config)
    for name, records in cv_records.items():
        _write_jsonl(
            output_root / f"lomo_predictions__{name}.jsonl",
            [records[key] for key in sorted(records)],
        )
    models, in_sample, in_sample_records = _fit_full_models(train, anchors, config)
    for name, records in in_sample_records.items():
        _write_jsonl(
            output_root / f"in_sample_predictions__{name}.jsonl",
            [records[key] for key in sorted(records)],
        )
    in_sample["generalization_gaps"] = {
        name: {
            "top1_gap": float(in_sample["summaries"][name]["pareto_top1_hit_rate"])
            - float(cross_validation["summaries"][name]["pareto_top1_hit_rate"]),
            "conflict_regret_gap": float(
                cross_validation["summaries"][name]["mean_conflict_regret"]
            )
            - float(in_sample["summaries"][name]["mean_conflict_regret"]),
        }
        for name in CAPACITY_ORDER
    }
    diagnosis = _diagnosis(cross_validation, in_sample, config)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "cross_validation": cross_validation,
        "in_sample": in_sample,
        "diagnosis": diagnosis,
        "validation_evaluated": False,
        "confirmation_generation_allowed": False,
        "static_context_used": False,
        "rl_trained": False,
        "new_data_collected": False,
    }
    if phase == "cross_validate" or not cross_validation["passed"]:
        report["decision"] = "stop_tabular_capacity_tuning"
        _write_json(output_root / "capacity_audit_report.json", report)
        return report
    winner = str(cross_validation["winner"])
    current_records = _evaluate_model(validation, models["current"], "current")
    offline_current = {
        str(row["state_id"]): row
        for row in _read_jsonl(
            offline_root / "offline_predictions__v2_realized_dynamic.jsonl"
        )
    }
    if {
        state: row["candidate_id"] for state, row in current_records.items()
    } != {
        state: row["candidate_id"] for state, row in offline_current.items()
    }:
        raise ValueError("current-capacity Validation baseline did not reproduce")
    winner_records = _evaluate_model(validation, models[winner], winner)
    current_summary = summarize_records(current_records)
    winner_summary = summarize_records(winner_records)
    no_worse, map_details = _maps_no_worse(current_records, winner_records)
    bootstrap = _map_bootstrap(
        current_records, winner_records, int(config["bootstrap_samples"])
    )
    acceptance = objective_acceptance(
        current_summary,
        winner_summary,
        bootstrap,
        no_worse,
        6,
        dict(config["thresholds"]),
    )
    validation_report = {
        "winner": winner,
        "validation_labels_used_for_training": False,
        "summaries": {"current": current_summary, winner: winner_summary},
        "bootstrap": bootstrap,
        "map_details": map_details,
        "acceptance": acceptance,
    }
    _write_jsonl(
        output_root / f"validation_predictions__{winner}.jsonl",
        [winner_records[key] for key in sorted(winner_records)],
    )
    _write_json(output_root / "validation_report.json", validation_report)
    report["development_validation"] = validation_report
    report["validation_evaluated"] = True
    if acceptance["passed"]:
        frozen_root = output_root / "frozen_winner"
        frozen_root.mkdir(parents=True, exist_ok=True)
        with (frozen_root / f"pairwise__{winner}.pkl").open("wb") as stream:
            pickle.dump(models[winner], stream)
        report["portable_manifest"] = export_capacity_model(
            winner, models[winner], anchors + train, frozen_root / "portable"
        )
        report["decision"] = "eligible_for_independent_confirmation"
        report["confirmation_generation_allowed"] = True
    else:
        report["decision"] = "stop_tabular_capacity_tuning"
    _write_json(output_root / "capacity_audit_report.json", report)
    return report


__all__ = [
    "CAPACITY_ORDER",
    "ELIGIBLE_CAPACITIES",
    "export_capacity_model",
    "run_model_capacity_audit",
]
