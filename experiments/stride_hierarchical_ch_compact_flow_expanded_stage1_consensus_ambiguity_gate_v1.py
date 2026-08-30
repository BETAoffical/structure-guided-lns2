"""Post-hoc consensus-only control with a decisive-vs-ambiguous gate.

This successor changes exactly one method choice relative to frozen
``consensus_only_v1``: Opportunity fits all 93 consensus rows with target
``decisive(S/V)=1 vs ambiguous=0`` and preserves natural state-equal class
mass.  Direction and every evaluation/safety boundary remain frozen.
"""

from __future__ import annotations

import collections
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
import experiments.stride_hierarchical_ch_compact_flow_stage1_router_v1 as base
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1 as frozen
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1 as two_head


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_config.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_report.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_prediction.v1"
)
EXPERIMENT_ID = (
    "stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1"
)
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-expanded-stage1-consensus-ambiguity-gate-v1"
)

FEATURE_NAMES = frozen.FEATURE_NAMES
FOLDS = frozen.FOLDS
EXPANDED_MAPS = frozen.EXPANDED_MAPS
ALL_LABELS = frozen.ALL_LABELS
DECISIVE_LABELS = frozen.DECISIVE_LABELS
GATE_THRESHOLDS = frozen.GATE_THRESHOLDS
DIRECTION_THRESHOLDS = frozen.DIRECTION_THRESHOLDS
DELTA_INDICES = frozen.DELTA_INDICES
EXPECTED_CONSENSUS_SUPPORT = frozen.EXPECTED_CONSENSUS_SUPPORT
LOGISTIC_MODEL = frozen.LOGISTIC_MODEL
OPPORTUNITY_VIEWS = frozen.OPPORTUNITY_VIEWS
CANDIDATE_ARMS = frozen.CANDIDATE_ARMS
INNER_GATES = frozen.INNER_GATES
FINAL_GATES = frozen.FINAL_GATES
THRESHOLD_SELECTION_RULE = frozen.THRESHOLD_SELECTION_RULE
ARM_SELECTION_RULE = frozen.ARM_SELECTION_RULE
STUDY_ROLE = "posthoc_same_cohort_ambiguity_gate_mechanism_control"
PREDECESSOR_ID = frozen.EXPERIMENT_ID
PREDECESSOR_CONFIG_SHA256 = (
    "208a74d07b3adf1f2bf45b8b8e62986856cac48b7356dd03253d5c01e7ffe5e8"
)
PREDECESSOR_REPORT_SHA256 = (
    "ebc5c0b103e354a22fd464d884a9967d68dab89ee915fc9856e01c4ce1761122"
)
PREDECESSOR_OOF_SHA256 = (
    "a6f773a0f53bdfdbae0d18ed658e373b01147a4bbcd399764fa3a8cc2e591654"
)
ALLOWED_METHOD_DIFFERENCES = (
    "opportunity_target",
    "opportunity_fit_population",
    "opportunity_class_balance",
)
CLAIM_BOUNDARY = {
    **frozen.CLAIM_BOUNDARY,
    "study_role": STUDY_ROLE,
    "passing_status": "POSTHOC_MECHANISM_SUPPORT_ONLY",
    "failure_status": "POSTHOC_CONTROL_NO_GO",
    "protocol_mismatch_status": "INVALID_CONTROL",
    "challenger_claim_allowed": False,
}


class HeadClassSupportError(ValueError):
    """A nested control fit split lacks one of the frozen classes."""


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_posthoc_consensus_ambiguity_gate_control"
        or config.get("registration_status") != "REGISTERED_NOT_RUN"
    ):
        raise ValueError("consensus ambiguity-gate identity changed")
    if set(config.get("inputs") or {}) != {
        "expanded_train_report",
        "expanded_stage1_pair_labels",
        "h1_collection_report",
        "predecessor_config",
        "predecessor_evaluation_report",
        "predecessor_oof_predictions",
    }:
        raise ValueError("consensus ambiguity-gate inputs changed")
    if dict(config.get("predecessor_contract") or {}) != {
        "study_role": STUDY_ROLE,
        "predecessor_experiment_id": PREDECESSOR_ID,
        "predecessor_config_sha256": PREDECESSOR_CONFIG_SHA256,
        "predecessor_report_sha256": PREDECESSOR_REPORT_SHA256,
        "predecessor_oof_sha256": PREDECESSOR_OOF_SHA256,
        "allowed_method_differences": list(ALLOWED_METHOD_DIFFERENCES),
        "control_reference_used_for_fit_or_threshold_selection": False,
        "paired_direction_probability_tolerance": 1e-12,
    }:
        raise ValueError("consensus ambiguity-gate predecessor contract changed")
    if dict(config.get("consensus_filter") or {}) != {
        "hierarchical_stratum": "consensus_structural",
        "structural_role_aliases_exact": ["component16", "hotspot16"],
        "canonical_partition": "consensus_structural",
        "unique_action_count": 2,
        "stage2_eligible": False,
        "component_equals_hotspot": True,
        "structural_differs_from_v2": True,
        "exact_component_hotspot_agents_equal": True,
        "exact_structural_v2_agents_differ": True,
        "exactly_one_pair_per_state": True,
    }:
        raise ValueError("consensus ambiguity-gate exact filter changed")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("expanded_map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("consensus ambiguity-gate folds changed")
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("source") != "expanded_row.model_feature_vector"
        or tuple(features.get("signed_feature_names") or ()) != FEATURE_NAMES
        or int(features.get("dimension", -1)) != 18
        or tuple(features.get("abs_delta_indices") or ()) != DELTA_INDICES
        or features.get("direction_view") != "signed18"
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
    ):
        raise ValueError("consensus ambiguity-gate feature contract changed")
    forbidden = set(map(str, features.get("forbidden_input_tokens") or ()))
    if not {
        "map_id",
        "task_id",
        "solver_seed",
        "trial",
        "outcome",
        "runtime",
        "pp_seconds",
        "rollback",
        "time_limit",
        "normalized_conflict_reduction",
    } <= forbidden:
        raise ValueError("consensus ambiguity-gate forbidden features weakened")
    if any(token in name.lower() for name in FEATURE_NAMES for token in forbidden):
        raise ValueError("consensus ambiguity-gate registered feature leaks")
    if dict(config.get("model") or {}) != LOGISTIC_MODEL:
        raise ValueError("consensus ambiguity-gate Logistic model changed")
    if tuple(config.get("opportunity_feature_views") or ()) != OPPORTUNITY_VIEWS:
        raise ValueError("consensus ambiguity-gate Opportunity views changed")
    if tuple(config.get("candidate_arms") or ()) != CANDIDATE_ARMS:
        raise ValueError("consensus ambiguity-gate candidate arms changed")
    if dict(config.get("head_contract") or {}) != {
        "opportunity": {
            "target": "decisive_structural_or_v2_vs_ambiguous",
            "fit_labels": ["structural_win", "v2_win", "ambiguous"],
            "fit_population_count": 93,
            "state_equal_weighting": True,
            "class_balance": False,
        },
        "direction": {
            "target": "structural_win_vs_v2_win",
            "fit_labels": ["structural_win", "v2_win"],
            "ambiguous_rows_in_fit": False,
            "state_equal_before_class_balance": True,
            "class_balance": True,
        },
    }:
        raise ValueError("consensus ambiguity-gate head contract changed")
    thresholds = dict(config.get("threshold_calibration") or {})
    if (
        tuple(map(float, thresholds.get("opportunity_threshold_grid") or ()))
        != GATE_THRESHOLDS
        or tuple(map(float, thresholds.get("direction_threshold_grid") or ()))
        != DIRECTION_THRESHOLDS
        or tuple(thresholds.get("selection_rule") or ())
        != THRESHOLD_SELECTION_RULE
    ):
        raise ValueError("consensus ambiguity-gate thresholds changed")
    if dict(config.get("inner_gates") or {}) != INNER_GATES:
        raise ValueError("consensus ambiguity-gate inner gates changed")
    if dict(config.get("final_gates") or {}) != FINAL_GATES:
        raise ValueError("consensus ambiguity-gate final gates changed")
    if tuple(config.get("arm_selection_rule") or ()) != ARM_SELECTION_RULE:
        raise ValueError("consensus ambiguity-gate arm selection changed")
    if dict(config.get("expected_support") or {}) != EXPECTED_CONSENSUS_SUPPORT:
        raise ValueError("consensus ambiguity-gate support changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "parallel_head_jobs": 3,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("consensus ambiguity-gate execution changed")
    if dict(config.get("outputs") or {}) != {
        "root": DEFAULT_OUTPUT,
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "consensus_ambiguity_gate_oof_predictions.jsonl",
        "evaluation_report": "consensus_ambiguity_gate_evaluation_report.json",
    }:
        raise ValueError("consensus ambiguity-gate outputs changed")
    if dict(config.get("claim_boundary") or {}) != CLAIM_BOUNDARY:
        raise ValueError("consensus ambiguity-gate claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def build_control_example(
    label_row: Mapping[str, Any],
    state: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    example = frozen.build_consensus_example(label_row, state, manifest_row)
    label = str(example["label"])
    return {
        **example,
        "opportunity_target": int(label in DECISIVE_LABELS),
        "opportunity_target_semantics": "decisive_structural_or_v2_vs_ambiguous",
    }


def _head_rows(
    examples: Sequence[Mapping[str, Any]], head: str, view_id: str
) -> list[dict[str, Any]]:
    if head == "opportunity":
        rows = list(examples)
        target = "opportunity_target"
        feature_key = f"opportunity_features_{view_id}"
    elif head == "direction":
        if view_id != "signed18":
            raise ValueError("Direction feature view must remain signed18")
        rows = [row for row in examples if row["label"] in DECISIVE_LABELS]
        target = "direction_target"
        feature_key = "direction_features_signed18"
    else:
        raise ValueError(f"unknown ambiguity-gate head: {head}")
    fit_rows = [
        {**row, "target": int(row[target]), "fit_features": list(row[feature_key])}
        for row in rows
    ]
    if {int(row["target"]) for row in fit_rows} != {0, 1}:
        raise HeadClassSupportError(f"ambiguity-gate {head} fit lacks both classes")
    if head == "opportunity" and len(fit_rows) != len(examples):
        raise ValueError("Opportunity did not fit all consensus rows")
    if head == "direction" and any(row["label"] == "ambiguous" for row in fit_rows):
        raise ValueError("ambiguous row entered Direction fit")
    return fit_rows


def _head_weights(
    rows: Sequence[Mapping[str, Any]], head: str
) -> tuple[list[float], dict[str, Any]]:
    state_counts = collections.Counter(str(row["state_occurrence_id"]) for row in rows)
    if set(state_counts.values()) != {1}:
        raise ValueError("ambiguity-gate population is not one pair per state")
    base = [1.0 for _ in rows]
    base_class_totals = {
        str(target): sum(
            weight
            for row, weight in zip(rows, base)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    if min(base_class_totals.values()) <= 0.0:
        raise HeadClassSupportError(f"ambiguity-gate {head} class has zero weight")
    if head == "opportunity":
        weights = base
        balanced = False
    elif head == "direction":
        raw = [
            weight / base_class_totals[str(int(row["target"]))]
            for row, weight in zip(rows, base)
        ]
        scale = len(raw) / sum(raw)
        weights = [weight * scale for weight in raw]
        balanced = True
    else:
        raise ValueError(f"unknown ambiguity-gate head: {head}")
    effective = {
        str(target): sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    if head == "opportunity" and weights != base:
        raise ValueError("Opportunity class balancing was applied")
    if head == "direction" and abs(effective["0"] - effective["1"]) > 1e-9:
        raise ValueError("Direction class balancing failed")
    return weights, {
        "head": head,
        "row_count": len(rows),
        "state_count": len(state_counts),
        "base_state_total_min": 1.0,
        "base_state_total_max": 1.0,
        "base_class_weight_totals": base_class_totals,
        "class_balance_applied": balanced,
        "effective_total_weight": sum(weights),
        "effective_class_weight_totals": effective,
        "state_equal_before_optional_class_balance": True,
    }


def _fit_logistic(rows: Sequence[Mapping[str, Any]], head: str) -> Any:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    weights, _ = _head_weights(rows, head)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_MODEL["parameters"])),
        ]
    )
    estimator.fit(
        np.asarray([row["fit_features"] for row in rows], dtype=np.float64),
        np.asarray([int(row["target"]) for row in rows], dtype=np.int8),
        model__sample_weight=np.asarray(weights, dtype=np.float64),
    )
    return estimator


def cross_validate_head(
    examples: Sequence[Mapping[str, Any]],
    head: str,
    view_id: str,
    fold_names: Sequence[str],
    *,
    fit: Callable[[Sequence[Mapping[str, Any]], str], Any] = _fit_logistic,
    predict: Callable[[Any, Sequence[Mapping[str, Any]], str], list[float]] = frozen._predict_probability,
) -> dict[str, Any]:
    active_folds = tuple(fold_names)
    if {str(row["train_fold"]) for row in examples} != set(active_folds):
        raise ValueError("ambiguity-gate inner fold population changed")
    feature_key = (
        f"opportunity_features_{view_id}"
        if head == "opportunity"
        else "direction_features_signed18"
    )
    probabilities: dict[str, float] = {}
    audits = []
    for held_fold in active_folds:
        train = [row for row in examples if row["train_fold"] != held_fold]
        held = [row for row in examples if row["train_fold"] == held_fold]
        fit_rows = _head_rows(train, head, view_id)
        _, weight_audit = _head_weights(fit_rows, head)
        train_states = {row["state_occurrence_id"] for row in fit_rows}
        held_states = {row["state_occurrence_id"] for row in held}
        train_maps = {row["map_id"] for row in fit_rows}
        held_maps = {row["map_id"] for row in held}
        if train_states & held_states or train_maps & held_maps:
            raise ValueError(f"ambiguity-gate inner leakage: {head}/{view_id}/{held_fold}")
        if not held_maps or not held_maps <= set(FOLDS[held_fold]):
            raise ValueError(f"ambiguity-gate held-map support changed: {held_fold}")
        estimator = fit(fit_rows, head)
        values = predict(estimator, held, feature_key)
        for row, value in zip(held, values):
            pair_id = str(row["pair_id"])
            if pair_id in probabilities:
                raise ValueError("duplicate ambiguity-gate inner OOF prediction")
            probabilities[pair_id] = float(value)
        audits.append(
            {
                "head": head,
                "view_id": view_id,
                "held_fold": held_fold,
                "train_map_count": len(train_maps),
                "held_map_count": len(held_maps),
                "train_state_count": len(train_states),
                "held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "fit_weight_audit": weight_audit,
            }
        )
    if set(probabilities) != {str(row["pair_id"]) for row in examples}:
        raise ValueError("ambiguity-gate inner OOF coverage changed")
    return {"probabilities": probabilities, "fold_audits": audits}


def _fit_outer_head(
    examples: Sequence[Mapping[str, Any]], head: str, view_id: str
) -> tuple[Any, dict[str, Any]]:
    rows = _head_rows(examples, head, view_id)
    _, audit = _head_weights(rows, head)
    return _fit_logistic(rows, head), audit


def _validate_frozen_equivalence(
    config: Mapping[str, Any], predecessor_config: Mapping[str, Any]
) -> None:
    frozen.validate_config(predecessor_config)
    core_inputs = (
        "expanded_train_report",
        "expanded_stage1_pair_labels",
        "h1_collection_report",
    )
    if any(
        dict(config["inputs"])[name] != dict(predecessor_config["inputs"])[name]
        for name in core_inputs
    ):
        raise ValueError("control core input differs from predecessor")
    shared_fields = (
        "consensus_filter",
        "expanded_map_folds",
        "feature_contract",
        "model",
        "opportunity_feature_views",
        "candidate_arms",
        "threshold_calibration",
        "inner_gates",
        "final_gates",
        "arm_selection_rule",
        "expected_support",
        "execution",
    )
    for field in shared_fields:
        if config.get(field) != predecessor_config.get(field):
            raise ValueError(f"control unauthorized method difference: {field}")
    if (
        dict(config["head_contract"])["direction"]
        != dict(predecessor_config["head_contract"])["direction"]
    ):
        raise ValueError("control Direction contract differs from predecessor")


def _prediction_output(row: Mapping[str, Any], run_fingerprint: str) -> dict[str, Any]:
    payload = frozen._prediction_output(row, run_fingerprint)
    decisive_probability = payload.pop("opportunity_structural_probability")
    payload.update(
        {
            "schema": PREDICTION_SCHEMA,
            "study_role": STUDY_ROLE,
            "opportunity_target": "decisive_structural_or_v2_vs_ambiguous",
            "opportunity_fit_population": "all_93_consensus_rows",
            "opportunity_class_balance": False,
            "opportunity_decisive_probability": decisive_probability,
            "opportunity_decisive_probability_semantics": (
                "probability_of_structural_or_v2_decisive_vs_ambiguous"
            ),
            "predecessor_experiment_id": PREDECESSOR_ID,
            "control_reference_used_for_fit_or_threshold_selection": False,
        }
    )
    return payload


def _paired_reference_audit(
    predictions: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    new_metrics: Mapping[str, Any],
    reference_report: Mapping[str, Any],
) -> dict[str, Any]:
    reference_by_pair = {str(row.get("pair_id", "")): row for row in reference_rows}
    if len(reference_by_pair) != len(reference_rows):
        raise ValueError("predecessor OOF pair IDs are duplicated")
    new_ids = {str(row["pair_id"]) for row in predictions}
    if set(reference_by_pair) != new_ids:
        raise ValueError("control/predecessor paired cohort differs")
    comparable = []
    inactive_or_incomparable = []
    for row in predictions:
        pair_id = str(row["pair_id"])
        reference = reference_by_pair[pair_id]
        new_probability = row.get("direction_structural_probability")
        old_probability = reference.get("direction_structural_probability")
        if new_probability is None or old_probability is None:
            inactive_or_incomparable.append(pair_id)
            continue
        comparable.append(abs(float(new_probability) - float(old_probability)))
    maximum_difference = max(comparable) if comparable else None
    expected_comparable = int(EXPECTED_CONSENSUS_SUPPORT["row_count"])
    direction_passed = bool(
        len(comparable) == expected_comparable
        and not inactive_or_incomparable
        and maximum_difference is not None
        and maximum_difference <= 1e-12
    )
    reference_metrics = dict(
        reference_report.get("primary_full_pooled_selected_action_metrics") or {}
    )
    metric_fields = (
        "full_pooled_decisive_balanced_accuracy",
        "selected_action_precision",
        "structural_win_override_recall",
        "v2_win_protection_recall",
        "ambiguous_override_rate",
        "total_override_count",
        "correct_structural_override_count",
        "v2_wrong_override_count",
        "correct_structural_override_map_count",
        "correct_structural_override_fold_count",
    )
    deltas = {}
    for field in metric_fields:
        new_value = new_metrics.get(field)
        old_value = reference_metrics.get(field)
        deltas[field] = (
            float(new_value) - float(old_value)
            if new_value is not None and old_value is not None
            else None
        )
    return {
        "paired_row_count": len(predictions),
        "expected_direction_comparable_pair_count": expected_comparable,
        "direction_comparable_pair_count": len(comparable),
        "direction_inactive_or_incomparable_pair_count": len(inactive_or_incomparable),
        "direction_inactive_or_missing_causes_protocol_mismatch": True,
        "direction_complete_cohort_comparison_passed": (
            len(comparable) == expected_comparable and not inactive_or_incomparable
        ),
        "direction_probability_tolerance": 1e-12,
        "maximum_comparable_direction_probability_abs_difference": maximum_difference,
        "direction_probability_reproduction_passed": direction_passed,
        "paired_metric_delta_new_minus_predecessor": deltas,
        "predecessor_metrics": reference_metrics,
        "control_reference_used_for_fit_or_threshold_selection": False,
    }


def run_consensus_ambiguity_gate(
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path | None = None,
    *,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path, project_root, config = load_config(config_path)
    execution = dict(config["execution"])
    worker_count = int(execution["workers"] if workers is None else workers)
    if worker_count <= 0 or worker_count > int(execution["maximum_workers"]):
        raise ValueError("workers must be in 1..20")
    inputs = {
        name: base._registered_input(project_root, specification, field=name)
        for name, specification in dict(config["inputs"]).items()
    }
    predecessor_config = _read_json(inputs["predecessor_config"])
    _validate_frozen_equivalence(config, predecessor_config)
    adapter_report = _read_json(inputs["expanded_train_report"])
    two_head._validate_adapter_report(
        adapter_report,
        pair_labels_path=inputs["expanded_stage1_pair_labels"],
    )
    states, state_integrity = base._load_states(
        inputs["h1_collection_report"], worker_count
    )
    manifest = frozen._load_manifest_index(inputs["h1_collection_report"])
    state_manifest_sha = str(state_integrity["state_manifest_sha256"])
    label_rows = _read_jsonl(inputs["expanded_stage1_pair_labels"])
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in label_rows:
        by_state[str(row.get("state_occurrence_id", ""))].append(row)
    consensus_state_ids = [
        state_id
        for state_id, state in states.items()
        if frozen._consensus_state_contract(state) is not None
    ]
    examples = []
    for state_id in sorted(consensus_state_ids):
        rows = by_state.get(state_id, [])
        if len(rows) != 1:
            raise ValueError(f"control consensus state must have one pair: {state_id}")
        row = rows[0]
        if str(row.get("source_h1_manifest_sha256", "")) != state_manifest_sha:
            raise ValueError(f"control label manifest SHA changed: {state_id}")
        examples.append(build_control_example(row, states[state_id], manifest[state_id]))
    label_counts = collections.Counter(row["label"] for row in examples)
    fold_counts = collections.Counter(row["train_fold"] for row in examples)
    active_maps = {str(row["map_id"]) for row in examples}
    observed_support = {
        "row_count": len(examples),
        "state_count": len({row["state_occurrence_id"] for row in examples}),
        "active_map_count": len(active_maps),
        "inactive_registered_maps": sorted(set(EXPANDED_MAPS) - active_maps),
        "label_counts": dict(sorted(label_counts.items())),
        "fold_row_counts": {fold: int(fold_counts[fold]) for fold in FOLDS},
    }
    if observed_support != EXPECTED_CONSENSUS_SUPPORT:
        raise ValueError("control consensus support changed")
    if len({row["pair_id"] for row in examples}) != len(examples):
        raise ValueError("control pair IDs are duplicated")
    if any(row["research_split"] != "expanded_train" for row in examples):
        raise ValueError("development/final row entered control fit")
    cohort_fingerprint = _fingerprint(
        [
            {
                "pair_id": row["pair_id"],
                "state_occurrence_id": row["state_occurrence_id"],
                "map_id": row["map_id"],
                "train_fold": row["train_fold"],
                "label": row["label"],
                "features": row["opportunity_features_signed18"],
            }
            for row in sorted(examples, key=lambda item: str(item["pair_id"]))
        ]
    )
    feature_matrix_fingerprint = _fingerprint(
        [
            {
                "pair_id": row["pair_id"],
                "signed18": row["opportunity_features_signed18"],
                "abs_delta18": row["opportunity_features_abs_delta18"],
            }
            for row in sorted(examples, key=lambda item: str(item["pair_id"]))
        ]
    )

    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if not dry_run and output_root.exists() and any(output_root.iterdir()):
        raise ValueError("consensus ambiguity-gate output already exists")

    import numpy  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.pipeline import Pipeline  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    outer_records = []
    predictions: list[dict[str, Any]] = []
    all_inner_direction_s_counts = []
    all_inner_direction_v_counts = []
    all_inner_opportunity_a_counts = []
    all_inner_opportunity_d_counts = []
    for outer_fold, registered_held_maps in FOLDS.items():
        outer_train = [row for row in examples if row["train_fold"] != outer_fold]
        outer_held = [row for row in examples if row["train_fold"] == outer_fold]
        inner_folds = tuple(fold for fold in FOLDS if fold != outer_fold)
        train_states = {row["state_occurrence_id"] for row in outer_train}
        held_states = {row["state_occurrence_id"] for row in outer_held}
        train_maps = {row["map_id"] for row in outer_train}
        held_maps = {row["map_id"] for row in outer_held}
        if train_states & held_states or train_maps & held_maps:
            raise ValueError(f"control outer leakage: {outer_fold}")
        if not held_maps or not held_maps <= set(registered_held_maps):
            raise ValueError(f"control outer held maps changed: {outer_fold}")
        jobs = (
            ("opportunity", "signed18"),
            ("opportunity", "abs_delta18"),
            ("direction", "signed18"),
        )
        try:
            with ThreadPoolExecutor(
                max_workers=min(worker_count, int(execution["parallel_head_jobs"]))
            ) as pool:
                futures = {
                    key: pool.submit(
                        cross_validate_head,
                        outer_train,
                        key[0],
                        key[1],
                        inner_folds,
                    )
                    for key in jobs
                }
                head_results = {
                    key: future.result() for key, future in futures.items()
                }
            for audit in head_results[("direction", "signed18")]["fold_audits"]:
                totals = audit["fit_weight_audit"]["base_class_weight_totals"]
                all_inner_direction_v_counts.append(int(totals["0"]))
                all_inner_direction_s_counts.append(int(totals["1"]))
            for audit in head_results[("opportunity", "signed18")]["fold_audits"]:
                totals = audit["fit_weight_audit"]["base_class_weight_totals"]
                all_inner_opportunity_a_counts.append(int(totals["0"]))
                all_inner_opportunity_d_counts.append(int(totals["1"]))
            selection = frozen.select_inner_arm(outer_train, head_results, config)
        except HeadClassSupportError as error:
            head_results = {}
            selection = {
                "status": "missing_inner_fit_class_exact_v2_outer_fold",
                "selected_arm": None,
                "candidate_arms": [],
                "error": str(error),
            }
        selected = selection["selected_arm"]
        outer_fit_weight_audits: dict[str, Any] = {}
        direction_fit_error = None
        try:
            direction_estimator, direction_audit = frozen._fit_outer_head(
                outer_train, "direction", "signed18"
            )
            outer_fit_weight_audits["direction"] = direction_audit
            direction_probabilities: list[float | None] = [
                float(value)
                for value in frozen._predict_probability(
                    direction_estimator,
                    outer_held,
                    "direction_features_signed18",
                )
            ]
        except frozen.HeadClassSupportError as error:
            direction_fit_error = str(error)
            direction_probabilities = [None for _ in outer_held]
            selection = {
                **selection,
                "outer_direction_fit_error": direction_fit_error,
            }
        if selected is None:
            held_predictions = [
                frozen.materialize_action(
                    row,
                    opportunity_probability=None,
                    direction_probability=direction_probability,
                    opportunity_threshold=1.01,
                    direction_threshold=1.01,
                    arm=None,
                    evaluation_active=False,
                    selection_status=str(selection["status"]),
                )
                for row, direction_probability in zip(
                    outer_held, direction_probabilities
                )
            ]
            outer_status = "INACTIVE_EXACT_V2_FAIL_CLOSED"
        elif direction_fit_error is not None:
            held_predictions = [
                frozen.materialize_action(
                    row,
                    opportunity_probability=None,
                    direction_probability=None,
                    opportunity_threshold=1.01,
                    direction_threshold=1.01,
                    arm=None,
                    evaluation_active=False,
                    selection_status="missing_outer_direction_fit_class_exact_v2",
                )
                for row in outer_held
            ]
            outer_status = "INACTIVE_EXACT_V2_MISSING_OUTER_DIRECTION_FIT_CLASS"
        else:
            try:
                opportunity_estimator, opportunity_audit = _fit_outer_head(
                    outer_train,
                    "opportunity",
                    str(selected["opportunity_view_id"]),
                )
                outer_fit_weight_audits["opportunity"] = opportunity_audit
                opportunity_probabilities = frozen._predict_probability(
                    opportunity_estimator,
                    outer_held,
                    f"opportunity_features_{selected['opportunity_view_id']}",
                )
                held_predictions = [
                    frozen.materialize_action(
                        row,
                        opportunity_probability=float(opportunity_probability),
                        direction_probability=float(direction_probability),
                        opportunity_threshold=float(selected["opportunity_threshold"]),
                        direction_threshold=float(selected["direction_threshold"]),
                        arm=selected,
                        evaluation_active=True,
                        selection_status=str(selection["status"]),
                    )
                    for row, opportunity_probability, direction_probability in zip(
                        outer_held,
                        opportunity_probabilities,
                        direction_probabilities,
                    )
                ]
                outer_status = "ACTIVE_NESTED_OUTER_PREDICTION"
            except HeadClassSupportError as error:
                held_predictions = [
                    frozen.materialize_action(
                        row,
                        opportunity_probability=None,
                        direction_probability=direction_probability,
                        opportunity_threshold=1.01,
                        direction_threshold=1.01,
                        arm=None,
                        evaluation_active=False,
                        selection_status="missing_outer_fit_class_exact_v2_outer_fold",
                    )
                    for row, direction_probability in zip(
                        outer_held, direction_probabilities
                    )
                ]
                selection = {**selection, "outer_fit_error": str(error)}
                outer_status = "INACTIVE_EXACT_V2_MISSING_OUTER_FIT_CLASS"
        predictions.extend(held_predictions)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "registered_held_maps": list(registered_held_maps),
                "observed_held_maps": sorted(held_maps),
                "outer_train_state_count": len(train_states),
                "outer_held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "outer_status": outer_status,
                "inner_selection": selection,
                "inner_head_oof_audits": {
                    f"{head}:{view}": result["fold_audits"]
                    for (head, view), result in sorted(head_results.items())
                },
                "outer_fit_weight_audits": outer_fit_weight_audits,
                "outer_labels_never_used_for_selection": True,
            }
        )

    if len(predictions) != len(examples) or {
        row["pair_id"] for row in predictions
    } != {row["pair_id"] for row in examples}:
        raise ValueError("control nested outer OOF coverage changed")
    overall = frozen.selected_action_metrics(predictions)
    active_predictions = [row for row in predictions if row["evaluation_active"]]
    active_diagnostic = frozen.selected_action_metrics(active_predictions)
    by_fold = frozen._group_metrics(predictions, "train_fold")
    by_map = frozen._group_metrics(predictions, "map_id")
    checks = frozen._final_checks(overall, config["final_gates"])

    # Control references are first read here, after every fit/threshold decision.
    predecessor_report = _read_json(inputs["predecessor_evaluation_report"])
    predecessor_rows = _read_jsonl(inputs["predecessor_oof_predictions"])
    if (
        predecessor_report.get("schema") != frozen.REPORT_SCHEMA
        or predecessor_report.get("experiment_id") != PREDECESSOR_ID
        or predecessor_report.get("complete") is not True
        or len(predecessor_rows) != EXPECTED_CONSENSUS_SUPPORT["row_count"]
    ):
        raise ValueError("predecessor control-reference identity changed")
    reference_audit = _paired_reference_audit(
        predictions, predecessor_rows, overall, predecessor_report
    )
    protocol_valid = bool(reference_audit["direction_probability_reproduction_passed"])
    passed = protocol_valid and all(checks.values())
    status = (
        "INVALID_CONTROL"
        if not protocol_valid
        else "POSTHOC_MECHANISM_SUPPORT_ONLY"
        if passed
        else "POSTHOC_CONTROL_NO_GO"
    )
    input_sha = {name: sha256_file(path) for name, path in inputs.items()}
    control_config_sha256 = sha256_file(config_path)
    run_fingerprint = _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "study_role": STUDY_ROLE,
            "config_sha256": control_config_sha256,
            "input_sha256": input_sha,
            "workers": worker_count,
            "predecessor": config["predecessor_contract"],
            "allowed_method_differences": list(ALLOWED_METHOD_DIFFERENCES),
            "cohort_fingerprint": cohort_fingerprint,
            "feature_matrix_fingerprint": feature_matrix_fingerprint,
            "fixed_map_folds": {fold: list(maps) for fold, maps in FOLDS.items()},
            "candidate_arms": config["candidate_arms"],
            "opportunity_thresholds": list(GATE_THRESHOLDS),
            "direction_thresholds": list(DIRECTION_THRESHOLDS),
            "inner_gates": config["inner_gates"],
            "final_gates": config["final_gates"],
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "study_role": STUDY_ROLE,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "input_integrity": {
            "registered_sha256": input_sha,
            **state_integrity,
            "source_h1_manifest_sha256": state_manifest_sha,
            "consensus_support": observed_support,
            "cohort_fingerprint": cohort_fingerprint,
            "feature_matrix_fingerprint": feature_matrix_fingerprint,
            "pair_count": len(examples),
            "state_count": len(examples),
            "development_row_count": 0,
            "stage2_row_count": 0,
        },
        "predecessor_control_reference": {
            **dict(config["predecessor_contract"]),
            "predecessor_config_file_sha256": input_sha["predecessor_config"],
            "predecessor_report_file_sha256": input_sha[
                "predecessor_evaluation_report"
            ],
            "predecessor_oof_file_sha256": input_sha["predecessor_oof_predictions"],
            "predecessor_run_fingerprint": predecessor_report.get("run_fingerprint"),
            "paired_audit": reference_audit,
        },
        "method_difference_audit": {
            "control_config_file_sha256": control_config_sha256,
            "predecessor_config_file_sha256": input_sha["predecessor_config"],
            "allowed_method_differences": list(ALLOWED_METHOD_DIFFERENCES),
            "opportunity_target": "decisive_structural_or_v2_vs_ambiguous",
            "opportunity_fit_population": "all_93_consensus_rows",
            "opportunity_class_balance": False,
            "direction_contract_unchanged": True,
            "inputs_folds_features_models_grids_gates_equal_predecessor": True,
            "control_reference_used_for_fit_or_threshold_selection": False,
        },
        "head_fit_contract": dict(config["head_contract"]),
        "fit_support": {
            "opportunity_all_rows": 93,
            "opportunity_ambiguous_rows": 69,
            "opportunity_decisive_rows": 24,
            "opportunity_full_cohort_weights_all_one": True,
            "opportunity_full_cohort_base_class_weight_totals": {
                "0_ambiguous": 69.0,
                "1_decisive": 24.0,
            },
            "opportunity_class_balance_applied": False,
            "direction_all_rows": 24,
            "direction_structural_rows": 14,
            "direction_v2_rows": 10,
            "direction_class_balance_applied": True,
            "minimum_inner_opportunity_fit_ambiguous_count": (
                min(all_inner_opportunity_a_counts)
                if all_inner_opportunity_a_counts
                else None
            ),
            "minimum_inner_opportunity_fit_decisive_count": (
                min(all_inner_opportunity_d_counts)
                if all_inner_opportunity_d_counts
                else None
            ),
            "minimum_inner_direction_fit_structural_count": (
                min(all_inner_direction_s_counts)
                if all_inner_direction_s_counts
                else None
            ),
            "minimum_inner_direction_fit_v2_count": (
                min(all_inner_direction_v_counts)
                if all_inner_direction_v_counts
                else None
            ),
            "ambiguous_rows_in_direction_fit": 0,
        },
        "nested_cv": {
            "outer_fold_count": 4,
            "inner_fold_count_per_outer": 3,
            "outer_labels_never_used_for_selection": True,
            "single_global_threshold_pair_per_outer_fold": True,
            "inactive_outer_fold_exact_v2_fail_closed": True,
            "inactive_outer_fold_not_a_breadth_win": True,
            "inactive_outer_fold_still_in_full_pooled_metrics": True,
            "outer_folds": outer_records,
        },
        "primary_full_pooled_selected_action_metrics": overall,
        "diagnostic_active_outer_subset_metrics": active_diagnostic,
        "diagnostic_by_fold": by_fold,
        "diagnostic_by_map": by_map,
        "final_gate_checks": checks,
        "protocol_valid": protocol_valid,
        "hard_gates_passed": passed,
        "threshold_selection_rule": list(THRESHOLD_SELECTION_RULE),
        "arm_selection_rule": list(ARM_SELECTION_RULE),
        "full_fit_model_executed": False,
        "model_exported": False,
        "development_rows_loaded": False,
        "stage2_rows_loaded": False,
        "promotion_or_default_exported": False,
        "v2_replaced": False,
        "challenger_claimed": False,
        "runtime_or_ttf_claim_authorized": False,
        "final_claim_authorized": False,
        "next_decision": (
            "record_posthoc_ambiguity_gate_mechanism_support_only"
            if passed
            else "invalidate_posthoc_control_protocol"
            if not protocol_valid
            else "record_posthoc_ambiguity_gate_control_no_go"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    feature_manifest = {
        "schema": (
            "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_feature_manifest.v1"
        ),
        "run_fingerprint": run_fingerprint,
        "study_role": STUDY_ROLE,
        "dimension": 18,
        "signed_feature_names": list(FEATURE_NAMES),
        "opportunity_feature_views": list(OPPORTUNITY_VIEWS),
        "direction_feature_view": "signed18",
        "cohort_fingerprint": cohort_fingerprint,
        "feature_matrix_fingerprint": feature_matrix_fingerprint,
        "leakage_audit_passed": True,
        "stage2_rows_loaded": False,
        "development_partition_present": False,
    }
    feature_path = output_root / outputs["feature_manifest"]
    prediction_path = output_root / outputs["oof_predictions"]
    report_path = output_root / outputs["evaluation_report"]
    _write_json(feature_path, feature_manifest)
    output_rows = [
        _prediction_output(row, run_fingerprint)
        for row in sorted(predictions, key=lambda item: str(item["pair_id"]))
    ]
    _write_jsonl(prediction_path, output_rows)
    report["artifacts"] = {
        "feature_manifest": {
            "file": feature_path.name,
            "sha256": sha256_file(feature_path),
        },
        "oof_predictions": {
            "file": prediction_path.name,
            "schema": PREDICTION_SCHEMA,
            "row_count": len(output_rows),
            "sha256": sha256_file(prediction_path),
        },
        "model": None,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "ALLOWED_METHOD_DIFFERENCES",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "PREDICTION_SCHEMA",
    "REPORT_SCHEMA",
    "STUDY_ROLE",
    "build_control_example",
    "cross_validate_head",
    "run_consensus_ambiguity_gate",
    "validate_config",
]
