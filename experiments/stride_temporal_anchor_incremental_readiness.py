from __future__ import annotations

import collections
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.run_output_guard import prepare_run_output
from experiments.stride_safeslot_gate import (
    FEATURE_NAMES as STATIC_FEATURE_NAMES,
    _calibration_checks,
    _fit_model,
    _inner_select,
    _map_folds,
    _parameters,
    _state_path,
    build_safeslot_gate_rows,
    candidate_classification_metrics,
    evaluate_gate_policy,
    summarize_gate_policy,
    validate_safeslot_gate_config,
)
from experiments.trace_replay import _initial_state, recorded_replay_action, state_before_decision


CONFIG_SCHEMA = "lns2.stride.temporal_anchor_incremental_readiness_config.v1"
ROW_SCHEMA = "lns2.stride.temporal_anchor_incremental_readiness_row.v1"
PREDICTION_SCHEMA = "lns2.stride.temporal_anchor_incremental_oof_prediction.v1"
REPORT_SCHEMA = "lns2.stride.temporal_anchor_incremental_readiness_report.v1"
IMPLEMENTATION_ID = "temporal-anchor-incremental-readiness-v1"
HISTORY_FEATURE_NAMES = (
    "history.available_steps",
    "history.replan_failure_count4",
    "history.accepted_nondecrease_count4",
    "history.strict_decrease_count4",
    "history.consecutive_replan_failure",
    "history.consecutive_no_strict_decrease",
    "history.last_normalized_conflict_reduction",
    "history.mean4_normalized_conflict_reduction",
    "delta:last_attempt_jaccard",
    "delta:max_failure_jaccard",
    "delta:max_progress_jaccard",
    "delta:failure_union_coverage",
    "delta:progress_union_coverage",
    "delta:mean_recent_repair_exposure",
    "delta:exact_prior_neighborhood_repeat_count",
    "delta:outcome_signed_agent_exposure",
)
FEATURE_NAMES = (*STATIC_FEATURE_NAMES, *HISTORY_FEATURE_NAMES)
PRODUCER_FILES = (
    "experiments/stride_temporal_anchor_incremental_readiness.py",
    "experiments/stride_safeslot_gate.py",
    "experiments/trace_replay.py",
    "scripts/run_stride_temporal_anchor_incremental_readiness.py",
)


def _registered(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {
        name: registered_input(project_root, dict(spec), label=IMPLEMENTATION_ID)
        for name, spec in dict(config["inputs"]).items()
    }


def validate_config(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != "preregistered_minimal_offline_readiness_audit"
        or config.get("implementation_id") != IMPLEMENTATION_ID
    ):
        raise ValueError("temporal readiness identity changed")
    expected_inputs = {
        "safeslot_config", "safeslot_oof_predictions", "state_selection",
        "state_selection_report", "source_v4_official_manifest", "source_v4_v2_manifest",
        "da2_v2_official_manifest", "da2_v2_v2_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("temporal readiness input registry changed")
    if dict(config.get("cohort") or {}) != {
        "state_count": 98, "map_count": 16, "candidate_count": 587,
        "positive_candidate_count": 201, "decision0_state_count": 30,
        "history_state_count": 68, "history_map_count": 15,
        "history_candidate_count": 407, "history_positive_candidate_count": 149,
        "source_trace_count": 80, "history_unique_source_trace_count": 56,
    }:
        raise ValueError("temporal readiness cohort changed")
    features = dict(config.get("features") or {})
    if (
        int(features.get("static_dimension", -1)) != 193
        or int(features.get("history_dimension", -1)) != 16
        or int(features.get("augmented_dimension", -1)) != 209
        or int(features.get("history_window", -1)) != 4
        or tuple(features.get("history_feature_names") or ()) != HISTORY_FEATURE_NAMES
        or list(features.get("occurrence_key") or ())
        != ["episode_id", "decision_index", "before_fingerprint"]
        or features.get("historical_union_coverage_denominator") != "historical_union_size"
        or features.get("candidate_exposure_denominator") != "available_prefix_steps"
        or features.get("candidate_relative_representation")
        != "challenger_minus_exact_v2_anchor"
        or set(features.get("forbidden_inputs") or ())
        != {"map_id", "source_policy", "pp_seconds", "pp_seed", "repair_order", "ttf", "current_outcome", "future_transition"}
    ):
        raise ValueError("temporal readiness feature contract changed")
    model = dict(config.get("model") or {})
    if (
        model.get("class") != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(model.get("fixed_parameters") or {}) != {
            "early_stopping": False, "learning_rate": 0.05, "max_iter": 100,
            "min_samples_leaf": 20, "random_state": 20260809,
        }
        or list(model.get("parameter_grid") or ()) != [
            {"max_leaf_nodes": 7, "l2_regularization": 0.1},
            {"max_leaf_nodes": 7, "l2_regularization": 1.0},
            {"max_leaf_nodes": 15, "l2_regularization": 0.1},
            {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        ]
        or int(model.get("inner_map_folds", 0)) != 3
        or list(model.get("threshold_grid") or ()) != [.70, .75, .80, .85, .90, .95, 1.01]
        or list(model.get("outer_map_folds") or ()) != [
            ["ca_cave", "ca_caverns2", "dr_primevalentrance", "lak106d", "lt_hangedman"],
            ["den900d", "lak203d", "lgt604d", "lt_undercitydungeon", "orz201d"],
            ["hrt001d", "lak250d", "orz601d"],
            ["ht_bartrand_n", "lak526d", "w_encounter3"],
        ]
    ):
        raise ValueError("temporal readiness model contract changed")
    policy = dict(config.get("policy") or {})
    if policy != {
        "decision0_action": "force_abstain_to_exact_v2_anchor",
        "calibration_population": "history_bearing_states_only",
        "minimum_safe_replace_precision": 0.80,
        "minimum_replacement_state_fraction": 0.10,
        "minimum_selected_map_fraction": 0.50,
        "minimum_mean_current_step_advantage": 0.0,
        "minimum_first_fixed_half_advantage": 0.0,
        "minimum_second_fixed_half_advantage": 0.0,
        "maximum_mean_no_progress_rate_delta": 0.0,
        "maximum_mean_residual_risk_delta": 0.0,
        "maximum_map_mean_current_step_regret": 0.02,
        "maximum_map_mean_residual_risk_degradation": 0.02,
    }:
        raise ValueError("temporal readiness policy contract changed")
    if dict(config.get("offline_acceptance") or {}) != {
        "minimum_history_candidate_roc_auc": 0.68,
        "minimum_history_auc_gain_over_static": 0.03,
        "minimum_each_fold_history_roc_auc": 0.55,
        "minimum_each_fold_auc_gain_over_static": -0.02,
        "minimum_selected_safe_replace_precision": 0.80,
        "minimum_history_replacement_state_fraction": 0.10,
        "minimum_selected_map_count": 8,
        "minimum_source_policy_mean_current_step_advantage": 0.0,
        "require_all_outer_calibrations_feasible": True,
        "maximum_static_probability_reproduction_error": 1e-12,
    }:
        raise ValueError("temporal readiness hard gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "offline_readiness_only": True,
        "solver_native_or_pp_executed": False,
        "runtime_integration_allowed": False,
        "model_export_allowed": False,
        "formal_ttf_claim": False,
        "plateau_or_repeated_rollback_claim": False,
        "component_hotspot_policy_history_matched": False,
        "pass_allows_only": "fresh_matched_component_hotspot_history_collection",
        "fail_action": "stop_without_window_feature_threshold_retuning",
    }:
        raise ValueError("temporal readiness claim boundary changed")
    if int(config.get("analysis_wall_fuse_seconds", 0)) != 600:
        raise ValueError("temporal readiness wall fuse changed")
    if dict(config.get("source_roots") or {}) != {
        "source-v4": "build/stride-robustaction-structpool-source-episodes-v4",
        "da2-stability-v2": "build/stride-robustaction-da2-source-stability-episodes-v2",
    } or dict(config.get("outputs") or {}) != {
        "readiness": "build/stride-temporal-anchor-incremental-readiness-v1"
    }:
        raise ValueError("temporal readiness source/output contract changed")
    if project_root is not None:
        _registered(project_root, config)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _union_coverage(candidate: set[int], histories: Iterable[set[int]]) -> float:
    union: set[int] = set().union(*histories)
    return len(candidate & union) / len(union) if union else 0.0


def _candidate_history_values(candidate: set[int], transitions: list[dict[str, Any]]) -> list[float]:
    if not candidate:
        raise ValueError("history candidate is empty")
    if not transitions:
        return [0.0] * 8
    neighborhoods = [set(map(int, row["neighborhood"])) for row in transitions]
    failures = [agents for agents, row in zip(neighborhoods, transitions) if not bool(row["replan_success"])]
    progress = [agents for agents, row in zip(neighborhoods, transitions) if bool(row["replan_success"]) and float(row["conflict_delta"]) > 0.0]
    exposure = statistics.fmean(
        sum(agent in agents for agents in neighborhoods) / len(neighborhoods)
        for agent in candidate
    )
    signed = statistics.fmean(
        sum(
            (1 if bool(row["replan_success"]) and float(row["conflict_delta"]) > 0.0 else -1 if not bool(row["replan_success"]) else 0)
            * (agent in agents)
            for agents, row in zip(neighborhoods, transitions)
        ) / len(neighborhoods)
        for agent in candidate
    )
    return [
        _jaccard(candidate, neighborhoods[-1]),
        max((_jaccard(candidate, agents) for agents in failures), default=0.0),
        max((_jaccard(candidate, agents) for agents in progress), default=0.0),
        _union_coverage(candidate, failures),
        _union_coverage(candidate, progress),
        exposure,
        float(sum(candidate == agents for agents in neighborhoods)),
        signed,
    ]


def history_feature_vector(
    challenger: set[int], anchor: set[int], transitions: list[dict[str, Any]], *, window: int = 4
) -> list[float]:
    recent = list(transitions[-int(window):])
    reductions = [float(row["conflict_delta"]) / max(1.0, float(row["conflicts_before"])) for row in recent]
    failures = [not bool(row["replan_success"]) for row in recent]
    strict = [bool(row["replan_success"]) and float(row["conflict_delta"]) > 0.0 for row in recent]
    consecutive_failure = 0
    consecutive_no_strict = 0
    for value in reversed(failures):
        if not value: break
        consecutive_failure += 1
    for value in reversed(strict):
        if value: break
        consecutive_no_strict += 1
    shared = [
        float(len(recent)), float(sum(failures)),
        float(sum(bool(row["replan_success"]) and float(row["conflict_delta"]) <= 0.0 for row in recent)),
        float(sum(strict)), float(consecutive_failure), float(consecutive_no_strict),
        reductions[-1] if reductions else 0.0,
        statistics.fmean(reductions) if reductions else 0.0,
    ]
    challenger_values = _candidate_history_values(challenger, recent)
    anchor_values = _candidate_history_values(anchor, recent)
    values = [*shared, *(left - right for left, right in zip(challenger_values, anchor_values))]
    if len(values) != len(HISTORY_FEATURE_NAMES) or any(not math.isfinite(value) for value in values):
        raise ValueError("temporal history vector is invalid")
    return values


def _target_blind_prefix(events: list[dict[str, Any]], decision_index: int) -> list[dict[str, Any]]:
    """Extract only completed transitions; target and future payloads are never read."""
    result: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "transition":
            continue
        current = int(event["decision_index"])
        if current >= int(decision_index):
            break
        metrics = dict(event["metrics"])
        result.append({
            "decision_index": current,
            "neighborhood": list(map(int, metrics["neighborhood"])),
            "replan_success": bool(metrics["replan_success"]),
            "conflict_delta": float(metrics["conflict_delta"]),
            "conflicts_before": int(metrics["conflicts_before"]),
            "before_fingerprint": str(event["before_fingerprint"]),
            "after_fingerprint": str(event["after_fingerprint"]),
        })
    if len(result) != int(decision_index):
        raise ValueError("source trace prefix length changed")
    return result


def _unique(rows: Iterable[dict[str, Any]], *fields: str) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"duplicate temporal readiness key: {key}")
        result[key] = row
    return result


def _manifest_input_name(namespace: str, policy: str) -> str:
    names = {
        ("source-v4", "official_adaptive"): "source_v4_official_manifest",
        ("source-v4", "realized_dynamic"): "source_v4_v2_manifest",
        ("da2-stability-v2", "official_adaptive"): "da2_v2_official_manifest",
        ("da2-stability-v2", "realized_dynamic"): "da2_v2_v2_manifest",
    }
    try:
        return names[(namespace, policy)]
    except KeyError as error:
        raise ValueError(f"unregistered source stream: {(namespace, policy)}") from error


def _candidate_agent_sets(
    safeslot_config: dict[str, Any], base_rows: list[dict[str, Any]], *, project_root: Path
) -> dict[tuple[str, str], set[int]]:
    grid_manifest = _read_jsonl(
        registered_input(project_root, dict(safeslot_config["inputs"]["grid_manifest"]), label="SafeSlot grid")
    )
    grid_root = (project_root / str(safeslot_config["grid_state_artifact_root"])).resolve()
    required: dict[str, set[str]] = collections.defaultdict(set)
    for row in base_rows:
        required[str(row["state_id"])].update(
            (str(row["anchor_candidate_id"]), str(row["challenger_candidate_id"]))
        )
    result: dict[tuple[str, str], set[int]] = {}
    for manifest_row in grid_manifest:
        state_id = str(manifest_row["state_id"])
        if state_id not in required:
            continue
        grid = _read_json(_state_path(grid_root, manifest_row))
        candidates = {
            str(row["candidate_id"]): set(map(int, row["agents"]))
            for row in list(grid["candidates"])
        }
        anchor = dict(grid["v2_base_anchor"])
        candidates[str(anchor["candidate_id"])] = set(map(int, anchor["agents"]))
        for candidate_id in required[state_id]:
            agents = candidates.get(candidate_id)
            if not agents or len(agents) != len(set(agents)):
                raise ValueError(f"candidate agents changed: {(state_id, candidate_id)}")
            result[(state_id, candidate_id)] = agents
    if len(result) != sum(len(value) for value in required.values()):
        raise ValueError("candidate agent-set coverage changed")
    return result


def build_temporal_rows(
    config: dict[str, Any], *, project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = _registered(project_root, config)
    safeslot_config = _read_json(inputs["safeslot_config"])
    validate_safeslot_gate_config(safeslot_config, project_root=project_root)
    base_rows, base_audit = build_safeslot_gate_rows(safeslot_config, project_root=project_root)
    agents = _candidate_agent_sets(safeslot_config, base_rows, project_root=project_root)
    state_ids = {str(row["state_id"]) for row in base_rows}
    selections = _unique(
        (row for row in _read_jsonl(inputs["state_selection"]) if str(row["state_id"]) in state_ids),
        "state_id",
    )
    if set(key[0] for key in selections) != state_ids:
        raise ValueError("SafeSlot occurrence coverage changed")
    selection_report = _read_json(inputs["state_selection_report"])
    if selection_report.get("passed") is not True:
        raise ValueError("registered state selection did not pass")

    manifest_indices: dict[str, dict[tuple[str], dict[str, Any]]] = {}
    for name in (
        "source_v4_official_manifest", "source_v4_v2_manifest",
        "da2_v2_official_manifest", "da2_v2_v2_manifest",
    ):
        manifest_indices[name] = _unique(
            (row for row in _read_jsonl(inputs[name]) if str(row.get("status")) == "ok"),
            "episode_id",
        )

    trace_cache: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]], str]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    occurrences: set[tuple[str, int, str]] = set()
    prefix_failure_states = prefix_accepted_nondecrease_states = prefix_no_strict_decrease_states = prefix_progress_states = 0
    mixed_failure_progress_states = long_prefix_states = unchanged_transition_states = 0
    for state_id, selection in sorted((key[0], row) for key, row in selections.items()):
        namespace = str(selection["source_namespace"])
        manifest_policy = str(selection["source_manifest_policy"])
        manifest_name = _manifest_input_name(namespace, manifest_policy)
        manifest = manifest_indices[manifest_name].get((str(selection["episode_id"]),))
        if manifest is None:
            raise ValueError(f"source episode missing: {selection['episode_id']}")
        root = (project_root / str(config["source_roots"][namespace])).resolve()
        cache_key = (namespace, str(selection["episode_id"]))
        if cache_key not in trace_cache:
            trace_path = contained_file(root, manifest["trace_file"], field="source trace")
            observed_hash = sha256_file(trace_path)
            if observed_hash != str(manifest["trace_sha256"]):
                raise ValueError(f"source trace hash changed: {selection['episode_id']}")
            events = read_trace_events(trace_path)
            if not events or events[0].get("event") != "initial":
                raise ValueError("source trace initial event changed")
            initial = _initial_state(root, trace_path, events[0])
            trace_cache[cache_key] = (initial, events, observed_hash)
        initial, events, _trace_hash = trace_cache[cache_key]
        decision_index = int(selection["decision_index"])
        transitions = [row for row in events if row.get("event") == "transition"]
        target_state = state_before_decision(
            initial, transitions, decision_index=decision_index,
            expected_fingerprint=str(selection["before_fingerprint"]),
        )
        if int(target_state["num_of_colliding_pairs"]) != int(selection["before_conflicts"]):
            raise ValueError(f"target conflict count changed: {state_id}")
        prefix = _target_blind_prefix(events, decision_index)
        replay_prefix = [recorded_replay_action(row) for row in transitions[:decision_index]]
        if replay_prefix != list(selection["prefix_actions"]):
            raise ValueError(f"stored source prefix changed: {state_id}")
        occurrence = (str(selection["episode_id"]), decision_index, str(selection["before_fingerprint"]))
        if occurrence in occurrences:
            raise ValueError(f"duplicate exact occurrence: {occurrence}")
        occurrences.add(occurrence)
        histories[state_id] = prefix
        recent = prefix[-int(config["features"]["history_window"]):]
        failures = [not bool(row["replan_success"]) for row in recent]
        accepted_nondecrease = [bool(row["replan_success"]) and float(row["conflict_delta"]) <= 0 for row in recent]
        no_strict_decrease = [not (bool(row["replan_success"]) and float(row["conflict_delta"]) > 0) for row in recent]
        progress = [bool(row["replan_success"]) and float(row["conflict_delta"]) > 0 for row in recent]
        prefix_failure_states += any(failures)
        prefix_accepted_nondecrease_states += any(accepted_nondecrease)
        prefix_no_strict_decrease_states += any(no_strict_decrease)
        prefix_progress_states += any(progress)
        mixed_failure_progress_states += any(failures) and any(progress)
        long_prefix_states += len(prefix) >= 4
        unchanged_transition_states += any(row["before_fingerprint"] == row["after_fingerprint"] for row in prefix)

    rows: list[dict[str, Any]] = []
    for base in base_rows:
        state_id = str(base["state_id"])
        selection = selections[(state_id,)]
        history_values = history_feature_vector(
            agents[(state_id, str(base["challenger_candidate_id"]))],
            agents[(state_id, str(base["anchor_candidate_id"]))],
            histories[state_id], window=int(config["features"]["history_window"]),
        )
        row = {
            **base,
            "schema": ROW_SCHEMA,
            "episode_id": str(selection["episode_id"]),
            "decision_index": int(selection["decision_index"]),
            "before_fingerprint": str(selection["before_fingerprint"]),
            "history_available": bool(histories[state_id]),
            "static_feature_values": list(base["feature_values"]),
            "history_feature_values": history_values,
            "feature_values": [*base["feature_values"], *history_values],
        }
        rows.append(row)
    cohort = dict(config["cohort"])
    history_rows = [row for row in rows if bool(row["history_available"])]
    history_states = {str(row["state_id"]) for row in history_rows}
    audit = {
        **base_audit,
        "decision0_state_count": len(state_ids - history_states),
        "history_state_count": len(history_states),
        "history_map_count": len({str(row["map_id"]) for row in history_rows}),
        "history_candidate_count": len(history_rows),
        "history_positive_candidate_count": sum(bool(row["label"]) for row in history_rows),
        "source_trace_count": len(trace_cache),
        "history_unique_source_trace_count": len({
            (str(selections[(state_id,)]["source_namespace"]), str(selections[(state_id,)]["episode_id"]))
            for state_id in history_states
        }),
        "exact_occurrence_count": len(occurrences),
        "prefix_failure_state_count": prefix_failure_states,
        "prefix_accepted_nondecrease_state_count": prefix_accepted_nondecrease_states,
        "prefix_no_strict_decrease_state_count": prefix_no_strict_decrease_states,
        "prefix_strict_progress_state_count": prefix_progress_states,
        "mixed_failure_progress_state_count": mixed_failure_progress_states,
        "prefix_at_least_four_state_count": long_prefix_states,
        "exact_unchanged_transition_state_count": unchanged_transition_states,
        "source_trace_sha256_verified": True,
        "target_state_reconstructed_without_target_outcome": True,
    }
    expected = {
        "state_count": cohort["state_count"], "map_count": cohort["map_count"],
        "eligible_challenger_count": cohort["candidate_count"],
        "positive_challenger_count": cohort["positive_candidate_count"],
        "decision0_state_count": cohort["decision0_state_count"],
        "history_state_count": cohort["history_state_count"],
        "history_map_count": cohort["history_map_count"],
        "history_candidate_count": cohort["history_candidate_count"],
        "history_positive_candidate_count": cohort["history_positive_candidate_count"],
        "source_trace_count": cohort["source_trace_count"],
        "history_unique_source_trace_count": cohort["history_unique_source_trace_count"],
    }
    if any(int(audit[name]) != int(value) for name, value in expected.items()):
        raise ValueError(f"temporal readiness row audit changed: {audit}")
    if any(len(row["feature_values"]) != len(FEATURE_NAMES) for row in rows):
        raise ValueError("augmented feature dimension changed")
    return rows, audit


def _history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["decision_index"]) > 0]


def _history_probabilities(
    rows: list[dict[str, Any]], probabilities: list[float]
) -> tuple[list[dict[str, Any]], list[float]]:
    pairs = [(row, float(value)) for row, value in zip(rows, probabilities) if int(row["decision_index"]) > 0]
    return [row for row, _ in pairs], [value for _, value in pairs]


def force_decision0_abstention_probabilities(
    rows: list[dict[str, Any]], probabilities: list[float]
) -> list[float]:
    if len(rows) != len(probabilities):
        raise ValueError("decision0 abstention inputs are incomplete")
    return [
        0.0 if int(row["decision_index"]) == 0 else float(probability)
        for row, probability in zip(rows, probabilities)
    ]


def _inner_temporal_select(
    *, config: dict[str, Any], rows: list[dict[str, Any]], values: Any,
    labels: Any, weights: Any, train_maps: set[str],
) -> tuple[int, float, bool, list[dict[str, Any]]]:
    folds = _map_folds(rows, train_maps, int(config["model"]["inner_map_folds"]))
    target_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in train_maps]
    summaries: list[dict[str, Any]] = []
    for parameter_index in range(len(config["model"]["parameter_grid"])):
        predictions: dict[int, float] = {}
        for fold in folds:
            fit_maps = set(map(str, fold["train_maps"]))
            validation_maps = set(map(str, fold["validation_maps"]))
            if fit_maps & validation_maps or fit_maps | validation_maps != train_maps:
                raise RuntimeError("temporal inner map leakage")
            fit_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in fit_maps]
            validation_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in validation_maps]
            estimator = _fit_model(values, labels, weights, fit_indices, _parameters(config, parameter_index))
            probabilities = estimator.predict_proba(values[validation_indices])[:, 1]
            predictions.update({index: float(value) for index, value in zip(validation_indices, probabilities)})
        if set(predictions) != set(target_indices):
            raise RuntimeError("temporal inner OOF coverage changed")
        inner_rows = [rows[index] for index in target_indices if int(rows[index]["decision_index"]) > 0]
        inner_probabilities = [predictions[index] for index in target_indices if int(rows[index]["decision_index"]) > 0]
        classification = candidate_classification_metrics(inner_rows, inner_probabilities)
        threshold_summaries: list[dict[str, Any]] = []
        selected_threshold = 1.01
        selected_metrics: dict[str, Any] | None = None
        for threshold in map(float, config["model"]["threshold_grid"]):
            _records, policy_metrics = evaluate_gate_policy(inner_rows, inner_probabilities, threshold)
            checks = _calibration_checks(policy_metrics, threshold, dict(config["policy"]))
            feasible = all(checks.values())
            threshold_summaries.append({
                "threshold": threshold, "metrics": policy_metrics,
                "checks": checks, "feasible": feasible,
            })
            if feasible and selected_metrics is None:
                selected_threshold = threshold
                selected_metrics = policy_metrics
        summaries.append({
            "parameter_index": parameter_index,
            "parameters": _parameters(config, parameter_index),
            "history_candidate_metrics": classification,
            "selected_threshold": selected_threshold,
            "calibration_feasible": selected_metrics is not None,
            "selected_policy_metrics": selected_metrics,
            "threshold_summaries": threshold_summaries,
        })
    feasible = [row for row in summaries if bool(row["calibration_feasible"])]
    if not feasible:
        return 0, 1.01, False, summaries
    selected = min(feasible, key=lambda row: (
        -float(row["selected_policy_metrics"]["replacement_state_fraction"]),
        -float(row["selected_policy_metrics"]["safe_replace_precision"]),
        -float(row["history_candidate_metrics"]["roc_auc"]),
        int(row["parameter_index"]),
    ))
    return int(selected["parameter_index"]), float(selected["selected_threshold"]), True, summaries


def _static_oof(
    rows: list[dict[str, Any]], safeslot_config: dict[str, Any]
) -> tuple[list[float], list[dict[str, Any]]]:
    import numpy as np

    values = np.asarray([row["static_feature_values"] for row in rows], dtype=np.float32)
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    all_maps = {str(row["map_id"]) for row in rows}
    folds = _map_folds(rows, all_maps, int(safeslot_config["model"]["outer_map_folds"]))
    probabilities: dict[int, float] = {}
    diagnostics: list[dict[str, Any]] = []
    for fold in folds:
        train_maps = set(map(str, fold["train_maps"]))
        parameter_index, threshold, feasible, _inner = _inner_select(
            config=safeslot_config, rows=rows, values=values, labels=labels,
            weights=weights, train_maps=train_maps,
        )
        train_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in train_maps]
        validation_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in set(map(str, fold["validation_maps"]))]
        estimator = _fit_model(values, labels, weights, train_indices, _parameters(safeslot_config, parameter_index))
        predicted = estimator.predict_proba(values[validation_indices])[:, 1]
        probabilities.update({index: float(value) for index, value in zip(validation_indices, predicted)})
        diagnostics.append({
            "outer_fold": int(fold["fold"]), "validation_maps": list(fold["validation_maps"]),
            "selected_parameter_index": parameter_index, "selected_threshold": threshold,
            "calibration_feasible": feasible,
        })
    if set(probabilities) != set(range(len(rows))):
        raise RuntimeError("static OOF coverage changed")
    return [probabilities[index] for index in range(len(rows))], diagnostics


def _temporal_oof(
    rows: list[dict[str, Any]], config: dict[str, Any], *, feature_key: str
) -> tuple[list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np

    values = np.asarray([row[feature_key] for row in rows], dtype=np.float32)
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    all_maps = {str(row["map_id"]) for row in rows}
    folds = _map_folds(rows, all_maps, 4)
    expected_folds = [sorted(map(str, fold)) for fold in config["model"]["outer_map_folds"]]
    if [sorted(map(str, fold["validation_maps"])) for fold in folds] != expected_folds:
        raise ValueError("frozen outer map folds changed")
    probabilities: dict[int, float] = {}
    diagnostics: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    for fold in folds:
        train_maps = set(map(str, fold["train_maps"]))
        validation_maps = set(map(str, fold["validation_maps"]))
        parameter_index, threshold, feasible, inner = _inner_temporal_select(
            config=config, rows=rows, values=values, labels=labels,
            weights=weights, train_maps=train_maps,
        )
        train_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in train_maps]
        validation_indices = [index for index, row in enumerate(rows) if str(row["map_id"]) in validation_maps]
        estimator = _fit_model(values, labels, weights, train_indices, _parameters(config, parameter_index))
        predicted = list(map(float, estimator.predict_proba(values[validation_indices])[:, 1]))
        probabilities.update({index: value for index, value in zip(validation_indices, predicted)})
        fold_rows = [rows[index] for index in validation_indices]
        policy_probabilities = force_decision0_abstention_probabilities(fold_rows, predicted)
        records, all_policy = evaluate_gate_policy(fold_rows, policy_probabilities, threshold)
        source_by_state = {str(row["state_id"]): str(row["source_policy"]) for row in fold_rows}
        decision_by_state = {str(row["state_id"]): int(row["decision_index"]) for row in fold_rows}
        for record in records:
            record["outer_fold"] = int(fold["fold"])
            record["calibration_feasible"] = feasible
            record["source_policy"] = source_by_state[str(record["state_id"])]
            record["decision_index"] = decision_by_state[str(record["state_id"])]
            if int(record["decision_index"]) == 0 and bool(record["selected_challenger"]):
                raise RuntimeError("decision0 replacement escaped forced abstention")
        policy_records.extend(records)
        history_fold_rows, history_fold_probabilities = _history_probabilities(fold_rows, predicted)
        diagnostics.append({
            "feature_key": feature_key,
            "outer_fold": int(fold["fold"]), "train_maps": sorted(train_maps),
            "validation_maps": sorted(validation_maps),
            "selected_parameter_index": parameter_index, "selected_threshold": threshold,
            "calibration_feasible": feasible,
            "history_candidate_metrics": candidate_classification_metrics(history_fold_rows, history_fold_probabilities),
            "all_state_policy_metrics": all_policy,
            "history_state_policy_metrics": summarize_gate_policy([row for row in records if int(row["decision_index"]) > 0]),
            "inner_parameter_summaries": inner,
        })
    if set(probabilities) != set(range(len(rows))):
        raise RuntimeError("temporal OOF coverage changed")
    return [probabilities[index] for index in range(len(rows))], diagnostics, policy_records


def _reproduction_error(
    rows: list[dict[str, Any]], probabilities: list[float], registered_rows: list[dict[str, Any]],
    fold_diagnostics: list[dict[str, Any]],
) -> tuple[float, bool]:
    expected = _unique(registered_rows, "state_id", "challenger_candidate_id")
    differences = []
    folds_equal = True
    fold_by_map = {
        str(map_id): int(fold["outer_fold"])
        for fold in fold_diagnostics for map_id in fold["validation_maps"]
    }
    for row, probability in zip(rows, probabilities):
        old = expected.get((str(row["state_id"]), str(row["challenger_candidate_id"])))
        if old is None:
            raise ValueError("registered static OOF row coverage changed")
        differences.append(abs(float(probability) - float(old["probability"])))
        folds_equal &= int(old["outer_fold"]) == int(fold_by_map[str(row["map_id"])])
    if len(expected) != len(rows):
        raise ValueError("registered static OOF count changed")
    return max(differences, default=0.0), folds_equal


def run_temporal_anchor_incremental_readiness(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_config(config, project_root=project_root)
    inputs = _registered(project_root, config)
    safeslot_config = _read_json(inputs["safeslot_config"])
    producer = producer_identity(
        project_root=project_root, source_files=PRODUCER_FILES,
        native_required=False, package_names=("numpy", "scikit-learn"),
    )
    output = Path(output).resolve()
    prepare_run_output(output, resume=False, identity={
        "runner": IMPLEMENTATION_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: str(spec["sha256"]) for name, spec in sorted(config["inputs"].items())},
        "producer_identity": producer,
    })
    rows, row_audit = build_temporal_rows(config, project_root=project_root)
    if time.monotonic() - started > float(config["analysis_wall_fuse_seconds"]):
        raise TimeoutError("temporal readiness analysis exceeded its wall fuse before CV")

    identity_static_probabilities, identity_static_fold_diagnostics = _static_oof(rows, safeslot_config)
    reproduction_error, reproduction_folds_equal = _reproduction_error(
        rows, identity_static_probabilities, _read_jsonl(inputs["safeslot_oof_predictions"]),
        identity_static_fold_diagnostics,
    )
    comparison_static_probabilities, comparison_static_fold_diagnostics, _comparison_static_policy_records = _temporal_oof(
        rows, config, feature_key="static_feature_values"
    )
    temporal_probabilities, fold_diagnostics, policy_records = _temporal_oof(
        rows, config, feature_key="feature_values"
    )
    if time.monotonic() - started > float(config["analysis_wall_fuse_seconds"]):
        raise TimeoutError("temporal readiness analysis exceeded its wall fuse")

    history_rows, history_static = _history_probabilities(rows, comparison_static_probabilities)
    _same_history_rows, history_temporal = _history_probabilities(rows, temporal_probabilities)
    static_history_metrics = candidate_classification_metrics(history_rows, history_static)
    temporal_history_metrics = candidate_classification_metrics(history_rows, history_temporal)
    all_temporal_metrics = candidate_classification_metrics(rows, temporal_probabilities)
    history_auc_gain = float(temporal_history_metrics["roc_auc"]) - float(static_history_metrics["roc_auc"])

    fold_comparisons = []
    for fold_index, maps in enumerate(config["model"]["outer_map_folds"]):
        indices = [
            index for index, row in enumerate(rows)
            if str(row["map_id"]) in set(map(str, maps)) and int(row["decision_index"]) > 0
        ]
        fold_rows = [rows[index] for index in indices]
        static_metrics = candidate_classification_metrics(fold_rows, [comparison_static_probabilities[index] for index in indices])
        temporal_metrics = candidate_classification_metrics(fold_rows, [temporal_probabilities[index] for index in indices])
        fold_comparisons.append({
            "outer_fold": fold_index, "validation_maps": list(maps),
            "history_state_count": len({str(row["state_id"]) for row in fold_rows}),
            "history_candidate_count": len(fold_rows),
            "static_metrics": static_metrics, "temporal_metrics": temporal_metrics,
            "roc_auc_gain": float(temporal_metrics["roc_auc"]) - float(static_metrics["roc_auc"]),
        })

    all_policy_metrics = summarize_gate_policy(policy_records)
    history_policy_records = [row for row in policy_records if int(row["decision_index"]) > 0]
    history_policy_metrics = summarize_gate_policy(history_policy_records)
    source_policy_metrics = {
        source: summarize_gate_policy([row for row in history_policy_records if str(row["source_policy"]) == source])
        for source in sorted({str(row["source_policy"]) for row in history_policy_records})
    }
    gates = dict(config["offline_acceptance"])
    policy = dict(config["policy"])
    checks = {
        "static_probability_exact_reproduction": reproduction_error <= float(gates["maximum_static_probability_reproduction_error"]),
        "static_outer_fold_identity": reproduction_folds_equal,
        "all_temporal_outer_calibrations_feasible": all(bool(row["calibration_feasible"]) for row in fold_diagnostics),
        "history_candidate_roc_auc": float(temporal_history_metrics["roc_auc"]) >= float(gates["minimum_history_candidate_roc_auc"]),
        "history_auc_gain_over_static": history_auc_gain >= float(gates["minimum_history_auc_gain_over_static"]),
        "each_fold_history_roc_auc": all(float(row["temporal_metrics"]["roc_auc"]) >= float(gates["minimum_each_fold_history_roc_auc"]) for row in fold_comparisons),
        "each_fold_auc_gain_over_static": all(float(row["roc_auc_gain"]) >= float(gates["minimum_each_fold_auc_gain_over_static"]) for row in fold_comparisons),
        "selected_safe_replace_precision": float(history_policy_metrics["safe_replace_precision"]) >= float(gates["minimum_selected_safe_replace_precision"]),
        "history_replacement_state_fraction": float(history_policy_metrics["replacement_state_fraction"]) >= float(gates["minimum_history_replacement_state_fraction"]),
        "selected_map_count": int(history_policy_metrics["selected_map_count"]) >= int(gates["minimum_selected_map_count"]),
        "mean_current_step_advantage": float(history_policy_metrics["mean_current_step_advantage"]) >= float(policy["minimum_mean_current_step_advantage"]),
        "first_fixed_half_advantage": float(history_policy_metrics["first_fixed_half_advantage"]) >= float(policy["minimum_first_fixed_half_advantage"]),
        "second_fixed_half_advantage": float(history_policy_metrics["second_fixed_half_advantage"]) >= float(policy["minimum_second_fixed_half_advantage"]),
        "mean_no_progress_rate_delta": float(history_policy_metrics["mean_no_progress_rate_delta"]) <= float(policy["maximum_mean_no_progress_rate_delta"]),
        "mean_residual_risk_delta": float(history_policy_metrics["mean_residual_risk_delta"]) <= float(policy["maximum_mean_residual_risk_delta"]),
        "maximum_map_current_step_regret": float(history_policy_metrics["maximum_map_mean_current_step_regret"]) <= float(policy["maximum_map_mean_current_step_regret"]),
        "maximum_map_residual_degradation": float(history_policy_metrics["maximum_map_mean_residual_risk_degradation"]) <= float(policy["maximum_map_mean_residual_risk_degradation"]),
        "source_policy_current_step_nonnegative": set(source_policy_metrics) == {"official_adaptive", "v2-full"} and all(float(value["mean_current_step_advantage"]) >= float(gates["minimum_source_policy_mean_current_step_advantage"]) for value in source_policy_metrics.values()),
        "decision0_forced_abstention": not any(bool(row["selected_challenger"]) for row in policy_records if int(row["decision_index"]) == 0),
        "zero_execution_errors": True,
        "no_solver_native_or_pp_executed": True,
    }
    passed = all(checks.values())

    prediction_rows = [{
        "schema": PREDICTION_SCHEMA,
        "state_id": str(row["state_id"]), "map_id": str(row["map_id"]),
        "decision_index": int(row["decision_index"]),
        "challenger_candidate_id": str(row["challenger_candidate_id"]),
        "label": bool(row["label"]), "sample_weight": float(row["sample_weight"]),
        "identity_static_probability": float(identity_probability),
        "comparison_static_probability": float(static_probability),
        "temporal_probability": float(temporal_probability),
    } for row, identity_probability, static_probability, temporal_probability in zip(
        rows, identity_static_probabilities, comparison_static_probabilities, temporal_probabilities
    )]
    rows_path = output / "training_rows.jsonl"
    predictions_path = output / "oof_candidate_predictions.jsonl"
    policy_path = output / "oof_state_policy.jsonl"
    folds_path = output / "fold_diagnostics.jsonl"
    _write_jsonl(rows_path, rows)
    _write_jsonl(predictions_path, sorted(prediction_rows, key=lambda row: (row["state_id"], row["challenger_candidate_id"])))
    _write_jsonl(policy_path, sorted(policy_records, key=lambda row: str(row["state_id"])))
    _write_jsonl(folds_path, fold_diagnostics)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "offline_incremental_signal_passed_fresh_matched_history_only" if passed else "offline_incremental_signal_failed_hard_stop",
        "implementation_id": IMPLEMENTATION_ID,
        "config_sha256": sha256_file(config_path), "producer": producer,
        "analysis_wall_seconds": time.monotonic() - started,
        "solver_native_or_pp_executed": False,
        "feature_dimension": len(FEATURE_NAMES), "history_feature_names": list(HISTORY_FEATURE_NAMES),
        "row_audit": row_audit,
        "identity_static_baseline": {
            "feature_dimension": len(STATIC_FEATURE_NAMES),
            "probability_reproduction_maximum_absolute_error": reproduction_error,
            "outer_fold_identity": reproduction_folds_equal,
            "fold_diagnostics": identity_static_fold_diagnostics,
        },
        "fair_comparison_static_baseline": {
            "feature_dimension": len(STATIC_FEATURE_NAMES),
            "same_history_calibration_population_as_temporal": True,
            "fold_diagnostics": comparison_static_fold_diagnostics,
            "history_candidate_metrics": static_history_metrics,
        },
        "temporal_model": {
            "feature_dimension": len(FEATURE_NAMES),
            "all_candidate_metrics": all_temporal_metrics,
            "history_candidate_metrics": temporal_history_metrics,
            "history_roc_auc_gain_over_static": history_auc_gain,
        },
        "fold_comparisons": fold_comparisons,
        "all_98_state_policy_metrics": all_policy_metrics,
        "history_68_state_policy_metrics": history_policy_metrics,
        "source_policy_metrics": source_policy_metrics,
        "offline_checks": checks, "offline_passed": passed,
        "model_exported": False, "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "claim_boundary": dict(config["claim_boundary"]),
        "limitations": [
            "source prefixes are Official Adaptive or V2, not matched Component16/Hotspot16/Dual16 actions",
            "the cohort contains no exact unchanged before/after transition, so plateau or repeated-rollback readiness is untested",
            "this is candidate-level offline readiness evidence, not end-to-end runtime or TTF evidence",
        ],
        "next_decision": "collect_fresh_matched_component_hotspot_prefixes_without_exporting_this_model" if passed else "stop_without_window_feature_or_threshold_retuning",
        "artifacts": {
            "training_rows_sha256": sha256_file(rows_path),
            "oof_candidate_predictions_sha256": sha256_file(predictions_path),
            "oof_state_policy_sha256": sha256_file(policy_path),
            "fold_diagnostics_sha256": sha256_file(folds_path),
        },
    }
    _write_json(output / "temporal_anchor_incremental_readiness_report.json", report)
    return report
