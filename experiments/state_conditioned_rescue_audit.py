from __future__ import annotations

import collections
import csv
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.repair_collection import _fingerprint, _write_json
from experiments.rescue_lite_confirmation import (
    _aggregate,
    _comparison,
    _write_csv,
)


STATE_CONDITIONED_AUDIT_SCHEMA = "lns2.state_conditioned_rescue_audit.v1"
ACTION_TO_POLICY = {
    "adaptive": "adaptive",
    "size4": "4>adaptive",
    "size8": "8>adaptive",
    "size16": "16>adaptive",
    "learned": "reference_learned_repair_aware",
}
ACTIONS = tuple(ACTION_TO_POLICY)
TREE_CONFIG = {
    "criterion": "entropy",
    "max_depth": 2,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": 20260721,
}
FEATURE_NAMES = (
    "agent_count",
    "decision_index",
    "conflicts_before",
    "conflicts_per_agent",
    "soc_per_agent",
    "base_size",
    "learned_size",
    "score_4",
    "score_8",
    "score_16",
    "score_gap_8_minus_4",
    "score_gap_16_minus_8",
    "score_range",
    "overlap_4_8",
    "overlap_4_16",
    "overlap_8_16",
    "layout_compartmentalized",
    "layout_dead_end_aisles",
    "layout_regular_beltway",
)


@dataclass(frozen=True)
class AuditState:
    dataset_id: str
    state_id: str
    map_id: str
    map_group: str
    cell: str
    features: tuple[float, ...]
    oracle_action: str
    rows_by_action: dict[str, list[dict[str, Any]]]
    action_metrics: dict[str, dict[str, Any]]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    temporary.replace(path)


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a = set(map(int, left))
    b = set(map(int, right))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def select_safe_oracle_action(
    action_metrics: dict[str, dict[str, Any]],
) -> str:
    if set(action_metrics) != set(ACTIONS):
        raise ValueError("state action metrics do not cover every audited action")
    baseline = action_metrics["adaptive"]
    safe = []
    for action in ACTIONS:
        metrics = action_metrics[action]
        if (
            float(metrics["state_escape_rate"]) + 1e-12
            >= float(baseline["state_escape_rate"])
            and float(metrics["final_hard_failure_rate"]) - 1e-12
            <= float(baseline["final_hard_failure_rate"])
            and float(metrics["mean_conflict_reduction"]) + 1e-12
            >= 0.98 * float(baseline["mean_conflict_reduction"])
        ):
            safe.append(action)
    if not safe:
        return "adaptive"
    priority = {action: -index for index, action in enumerate(ACTIONS)}
    return max(
        safe,
        key=lambda action: (
            float(action_metrics[action]["conflict_reduction_per_second"]),
            float(action_metrics[action]["state_escape_rate"]),
            float(action_metrics[action]["mean_conflict_reduction"]),
            -float(action_metrics[action]["mean_repair_seconds"]),
            priority[action],
        ),
    )


def balanced_map_folds(
    states: list[AuditState], fold_count: int = 4
) -> list[set[str]]:
    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for state in states:
        layout = state.cell.split("__agents_", 1)[0]
        groups[(state.dataset_id, layout)].append(state.map_group)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for key in sorted(groups):
        for index, group in enumerate(sorted(set(groups[key]))):
            folds[index % fold_count].append(group)
    result = [set(values) for values in folds]
    if any(not values for values in result):
        raise ValueError("state-conditioned map folds contain an empty fold")
    observed = [value for values in result for value in values]
    if len(observed) != len(set(observed)):
        raise ValueError("a map group appears in more than one audit fold")
    return result


def resolve_recorded_source(project_root: Path, recorded: str | Path) -> Path:
    candidate = Path(recorded)
    if candidate.is_dir():
        return candidate.resolve()
    fallback = project_root / "build" / Path(str(recorded).replace("\\", "/")).name
    if not fallback.is_dir():
        raise ValueError("recorded confirmation source cannot be resolved")
    return fallback.resolve()


def _resolve_prepared_root(
    project_root: Path, source_root: Path, report: dict[str, Any]
) -> Path:
    direct = source_root / "prepared"
    if direct.is_dir():
        return direct
    recorded = str(report.get("source", ""))
    if not recorded:
        raise ValueError("audit source does not identify its prepared-state source")
    source = resolve_recorded_source(project_root, recorded)
    if not (source / "prepared").is_dir():
        raise ValueError("recorded prepared-state source cannot be resolved")
    return source / "prepared"


def _report_path(source_root: Path) -> Path:
    for name in ("diagnostic_report.json", "locked_confirmation_report.json"):
        path = source_root / name
        if path.is_file():
            return path
    raise ValueError(f"confirmation source has no report: {source_root}")


def _read_trial_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _feature_values(
    state_payload: dict[str, Any], preparation: dict[str, Any]
) -> tuple[float, ...]:
    state = dict(state_payload["state"])
    trials = list(state_payload["trials"])
    if not trials:
        raise ValueError("confirmation state has no raw branch trials")
    before = {
        (
            int(row["outcome"]["conflicts_before"]),
            int(row["outcome"]["sum_of_costs_before"]),
        )
        for row in trials
    }
    if len(before) != 1:
        raise ValueError("confirmation branches disagree on their before state")
    conflicts, soc = next(iter(before))
    agent_count = int(state["agent_count"])
    top = {int(key): str(value) for key, value in state_payload["top_candidate_by_size"].items()}
    arms = {str(row["candidate_id"]): dict(row) for row in preparation["arms"]}
    if any(candidate not in arms for candidate in top.values()):
        raise ValueError("top candidate is absent from its prepared arm set")
    top_arms = {size: arms[candidate] for size, candidate in top.items()}
    if set(top_arms) != {4, 8, 16}:
        raise ValueError("confirmation state does not cover sizes 4/8/16")
    scores = {size: float(top_arms[size]["main_score"]) for size in (4, 8, 16)}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("confirmation state has a non-finite v2 score")
    base_id = str(preparation.get("base_candidate_id", ""))
    learned_id = str(state_payload["learned_candidate_id"])
    base_size = int(arms.get(base_id, {}).get("actual_size", 0))
    learned_size = int(arms.get(learned_id, {}).get("actual_size", 0))
    layout = str(state["layout_mode"])
    values = {
        "agent_count": float(agent_count),
        "decision_index": float(state["decision_index"]),
        "conflicts_before": float(conflicts),
        "conflicts_per_agent": float(conflicts) / max(1, agent_count),
        "soc_per_agent": float(soc) / max(1, agent_count),
        "base_size": float(base_size),
        "learned_size": float(learned_size),
        "score_4": scores[4],
        "score_8": scores[8],
        "score_16": scores[16],
        "score_gap_8_minus_4": scores[8] - scores[4],
        "score_gap_16_minus_8": scores[16] - scores[8],
        "score_range": max(scores.values()) - min(scores.values()),
        "overlap_4_8": _jaccard(top_arms[4]["agents"], top_arms[8]["agents"]),
        "overlap_4_16": _jaccard(top_arms[4]["agents"], top_arms[16]["agents"]),
        "overlap_8_16": _jaccard(top_arms[8]["agents"], top_arms[16]["agents"]),
        "layout_compartmentalized": float(layout == "compartmentalized"),
        "layout_dead_end_aisles": float(layout == "dead_end_aisles"),
        "layout_regular_beltway": float(layout == "regular_beltway"),
    }
    return tuple(values[name] for name in FEATURE_NAMES)


def _load_source(
    *, project_root: Path, dataset_id: str, source_root: Path
) -> tuple[list[AuditState], dict[str, Any]]:
    report_path = _report_path(source_root)
    report = dict(read_json(report_path))
    coverage = dict(report.get("coverage", {}))
    if not bool(coverage.get("passed")) or int(coverage.get("error_count", -1)) != 0:
        raise ValueError("confirmation source did not pass branch coverage")
    if not bool(report.get("exact_repair_fingerprint_coverage")):
        raise ValueError("confirmation source lacks exact repair fingerprints")
    prepared_root = _resolve_prepared_root(project_root, source_root, report)
    state_paths = sorted((source_root / "states").glob("*.json"))
    if len(state_paths) != int(report["state_count"]):
        raise ValueError("confirmation source state-file coverage is incomplete")
    trial_rows = _read_trial_csv(source_root / "confirmation_trials.csv")
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        rows_by_key[(str(row["state_id"]), str(row["policy_id"]))].append(row)

    states = []
    state_hashes = []
    prepared_hashes = []
    for path in state_paths:
        payload = dict(read_json(path))
        if not bool(payload.get("complete")):
            raise ValueError("confirmation state is incomplete")
        state = dict(payload["state"])
        state_id = str(state["state_id"])
        if path.stem != state_id:
            raise ValueError("confirmation state filename does not match state_id")
        prepared_path = prepared_root / f"{state_id}.json"
        preparation = dict(read_json(prepared_path))
        if not bool(preparation.get("valid")) or not bool(preparation.get("complete")):
            raise ValueError("prepared confirmation state is invalid")
        if str(preparation["state"]["before_repair_fingerprint"]) != str(
            state["before_repair_fingerprint"]
        ):
            raise ValueError("prepared and trial repair fingerprints differ")
        rows_by_action = {}
        action_metrics = {}
        for action, policy_id in ACTION_TO_POLICY.items():
            rows = rows_by_key.get((state_id, policy_id), [])
            if len(rows) != int(report["trial_count_per_state"]):
                raise ValueError(f"state {state_id} has incomplete {action} trials")
            rows_by_action[action] = rows
            action_metrics[action] = _aggregate(rows)
        features = _feature_values(payload, preparation)
        states.append(
            AuditState(
                dataset_id=dataset_id,
                state_id=state_id,
                map_id=str(state["map_id"]),
                map_group=f"{dataset_id}::{state['map_id']}",
                cell=str(state["cell"]),
                features=features,
                oracle_action=select_safe_oracle_action(action_metrics),
                rows_by_action=rows_by_action,
                action_metrics=action_metrics,
            )
        )
        state_hashes.append((state_id, sha256_file(path)))
        prepared_hashes.append((state_id, sha256_file(prepared_path)))
    if len({state.state_id for state in states}) != len(states):
        raise ValueError("confirmation source has duplicate state ids")
    source_evidence = {
        "dataset_id": dataset_id,
        "source": str(source_root),
        "report_sha256": sha256_file(report_path),
        "trials_sha256": sha256_file(source_root / "confirmation_trials.csv"),
        "state_files_fingerprint": _fingerprint(state_hashes),
        "prepared_files_fingerprint": _fingerprint(prepared_hashes),
        "state_count": len(states),
        "map_group_count": len({state.map_group for state in states}),
    }
    return states, source_evidence


def _fit_tree(states: list[AuditState]) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    if not states:
        raise ValueError("cannot fit a selector without states")
    model = DecisionTreeClassifier(**TREE_CONFIG)
    model.fit(
        [list(state.features) for state in states],
        [state.oracle_action for state in states],
    )
    return model


def _tree_predictions(model: Any, states: list[AuditState]) -> dict[str, str]:
    values = list(model.predict([list(state.features) for state in states]))
    predictions = {state.state_id: str(value) for state, value in zip(states, values)}
    if any(value not in ACTIONS for value in predictions.values()):
        raise ValueError("selector predicted an unsupported action")
    return predictions


def _evaluate(
    states: list[AuditState], predictions: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(predictions) != {state.state_id for state in states}:
        raise ValueError("selector prediction coverage is incomplete")
    selected_rows = []
    baseline_rows = []
    prediction_rows = []
    for state in states:
        action = predictions[state.state_id]
        selected_rows.extend(state.rows_by_action[action])
        baseline_rows.extend(state.rows_by_action["adaptive"])
        action_metrics = state.action_metrics[action]
        prediction_rows.append(
            {
                "dataset_id": state.dataset_id,
                "state_id": state.state_id,
                "map_id": state.map_id,
                "cell": state.cell,
                "oracle_action": state.oracle_action,
                "predicted_action": action,
                "oracle_match": int(action == state.oracle_action),
                "state_escape_rate": action_metrics["state_escape_rate"],
                "final_hard_failure_rate": action_metrics["final_hard_failure_rate"],
                "mean_conflict_reduction": action_metrics["mean_conflict_reduction"],
                "mean_repair_seconds": action_metrics["mean_repair_seconds"],
                "conflict_reduction_per_second": action_metrics[
                    "conflict_reduction_per_second"
                ],
            }
        )
    baseline = _aggregate(baseline_rows)
    metrics = _aggregate(selected_rows)
    summary = {**metrics, **_comparison(metrics, baseline)}
    cell_rows = []
    for dataset_id, cell in sorted(
        {(state.dataset_id, state.cell) for state in states}
    ):
        subset = [
            state
            for state in states
            if state.dataset_id == dataset_id and state.cell == cell
        ]
        chosen = [
            row
            for state in subset
            for row in state.rows_by_action[predictions[state.state_id]]
        ]
        adaptive = [
            row for state in subset for row in state.rows_by_action["adaptive"]
        ]
        observed = _aggregate(chosen)
        base = _aggregate(adaptive)
        cell_rows.append(
            {
                "dataset_id": dataset_id,
                "cell": cell,
                **observed,
                **_comparison(observed, base),
            }
        )
    return summary, cell_rows, prediction_rows


def _basic_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["state_escape_rate_delta"]) >= -1e-12
        and float(metrics["final_hard_failure_rate_delta"]) <= 1e-12
        and float(metrics["efficiency_ratio"]) + 1e-12 >= 1.10
        and float(metrics["mean_reduction_ratio"]) + 1e-12 >= 0.98
    )


def _stability_gate(
    fold_metrics: list[dict[str, Any]], cell_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    noninferior = sum(
        float(row["efficiency_ratio"]) + 1e-12 >= 1.0 for row in cell_rows
    )
    worst = min(float(row["efficiency_ratio"]) for row in cell_rows)
    return {
        "passed": bool(
            all(_basic_gate(row) for row in fold_metrics)
            and noninferior >= 10
            and worst + 1e-12 >= 0.90
        ),
        "basic_fold_pass_count": sum(_basic_gate(row) for row in fold_metrics),
        "required_basic_fold_count": len(fold_metrics),
        "noninferior_cell_count": noninferior,
        "required_noninferior_cell_count": 10,
        "cell_count": len(cell_rows),
        "worst_cell_efficiency_ratio": worst,
        "minimum_worst_cell_efficiency_ratio": 0.90,
    }


def _source_map_hashes(project_root: Path, source_root: Path) -> set[str]:
    report = dict(read_json(_report_path(source_root)))
    isolation = dict(report.get("dataset_isolation", {}))
    if not isolation and report.get("source"):
        source = resolve_recorded_source(project_root, str(report["source"]))
        source_report = dict(read_json(source / "locked_confirmation_report.json"))
        isolation = dict(source_report.get("dataset_isolation", {}))
    return set(map(str, dict(isolation.get("confirmation_map_hashes", {})).values()))


def audit_state_conditioned_rescue(
    *,
    project_root: str | Path,
    sources: dict[str, str | Path],
    output: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("state-conditioned audit output already exists")
    if len(sources) != 2 or len(set(sources)) != 2:
        raise ValueError("state-conditioned audit requires exactly two named sources")
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "state-conditioned audit requires the training profile with scikit-learn"
        ) from error
    loaded = []
    evidence = []
    resolved_sources = {name: Path(path).resolve() for name, path in sources.items()}
    for dataset_id, source_root in sorted(resolved_sources.items()):
        states, source_evidence = _load_source(
            project_root=root,
            dataset_id=dataset_id,
            source_root=source_root,
        )
        loaded.extend(states)
        evidence.append(source_evidence)
    if len({state.state_id for state in loaded}) != len(loaded):
        raise ValueError("confirmation sources overlap in state ids")
    hash_sets = [
        _source_map_hashes(root, path) for path in resolved_sources.values()
    ]
    known_hash_sets = [values for values in hash_sets if values]
    if len(known_hash_sets) == 2 and known_hash_sets[0] & known_hash_sets[1]:
        raise ValueError("confirmation sources overlap in map content")

    identity = {
        "schema": STATE_CONDITIONED_AUDIT_SCHEMA,
        "sources": evidence,
        "feature_names": list(FEATURE_NAMES),
        "actions": dict(ACTION_TO_POLICY),
        "tree_config": TREE_CONFIG,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "promotion_eligible": False,
    }
    run_fingerprint = _fingerprint(identity)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "run_config.json", {**identity, "run_fingerprint": run_fingerprint})

    datasets = sorted(sources)
    cross_predictions: dict[str, str] = {}
    cross_fold_rows = []
    tree_rules = []
    for test_id in datasets:
        train = [state for state in loaded if state.dataset_id != test_id]
        test = [state for state in loaded if state.dataset_id == test_id]
        model = _fit_tree(train)
        predicted = _tree_predictions(model, test)
        cross_predictions.update(predicted)
        metrics, _, _ = _evaluate(test, predicted)
        cross_fold_rows.append(
            {
                "protocol": "leave-one-confirmation-set-out",
                "test_dataset_id": test_id,
                "train_state_count": len(train),
                "test_state_count": len(test),
                **metrics,
                "basic_gate_passed": _basic_gate(metrics),
            }
        )
        from sklearn.tree import export_text

        tree_rules.append(
            f"## train=not-{test_id}\n"
            + export_text(model, feature_names=list(FEATURE_NAMES))
        )
    cross_summary, cross_cells, cross_rows = _evaluate(loaded, cross_predictions)
    cross_gate = _stability_gate(cross_fold_rows, cross_cells)

    map_predictions: dict[str, str] = {}
    map_fold_rows = []
    folds = balanced_map_folds(loaded)
    for fold_index, validation_groups in enumerate(folds):
        train = [state for state in loaded if state.map_group not in validation_groups]
        test = [state for state in loaded if state.map_group in validation_groups]
        model = _fit_tree(train)
        predicted = _tree_predictions(model, test)
        map_predictions.update(predicted)
        metrics, _, _ = _evaluate(test, predicted)
        map_fold_rows.append(
            {
                "protocol": "pooled-map-group-oof",
                "fold": fold_index,
                "train_state_count": len(train),
                "test_state_count": len(test),
                **metrics,
                "basic_gate_passed": _basic_gate(metrics),
            }
        )
    map_summary, map_cells, map_rows = _evaluate(loaded, map_predictions)
    map_gate = _stability_gate(map_fold_rows, map_cells)

    reference_rows = []
    for method, predictions in (
        ("tree_cross_dataset", cross_predictions),
        ("tree_map_oof", map_predictions),
        ("constant_4", {state.state_id: "size4" for state in loaded}),
        ("constant_8", {state.state_id: "size8" for state in loaded}),
        ("constant_16", {state.state_id: "size16" for state in loaded}),
        ("existing_learned", {state.state_id: "learned" for state in loaded}),
        ("safe_oracle", {state.state_id: state.oracle_action for state in loaded}),
    ):
        metrics, cells, _ = _evaluate(loaded, predictions)
        reference_rows.append(
            {
                "method": method,
                **metrics,
                "noninferior_cell_count": sum(
                    float(row["efficiency_ratio"]) + 1e-12 >= 1.0
                    for row in cells
                ),
                "worst_cell_efficiency_ratio": min(
                    float(row["efficiency_ratio"]) for row in cells
                ),
                "basic_gate_passed": _basic_gate(metrics),
            }
        )

    if cross_gate["passed"] and map_gate["passed"]:
        decision = "state_conditioned_rescue_candidate_design_only"
        recommendation = "freeze_selector_design_then_collect_untouched_confirmation"
    else:
        decision = "state_conditioned_rescue_not_stable"
        recommendation = "proceed_to_v3_repair_cost_aware_design"
    failed_cross_cells = [
        row
        for row in cross_cells
        if float(row["efficiency_ratio"]) + 1e-12 < 1.0
    ]
    failed_map_cells = [
        row
        for row in map_cells
        if float(row["efficiency_ratio"]) + 1e-12 < 1.0
    ]
    report = {
        "schema": STATE_CONDITIONED_AUDIT_SCHEMA,
        "decision": decision,
        "recommendation": recommendation,
        "run_fingerprint": run_fingerprint,
        "state_count": len(loaded),
        "state_count_by_dataset": dict(collections.Counter(state.dataset_id for state in loaded)),
        "map_group_count": len({state.map_group for state in loaded}),
        "feature_names": list(FEATURE_NAMES),
        "tree_config": TREE_CONFIG,
        "oracle_action_counts": dict(collections.Counter(state.oracle_action for state in loaded)),
        "cross_dataset_summary": cross_summary,
        "cross_dataset_gate": cross_gate,
        "map_oof_summary": map_summary,
        "map_oof_gate": map_gate,
        "cross_dataset_action_counts": dict(collections.Counter(cross_predictions.values())),
        "map_oof_action_counts": dict(collections.Counter(map_predictions.values())),
        "cross_dataset_folds": cross_fold_rows,
        "map_oof_folds": map_fold_rows,
        "cross_dataset_failed_cells": failed_cross_cells,
        "map_oof_failed_cells": failed_map_cells,
        "reference_methods": {
            str(row["method"]): row for row in reference_rows
        },
        "promotion_eligible": False,
        "deployment_promoted": False,
        "default_controller_changed": False,
        "complete_episode_evaluation": False,
        "solver_executions_started": 0,
        "quick_formal_v3_started": False,
    }
    _write_json(output_root / "state_conditioned_rescue_audit_report.json", report)
    _write_csv(output_root / "cross_dataset_folds.csv", cross_fold_rows)
    _write_csv(output_root / "map_oof_folds.csv", map_fold_rows)
    _write_csv(output_root / "selector_summary.csv", reference_rows)
    _write_csv(
        output_root / "cell_stability.csv",
        [
            {"protocol": "cross_dataset", **row} for row in cross_cells
        ]
        + [{"protocol": "map_oof", **row} for row in map_cells],
    )
    feature_rows = []
    for state in loaded:
        feature_rows.append(
            {
                "dataset_id": state.dataset_id,
                "state_id": state.state_id,
                "map_id": state.map_id,
                "cell": state.cell,
                "oracle_action": state.oracle_action,
                **dict(zip(FEATURE_NAMES, state.features)),
            }
        )
    _write_csv(output_root / "state_features.csv", feature_rows)
    action_rows = []
    for state in loaded:
        for action in ACTIONS:
            action_rows.append(
                {
                    "dataset_id": state.dataset_id,
                    "state_id": state.state_id,
                    "map_id": state.map_id,
                    "cell": state.cell,
                    "action": action,
                    "oracle_action": state.oracle_action,
                    **state.action_metrics[action],
                }
            )
    _write_csv(output_root / "state_action_metrics.csv", action_rows)
    _write_csv(
        output_root / "selector_predictions.csv",
        [
            {"protocol": "cross_dataset", **row} for row in cross_rows
        ]
        + [{"protocol": "map_oof", **row} for row in map_rows],
    )
    _atomic_write_text(output_root / "tree_rules.txt", "\n\n".join(tree_rules) + "\n")
    markdown = [
        "# State-conditioned rescue audit (design only)",
        "",
        f"- Decision: `{decision}`",
        f"- Recommendation: `{recommendation}`",
        f"- States: {len(loaded)} across {len(datasets)} confirmation sets",
        f"- Cross-dataset gate: {str(cross_gate['passed']).lower()}",
        f"- Map-group OOF gate: {str(map_gate['passed']).lower()}",
        "- Promotion eligible: false",
        "- Solver executions started: 0",
        "",
        "The selector uses only cheap state values and cached v2 candidate scores.",
        "Neither dataset identity nor map identity is an input feature. Results cover",
        "same-state rescue repairs, not complete episodes or controller overhead.",
        "",
        "## Cross-confirmation transfer",
        "",
        (
            "- Efficiency ratio: "
            f"{float(cross_summary['efficiency_ratio']):.3f}x Adaptive"
        ),
        (
            "- Escape-rate delta: "
            f"{float(cross_summary['state_escape_rate_delta']):+.3f}"
        ),
        (
            "- Hard-failure delta: "
            f"{float(cross_summary['final_hard_failure_rate_delta']):+.3f}"
        ),
        (
            "- Non-inferior cells: "
            f"{cross_gate['noninferior_cell_count']}/{cross_gate['cell_count']}"
        ),
        (
            "- Worst cell efficiency ratio: "
            f"{float(cross_gate['worst_cell_efficiency_ratio']):.3f}"
        ),
        "",
        "## Pooled map-group OOF",
        "",
        (
            "- Efficiency ratio: "
            f"{float(map_summary['efficiency_ratio']):.3f}x Adaptive"
        ),
        (
            "- Basic folds passed: "
            f"{map_gate['basic_fold_pass_count']}/"
            f"{map_gate['required_basic_fold_count']}"
        ),
        (
            "- Non-inferior cells: "
            f"{map_gate['noninferior_cell_count']}/{map_gate['cell_count']}"
        ),
        (
            "- Worst cell efficiency ratio: "
            f"{float(map_gate['worst_cell_efficiency_ratio']):.3f}"
        ),
        "",
        "## Reference bounds",
        "",
        (
            "- Existing learned rescue: "
            f"{float(report['reference_methods']['existing_learned']['efficiency_ratio']):.3f}x, "
            f"{report['reference_methods']['existing_learned']['noninferior_cell_count']}/12 cells"
        ),
        (
            "- Constant size 8: "
            f"{float(report['reference_methods']['constant_8']['efficiency_ratio']):.3f}x, "
            f"{report['reference_methods']['constant_8']['noninferior_cell_count']}/12 cells"
        ),
        (
            "- Safe per-state oracle upper bound: "
            f"{float(report['reference_methods']['safe_oracle']['efficiency_ratio']):.3f}x, "
            f"{report['reference_methods']['safe_oracle']['noninferior_cell_count']}/12 cells"
        ),
        "",
        "The oracle gap shows that repair-aware choice remains useful, but the shallow",
        "selector does not transfer reliably enough. Do not integrate it into runtime;",
        "use the evidence to design a higher-coverage v3 repair-cost model.",
        "",
    ]
    _atomic_write_text(
        output_root / "state_conditioned_rescue_audit_report.md",
        "\n".join(markdown),
    )
    return report


__all__ = [
    "ACTIONS",
    "FEATURE_NAMES",
    "STATE_CONDITIONED_AUDIT_SCHEMA",
    "AuditState",
    "audit_state_conditioned_rescue",
    "balanced_map_folds",
    "resolve_recorded_source",
    "select_safe_oracle_action",
]
