from __future__ import annotations

import collections
import itertools
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.closed_loop_confirmation import score_online_candidates
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import (
    PROFILE_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    canonicalize_features,
)
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_aware_training import _balanced_map_folds
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


V2_FACTORIAL_AUDIT_SCHEMA = "lns2.v2_factorial_audit.v1"
MODEL_PARAMETERS = {
    "early_stopping": False,
    "l2_regularization": 0.1,
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "random_state": 20260714,
}


def _outcome_name(outcome: dict[str, Any]) -> str:
    registered = outcome.get("repair_outcome")
    if registered is not None:
        return str(registered)
    before = int(outcome["conflicts_before"])
    after = int(outcome["conflicts_after"])
    changed = bool(outcome.get("repair_state_changed", after != before))
    return classify_repair_outcome(
        before_fingerprint="before",
        after_fingerprint="after" if changed else "before",
        replan_success=not bool(outcome.get("hard_failure")),
        conflicts_before=before,
        conflicts_after=after,
        feasible=after == 0,
    )


def effectiveness_dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return the frozen v2 effectiveness-only Pareto relation.

    The original ranker maximized solved rate and minimized remaining conflicts.
    Runtime and low-level search counters were deliberately not label inputs.
    """

    no_worse = bool(
        float(left["solved_rate"]) + 1e-12 >= float(right["solved_rate"])
        and float(left["conflicts_after"])
        <= float(right["conflicts_after"]) + 1e-12
    )
    strictly_better = bool(
        float(left["solved_rate"]) > float(right["solved_rate"]) + 1e-12
        or float(left["conflicts_after"]) + 1e-12
        < float(right["conflicts_after"])
    )
    return no_worse and strictly_better


def pair_vector(
    left: dict[str, float],
    right: dict[str, float],
    input_specs: tuple[tuple[str, str], ...],
) -> list[float]:
    values = []
    for mode, name in input_specs:
        first = float(left.get(name, 0.0))
        second = float(right.get(name, 0.0))
        if mode == "delta":
            values.append(first - second)
        elif mode == "shared":
            values.append((first + second) / 2.0)
        else:
            raise ValueError(f"unsupported pairwise input mode: {mode}")
    return values


def equal_state_weights(rows: list[dict[str, Any]]) -> list[float]:
    counts = collections.Counter(str(row["state_id"]) for row in rows)
    if not rows or any(count <= 0 for count in counts.values()):
        raise ValueError("pairwise training rows are empty")
    raw = [1.0 / counts[str(row["state_id"])] for row in rows]
    scale = len(rows) / math.fsum(raw)
    return [value * scale for value in raw]


def _aggregate_high_load(
    feature_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    model_features = [
        row for row in feature_rows if str(row.get("route")) == "model"
    ]
    expected = {
        (str(row["split"]), str(row["state_id"]), str(row["candidate_id"]))
        for row in model_features
    }
    if not expected:
        raise ValueError("high-load collection has no model candidates")
    trials: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    seeds: dict[tuple[str, str, int], set[int]] = collections.defaultdict(set)
    for row in trial_rows:
        if str(row.get("status")) not in {"ok", "resumed"} or not bool(
            row.get("complete")
        ):
            raise ValueError("high-load v2 audit found an incomplete trial")
        key = (str(row["split"]), str(row["state_id"]), str(row["candidate_id"]))
        # Adaptive is present as a V3 reference arm, but it was never part of
        # the frozen v2 candidate pool and must not enter same-semantics labels.
        if key not in expected:
            continue
        trials[key].append(row)
        seeds[(key[0], key[1], int(row["trial_index"]))].add(
            int(row["random_seed"])
        )
    if any(len(values) != 1 for values in seeds.values()):
        raise ValueError("high-load candidates do not share paired PP seeds")

    result = []
    for feature in model_features:
        split = str(feature["split"])
        state_id = str(feature["state_id"])
        candidate_id = str(feature["candidate_id"])
        key = (split, state_id, candidate_id)
        values = trials.get(key, [])
        indices = [int(row["trial_index"]) for row in values]
        if len(indices) != len(set(indices)) or not {0, 1}.issubset(indices):
            raise ValueError(f"invalid paired trial coverage for {key}")
        outcomes = [dict(row["outcome"]) for row in values]
        before_values = {int(row["conflicts_before"]) for row in outcomes}
        if len(before_values) != 1:
            raise ValueError(f"candidate trials disagree on initial conflicts: {key}")
        before = before_values.pop()
        names = [_outcome_name(row) for row in outcomes]
        realized = canonicalize_features(
            dict(feature["features"]["realized_dynamic"]), "realized_dynamic"
        )
        result.append(
            {
                "source": "high_load",
                "split": split,
                "state_id": state_id,
                "candidate_id": candidate_id,
                "candidate_key": str(feature.get("candidate_key", candidate_id)),
                "map_id": str(feature["map_id"]),
                "layout_mode": str(feature.get("layout_mode", "unknown")),
                "agent_count": int(feature.get("agent_count", 0)),
                "features": realized,
                "frozen_selected": bool(feature.get("base_selected")),
                "frozen_score": float(feature.get("main_score", -1e30)),
                "outcome": {
                    "solved_rate": statistics.fmean(
                        float(bool(row.get("feasible"))) for row in outcomes
                    ),
                    "conflicts_after": statistics.fmean(
                        float(row["conflicts_after"]) for row in outcomes
                    ),
                    "conflicts_before": float(before),
                    "effective_rate": statistics.fmean(
                        float(name in {"conflict_reduced", "feasible"})
                        for name in names
                    ),
                    "no_progress_rate": statistics.fmean(
                        float(name in {"hard_failure", "accepted_noop"})
                        for name in names
                    ),
                    "conflict_reduction": statistics.fmean(
                        max(0.0, before - float(row["conflicts_after"]))
                        for row in outcomes
                    ),
                    "repair_seconds": statistics.fmean(
                        max(1e-9, float(row["repair_seconds"])) for row in outcomes
                    ),
                },
            }
        )
    if set(trials) != expected:
        raise ValueError("high-load feature/trial candidate coverage differs")
    return result


def _legacy_candidates(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if str(row.get("split")) != split:
            continue
        raw_features = dict(row["features"]["realized_dynamic"])
        features = canonicalize_features(raw_features, "realized_dynamic")
        outcome = dict(row["outcome"])
        before = float(features["state.colliding_pairs"])
        after = float(outcome["conflicts_after"])
        solved_rate = float(outcome["solved_rate"])
        result.append(
            {
                "source": "legacy",
                "split": split,
                "state_id": str(row["state_id"]),
                "candidate_id": str(row["candidate_id"]),
                "candidate_key": str(row.get("candidate_key", row["candidate_id"])),
                "map_id": str(row["map_id"]),
                "layout_mode": str(row.get("layout_mode", "unknown")),
                "agent_count": int(row.get("agent_count", 0)),
                "features": features,
                "legacy_features": raw_features,
                "outcome": {
                    "solved_rate": solved_rate,
                    "conflicts_after": after,
                    "conflicts_before": before,
                    "effective_rate": float(solved_rate > 0.0 or after < before),
                    # Historical aggregates cannot distinguish a changed state
                    # with equal conflict count, so this is explicitly a proxy.
                    "no_progress_rate": float(solved_rate == 0.0 and after >= before),
                    "conflict_reduction": max(0.0, before - after),
                    "repair_seconds": max(1e-9, float(outcome["runtime"])),
                },
            }
        )
    if not result:
        raise ValueError(f"legacy v2 split is empty: {split}")
    return result


def _group_states(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    states = []
    for state_id, candidates in sorted(grouped.items()):
        candidate_ids = [str(row["candidate_id"]) for row in candidates]
        if len(candidate_ids) != len(set(candidate_ids)) or len(candidates) < 2:
            raise ValueError(f"invalid candidate pool for state {state_id}")
        states.append(candidates)
    return states


def build_pair_rows(
    candidates: list[dict[str, Any]],
    input_specs: tuple[tuple[str, str], ...],
    *,
    legacy_features: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    diagnostics = collections.Counter()
    for state in _group_states(candidates):
        state_pairs = 0
        for first, second in itertools.combinations(state, 2):
            left = first["outcome"]
            right = second["outcome"]
            first_wins = effectiveness_dominates(left, right)
            second_wins = effectiveness_dominates(right, left)
            if first_wins == second_wins:
                diagnostics["indistinguishable_or_non_dominating"] += 1
                continue
            winner, loser = (first, second) if first_wins else (second, first)
            winner_values = winner["legacy_features"] if legacy_features else winner["features"]
            loser_values = loser["legacy_features"] if legacy_features else loser["features"]
            common = {
                "state_id": str(winner["state_id"]),
                "map_id": str(winner["map_id"]),
            }
            rows.append(
                {
                    **common,
                    "label": 1,
                    "features": pair_vector(winner_values, loser_values, input_specs),
                }
            )
            rows.append(
                {
                    **common,
                    "label": 0,
                    "features": pair_vector(loser_values, winner_values, input_specs),
                }
            )
            state_pairs += 1
        diagnostics["dominance_pairs"] += state_pairs
    diagnostics["training_examples"] = len(rows)
    diagnostics["state_count"] = len(_group_states(candidates))
    return rows, dict(diagnostics)


def _fit_pairwise(rows: list[dict[str, Any]]) -> Any:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    if {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("pairwise training requires both classes")
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(
        np.asarray([row["features"] for row in rows], dtype=float),
        np.asarray([row["label"] for row in rows], dtype=int),
        sample_weight=np.asarray(equal_state_weights(rows), dtype=float),
    )
    return model


def _positive_probability(model: Any, values: list[list[float]]) -> list[float]:
    import numpy as np

    return list(
        map(float, model.predict_proba(np.asarray(values, dtype=float))[:, 1])
    )


def _select_model(
    state: list[dict[str, Any]],
    model: Any,
    input_specs: tuple[tuple[str, str], ...],
    *,
    legacy_features: bool = False,
) -> dict[str, Any]:
    scores = [0.0] * len(state)
    pairs = list(itertools.combinations(range(len(state)), 2))
    forward = []
    reverse = []
    for left, right in pairs:
        first = state[left]["legacy_features"] if legacy_features else state[left]["features"]
        second = state[right]["legacy_features"] if legacy_features else state[right]["features"]
        forward.append(pair_vector(first, second, input_specs))
        reverse.append(pair_vector(second, first, input_specs))
    probabilities = [
        (first + (1.0 - second)) / 2.0
        for first, second in zip(
            _positive_probability(model, forward),
            _positive_probability(model, reverse),
        )
    ]
    for probability, (left, right) in zip(probabilities, pairs):
        scores[left] += probability
        scores[right] += 1.0 - probability
    stable = [round(value, 12) for value in scores]
    index = min(
        range(len(state)),
        key=lambda value: (-stable[value], str(state[value]["candidate_key"])),
    )
    return state[index]


def _frozen_selection(
    states: list[list[dict[str, Any]]], bundle: Any
) -> list[dict[str, Any]]:
    model = bundle.main_models["realized_dynamic"]
    selected = []
    for state in states:
        registered = [row for row in state if bool(row.get("frozen_selected"))]
        if len(registered) == 1:
            selected.append(registered[0])
            continue
        online_rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        index, _, _ = score_online_candidates(online_rows, model)
        selected.append(state[index])
    return selected


def selection_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty selection")
    outcome = [dict(row["outcome"]) for row in rows]
    reduction = statistics.fmean(float(row["conflict_reduction"]) for row in outcome)
    seconds = statistics.fmean(float(row["repair_seconds"]) for row in outcome)
    return {
        "state_count": float(len(rows)),
        "effective_rate": statistics.fmean(float(row["effective_rate"]) for row in outcome),
        "no_progress_rate": statistics.fmean(float(row["no_progress_rate"]) for row in outcome),
        "mean_conflict_reduction": reduction,
        "mean_repair_seconds": seconds,
        "conflict_reduction_per_repair_second": reduction / max(1e-9, seconds),
    }


def compare_metrics(selected: dict[str, float], frozen: dict[str, float]) -> dict[str, float]:
    return {
        "effective_rate_delta": selected["effective_rate"] - frozen["effective_rate"],
        "no_progress_rate_delta": selected["no_progress_rate"] - frozen["no_progress_rate"],
        "conflict_reduction_ratio": selected["mean_conflict_reduction"]
        / max(1e-9, frozen["mean_conflict_reduction"]),
        "repair_seconds_ratio": selected["mean_repair_seconds"]
        / max(1e-9, frozen["mean_repair_seconds"]),
        "efficiency_ratio": selected["conflict_reduction_per_repair_second"]
        / max(1e-9, frozen["conflict_reduction_per_repair_second"]),
    }


def quality_checks(comparison: dict[str, float]) -> dict[str, bool]:
    return {
        "effective_rate": comparison["effective_rate_delta"] + 1e-12 >= -0.01,
        "no_progress_rate": comparison["no_progress_rate_delta"] <= 0.01 + 1e-12,
        "conflict_reduction": comparison["conflict_reduction_ratio"] + 1e-12 >= 0.98,
    }


def _cell_gate(evaluation: dict[str, Any], required: int = 5) -> bool:
    cell_count = int(evaluation["cell_count"])
    return int(evaluation["quality_noninferior_cell_count"]) >= min(
        required, cell_count
    )


def _evaluation(
    states: list[list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(states) != len(selected) or len(states) != len(frozen):
        raise ValueError("selection evaluation coverage differs")
    chosen_metrics = selection_metrics(selected)
    frozen_metrics = selection_metrics(frozen)
    comparison = compare_metrics(chosen_metrics, frozen_metrics)
    by_cell: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for index, state in enumerate(states):
        by_cell[(str(state[0]["layout_mode"]), int(state[0]["agent_count"]))].append(index)
    cells = []
    for (layout, agents), indices in sorted(by_cell.items()):
        cell_selected = selection_metrics([selected[index] for index in indices])
        cell_frozen = selection_metrics([frozen[index] for index in indices])
        cell_comparison = compare_metrics(cell_selected, cell_frozen)
        cells.append(
            {
                "layout_mode": layout,
                "agent_count": agents,
                "state_count": len(indices),
                **cell_comparison,
                "quality_noninferior": all(quality_checks(cell_comparison).values()),
            }
        )
    changed = sum(
        str(left["candidate_id"]) != str(right["candidate_id"])
        for left, right in zip(selected, frozen)
    )
    return {
        "selected": chosen_metrics,
        "frozen_v2": frozen_metrics,
        "comparison": comparison,
        "checks": quality_checks(comparison),
        "quality_passed": all(quality_checks(comparison).values()),
        "selection_change_count": changed,
        "selection_change_fraction": changed / len(states),
        "cell_count": len(cells),
        "quality_noninferior_cell_count": sum(bool(row["quality_noninferior"]) for row in cells),
        "cells": cells,
    }


def _full_v2_specs() -> tuple[tuple[str, str], ...]:
    names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    return tuple(("delta", name) for name in names) + tuple(
        ("shared", name) for name in STATE_FEATURE_NAMES
    )


def _legacy_specs(feature_names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    names = tuple(map(str, feature_names))
    return tuple(("delta", name) for name in names) + tuple(
        ("shared", name) for name in names if name.startswith(("state.", "context."))
    )


def _runtime_specs(controller_root: Path) -> tuple[tuple[str, str], ...]:
    payload = _read_json(controller_root / "main__realized_dynamic.json")
    values = tuple(
        (str(row["mode"]), str(row["name"])) for row in payload["input_features"]
    )
    if len(values) != int(payload["input_dimension"]):
        raise ValueError("runtime v2 input specification is inconsistent")
    return values


def _fit_and_select(
    training: list[dict[str, Any]],
    evaluation_states: list[list[dict[str, Any]]],
    input_specs: tuple[tuple[str, str], ...],
    *,
    legacy_features: bool = False,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], float]:
    pair_rows, pair_diagnostics = build_pair_rows(
        training, input_specs, legacy_features=legacy_features
    )
    started = time.perf_counter()
    model = _fit_pairwise(pair_rows)
    training_seconds = time.perf_counter() - started
    selected = [
        _select_model(
            state, model, input_specs, legacy_features=legacy_features
        )
        for state in evaluation_states
    ]
    return model, selected, pair_diagnostics, training_seconds


def _variant_report(
    *,
    name: str,
    legacy_anchor: list[dict[str, Any]],
    high_train: list[dict[str, Any]],
    high_diagnostic_states: list[list[dict[str, Any]]],
    legacy_validation_states: list[list[dict[str, Any]]],
    frozen_high_train: list[dict[str, Any]],
    frozen_high_diagnostic: list[dict[str, Any]],
    frozen_legacy_validation: list[dict[str, Any]],
    input_specs: tuple[tuple[str, str], ...],
    include_legacy_anchor: bool,
) -> dict[str, Any]:
    folds = _balanced_map_folds(high_train)
    oof_by_state: dict[str, dict[str, Any]] = {}
    fold_reports = []
    total_training_seconds = 0.0
    for fold_index, fold in enumerate(folds):
        validation_maps = set(map(str, fold["validation_maps"]))
        held_candidates = [
            row for row in high_train if str(row["map_id"]) in validation_maps
        ]
        held_states = _group_states(held_candidates)
        training = [
            row for row in high_train if str(row["map_id"]) not in validation_maps
        ]
        if include_legacy_anchor:
            training = [*legacy_anchor, *training]
        _, selected, pair_diagnostics, elapsed = _fit_and_select(
            training, held_states, input_specs
        )
        total_training_seconds += elapsed
        for row in selected:
            state_id = str(row["state_id"])
            if state_id in oof_by_state:
                raise ValueError("OOF state was selected more than once")
            oof_by_state[state_id] = row
        held_frozen = [
            next(
                row
                for row in frozen_high_train
                if str(row["state_id"]) == str(state[0]["state_id"])
            )
            for state in held_states
        ]
        fold_reports.append(
            {
                "fold": fold_index,
                "train_maps": list(fold["train_maps"]),
                "validation_maps": list(fold["validation_maps"]),
                "training_seconds": elapsed,
                "pair_diagnostics": pair_diagnostics,
                "evaluation": _evaluation(held_states, selected, held_frozen),
            }
        )
    high_train_states = _group_states(high_train)
    if set(oof_by_state) != {str(state[0]["state_id"]) for state in high_train_states}:
        raise ValueError("OOF predictions do not cover all high-load training states")
    oof_selected = [oof_by_state[str(state[0]["state_id"])] for state in high_train_states]
    oof = _evaluation(high_train_states, oof_selected, frozen_high_train)

    final_training = [*legacy_anchor, *high_train] if include_legacy_anchor else list(high_train)
    pair_rows, pair_diagnostics = build_pair_rows(final_training, input_specs)
    started = time.perf_counter()
    final_model = _fit_pairwise(pair_rows)
    elapsed = time.perf_counter() - started
    diagnostic_selected = [
        _select_model(state, final_model, input_specs)
        for state in high_diagnostic_states
    ]
    legacy_selected = [
        _select_model(state, final_model, input_specs)
        for state in legacy_validation_states
    ]
    total_training_seconds += elapsed
    diagnostic = _evaluation(
        high_diagnostic_states, diagnostic_selected, frozen_high_diagnostic
    )
    legacy = _evaluation(
        legacy_validation_states, legacy_selected, frozen_legacy_validation
    )
    candidate_checks = {
        "oof_quality": bool(oof["quality_passed"]),
        "oof_map_folds": sum(
            bool(row["evaluation"]["quality_passed"]) for row in fold_reports
        )
        >= 3,
        "diagnostic_quality": bool(diagnostic["quality_passed"]),
        "legacy_quality": bool(legacy["quality_passed"]),
        "oof_cell_noninferiority": _cell_gate(oof),
        "diagnostic_cell_noninferiority": _cell_gate(diagnostic),
        "legacy_cell_noninferiority": _cell_gate(legacy),
        "diagnostic_efficiency_potential": float(
            diagnostic["comparison"]["efficiency_ratio"]
        )
        >= 1.05,
    }
    return {
        "name": name,
        "input_dimension": len(input_specs),
        "include_legacy_anchor": include_legacy_anchor,
        "total_training_seconds": total_training_seconds,
        "pair_diagnostics": pair_diagnostics,
        "folds": fold_reports,
        "oof_high_load": oof,
        "diagnostic_high_load": diagnostic,
        "legacy_validation": legacy,
        "candidate_checks": candidate_checks,
        "candidate_passed": all(candidate_checks.values()),
    }


def _reproduction_report(
    legacy_anchor: list[dict[str, Any]],
    legacy_validation_states: list[list[dict[str, Any]]],
    frozen_validation: list[dict[str, Any]],
    portable_root: Path,
) -> dict[str, Any]:
    payload = _read_json(portable_root / "pairwise__realized_dynamic.json")
    specs = _legacy_specs(payload["feature_names"])
    _, selected, pair_diagnostics, elapsed = _fit_and_select(
        legacy_anchor,
        legacy_validation_states,
        specs,
        legacy_features=True,
    )
    evaluation = _evaluation(
        legacy_validation_states, selected, frozen_validation
    )
    return {
        "input_dimension": len(specs),
        "training_seconds": elapsed,
        "pair_diagnostics": pair_diagnostics,
        "expected_dominance_pair_count": 38564,
        "dominance_pair_count_matches": int(pair_diagnostics["dominance_pairs"])
        == 38564,
        "exact_model_reproduction": bool(
            float(evaluation["selection_change_fraction"]) == 0.0
        ),
        "evaluation": evaluation,
        "selection_agreement": 1.0
        - float(evaluation["selection_change_fraction"]),
    }


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in report["variants"]:
        for cohort in ("oof_high_load", "diagnostic_high_load", "legacy_validation"):
            evaluation = variant[cohort]
            rows.append(
                {
                    "variant": variant["name"],
                    "input_dimension": variant["input_dimension"],
                    "cohort": cohort,
                    **evaluation["comparison"],
                    "quality_passed": evaluation["quality_passed"],
                    "selection_change_fraction": evaluation["selection_change_fraction"],
                }
            )
    return rows


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# v2 同语义训练分布与特征因子审计",
        "",
        "> 证据等级：离线 pilot。未运行完整 episode，不能据此宣称端到端加速。",
        "",
        f"- 决策：`{report['decision']}`",
        f"- 旧训练状态：{report['coverage']['legacy_training_states']}",
        f"- 高负载训练/诊断状态：{report['coverage']['high_training_states']} / {report['coverage']['high_diagnostic_states']}",
        f"- v2 标签重建 dominance pairs：{report['reproduction']['pair_diagnostics']['dominance_pairs']:,}",
        f"- 旧模型选择复现一致率：{report['reproduction']['selection_agreement']:.3%}（仅诊断，不宣称精确复现）",
        "",
        "## 因子结果",
        "",
        "| 变体 | 输入维度 | 高负载 OOF 效率比 | 高负载诊断效率比 | 冲突下降保留 | 旧验证保留 | 通过 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in report["variants"]:
        oof = variant["oof_high_load"]["comparison"]
        diagnostic = variant["diagnostic_high_load"]["comparison"]
        legacy = variant["legacy_validation"]["comparison"]
        lines.append(
            f"| {variant['name']} | {variant['input_dimension']} | "
            f"{oof['efficiency_ratio']:.3f} | {diagnostic['efficiency_ratio']:.3f} | "
            f"{diagnostic['conflict_reduction_ratio']:.3f} | "
            f"{legacy['conflict_reduction_ratio']:.3f} | "
            f"{str(bool(variant['candidate_passed'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `mixed_full_v2` 只改变训练分布，标签、候选池和 pairwise 聚合语义保持 v2。",
            "- `mixed_runtime_projection_86` 是特征删减负对照；当前树只使用这些输入，不代表重新训练时其余输入无价值。",
            "- `high_only_full_v2` 用于识别遗忘风险，不具备部署资格。",
            "- repair 时间来自同状态配对试验；最终晋级仍需完整 episode 的 TTF、wall-AUC、选择时间和 PP 时间。",
            "- 本审计不导出 bundle，不修改默认 `v2-full`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_v2_factorial_audit(
    *,
    legacy_training_index: str | Path,
    legacy_validation_index: str | Path,
    high_load_collection: str | Path,
    controller_bundle: str | Path,
    portable_bundle: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    legacy_training_path = Path(legacy_training_index).resolve()
    legacy_validation_path = Path(legacy_validation_index).resolve()
    high_root = Path(high_load_collection).resolve()
    controller_root = Path(controller_bundle).resolve()
    portable_root = Path(portable_bundle).resolve()
    output_root = Path(output).resolve()

    legacy_training = _legacy_candidates(
        _read_jsonl(legacy_training_path), "policy_train"
    )
    # aggregate_train_index includes 23 registered historical development states.
    # Preserve them even though their split names predate policy_train.
    aggregate_rows = _read_jsonl(legacy_training_path)
    if len({str(row["state_id"]) for row in aggregate_rows}) > len(
        {str(row["state_id"]) for row in legacy_training}
    ):
        legacy_training = _legacy_candidates_any_split(aggregate_rows)
    legacy_validation = _legacy_candidates(
        _read_jsonl(legacy_validation_path), "policy_validation"
    )
    high_features = _read_jsonl(high_root / "feature_index.jsonl")
    high_trials = _read_jsonl(high_root / "trial_manifest.jsonl")
    high = _aggregate_high_load(high_features, high_trials)
    high_train = [row for row in high if str(row["split"]) == "policy_train"]
    high_diagnostic = [
        row for row in high if str(row["split"]) != "policy_train"
    ]
    if not high_train or not high_diagnostic:
        raise ValueError("high-load collection lacks train/diagnostic separation")
    if {row["map_id"] for row in high_train} & {
        row["map_id"] for row in high_diagnostic
    }:
        raise ValueError("high-load train and diagnostic maps overlap")

    bundle = load_controller_bundle(controller_root)
    high_train_states = _group_states(high_train)
    high_diagnostic_states = _group_states(high_diagnostic)
    legacy_validation_states = _group_states(legacy_validation)
    frozen_high_train = _frozen_selection(high_train_states, bundle)
    frozen_high_diagnostic = _frozen_selection(high_diagnostic_states, bundle)
    frozen_legacy_validation = _frozen_selection(legacy_validation_states, bundle)

    reproduction = _reproduction_report(
        legacy_training,
        legacy_validation_states,
        frozen_legacy_validation,
        portable_root,
    )
    full_specs = _full_v2_specs()
    runtime_specs = _runtime_specs(controller_root)
    variants = [
        _variant_report(
            name="mixed_full_v2",
            legacy_anchor=legacy_training,
            high_train=high_train,
            high_diagnostic_states=high_diagnostic_states,
            legacy_validation_states=legacy_validation_states,
            frozen_high_train=frozen_high_train,
            frozen_high_diagnostic=frozen_high_diagnostic,
            frozen_legacy_validation=frozen_legacy_validation,
            input_specs=full_specs,
            include_legacy_anchor=True,
        ),
        _variant_report(
            name="mixed_runtime_projection_86",
            legacy_anchor=legacy_training,
            high_train=high_train,
            high_diagnostic_states=high_diagnostic_states,
            legacy_validation_states=legacy_validation_states,
            frozen_high_train=frozen_high_train,
            frozen_high_diagnostic=frozen_high_diagnostic,
            frozen_legacy_validation=frozen_legacy_validation,
            input_specs=runtime_specs,
            include_legacy_anchor=True,
        ),
        _variant_report(
            name="high_only_full_v2",
            legacy_anchor=legacy_training,
            high_train=high_train,
            high_diagnostic_states=high_diagnostic_states,
            legacy_validation_states=legacy_validation_states,
            frozen_high_train=frozen_high_train,
            frozen_high_diagnostic=frozen_high_diagnostic,
            frozen_legacy_validation=frozen_legacy_validation,
            input_specs=full_specs,
            include_legacy_anchor=False,
        ),
    ]
    eligible = [
        row
        for row in variants
        if row["name"] != "high_only_full_v2" and bool(row["candidate_passed"])
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["oof_high_load"]["comparison"]["efficiency_ratio"]),
            -int(row["input_dimension"]),
            str(row["name"]),
        ),
        default=None,
    )
    decision = (
        "v2_1_same_semantics_pilot_candidate"
        if selected is not None
        else "keep_frozen_v2"
    )
    report = {
        "schema": V2_FACTORIAL_AUDIT_SCHEMA,
        "evidence_level": "offline_pilot",
        "decision": decision,
        "selected_variant": selected["name"] if selected is not None else None,
        "default_controller_changed": False,
        "bundle_exported": False,
        "movingai_or_formal_labels_seen": False,
        "label_semantics": {
            "dominance": "maximize_solved_rate_and_minimize_conflicts_after",
            "runtime_in_training_label": False,
            "trial_aggregation": "candidate_before_pair_construction",
            "state_weighting": "equal_total_weight_per_state",
        },
        "coverage": {
            "legacy_training_states": len(_group_states(legacy_training)),
            "legacy_validation_states": len(legacy_validation_states),
            "high_training_states": len(high_train_states),
            "high_diagnostic_states": len(high_diagnostic_states),
            "high_training_agent_counts": sorted(
                {int(row["agent_count"]) for row in high_train}
            ),
        },
        "inputs": {
            "legacy_training_index": str(legacy_training_path),
            "legacy_training_sha256": sha256_file(legacy_training_path),
            "legacy_validation_index": str(legacy_validation_path),
            "legacy_validation_sha256": sha256_file(legacy_validation_path),
            "high_load_collection": str(high_root),
            "high_feature_index_sha256": sha256_file(high_root / "feature_index.jsonl"),
            "high_trial_manifest_sha256": sha256_file(high_root / "trial_manifest.jsonl"),
            "controller_manifest_sha256": sha256_file(
                controller_root / "controller_manifest.json"
            ),
        },
        "model_parameters": MODEL_PARAMETERS,
        "reproduction": reproduction,
        "variants": variants,
        "limitations": [
            "High-load diagnostic states were used by prior V3 diagnostics and are not a new locked validation set.",
            "Historical aggregate no-progress is a conflict-count proxy because state fingerprints were not stored per candidate.",
            "Offline repair efficiency is not an end-to-end wall-clock result.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "v2_factorial_audit_report.json", report)
    atomic_write_csv(
        output_root / "variant_summary.csv",
        _summary_rows(report),
    )
    (output_root / "v2_factorial_audit_report.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    return report


def _legacy_candidates_any_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load the frozen development union while preserving legacy split names."""

    result = []
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "historical"))].append(row)
    for split_rows in by_split.values():
        split_name = str(split_rows[0].get("split", "historical"))
        if split_name in {"policy_validation", "formal", "ood"}:
            raise ValueError("development aggregate contains a forbidden split")
        for row in split_rows:
            patched = dict(row)
            patched["split"] = "development"
            result.extend(_legacy_candidates([patched], "development"))
    return result


__all__ = [
    "V2_FACTORIAL_AUDIT_SCHEMA",
    "compare_metrics",
    "effectiveness_dominates",
    "equal_state_weights",
    "pair_vector",
    "quality_checks",
    "run_v2_factorial_audit",
    "selection_metrics",
]
