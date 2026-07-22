from __future__ import annotations

import collections
import csv
import hashlib
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_aware_training import _balanced_map_folds
from experiments.repair_collection import _read_jsonl, _write_json
from experiments.v3_controller import load_v3_controller_bundle
from experiments.v3_training import (
    _fit,
    _index_predictions,
    _metrics,
    _predict,
    _prediction_metrics,
    _rows,
    _states,
    _with_runtime_predictions,
)


V3_FEATURE_AUDIT_SCHEMA = "lns2.v3_feature_audit.v1"
MODEL_OUTPUTS = (
    "effective_progress_probability",
    "no_progress_probability",
    "conflict_reduction",
    "log_repair_seconds",
)


def feature_group(name: str) -> str:
    if name.startswith("state."):
        return "state"
    if name.startswith("proposal.actual_size"):
        return "proposal.size"
    if name.startswith(
        (
            "proposal.family_",
            "proposal.selection_family",
            "proposal.support_family",
            "proposal.total_count",
        )
    ):
        return "proposal.family"
    if name.startswith("proposal.seed_"):
        return "proposal.seed"
    if name.startswith("proposal."):
        return "proposal.other"
    if name.startswith("realized.path_"):
        return "realized.path"
    if name.startswith("realized.delay_"):
        return "realized.delay"
    if name.startswith(
        (
            "realized.boundary_",
            "realized.component_",
            "realized.conflict_",
            "realized.conflicting_",
            "realized.incident_",
            "realized.internal_",
        )
    ):
        return "realized.conflict"
    if name.startswith("realized."):
        return "realized.other"
    raise ValueError(f"unregistered v3 feature group: {name}")


def _candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["state_id"]), str(row["candidate_id"])


def _permutation_shift(namespace: str, count: int) -> int:
    if count <= 1:
        return 0
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return 1 + int(digest[:16], 16) % (count - 1)


def permute_candidate_features(
    rows: list[dict[str, Any]],
    feature_indices: Iterable[int],
    *,
    namespace: str,
    stratify_by_actual_size: bool = True,
) -> list[dict[str, Any]]:
    """Permute candidate-level values while keeping paired PP trials identical."""

    indices = tuple(sorted(set(map(int, feature_indices))))
    if not indices:
        return [{**row, "features": list(row["features"])} for row in rows]
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _candidate_key(row)
        previous = candidates.setdefault(key, row)
        if any(
            float(previous["features"][index]) != float(row["features"][index])
            for index in indices
        ):
            raise ValueError("paired PP trials disagree on candidate features")
    strata: dict[tuple[str, str, int], list[tuple[str, str]]] = (
        collections.defaultdict(list)
    )
    for key, row in candidates.items():
        strata[
            (
                str(row["map_id"]),
                str(row["route"]),
                int(row["actual_size"]) if stratify_by_actual_size else -1,
            )
        ].append(key)
    replacements: dict[tuple[tuple[str, str], int], float] = {}
    for stratum, raw_keys in sorted(strata.items()):
        keys = sorted(raw_keys)
        for feature_index in indices:
            shift = _permutation_shift(
                f"{namespace}|{feature_index}|{stratum}", len(keys)
            )
            for position, key in enumerate(keys):
                source = keys[(position + shift) % len(keys)]
                replacements[(key, feature_index)] = float(
                    candidates[source]["features"][feature_index]
                )
    result = []
    for row in rows:
        values = list(map(float, row["features"]))
        key = _candidate_key(row)
        for feature_index in indices:
            values[feature_index] = replacements[(key, feature_index)]
        result.append({**row, "features": values})
    return result


def project_rows(
    rows: list[dict[str, Any]], retained_indices: Iterable[int]
) -> list[dict[str, Any]]:
    indices = tuple(map(int, retained_indices))
    return [
        {**row, "features": [float(row["features"][index]) for index in indices]}
        for row in rows
    ]


def _estimator_split_usage(
    models: dict[str, Any], feature_names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        name: {
            "split_count": 0,
            "gain_sum": 0.0,
            "root_split_count": 0,
            "minimum_depth": None,
            "models": set(),
        }
        for name in feature_names
    }
    for model_name, estimator in models.items():
        for stage in estimator._predictors:
            for predictor in stage:
                for node in predictor.nodes:
                    if bool(node["is_leaf"]):
                        continue
                    index = int(node["feature_idx"])
                    if index < 0 or index >= len(feature_names):
                        raise ValueError("v3 estimator has an invalid split feature")
                    row = result[feature_names[index]]
                    depth = int(node["depth"])
                    row["split_count"] += 1
                    row["gain_sum"] += max(0.0, float(node["gain"]))
                    row["root_split_count"] += int(depth == 0)
                    row["minimum_depth"] = (
                        depth
                        if row["minimum_depth"] is None
                        else min(int(row["minimum_depth"]), depth)
                    )
                    row["models"].add(str(model_name))
    return result


def _portable_split_counts(bundle: Any) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for model in bundle.models.values():
        for tree in model.trees:
            for node in tree:
                if not bool(node["is_leaf"]):
                    counts[model.feature_names[int(node["feature_idx"])]] += 1
    return counts


def _prediction_delta(
    baseline: dict[str, list[float]], changed: dict[str, list[float]]
) -> dict[str, float]:
    result = {}
    for name in MODEL_OUTPUTS:
        differences = [
            abs(float(left) - float(right))
            for left, right in zip(baseline[name], changed[name])
        ]
        result[f"{name}_mean_abs_delta"] = (
            statistics.fmean(differences) if differences else 0.0
        )
        result[f"{name}_max_abs_delta"] = max(differences, default=0.0)
    return result


def _selection_ids(states: list[dict[str, Any]], thresholds: dict[str, float]) -> list[str]:
    from experiments.v3_training import _selected

    return [
        str(_selected(state, "v3", thresholds)["candidate_id"]) for state in states
    ]


def _comparison(
    baseline_states: list[dict[str, Any]],
    changed_states: list[dict[str, Any]],
    thresholds: dict[str, float],
    overhead: float,
) -> dict[str, float]:
    baseline = _metrics(baseline_states, "v3", thresholds, overhead)
    changed = _metrics(changed_states, "v3", thresholds, overhead)
    baseline_ids = _selection_ids(baseline_states, thresholds)
    changed_ids = _selection_ids(changed_states, thresholds)
    changes = sum(left != right for left, right in zip(baseline_ids, changed_ids))
    return {
        "state_count": float(len(baseline_states)),
        "selection_change_count": float(changes),
        "selection_change_rate": changes / len(baseline_states),
        "efficiency_ratio": changed["conflict_reduction_per_total_second"]
        / max(1e-12, baseline["conflict_reduction_per_total_second"]),
        "conflict_reduction_ratio": changed["mean_conflict_reduction"]
        / max(1e-12, baseline["mean_conflict_reduction"]),
        "effective_rate_delta": changed["effective_rate"] - baseline["effective_rate"],
        "no_progress_rate_delta": changed["no_progress_rate"]
        - baseline["no_progress_rate"],
    }


def _prediction_metric_delta(
    rows: list[dict[str, Any]],
    baseline: dict[str, list[float]],
    changed: dict[str, list[float]],
) -> dict[str, float]:
    reference = _prediction_metrics(rows, baseline)
    perturbed = _prediction_metrics(rows, changed)
    return {
        "effective_auc_drop": reference["effective_progress_auc"]
        - perturbed["effective_progress_auc"],
        "no_progress_auc_drop": reference["no_progress_auc"]
        - perturbed["no_progress_auc"],
        "reduction_mae_increase": perturbed["reduction_mae"]
        - reference["reduction_mae"],
        "repair_log_mae_increase": perturbed["repair_log_mae"]
        - reference["repair_log_mae"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _oof_models_and_predictions(
    rows: list[dict[str, Any]],
    overhead: float,
    feature_names: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[float]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    held_rows: list[dict[str, Any]] = []
    predictions = {name: [] for name in MODEL_OUTPUTS}
    folds = []
    usage = {
        name: {
            "fold_count": 0,
            "fold_model_count": 0,
            "split_count": 0,
            "gain_sum": 0.0,
            "root_split_count": 0,
            "minimum_depth": None,
        }
        for name in feature_names
    }
    for fold_index, fold in enumerate(_balanced_map_folds(rows)):
        held_maps = set(map(str, fold["validation_maps"]))
        training = [row for row in rows if str(row["map_id"]) not in held_maps]
        held = [row for row in rows if str(row["map_id"]) in held_maps]
        models = _fit(training)
        raw = _predict(models, held)
        for name in MODEL_OUTPUTS:
            predictions[name].extend(raw[name])
        split = _estimator_split_usage(models, feature_names)
        for feature, row in split.items():
            target = usage[feature]
            target["fold_count"] += int(int(row["split_count"]) > 0)
            target["fold_model_count"] += len(row["models"])
            target["split_count"] += int(row["split_count"])
            target["gain_sum"] += float(row["gain_sum"])
            target["root_split_count"] += int(row["root_split_count"])
            if row["minimum_depth"] is not None:
                target["minimum_depth"] = (
                    int(row["minimum_depth"])
                    if target["minimum_depth"] is None
                    else min(
                        int(target["minimum_depth"]), int(row["minimum_depth"])
                    )
                )
        folds.append(
            {
                "fold_index": fold_index,
                "validation_maps": sorted(held_maps),
                "training_rows": training,
                "held_rows": held,
                "models": models,
                "baseline_raw": raw,
            }
        )
        held_rows.extend(held)
    runtime = _with_runtime_predictions(predictions, overhead)
    states = _states(held_rows, _index_predictions(held_rows, runtime))
    return held_rows, runtime, states, usage | {"__folds__": {"rows": folds}}


def _perturbed_oof(
    folds: list[dict[str, Any]],
    feature_indices: Iterable[int],
    overhead: float,
    namespace: str,
    *,
    stratify_by_actual_size: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predicted = {name: [] for name in MODEL_OUTPUTS}
    for fold in folds:
        held = permute_candidate_features(
            fold["held_rows"],
            feature_indices,
            namespace=f"{namespace}|fold={fold['fold_index']}",
            stratify_by_actual_size=stratify_by_actual_size,
        )
        values = _predict(fold["models"], held)
        rows.extend(held)
        for name in MODEL_OUTPUTS:
            predicted[name].extend(values[name])
    runtime = _with_runtime_predictions(predicted, overhead)
    states = _states(rows, _index_predictions(rows, runtime))
    return rows, runtime, states


def _retrained_projection(
    train_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    retained_indices: tuple[int, ...],
    thresholds: dict[str, float],
    overhead: float,
) -> dict[str, Any]:
    projected_train = project_rows(train_rows, retained_indices)
    projected_diagnostic = project_rows(diagnostic_rows, retained_indices)
    held_rows, oof_raw, oof_states, _usage = _oof_models_and_predictions(
        projected_train,
        overhead,
        tuple(str(index) for index in retained_indices),
    )
    models = _fit(projected_train)
    diagnostic_raw = _predict(models, projected_diagnostic)
    diagnostic_runtime = _with_runtime_predictions(diagnostic_raw, overhead)
    diagnostic_states = _states(
        projected_diagnostic,
        _index_predictions(projected_diagnostic, diagnostic_runtime),
    )
    return {
        "oof_rows": held_rows,
        "oof_raw": oof_raw,
        "oof_states": oof_states,
        "diagnostic_rows": projected_diagnostic,
        "diagnostic_raw": diagnostic_runtime,
        "diagnostic_states": diagnostic_states,
        "oof_metrics": _metrics(oof_states, "v3", thresholds, overhead),
        "diagnostic_metrics": _metrics(
            diagnostic_states, "v3", thresholds, overhead
        ),
    }


def _training_time_benchmark(
    full_rows: list[dict[str, Any]],
    reduced_rows: list[dict[str, Any]],
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("v3 training benchmark repeats must be positive")
    samples = {"full": [], "reduced": []}
    for repeat in range(repeats):
        order = ("full", "reduced") if repeat % 2 == 0 else ("reduced", "full")
        for name in order:
            started = time.perf_counter()
            _fit(full_rows if name == "full" else reduced_rows)
            samples[name].append(time.perf_counter() - started)
    full_median = statistics.median(samples["full"])
    reduced_median = statistics.median(samples["reduced"])
    return {
        "repeats": repeats,
        "full_seconds": samples["full"],
        "reduced_seconds": samples["reduced"],
        "full_median_seconds": full_median,
        "reduced_median_seconds": reduced_median,
        "median_speedup_fraction": 1.0 - reduced_median / max(1e-12, full_median),
    }


def audit_v3_features(source: str | Path, output: str | Path) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("v3 feature audit output directory is not empty")
    collection = source_root / "collection"
    controller = source_root / "controller"
    feature_names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    feature_rows = _read_jsonl(collection / "feature_index.jsonl")
    trial_rows = _read_jsonl(collection / "trial_manifest.jsonl")
    train_rows = _rows(feature_rows, trial_rows, "policy_train", feature_names)
    diagnostic_rows = _rows(
        feature_rows, trial_rows, "policy_validation", feature_names
    )
    train_maps = {str(row["map_id"]) for row in train_rows}
    diagnostic_maps = {str(row["map_id"]) for row in diagnostic_rows}
    if train_maps & diagnostic_maps:
        raise ValueError("v3 feature audit train and diagnostic maps overlap")
    bundle = load_v3_controller_bundle(controller)
    thresholds = dict(bundle.thresholds)
    overhead = float(bundle.selection_overhead_seconds)
    main = load_controller_bundle(
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "initlns-closed-loop-controller-v2"
    ).main_models["realized_dynamic"]

    held_rows, baseline_oof, baseline_states, usage_payload = (
        _oof_models_and_predictions(train_rows, overhead, feature_names)
    )
    folds = list(usage_payload.pop("__folds__")["rows"])
    full_models = _fit(train_rows)
    diagnostic_baseline_raw = _predict(full_models, diagnostic_rows)
    diagnostic_baseline = _with_runtime_predictions(
        diagnostic_baseline_raw, overhead
    )
    diagnostic_baseline_states = _states(
        diagnostic_rows,
        _index_predictions(diagnostic_rows, diagnostic_baseline),
    )
    portable_counts = _portable_split_counts(bundle)
    runtime_features = set(bundle.required_feature_names)
    combined_features = runtime_features | set(main.base_feature_names)

    feature_results = []
    for feature_index, feature in enumerate(feature_names):
        group = feature_group(feature)
        _rows_oof, changed_oof, changed_states = _perturbed_oof(
            folds,
            (feature_index,),
            overhead,
            f"feature={feature}",
            stratify_by_actual_size=group != "proposal.size",
        )
        changed_diagnostic_rows = permute_candidate_features(
            diagnostic_rows,
            (feature_index,),
            namespace=f"diagnostic|feature={feature}",
            stratify_by_actual_size=group != "proposal.size",
        )
        changed_diagnostic_raw = _predict(full_models, changed_diagnostic_rows)
        changed_diagnostic = _with_runtime_predictions(
            changed_diagnostic_raw, overhead
        )
        changed_diagnostic_states = _states(
            changed_diagnostic_rows,
            _index_predictions(changed_diagnostic_rows, changed_diagnostic),
        )
        oof_delta = _prediction_delta(baseline_oof, changed_oof)
        diagnostic_delta = _prediction_delta(
            diagnostic_baseline, changed_diagnostic
        )
        usage = usage_payload[feature]
        maximum_mean_delta = max(
            oof_delta[f"{name}_mean_abs_delta"] for name in MODEL_OUTPUTS
        )
        oof_comparison = _comparison(
            baseline_states, changed_states, thresholds, overhead
        )
        recommendation = "retain"
        if (
            feature not in runtime_features
            and int(usage["fold_count"]) == 0
            and oof_comparison["selection_change_count"] == 0
            and maximum_mean_delta <= 1e-12
        ):
            recommendation = "training_removal_candidate"
        elif (
            int(usage["fold_count"]) <= 1
            and oof_comparison["selection_change_rate"] <= 0.01
            and maximum_mean_delta <= 1e-4
        ):
            recommendation = "weak_review"
        feature_results.append(
            {
                "feature": feature,
                "group": group,
                "runtime_used_by_v3": feature in runtime_features,
                "runtime_used_by_v2_or_v3": feature in combined_features,
                "full_v3_split_count": int(portable_counts[feature]),
                **usage,
                **{f"oof_{key}": value for key, value in oof_delta.items()},
                **{
                    f"oof_{key}": value
                    for key, value in _prediction_metric_delta(
                        held_rows, baseline_oof, changed_oof
                    ).items()
                },
                **{f"oof_{key}": value for key, value in oof_comparison.items()},
                **{
                    f"diagnostic_{key}": value
                    for key, value in diagnostic_delta.items()
                },
                **{
                    f"diagnostic_{key}": value
                    for key, value in _comparison(
                        diagnostic_baseline_states,
                        changed_diagnostic_states,
                        thresholds,
                        overhead,
                    ).items()
                },
                "recommendation": recommendation,
            }
        )

    group_results = []
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for index, feature in enumerate(feature_names):
        by_group[feature_group(feature)].append(index)
    for group, indices in sorted(by_group.items()):
        _group_rows, group_predictions, group_states = _perturbed_oof(
            folds,
            indices,
            overhead,
            f"group={group}",
            stratify_by_actual_size=group != "proposal.size",
        )
        group_results.append(
            {
                "group": group,
                "feature_count": len(indices),
                **_comparison(
                    baseline_states, group_states, thresholds, overhead
                ),
                **_prediction_metric_delta(
                    held_rows, baseline_oof, group_predictions
                ),
            }
        )

    removal_candidates = sorted(
        row["feature"]
        for row in feature_results
        if row["recommendation"] == "training_removal_candidate"
    )
    retained_indices = tuple(
        index
        for index, feature in enumerate(feature_names)
        if feature not in set(removal_candidates)
    )
    reduced = _retrained_projection(
        train_rows,
        diagnostic_rows,
        retained_indices,
        thresholds,
        overhead,
    )
    reduced_oof_comparison = _comparison(
        baseline_states, reduced["oof_states"], thresholds, overhead
    )
    reduced_diagnostic_comparison = _comparison(
        diagnostic_baseline_states,
        reduced["diagnostic_states"],
        thresholds,
        overhead,
    )
    reduced_checks = {
        "oof_selection_change_at_most_1pct": reduced_oof_comparison[
            "selection_change_rate"
        ]
        <= 0.01 + 1e-12,
        "oof_efficiency_retained_98pct": reduced_oof_comparison[
            "efficiency_ratio"
        ]
        + 1e-12
        >= 0.98,
        "oof_effective_rate_not_down_1pct": reduced_oof_comparison[
            "effective_rate_delta"
        ]
        + 1e-12
        >= -0.01,
        "oof_no_progress_not_up_1pct": reduced_oof_comparison[
            "no_progress_rate_delta"
        ]
        <= 0.01 + 1e-12,
        "diagnostic_efficiency_retained_95pct": reduced_diagnostic_comparison[
            "efficiency_ratio"
        ]
        + 1e-12
        >= 0.95,
    }
    reduced_passed = bool(removal_candidates) and all(reduced_checks.values())
    training_benchmark = _training_time_benchmark(
        train_rows,
        project_rows(train_rows, retained_indices),
    )
    report = {
        "schema": V3_FEATURE_AUDIT_SCHEMA,
        "schema_version": 1,
        "source": str(source_root),
        "selection_basis": "policy_train_map_group_oof_only",
        "diagnostic_split_role": "reported_not_used_for_feature_selection",
        "solver_rerun": False,
        "feature_count": len(feature_names),
        "training_state_count": len({row["state_id"] for row in train_rows}),
        "diagnostic_state_count": len(
            {row["state_id"] for row in diagnostic_rows}
        ),
        "training_map_count": len(train_maps),
        "diagnostic_map_count": len(diagnostic_maps),
        "training_trial_count": len(train_rows),
        "diagnostic_trial_count": len(diagnostic_rows),
        "runtime_projection": {
            **bundle.runtime_projection,
            "v2_runtime_feature_count": len(main.base_feature_names),
            "combined_runtime_feature_count": len(combined_features),
        },
        "recommendation_counts": dict(
            collections.Counter(row["recommendation"] for row in feature_results)
        ),
        "training_removal_candidates": removal_candidates,
        "training_time_benchmark": training_benchmark,
        "retrained_reduced_schema": {
            "original_feature_count": len(feature_names),
            "reduced_feature_count": len(retained_indices),
            "removed_feature_count": len(removal_candidates),
            "oof_comparison": reduced_oof_comparison,
            "diagnostic_comparison": reduced_diagnostic_comparison,
            "checks": reduced_checks,
            "passed": reduced_passed,
        },
        "decision": (
            "reduced_training_schema_candidate"
            if reduced_passed
            else "keep_124_training_schema"
        ),
        "deployment_promoted": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "feature_audit.csv", feature_results)
    _write_csv(output_root / "feature_group_permutation.csv", group_results)
    _write_json(output_root / "v3_feature_audit_report.json", report)
    ranked = sorted(
        feature_results,
        key=lambda row: (
            -float(row["oof_selection_change_rate"]),
            -float(row["gain_sum"]),
            str(row["feature"]),
        ),
    )
    markdown = [
        "# v3 feature audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Runtime projection: 124 -> {len(combined_features)} combined v2/v3 features.",
        f"- Strict training-removal candidates: {len(removal_candidates)}.",
        f"- Retrained reduced schema: {len(feature_names)} -> {len(retained_indices)}.",
        f"- Reduced-schema audit passed: `{reduced_passed}`.",
        "- Median four-head fit time: "
        f"{training_benchmark['full_median_seconds']:.3f}s -> "
        f"{training_benchmark['reduced_median_seconds']:.3f}s "
        f"({training_benchmark['median_speedup_fraction']:.1%} faster).",
        "- No PP or complete solver episode was rerun.",
        "",
        "## Strict removal candidates",
        "",
        ", ".join(f"`{name}`" for name in removal_candidates) or "None.",
        "",
        "## Most selection-sensitive features",
        "",
        "| Feature | OOF selection change | OOF gain | Fold count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ranked[:20]:
        markdown.append(
            f"| `{row['feature']}` | {float(row['oof_selection_change_rate']):.3%} "
            f"| {float(row['gain_sum']):.6g} | {int(row['fold_count'])}/4 |"
        )
    markdown.extend(
        [
            "",
            "The diagnostic maps are reported for sensitivity only and are not used "
            "to choose the removal set. This audit does not promote a controller.",
            "",
        ]
    )
    (output_root / "v3_feature_audit_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return report


__all__ = [
    "V3_FEATURE_AUDIT_SCHEMA",
    "audit_v3_features",
    "feature_group",
    "permute_candidate_features",
    "project_rows",
]
