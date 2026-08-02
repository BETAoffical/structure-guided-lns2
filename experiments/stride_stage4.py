from __future__ import annotations

import gc
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import (
    compact_portable_payload,
    load_compact_model,
    load_controller_bundle,
)
from experiments.context_audit import PairwiseModel
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.mixed_full_v2 import (
    _atomic_pickle,
    _feature_ranges,
    _portable_payload,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_stage3 import _project_path
from experiments.v2_factorial_audit import _select_model
from lns2_selector.runtime.online_selection import score_online_candidates


STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA = "lns2.stride.stage4_training_config.v1"
STRIDE_STAGE4_FOLD_SCHEMA = "lns2.stride.stage4_fold_assignment.v1"
STRIDE_STAGE4_PROTOCOL_SCHEMA = "lns2.stride.stage4_protocol.v1"
STRIDE_STAGE4_TRAINING_SCHEMA = "lns2.stride.stage4_training.v1"
STRIDE_STAGE4_MODEL_SCHEMA = "lns2.stride.stage4_model.v1"

def _layout_family(layout_mode: str, aliases: dict[str, str]) -> str:
    return aliases.get(layout_mode, layout_mode)


def _count_dimensions(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["total"] += 1
        counts[f"source_policy:{row['source_policy']}"] += 1
        counts[f"agent_band:{row['agent_band']}"] += 1
        counts[f"decision_stage:{row['decision_stage']}"] += 1
    return counts


def _assign_grouped_folds(
    rows: list[dict[str, Any]],
    *,
    fold_count: int,
    layout_aliases: dict[str, str],
) -> tuple[dict[str, int], list[list[str]], dict[str, str]]:
    """Assign whole maps using outcome-blind, deterministic metadata balancing."""

    if fold_count < 2:
        raise ValueError("Stage 4 requires at least two folds")

    rows_by_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    map_families: dict[str, str] = {}
    for row in rows:
        map_id = str(row["map_id"])
        layout_mode = str(row["layout_mode"])
        family = _layout_family(layout_mode, layout_aliases)
        previous = map_families.setdefault(map_id, family)
        if previous != family:
            raise ValueError(f"map has inconsistent layout family: {map_id}")
        rows_by_map[map_id].append(row)

    maps_by_family: defaultdict[str, list[str]] = defaultdict(list)
    for map_id, family in map_families.items():
        maps_by_family[family].append(map_id)

    global_counts = _count_dimensions(rows)
    targets = {
        name: value / float(fold_count) for name, value in global_counts.items()
    }
    fold_counts = [Counter() for _ in range(fold_count)]
    fold_maps: list[list[str]] = [[] for _ in range(fold_count)]
    family_fold_counts = [Counter() for _ in range(fold_count)]
    map_to_fold: dict[str, int] = {}

    family_order = sorted(
        maps_by_family,
        key=lambda family: (
            -sum(len(rows_by_map[map_id]) for map_id in maps_by_family[family]),
            family,
        ),
    )
    for family in family_order:
        map_order = sorted(
            maps_by_family[family],
            key=lambda map_id: (-len(rows_by_map[map_id]), map_id),
        )
        for map_id in map_order:
            map_counts = _count_dimensions(rows_by_map[map_id])

            def assignment_score(fold_index: int) -> tuple[Any, ...]:
                ratios = [
                    (fold_counts[fold_index][name] + map_counts[name]) / target
                    for name, target in targets.items()
                    if target > 0.0
                ]
                return (
                    family_fold_counts[fold_index][family],
                    max(ratios, default=0.0),
                    sum(value * value for value in ratios),
                    len(fold_maps[fold_index]),
                    fold_index,
                )

            fold_index = min(range(fold_count), key=assignment_score)
            map_to_fold[map_id] = fold_index
            fold_maps[fold_index].append(map_id)
            fold_counts[fold_index].update(map_counts)
            family_fold_counts[fold_index][family] += 1

    state_to_fold = {
        str(row["state_id"]): map_to_fold[str(row["map_id"])] for row in rows
    }
    return state_to_fold, fold_maps, map_families


def _fold_counts(
    rows: list[dict[str, Any]], state_to_fold: dict[str, int], fold_count: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        selected = [
            row
            for row in rows
            if state_to_fold[str(row["state_id"])] == fold_index
        ]
        result.append(
            {
                "fold_index": fold_index,
                "state_count": len(selected),
                "source_policy_state_counts": dict(
                    sorted(Counter(str(row["source_policy"]) for row in selected).items())
                ),
                "agent_band_state_counts": dict(
                    sorted(Counter(str(row["agent_band"]) for row in selected).items())
                ),
                "decision_stage_state_counts": dict(
                    sorted(
                        Counter(str(row["decision_stage"]) for row in selected).items()
                    )
                ),
            }
        )
    return result


def prepare_stride_stage4_protocol(
    *, config_path: Path, output: Path, project_root: Path
) -> dict[str, Any]:
    """Freeze the Stage 4 protocol and map-grouped folds without reading labels."""

    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4 training-config schema")

    project_root = project_root.resolve()
    audit_path = _project_path(project_root, str(config["stage3_audit_report"]))
    audit = _read_json(audit_path)
    selection_paths = [
        _project_path(project_root, str(value))
        for value in config.get("selections", [])
    ]
    if not selection_paths:
        raise ValueError("Stage 4 requires registered selection files")

    selected_rows: list[dict[str, Any]] = []
    for path in selection_paths:
        selected_rows.extend(_read_jsonl(path))

    required_fields = {
        "state_id",
        "map_id",
        "layout_mode",
        "split",
        "source_policy",
        "agent_band",
        "decision_stage",
        "agent_count",
    }
    missing_field_count = sum(
        not required_fields.issubset(row) for row in selected_rows
    )
    state_ids = [str(row.get("state_id", "")) for row in selected_rows]
    duplicate_state_count = len(state_ids) - len(set(state_ids))

    protocol = dict(config.get("protocol", {}))
    models = dict(protocol.get("models", {}))
    feature_variants = dict(protocol.get("feature_variants", {}))
    registered_names_valid = (
        dict(models.get("frozen_anchor", {})).get("id") == "v2-full"
        and dict(models.get("frozen_anchor", {})).get("read_only") is True
        and dict(models.get("control", {})).get("id") == "stride-control-v1"
        and dict(models.get("quality", {})).get("id") == "stride-quality-v1"
    )
    feature_variants_valid = "full" in feature_variants and len(feature_variants) >= 2
    primary_label = str(protocol.get("primary_label", ""))
    forbidden_labels = {str(value) for value in protocol.get("forbidden_primary_labels", [])}

    fold_count = int(config["fold_count"])
    aliases = {
        str(name): str(value)
        for name, value in dict(config.get("layout_family_aliases", {})).items()
    }
    state_to_fold, fold_maps, map_families = _assign_grouped_folds(
        selected_rows, fold_count=fold_count, layout_aliases=aliases
    )
    counts = _fold_counts(selected_rows, state_to_fold, fold_count)

    manifest_rows = [
        {
            "schema": STRIDE_STAGE4_FOLD_SCHEMA,
            "state_id": str(row["state_id"]),
            "map_id": str(row["map_id"]),
            "layout_mode": str(row["layout_mode"]),
            "layout_family": map_families[str(row["map_id"])],
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "agent_band": str(row["agent_band"]),
            "decision_stage": str(row["decision_stage"]),
            "agent_count": int(row["agent_count"]),
            "fold_index": state_to_fold[str(row["state_id"])],
        }
        for row in sorted(selected_rows, key=lambda item: str(item["state_id"]))
    ]

    thresholds = dict(config.get("fold_gates", {}))
    expected_state_count = int(config["expected_state_count"])
    expected_policy_counts = {
        str(name): int(value)
        for name, value in dict(config["expected_state_count_per_policy"]).items()
    }
    policy_counts = Counter(str(row["source_policy"]) for row in selected_rows)
    required_families = {
        str(value) for value in config.get("required_layout_families", [])
    }
    fold_families = [
        {map_families[map_id] for map_id in map_ids} for map_ids in fold_maps
    ]
    min_policy = int(thresholds.get("min_policy_states_per_fold", 0))
    min_agent_band = int(thresholds.get("min_agent_band_states_per_fold", 0))
    expected_agent_bands = {
        str(row["agent_band"]) for row in selected_rows
    }
    stage_minima = {
        str(name): int(value)
        for name, value in dict(
            thresholds.get("min_decision_stage_states_per_fold", {})
        ).items()
    }
    map_fold_occurrences = Counter(
        map_id for map_ids in fold_maps for map_id in map_ids
    )
    audit_hash = sha256_file(audit_path)
    gates = {
        "stage3_audit_passed": audit.get("passed") is True,
        "stage3_audit_hash_frozen": audit_hash
        == str(config["stage3_audit_sha256"]),
        "stage3_state_count_matches": int(audit.get("state_count", -1))
        == expected_state_count,
        "selection_files_exist": all(path.is_file() for path in selection_paths),
        "selection_fields_complete": missing_field_count == 0,
        "selection_state_count": len(selected_rows) == expected_state_count,
        "selection_states_unique": duplicate_state_count == 0,
        "source_policy_balance": dict(policy_counts) == expected_policy_counts,
        "registered_model_names_valid": registered_names_valid,
        "feature_variants_registered": feature_variants_valid,
        "runtime_excluded_from_primary_label": (
            protocol.get("runtime_used_in_primary_label") is False
        ),
        "forbidden_primary_labels_excluded": primary_label not in forbidden_labels,
        "all_states_assigned": len(state_to_fold) == expected_state_count,
        "maps_grouped_exclusively": (
            len(map_fold_occurrences) == len(map_families)
            and all(value == 1 for value in map_fold_occurrences.values())
        ),
        "fold_state_minimum": all(
            row["state_count"] >= int(thresholds.get("min_states_per_fold", 0))
            for row in counts
        ),
        "fold_state_maximum": all(
            row["state_count"] <= int(
                thresholds.get("max_states_per_fold", expected_state_count)
            )
            for row in counts
        ),
        "fold_map_minimum": all(
            len(map_ids) >= int(thresholds.get("min_maps_per_fold", 0))
            for map_ids in fold_maps
        ),
        "each_layout_family_in_each_fold": all(
            required_families.issubset(families) for families in fold_families
        ),
        "fold_policy_minimum": all(
            all(value >= min_policy for value in row["source_policy_state_counts"].values())
            and set(row["source_policy_state_counts"]) == set(expected_policy_counts)
            for row in counts
        ),
        "fold_agent_band_minimum": all(
            set(row["agent_band_state_counts"]) == expected_agent_bands
            and all(
                value >= min_agent_band
                for value in row["agent_band_state_counts"].values()
            )
            for row in counts
        ),
        "fold_decision_stage_minimum": all(
            all(
                row["decision_stage_state_counts"].get(name, 0) >= minimum
                for name, minimum in stage_minima.items()
            )
            for row in counts
        ),
    }

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "fold_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    try:
        config_name = config_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        config_name = config_path.name
    report = {
        "schema": STRIDE_STAGE4_PROTOCOL_SCHEMA,
        "config": config_name,
        "config_sha256": _fingerprint(config),
        "passed": all(gates.values()),
        "gates": gates,
        "label_outcomes_read": False,
        "state_count": len(selected_rows),
        "map_count": len(map_families),
        "fold_count": fold_count,
        "folds": [
            {
                **row,
                "map_count": len(fold_maps[row["fold_index"]]),
                "maps": sorted(fold_maps[row["fold_index"]]),
                "layout_families": sorted(fold_families[row["fold_index"]]),
            }
            for row in counts
        ],
        "layout_family_map_counts": dict(
            sorted(Counter(map_families.values()).items())
        ),
        "source_policy_state_counts": dict(sorted(policy_counts.items())),
        "diagnostics": {
            "missing_selection_field_count": missing_field_count,
            "duplicate_state_count": duplicate_state_count,
        },
        "stage3_audit_sha256": audit_hash,
        "selection_sha256": [sha256_file(path) for path in selection_paths],
        "fold_manifest_sha256": sha256_file(manifest_path),
        "registered_protocol": protocol,
    }
    _write_json(output / "stage4_protocol_report.json", report)
    return report


def _variant_specifications(
    variants: dict[str, Any],
) -> dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]]:
    registered = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    result = {}
    for variant, row_value in variants.items():
        row = dict(row_value)
        prefixes = tuple(map(str, row.get("drop_prefixes", [])))
        names = tuple(
            name for name in registered if not name.startswith(prefixes)
        )
        if not names:
            raise ValueError(f"Stage 4 feature variant is empty: {variant}")
        specifications = tuple(("delta", name) for name in names) + tuple(
            ("shared", name)
            for name in names
            if name.startswith(("state.", "context."))
        )
        result[str(variant)] = (names, specifications)
    return result


def _load_candidates(
    aggregate_path: Path, manifest_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    metadata = {str(row["state_id"]): dict(row) for row in manifest_rows}
    registered_features = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    candidates = []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for source in _read_jsonl(aggregate_path):
        state_id = str(source["state_id"])
        candidate_id = str(source["candidate_id"])
        key = (state_id, candidate_id)
        if key in seen:
            raise ValueError(f"duplicate Stage 4 candidate: {key}")
        seen.add(key)
        if state_id not in metadata:
            raise ValueError(f"candidate state is absent from fold manifest: {state_id}")
        features = {str(name): float(value) for name, value in dict(source["features"]).items()}
        if set(features) != registered_features:
            raise ValueError(f"candidate has the wrong feature schema: {key}")
        row = {
            "candidate_index": len(candidates),
            "state_id": state_id,
            "candidate_id": candidate_id,
            "candidate_key": candidate_id,
            "features": features,
            "quality_score": float(source["quality_score"]),
            "feasible_rate": float(source["feasible_rate"]),
            "progress_rate": float(source["progress_rate"]),
            "mean_conflict_reduction": float(source["mean_conflict_reduction"]),
            "mean_reduction_ratio": float(source["mean_reduction_ratio"]),
            "structural_score": float(source["structural_score"]),
            "before_conflicts": int(source["before_conflicts"]),
            **metadata[state_id],
        }
        candidates.append(row)
        grouped[state_id].append(row)
    if set(grouped) != set(metadata):
        raise ValueError("candidate and fold-manifest state coverage differs")
    if any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("every Stage 4 state must have at least two candidates")
    return candidates, dict(grouped)


def _pair_table_from_quality(
    pair_path: Path,
    candidate_index: dict[tuple[str, str], int],
    fold_by_state: dict[str, int],
) -> dict[str, Any]:
    import numpy as np

    left = []
    right = []
    labels = []
    weights = []
    folds = []
    state_weights: Counter[str] = Counter()
    for row in _read_jsonl(pair_path):
        state_id = str(row["state_id"])
        left.append(candidate_index[(state_id, str(row["left_candidate_id"]))])
        right.append(candidate_index[(state_id, str(row["right_candidate_id"]))])
        labels.append(int(row["label"]))
        weight = float(row["sample_weight"])
        weights.append(weight)
        folds.append(fold_by_state[state_id])
        state_weights[state_id] += weight
    if set(labels) != {0, 1}:
        raise ValueError("quality pairs require both labels")
    if set(state_weights) != set(fold_by_state) or any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in state_weights.values()
    ):
        raise ValueError("quality pair weights are not equal by state")
    return {
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "folds": np.asarray(folds, dtype=np.int8),
        "state_count": len(state_weights),
    }


def _control_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        float(left["feasible_rate"]) + 1e-12 >= float(right["feasible_rate"])
        and float(left["mean_conflict_reduction"]) + 1e-12
        >= float(right["mean_conflict_reduction"])
    )
    strictly_better = (
        float(left["feasible_rate"])
        > float(right["feasible_rate"]) + 1e-12
        or float(left["mean_conflict_reduction"])
        > float(right["mean_conflict_reduction"]) + 1e-12
    )
    return bool(no_worse and strictly_better)


def _pair_table_from_control(
    grouped: dict[str, list[dict[str, Any]]], fold_by_state: dict[str, int]
) -> dict[str, Any]:
    import numpy as np

    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    folds: list[int] = []
    states_with_pairs = 0
    for state_id, state in sorted(grouped.items()):
        winners = []
        for first, second in itertools.combinations(state, 2):
            first_wins = _control_dominates(first, second)
            second_wins = _control_dominates(second, first)
            if first_wins == second_wins:
                continue
            winner, loser = (first, second) if first_wins else (second, first)
            winners.append((int(winner["candidate_index"]), int(loser["candidate_index"])))
        if not winners:
            continue
        states_with_pairs += 1
        state_weight = 1.0 / (2.0 * len(winners))
        for winner, loser in winners:
            left.extend((winner, loser))
            right.extend((loser, winner))
            labels.extend((1, 0))
            weights.extend((state_weight, state_weight))
            folds.extend((fold_by_state[state_id], fold_by_state[state_id]))
    if set(labels) != {0, 1}:
        raise ValueError("control pairs require both labels")
    return {
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "folds": np.asarray(folds, dtype=np.int8),
        "state_count": states_with_pairs,
    }


def _candidate_matrix(candidates: list[dict[str, Any]]) -> Any:
    import numpy as np

    names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in candidates],
        dtype=np.float32,
    )


def _pair_matrix(
    candidate_values: Any,
    pair_table: dict[str, Any],
    input_specs: tuple[tuple[str, str], ...],
) -> Any:
    import numpy as np

    registered = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    feature_index = {name: index for index, name in enumerate(registered)}
    left = pair_table["left"]
    right = pair_table["right"]
    values = np.empty((len(left), len(input_specs)), dtype=np.float32)
    for output_index, (mode, name) in enumerate(input_specs):
        index = feature_index[name]
        if mode == "delta":
            values[:, output_index] = (
                candidate_values[left, index] - candidate_values[right, index]
            )
        elif mode == "shared":
            values[:, output_index] = 0.5 * (
                candidate_values[left, index] + candidate_values[right, index]
            )
        else:
            raise ValueError(f"unsupported pair mode: {mode}")
    return values


def _fit_registered_model(
    values: Any, labels: Any, weights: Any, parameters: dict[str, Any]
) -> Any:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    if set(map(int, np.unique(labels))) != {0, 1}:
        raise ValueError("pairwise training split requires both labels")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized *= len(normalized) / float(np.sum(normalized))
    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(values, labels, sample_weight=normalized)
    return estimator


def _cross_validated_predictions(
    *,
    pair_values: Any,
    pair_table: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    fold_by_state: dict[str, int],
    fold_count: int,
    input_specs: tuple[tuple[str, str], ...],
    parameters: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    import numpy as np

    predictions: dict[str, str] = {}
    fold_rows = []
    correct_weight = 0.0
    total_weight = 0.0
    for fold_index in range(fold_count):
        train = pair_table["folds"] != fold_index
        validate = pair_table["folds"] == fold_index
        if not bool(np.any(train)) or not bool(np.any(validate)):
            raise ValueError(f"empty Stage 4 pair split: fold {fold_index}")
        estimator = _fit_registered_model(
            pair_values[train],
            pair_table["labels"][train],
            pair_table["weights"][train],
            parameters,
        )
        probabilities = estimator.predict_proba(pair_values[validate])[:, 1]
        labels = pair_table["labels"][validate]
        weights = pair_table["weights"][validate]
        correct = (probabilities >= 0.5) == (labels == 1)
        fold_correct_weight = float(np.sum(weights[correct]))
        fold_total_weight = float(np.sum(weights))
        correct_weight += fold_correct_weight
        total_weight += fold_total_weight
        fold_state_ids = sorted(
            state_id
            for state_id, assigned_fold in fold_by_state.items()
            if assigned_fold == fold_index
        )
        for state_id in fold_state_ids:
            chosen = _select_model(grouped[state_id], estimator, input_specs)
            predictions[state_id] = str(chosen["candidate_id"])
        fold_rows.append(
            {
                "fold_index": fold_index,
                "training_pair_count": int(np.sum(train)),
                "validation_pair_count": int(np.sum(validate)),
                "validation_state_count": len(fold_state_ids),
                "pairwise_accuracy": fold_correct_weight / fold_total_weight,
            }
        )
    if set(predictions) != set(grouped):
        raise ValueError("OOF prediction coverage differs from Stage 4 states")
    return predictions, {
        "pairwise_accuracy": correct_weight / total_weight,
        "pairwise_validation_weight": total_weight,
        "folds": fold_rows,
    }


def _frozen_predictions(
    grouped: dict[str, list[dict[str, Any]]], bundle_path: Path
) -> dict[str, str]:
    bundle = load_controller_bundle(bundle_path)
    if str(bundle.manifest.get("default_controller")) != "v2-full":
        raise ValueError("frozen Stage 4 anchor is not v2-full")
    model = bundle.main_models["realized_dynamic"]
    result = {}
    for state_id, state in sorted(grouped.items()):
        rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        index, _, _ = score_online_candidates(rows, model)
        result[state_id] = str(state[index]["candidate_id"])
    return result


def _selection_records(
    model_id: str,
    predictions: dict[str, str],
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for state_id, state in sorted(grouped.items()):
        by_id = {str(row["candidate_id"]): row for row in state}
        selected = by_id[predictions[state_id]]
        ranked = sorted(
            state,
            key=lambda row: (-float(row["quality_score"]), str(row["candidate_id"])),
        )
        best = float(ranked[0]["quality_score"])
        worst = min(float(row["quality_score"]) for row in state)
        selected_quality = float(selected["quality_score"])
        regret = max(0.0, best - selected_quality)
        quality_range = best - worst
        records.append(
            {
                "schema": "lns2.stride.stage4_oof_prediction.v1",
                "model_id": model_id,
                "state_id": state_id,
                "selected_candidate_id": str(selected["candidate_id"]),
                "fold_index": int(selected["fold_index"]),
                "map_id": str(selected["map_id"]),
                "layout_family": str(selected["layout_family"]),
                "source_policy": str(selected["source_policy"]),
                "agent_band": str(selected["agent_band"]),
                "decision_stage": str(selected["decision_stage"]),
                "candidate_count": len(state),
                "exact_best": selected_quality + 1e-12 >= best,
                "top3_hit": str(selected["candidate_id"])
                in {str(row["candidate_id"]) for row in ranked[:3]},
                "quality_score": selected_quality,
                "quality_regret": regret,
                "normalized_quality_regret": (
                    regret / quality_range if quality_range > 1e-12 else 0.0
                ),
                "feasible_rate": float(selected["feasible_rate"]),
                "progress_rate": float(selected["progress_rate"]),
                "mean_conflict_reduction": float(
                    selected["mean_conflict_reduction"]
                ),
                "mean_reduction_ratio": float(selected["mean_reduction_ratio"]),
                "structural_score": float(selected["structural_score"]),
            }
        )
    return records


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize empty Stage 4 predictions")
    mean = lambda name: statistics.fmean(float(row[name]) for row in records)
    return {
        "state_count": len(records),
        "exact_best_rate": mean("exact_best"),
        "top3_hit_rate": mean("top3_hit"),
        "mean_quality_score": mean("quality_score"),
        "mean_quality_regret": mean("quality_regret"),
        "mean_normalized_quality_regret": mean("normalized_quality_regret"),
        "mean_feasible_rate": mean("feasible_rate"),
        "mean_progress_rate": mean("progress_rate"),
        "mean_conflict_reduction": mean("mean_conflict_reduction"),
        "mean_reduction_ratio": mean("mean_reduction_ratio"),
        "mean_structural_score": mean("structural_score"),
    }


def _model_metrics(
    records: list[dict[str, Any]], pairwise: dict[str, Any] | None
) -> dict[str, Any]:
    result = _mean_metrics(records)
    result["pairwise_accuracy"] = (
        None if pairwise is None else float(pairwise["pairwise_accuracy"])
    )
    result["pairwise_folds"] = [] if pairwise is None else pairwise["folds"]
    result["by_fold"] = {
        str(fold): _mean_metrics(
            [row for row in records if int(row["fold_index"]) == fold]
        )
        for fold in sorted({int(row["fold_index"]) for row in records})
    }
    subgroups = []
    for field in ("source_policy", "agent_band", "decision_stage", "layout_family"):
        for value in sorted({str(row[field]) for row in records}):
            selected = [row for row in records if str(row[field]) == value]
            subgroups.append(
                {"field": field, "value": value, **_mean_metrics(selected)}
            )
    result["subgroups"] = subgroups
    return result


def _promotion_evaluation(
    model: dict[str, Any],
    control: dict[str, Any],
    frozen: dict[str, Any],
    gates_config: dict[str, Any],
) -> dict[str, Any]:
    normalized = float(model["mean_normalized_quality_regret"])
    control_normalized = float(control["mean_normalized_quality_regret"])
    frozen_normalized = float(frozen["mean_normalized_quality_regret"])
    absolute_improvement = control_normalized - normalized
    relative_improvement = absolute_improvement / max(control_normalized, 1e-12)
    fold_wins = sum(
        float(model["by_fold"][fold]["mean_normalized_quality_regret"])
        + 1e-12
        < float(control["by_fold"][fold]["mean_normalized_quality_regret"])
        for fold in model["by_fold"]
    )

    minimum_group = int(gates_config["subgroup_minimum_state_count"])
    subgroup_tolerance = float(
        gates_config["maximum_subgroup_normalized_regret_degradation"]
    )
    control_groups = {
        (str(row["field"]), str(row["value"])): row
        for row in control["subgroups"]
    }
    frozen_groups = {
        (str(row["field"]), str(row["value"])): row
        for row in frozen["subgroups"]
    }
    subgroup_rows = []
    for row in model["subgroups"]:
        if int(row["state_count"]) < minimum_group:
            continue
        key = (str(row["field"]), str(row["value"]))
        value = float(row["mean_normalized_quality_regret"])
        control_delta = value - float(
            control_groups[key]["mean_normalized_quality_regret"]
        )
        frozen_delta = value - float(
            frozen_groups[key]["mean_normalized_quality_regret"]
        )
        subgroup_rows.append(
            {
                "field": key[0],
                "value": key[1],
                "state_count": int(row["state_count"]),
                "control_regret_delta": control_delta,
                "frozen_regret_delta": frozen_delta,
                "passed": max(control_delta, frozen_delta) <= subgroup_tolerance + 1e-12,
            }
        )

    noninferiority = float(
        gates_config["frozen_anchor_normalized_regret_noninferiority_tolerance"]
    )
    best_tolerance = float(
        gates_config["exact_best_rate_noninferiority_tolerance"]
    )
    top3_tolerance = float(
        gates_config["top3_hit_rate_noninferiority_tolerance"]
    )
    gates = {
        "pairwise_accuracy": float(model["pairwise_accuracy"])
        + 1e-12
        >= float(gates_config["pairwise_accuracy_min"]),
        "quality_vs_control_normalized_regret": (
            relative_improvement + 1e-12
            >= float(
                gates_config[
                    "quality_vs_control_relative_normalized_regret_improvement_min"
                ]
            )
            or absolute_improvement + 1e-12
            >= float(
                gates_config[
                    "quality_vs_control_absolute_normalized_regret_improvement_alternative"
                ]
            )
        ),
        "frozen_anchor_normalized_regret_noninferior": normalized
        <= frozen_normalized + noninferiority + 1e-12,
        "exact_best_rate_noninferior": float(model["exact_best_rate"])
        + best_tolerance
        + 1e-12
        >= max(float(control["exact_best_rate"]), float(frozen["exact_best_rate"])),
        "top3_hit_rate_noninferior": float(model["top3_hit_rate"])
        + top3_tolerance
        + 1e-12
        >= max(float(control["top3_hit_rate"]), float(frozen["top3_hit_rate"])),
        "minimum_fold_wins_vs_control": fold_wins
        >= int(gates_config["minimum_fold_wins_vs_control"]),
        "subgroup_regret_noninferior": all(row["passed"] for row in subgroup_rows),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "control_absolute_normalized_regret_improvement": absolute_improvement,
        "control_relative_normalized_regret_improvement": relative_improvement,
        "fold_wins_vs_control": fold_wins,
        "subgroup_comparisons": subgroup_rows,
    }


def _export_selected_model(
    *,
    output: Path,
    controller_id: str,
    variant: str,
    feature_names: tuple[str, ...],
    input_specs: tuple[tuple[str, str], ...],
    estimator: Any,
    candidates: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    parameters: dict[str, Any],
    training_pair_count: int,
    training_state_count: int,
) -> dict[str, Any]:
    model_root = output / "selected_model"
    sklearn_model = PairwiseModel(
        profile="realized_dynamic",
        feature_names=list(feature_names),
        estimator=estimator,
    )
    sklearn_path = model_root / "sklearn.pkl"
    _atomic_pickle(sklearn_path, sklearn_model)
    sklearn_hash = sha256_file(sklearn_path)
    compact_payload = compact_portable_payload(
        _portable_payload(sklearn_model, sklearn_hash)
    )
    compact_path = model_root / "main__realized_dynamic.json"
    _write_json(compact_path, compact_payload)
    compact_model = load_compact_model(compact_payload)

    mismatch_count = 0
    maximum_score_delta = 0.0
    for state in grouped.values():
        rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        reference_index, reference_scores, _ = score_online_candidates(
            rows, sklearn_model
        )
        compact_index, compact_scores, _ = score_online_candidates(rows, compact_model)
        mismatch_count += int(reference_index != compact_index)
        maximum_score_delta = max(
            maximum_score_delta,
            *(abs(left - right) for left, right in zip(reference_scores, compact_scores)),
        )
    equivalence = {
        "state_count": len(grouped),
        "selection_mismatch_count": mismatch_count,
        "maximum_score_delta": maximum_score_delta,
        "passed": mismatch_count == 0 and maximum_score_delta <= 1e-10,
    }
    manifest = {
        "schema": STRIDE_STAGE4_MODEL_SCHEMA,
        "controller_id": controller_id,
        "variant": variant,
        "scientific_status": (
            "stage5_eligible" if equivalence["passed"] else "portable_equivalence_failed"
        ),
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "base_feature_names": list(feature_names),
        "base_feature_count": len(feature_names),
        "input_features": [
            {"mode": mode, "name": name} for mode, name in input_specs
        ],
        "input_dimension": len(input_specs),
        "model_parameters": parameters,
        "training_state_count": training_state_count,
        "training_pair_count": training_pair_count,
        "feature_ranges": _feature_ranges(candidates, list(feature_names)),
        "artifacts": {
            "sklearn": "sklearn.pkl",
            "sklearn_sha256": sklearn_hash,
            "portable": "main__realized_dynamic.json",
            "portable_sha256": sha256_file(compact_path),
            "portable_semantic_fingerprint": compact_payload[
                "semantic_fingerprint"
            ],
        },
        "equivalence": equivalence,
    }
    _write_json(model_root / "model_manifest.json", manifest)
    return manifest


def run_stride_stage4_training(
    *,
    config_path: Path,
    protocol_report_path: Path,
    output: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Run registered five-fold control, quality, and feature-ablation training."""

    import numpy as np

    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4 training-config schema")
    protocol_report = _read_json(protocol_report_path)
    if (
        protocol_report.get("schema") != STRIDE_STAGE4_PROTOCOL_SCHEMA
        or protocol_report.get("passed") is not True
        or protocol_report.get("label_outcomes_read") is not False
        or str(protocol_report.get("config_sha256")) != _fingerprint(config)
    ):
        raise ValueError("Stage 4 protocol report is not valid for this config")
    protocol = dict(config["protocol"])
    if protocol.get("formal_ood_data_allowed") is not False:
        raise ValueError("Stage 4 may not read formal OOD data")

    project_root = project_root.resolve()
    labels = _project_path(project_root, str(config["labels"]))
    aggregate_path = labels / "candidate_aggregates.jsonl"
    quality_pair_path = labels / "dominance_pairs.jsonl"
    manifest_path = protocol_report_path.parent / "fold_manifest.jsonl"
    if sha256_file(manifest_path) != str(protocol_report["fold_manifest_sha256"]):
        raise ValueError("Stage 4 fold-manifest SHA256 mismatch")
    manifest_rows = _read_jsonl(manifest_path)
    candidates, grouped = _load_candidates(aggregate_path, manifest_rows)
    fold_by_state = {
        str(row["state_id"]): int(row["fold_index"]) for row in manifest_rows
    }
    candidate_index = {
        (str(row["state_id"]), str(row["candidate_id"])): int(
            row["candidate_index"]
        )
        for row in candidates
    }
    candidate_values = _candidate_matrix(candidates)
    quality_pairs = _pair_table_from_quality(
        quality_pair_path, candidate_index, fold_by_state
    )
    control_pairs = _pair_table_from_control(grouped, fold_by_state)
    variants = _variant_specifications(dict(protocol["feature_variants"]))
    if "full" not in variants:
        raise ValueError("Stage 4 requires the full feature variant")
    parameters = dict(config["model_parameters"])
    fold_count = int(config["fold_count"])

    frozen_bundle = _project_path(
        project_root,
        str(dict(protocol["models"])["frozen_anchor"]["bundle"]),
    )
    frozen_id = str(dict(protocol["models"])["frozen_anchor"]["id"])
    control_id = str(dict(protocol["models"])["control"]["id"])
    quality_id = str(dict(protocol["models"])["quality"]["id"])
    predictions: dict[str, dict[str, str]] = {
        frozen_id: _frozen_predictions(grouped, frozen_bundle)
    }
    pairwise_results: dict[str, dict[str, Any] | None] = {frozen_id: None}

    full_names, full_specs = variants["full"]
    control_values = _pair_matrix(candidate_values, control_pairs, full_specs)
    control_predictions, control_pairwise = _cross_validated_predictions(
        pair_values=control_values,
        pair_table=control_pairs,
        grouped=grouped,
        fold_by_state=fold_by_state,
        fold_count=fold_count,
        input_specs=full_specs,
        parameters=parameters,
    )
    predictions[control_id] = control_predictions
    pairwise_results[control_id] = control_pairwise
    del control_values
    gc.collect()

    variant_dimensions = {}
    for variant in ["full", *sorted(set(variants) - {"full"})]:
        names, specs = variants[variant]
        model_id = quality_id if variant == "full" else f"{quality_id}/{variant}"
        values = _pair_matrix(candidate_values, quality_pairs, specs)
        variant_predictions, pairwise = _cross_validated_predictions(
            pair_values=values,
            pair_table=quality_pairs,
            grouped=grouped,
            fold_by_state=fold_by_state,
            fold_count=fold_count,
            input_specs=specs,
            parameters=parameters,
        )
        predictions[model_id] = variant_predictions
        pairwise_results[model_id] = pairwise
        variant_dimensions[variant] = {
            "base_feature_count": len(names),
            "input_dimension": len(specs),
        }
        del values
        gc.collect()

    records_by_model = {
        model_id: _selection_records(model_id, rows, grouped)
        for model_id, rows in predictions.items()
    }
    metrics = {
        model_id: _model_metrics(records_by_model[model_id], pairwise_results[model_id])
        for model_id in predictions
    }
    control_metrics = metrics[control_id]
    frozen_metrics = metrics[frozen_id]
    gates_config = dict(protocol["promotion_gates"])
    promotion = {}
    for variant in ["full", *sorted(set(variants) - {"full"})]:
        model_id = quality_id if variant == "full" else f"{quality_id}/{variant}"
        promotion[variant] = _promotion_evaluation(
            metrics[model_id], control_metrics, frozen_metrics, gates_config
        )

    offline_passed = bool(promotion["full"]["passed"])
    selected_variant = None
    if offline_passed:
        passing = [name for name, row in promotion.items() if bool(row["passed"])]
        best_regret = min(
            float(
                metrics[
                    quality_id if name == "full" else f"{quality_id}/{name}"
                ]["mean_normalized_quality_regret"]
            )
            for name in passing
        )
        tolerance = float(
            dict(protocol["feature_selection"])[
                "normalized_regret_best_tolerance"
            ]
        )
        within = [
            name
            for name in passing
            if float(
                metrics[
                    quality_id if name == "full" else f"{quality_id}/{name}"
                ]["mean_normalized_quality_regret"]
            )
            <= best_regret + tolerance + 1e-12
        ]
        selected_variant = min(
            within,
            key=lambda name: (
                variant_dimensions[name]["base_feature_count"],
                float(
                    metrics[
                        quality_id if name == "full" else f"{quality_id}/{name}"
                    ]["mean_normalized_quality_regret"]
                ),
                name,
            ),
        )

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_records = [
        row
        for model_id in sorted(records_by_model)
        for row in records_by_model[model_id]
    ]
    _write_jsonl(output / "oof_predictions.jsonl", all_records)

    selected_manifest = None
    if selected_variant is not None:
        selected_names, selected_specs = variants[selected_variant]
        selected_values = _pair_matrix(
            candidate_values, quality_pairs, selected_specs
        )
        estimator = _fit_registered_model(
            selected_values,
            quality_pairs["labels"],
            quality_pairs["weights"],
            parameters,
        )
        selected_manifest = _export_selected_model(
            output=output,
            controller_id=quality_id,
            variant=selected_variant,
            feature_names=selected_names,
            input_specs=selected_specs,
            estimator=estimator,
            candidates=candidates,
            grouped=grouped,
            parameters=parameters,
            training_pair_count=len(quality_pairs["labels"]),
            training_state_count=int(quality_pairs["state_count"]),
        )
        del selected_values
        gc.collect()

    portable_passed = bool(
        selected_manifest is not None
        and dict(selected_manifest["equivalence"])["passed"]
    )
    report = {
        "schema": STRIDE_STAGE4_TRAINING_SCHEMA,
        "passed": offline_passed and portable_passed,
        "stage5_eligible": offline_passed and portable_passed,
        "formal_ood_data_read": False,
        "test_data_read": False,
        "controller_ids": {
            "frozen_anchor": frozen_id,
            "same_data_control": control_id,
            "quality": quality_id,
        },
        "state_count": len(grouped),
        "candidate_count": len(candidates),
        "quality_oriented_pair_count": len(quality_pairs["labels"]),
        "quality_pair_state_count": int(quality_pairs["state_count"]),
        "control_oriented_pair_count": len(control_pairs["labels"]),
        "control_pair_state_count": int(control_pairs["state_count"]),
        "fold_count": fold_count,
        "model_parameters": parameters,
        "feature_variants": variant_dimensions,
        "metrics": metrics,
        "promotion": promotion,
        "offline_full_quality_passed": offline_passed,
        "selected_variant": selected_variant,
        "selected_model": selected_manifest,
        "portable_equivalence_passed": portable_passed,
        "inputs": {
            "config_sha256": _fingerprint(config),
            "protocol_report_sha256": sha256_file(protocol_report_path),
            "fold_manifest_sha256": sha256_file(manifest_path),
            "candidate_aggregates_sha256": sha256_file(aggregate_path),
            "dominance_pairs_sha256": sha256_file(quality_pair_path),
            "frozen_controller_manifest_sha256": sha256_file(
                frozen_bundle / "controller_manifest.json"
            ),
        },
        "artifacts": {
            "oof_predictions": "oof_predictions.jsonl",
            "oof_predictions_sha256": sha256_file(
                output / "oof_predictions.jsonl"
            ),
        },
    }
    _write_json(output / "stage4_training_report.json", report)
    return report


__all__ = [
    "STRIDE_STAGE4_FOLD_SCHEMA",
    "STRIDE_STAGE4_PROTOCOL_SCHEMA",
    "STRIDE_STAGE4_TRAINING_SCHEMA",
    "STRIDE_STAGE4_TRAINING_CONFIG_SCHEMA",
    "prepare_stride_stage4_protocol",
    "run_stride_stage4_training",
]
