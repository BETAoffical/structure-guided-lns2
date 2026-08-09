from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_quality_v2 import (
    STRIDE_QUALITY_V2_CANDIDATE_SCHEMA,
    STRIDE_QUALITY_V2_CONTROLLER_ID,
    STRIDE_QUALITY_V2_LABEL_SCHEMA,
    STRIDE_QUALITY_V2_STRUCTURE_WEIGHT,
)
from experiments.stride_lns import validate_post_structure_metrics
from lns2_selector.runtime.artifact_validation import finite_number


STRIDE_STAGE3_LABEL_AUDIT_CONFIG_SCHEMA = (
    "lns2.stride.stage3_label_audit_config.v1"
)
STRIDE_STAGE3_LABEL_AUDIT_SCHEMA = "lns2.stride.stage3_label_audit.v1"


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _formal_ood_ids(config: dict[str, Any]) -> set[str]:
    rows = config.get("cases", config.get("benchmarks"))
    if not isinstance(rows, list):
        raise ValueError("formal OOD config must contain cases or benchmarks")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("formal OOD entries must be objects")
        value = row.get("benchmark_id", row.get("id"))
        if not isinstance(value, str) or not value:
            raise ValueError("formal OOD entries require benchmark_id or id")
        if value in result:
            raise ValueError(f"duplicate formal OOD map ID: {value}")
        result.add(value)
    return result


def run_stride_stage3_label_audit(
    *, config_path: Path, output: Path, project_root: Path
) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema") != STRIDE_STAGE3_LABEL_AUDIT_CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 3 label-audit config schema")

    project_root = project_root.resolve()
    labels = _project_path(project_root, str(config["labels"]))
    selection_paths = [
        _project_path(project_root, str(value))
        for value in config.get("selections", [])
    ]
    if not selection_paths:
        raise ValueError("STRIDE Stage 3 audit requires selections")
    formal_ood_config = _project_path(
        project_root, str(config["formal_ood_config"])
    )
    aggregate_path = labels / "candidate_aggregates.jsonl"
    pair_path = labels / "dominance_pairs.jsonl"
    summary_path = labels / "label_build_summary.json"
    summary = _read_json(summary_path)

    expected_state_count = int(config["expected_state_count"])
    expected_policy_counts = {
        str(name): int(value)
        for name, value in dict(config["expected_state_count_per_policy"]).items()
    }
    expected_feature_dimension = int(config["expected_feature_dimension"])
    min_map_count = int(config["min_map_count"])
    required_split_map_counts = {
        str(name): int(value)
        for name, value in dict(config.get("required_split_map_counts", {})).items()
    }
    one_state_splits = {
        str(value) for value in config.get("one_state_per_episode_splits", [])
    }

    selected_rows: list[dict[str, Any]] = []
    for path in selection_paths:
        selected_rows.extend(_read_jsonl(path))
    selected_ids = [str(row["state_id"]) for row in selected_rows]
    selected_id_set = set(selected_ids)
    duplicate_selected_state_count = len(selected_ids) - len(selected_id_set)
    selected_metadata = {
        str(row["state_id"]): {
            "map_id": str(row["map_id"]),
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "decision_stage": str(row["decision_stage"]),
            "agent_count": int(row["agent_count"]),
            "before_conflicts": int(row["before_conflicts"]),
        }
        for row in selected_rows
    }
    extension_episode_counts: Counter[tuple[str, str]] = Counter()
    missing_episode_id_count = 0
    for row in selected_rows:
        if str(row["split"]) not in one_state_splits:
            continue
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            missing_episode_id_count += 1
        else:
            extension_episode_counts[(str(row["source_policy"]), episode_id)] += 1
    max_extension_states_per_episode = max(
        extension_episode_counts.values(), default=0
    )

    state_metadata: dict[str, dict[str, Any]] = {}
    candidate_ids: defaultdict[str, set[str]] = defaultdict(set)
    aggregate_row_count = 0
    duplicate_candidate_row_count = 0
    feature_dimension_errors = 0
    feature_schema_errors = 0
    nonfinite_feature_errors = 0
    aggregate_scientific_errors = 0
    aggregate_schema_errors = 0
    aggregate_metadata_errors = 0
    aggregate_parse_error: str | None = None
    try:
        aggregate_rows = _read_jsonl(aggregate_path)
    except (OSError, ValueError) as error:
        aggregate_rows = []
        aggregate_parse_error = f"{type(error).__name__}: {error}"
        if "non-finite JSON constant" in str(error):
            nonfinite_feature_errors += 1
    for row in aggregate_rows:
        aggregate_row_count += 1
        state_id = str(row["state_id"])
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids[state_id]:
            duplicate_candidate_row_count += 1
        candidate_ids[state_id].add(candidate_id)
        if (
            row.get("schema") != STRIDE_QUALITY_V2_CANDIDATE_SCHEMA
            or row.get("label_schema") != STRIDE_QUALITY_V2_LABEL_SCHEMA
        ):
            aggregate_schema_errors += 1
        features = row.get("features")
        if not isinstance(features, dict) or len(features) != expected_feature_dimension:
            feature_dimension_errors += 1
        if not isinstance(features, dict) or set(features) != set(
            PROFILE_FEATURE_NAMES["realized_dynamic"]
        ):
            feature_schema_errors += 1
        if not isinstance(features, dict) or any(
            not finite_number(value) for value in features.values()
        ):
            nonfinite_feature_errors += 1
        scientific_valid = (
            type(row.get("before_conflicts")) is int
            and int(row["before_conflicts"]) > 0
            and type(row.get("trial_count")) is int
            and int(row["trial_count"]) == int(config["expected_trials_per_candidate"])
            and isinstance(row.get("pp_seeds"), list)
            and len(row["pp_seeds"]) == int(config["expected_trials_per_candidate"])
            and all(type(seed) is int for seed in row["pp_seeds"])
            and len(set(row["pp_seeds"])) == len(row["pp_seeds"])
            and all(
                finite_number(row.get(name))
                for name in (
                    "feasible_rate",
                    "progress_rate",
                    "mean_conflict_reduction",
                    "mean_reduction_ratio",
                    "reduction_std",
                    "quality_score",
                    "structural_score",
                )
            )
            and 0.0 <= float(row.get("feasible_rate", -1.0)) <= 1.0
            and 0.0 <= float(row.get("progress_rate", -1.0)) <= 1.0
            and float(row.get("reduction_std", -1.0)) >= 0.0
            and 0.0 <= float(row.get("structural_score", -1.0)) <= 1.0
        )
        try:
            validate_post_structure_metrics(row.get("mean_post_structure"))
        except ValueError:
            scientific_valid = False
        if scientific_valid:
            expected_ratio = float(row["mean_conflict_reduction"]) / int(
                row["before_conflicts"]
            )
            expected_quality = expected_ratio - (
                STRIDE_QUALITY_V2_STRUCTURE_WEIGHT
                * float(row["structural_score"])
            )
            scientific_valid = (
                row["pp_seeds"] == sorted(row["pp_seeds"])
                and math.isclose(
                    float(row["mean_reduction_ratio"]),
                    expected_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    float(row["quality_score"]),
                    expected_quality,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        if not scientific_valid:
            aggregate_scientific_errors += 1
        current = {
            "map_id": str(row["map_id"]),
            "split": str(row["split"]),
            "source_policy": str(row["source_policy"]),
            "decision_stage": str(row["decision_stage"]),
            "agent_count": int(row["agent_count"]),
            "before_conflicts": int(row["before_conflicts"]),
        }
        if state_id in state_metadata and state_metadata[state_id] != current:
            aggregate_metadata_errors += 1
        state_metadata[state_id] = current

    labelled_ids = set(state_metadata)
    selection_metadata_mismatch_count = sum(
        selected_metadata.get(state_id) != metadata
        for state_id, metadata in state_metadata.items()
    )
    policy_counts = Counter(
        metadata["source_policy"] for metadata in state_metadata.values()
    )
    map_counts = Counter(metadata["map_id"] for metadata in state_metadata.values())
    split_counts = Counter(metadata["split"] for metadata in state_metadata.values())
    split_maps: defaultdict[str, set[str]] = defaultdict(set)
    agent_counts = Counter(metadata["agent_count"] for metadata in state_metadata.values())
    decision_stage_counts = Counter(
        metadata["decision_stage"] for metadata in state_metadata.values()
    )
    for metadata in state_metadata.values():
        split_maps[metadata["split"]].add(metadata["map_id"])
    candidate_count_distribution = Counter(
        len(values) for values in candidate_ids.values()
    )

    formal_ids = _formal_ood_ids(_read_json(formal_ood_config))
    exact_ood_overlap = sorted(set(map_counts) & formal_ids)
    conservative_ood_overlap = sorted(
        {
            (map_id, formal_id)
            for map_id in map_counts
            for formal_id in formal_ids
            if map_id in formal_id or formal_id in map_id
        }
    )

    pair_rows: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    state_weight = defaultdict(float)
    oriented_row_count = 0
    pair_schema_errors = 0
    pair_metadata_errors = 0
    invalid_label_count = 0
    invalid_weight_count = 0
    bad_candidate_reference_count = 0
    self_pair_count = 0
    for row in _read_jsonl(pair_path):
        oriented_row_count += 1
        state_id = str(row["state_id"])
        left = str(row["left_candidate_id"])
        right = str(row["right_candidate_id"])
        raw_label = row.get("label")
        label = raw_label if type(raw_label) is int else -1
        raw_weight = row.get("sample_weight")
        weight = float(raw_weight) if finite_number(raw_weight) else math.nan
        if row.get("schema") != STRIDE_QUALITY_V2_LABEL_SCHEMA:
            pair_schema_errors += 1
        if state_id not in state_metadata or (
            str(row["map_id"]) != state_metadata[state_id]["map_id"]
            or str(row["split"]) != state_metadata[state_id]["split"]
        ):
            pair_metadata_errors += 1
        if label not in (0, 1):
            invalid_label_count += 1
        if not math.isfinite(weight) or weight <= 0.0:
            invalid_weight_count += 1
        if left not in candidate_ids[state_id] or right not in candidate_ids[state_id]:
            bad_candidate_reference_count += 1
        if left == right:
            self_pair_count += 1
        state_weight[state_id] += weight
        low, high = sorted((left, right))
        pair_rows[(state_id, low, high)].append(row)

    invalid_oriented_pair_count = 0
    for rows in pair_rows.values():
        if len(rows) != 2 or {int(row["label"]) for row in rows} != {0, 1}:
            invalid_oriented_pair_count += 1
            continue
        first, second = rows
        if (
            str(first["left_candidate_id"]) != str(second["right_candidate_id"])
            or str(first["right_candidate_id"]) != str(second["left_candidate_id"])
            or not math.isclose(
                float(first["sample_weight"]),
                float(second["sample_weight"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            invalid_oriented_pair_count += 1
    bad_state_weight_count = sum(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in state_weight.values()
    )

    trial_files = [
        _project_path(project_root, str(value))
        for value in config.get("trial_sources", [])
    ]
    registered_trial_hashes = [
        str(value) for value in summary.get("trial_sha256", [])
    ]
    trial_files_exist = bool(trial_files) and all(path.is_file() for path in trial_files)
    actual_trial_hashes = (
        [sha256_file(path) for path in trial_files] if trial_files_exist else []
    )

    gates = {
        "selected_state_count": len(selected_id_set) == expected_state_count,
        "duplicate_selected_states_zero": duplicate_selected_state_count == 0,
        "labelled_state_count": len(labelled_ids) == expected_state_count,
        "selection_state_match": labelled_ids == selected_id_set,
        "selection_metadata_match": selection_metadata_mismatch_count == 0,
        "source_policy_balance": dict(policy_counts) == expected_policy_counts,
        "minimum_map_coverage": len(map_counts) >= min_map_count,
        "required_split_map_coverage": all(
            len(split_maps[name]) >= count
            for name, count in required_split_map_counts.items()
        ),
        "one_state_per_extension_episode": (
            missing_episode_id_count == 0 and max_extension_states_per_episode <= 1
        ),
        "formal_ood_overlap_zero": (
            not exact_ood_overlap and not conservative_ood_overlap
        ),
        "aggregate_json_valid": aggregate_parse_error is None,
        "aggregate_rows_unique": duplicate_candidate_row_count == 0,
        "aggregate_schema_valid": aggregate_schema_errors == 0,
        "aggregate_metadata_valid": aggregate_metadata_errors == 0,
        "feature_dimension_valid": feature_dimension_errors == 0,
        "feature_schema_registered": (
            expected_feature_dimension
            == len(PROFILE_FEATURE_NAMES["realized_dynamic"])
            and feature_schema_errors == 0
        ),
        "aggregate_features_finite": nonfinite_feature_errors == 0,
        "aggregate_scientific_fields_valid": aggregate_scientific_errors == 0,
        "all_states_have_pairs": set(state_weight) == labelled_ids,
        "state_sample_weight_one": bad_state_weight_count == 0,
        "pair_schema_valid": pair_schema_errors == 0,
        "pair_metadata_valid": pair_metadata_errors == 0,
        "pair_labels_valid": invalid_label_count == 0,
        "pair_weights_valid": invalid_weight_count == 0,
        "pair_candidates_valid": bad_candidate_reference_count == 0,
        "pair_self_comparisons_zero": self_pair_count == 0,
        "pair_orientations_valid": invalid_oriented_pair_count == 0,
        "summary_controller_valid": (
            summary.get("controller_id") == STRIDE_QUALITY_V2_CONTROLLER_ID
        ),
        "summary_label_schema_valid": (
            summary.get("label_schema") == STRIDE_QUALITY_V2_LABEL_SCHEMA
        ),
        "summary_counts_match": (
            int(summary.get("state_count", -1)) == len(labelled_ids)
            and int(summary.get("states_with_pairs", -1)) == len(state_weight)
            and int(summary.get("candidate_count", -1)) == aggregate_row_count
            and int(summary.get("dominance_pair_count", -1)) == len(pair_rows)
            and int(summary.get("oriented_training_row_count", -1))
            == oriented_row_count
        ),
        "summary_trial_count_valid": (
            int(summary.get("trials_per_candidate", -1))
            == int(config["expected_trials_per_candidate"])
        ),
        "runtime_excluded_from_label": summary.get("runtime_used_in_label") is False,
        "trial_source_hashes_valid": (
            trial_files_exist
            and len(actual_trial_hashes) == len(registered_trial_hashes)
            and actual_trial_hashes == registered_trial_hashes
        ),
    }
    try:
        config_name = config_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        config_name = config_path.name
    report = {
        "schema": STRIDE_STAGE3_LABEL_AUDIT_SCHEMA,
        "config": config_name,
        "config_sha256": _fingerprint(config),
        "passed": all(gates.values()),
        "gates": gates,
        "state_count": len(labelled_ids),
        "candidate_count": aggregate_row_count,
        "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "source_policy_state_counts": dict(sorted(policy_counts.items())),
        "map_count": len(map_counts),
        "map_state_counts": dict(sorted(map_counts.items())),
        "split_state_counts": dict(sorted(split_counts.items())),
        "split_map_counts": {
            name: len(values) for name, values in sorted(split_maps.items())
        },
        "agent_count_state_counts": dict(sorted(agent_counts.items())),
        "decision_stage_state_counts": dict(sorted(decision_stage_counts.items())),
        "dominance_pair_count": len(pair_rows),
        "oriented_training_row_count": oriented_row_count,
        "state_weight_min": min(state_weight.values(), default=0.0),
        "state_weight_max": max(state_weight.values(), default=0.0),
        "formal_ood_exact_overlap": exact_ood_overlap,
        "formal_ood_conservative_overlap": [
            {"label_map_id": left, "formal_map_id": right}
            for left, right in conservative_ood_overlap
        ],
        "max_extension_states_per_episode": max_extension_states_per_episode,
        "diagnostics": {
            "duplicate_selected_state_count": duplicate_selected_state_count,
            "selection_metadata_mismatch_count": selection_metadata_mismatch_count,
            "missing_extension_episode_id_count": missing_episode_id_count,
            "duplicate_candidate_row_count": duplicate_candidate_row_count,
            "feature_dimension_error_count": feature_dimension_errors,
            "feature_schema_error_count": feature_schema_errors,
            "nonfinite_feature_error_count": nonfinite_feature_errors,
            "aggregate_scientific_error_count": aggregate_scientific_errors,
            "aggregate_schema_error_count": aggregate_schema_errors,
            "aggregate_metadata_error_count": aggregate_metadata_errors,
            "aggregate_parse_error": aggregate_parse_error,
            "bad_state_weight_count": bad_state_weight_count,
            "pair_schema_error_count": pair_schema_errors,
            "pair_metadata_error_count": pair_metadata_errors,
            "invalid_label_count": invalid_label_count,
            "invalid_weight_count": invalid_weight_count,
            "bad_candidate_reference_count": bad_candidate_reference_count,
            "self_pair_count": self_pair_count,
            "invalid_oriented_pair_count": invalid_oriented_pair_count,
        },
        "artifact_sha256": {
            "candidate_aggregates": sha256_file(aggregate_path),
            "dominance_pairs": sha256_file(pair_path),
            "label_build_summary": sha256_file(summary_path),
        },
        "trial_source_sha256": actual_trial_hashes,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "stage3_label_audit_report.json", report)
    return report
