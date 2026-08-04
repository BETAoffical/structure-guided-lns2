from __future__ import annotations

import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _write_json, _write_jsonl
from experiments.stride_repairability import (
    _load_trials,
    _robust_winner,
    _supported_collection_audit_schemas,
)
from experiments.stride_maprank import _valid_sha256


DESIGN_SCHEMA = "lns2.stride.certguard_design.v1"
LABEL_CONFIG_SCHEMA = "lns2.stride.certguard_label_config.v1"
LABEL_SCHEMA = "lns2.stride.certguard_uncertainty_label.v1"
LABEL_REPORT_SCHEMA = "lns2.stride.certguard_label_build.v1"
TRAINING_CONFIG_SCHEMA = "lns2.stride.certguard_training_config.v1"
CONTROLLER_ID = "stride-certguard-v1"
LABEL_ARTIFACT_ID = "stride-certguard-labels-v1"
FEATURE_SCHEMA = "lns2.realized_features.v2"
BASE_FEATURE_DIMENSION = 124
PAIR_FEATURE_DIMENSION = 248


def _robust_rule() -> dict[str, Any]:
    return {
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "tie_epsilon": 1e-12,
    }


def validate_certguard_design(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "preregistered_after_maprank_development_failure_before_certainty_label_build"
        or config.get("design_id") != "stride-certguard-design-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_artifact_id") != LABEL_ARTIFACT_ID
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("STRIDE-CertGuard design identity changed")
    predecessor = dict(config.get("frozen_predecessor") or {})
    if (
        predecessor.get("controller_id") != "stride-maprank-v1"
        or predecessor.get("bundle")
        != "build/stride-maprank-training-v1/stride-maprank-v1"
        or predecessor.get("training_report")
        != "build/stride-maprank-training-v1/maprank_training_report.json"
        or predecessor.get("direction_semantics")
        != "paired_current_step_normalized_conflict_reduction"
        or any(
            not _valid_sha256(predecessor.get(field))
            for field in ("bundle_manifest_sha256", "training_report_sha256")
        )
    ):
        raise ValueError("STRIDE-CertGuard predecessor registration changed")
    if dict(config.get("architecture") or {}) != {
        "anchor": "frozen-v2-full",
        "candidate_pool": "frozen-maprank-base-plus-optional-topology-boundary",
        "direction_model": "frozen-maprank-direction-ranker",
        "uncertainty_model": "independent-symmetric-robust-pair-classifier",
        "decision_rule": "override_only_if_direction_and_certainty_gates_pass",
        "uncertainty_role": "abstention_only",
        "uncertainty_pair_representation": "absolute_candidate_delta_plus_pair_mean",
        "base_feature_dimension": 124,
        "uncertainty_input_dimension": 248,
    }:
        raise ValueError("STRIDE-CertGuard architecture changed")
    if dict(config.get("label_design") or {}) != {
        "schema": LABEL_SCHEMA,
        "target": "whether_the_unordered_candidate_pair_has_a_robust_current_step_winner",
        "positive": "robust_winner_exists_under_registered_16_seed_rule",
        "negative": "winner_is_uncertain_under_registered_16_seed_rule",
        "score": "conflict_reduction_normalized_by_before_conflicts",
        "trial_indices": list(range(16)),
        "paired_pp_seeds_within_state_and_trial_index": True,
        "robust_pair_rule": _robust_rule(),
        "sample_weighting": "equal_total_weight_per_state",
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
    }:
        raise ValueError("STRIDE-CertGuard label semantics changed")
    if list(config.get("evaluation_order") or ()) != [
        "nested_train_map_oof",
        "legacy_validation_descriptive",
        "paired_high_load_development_raw_ttf",
        "paired_fresh_map_raw_ttf_only_after_development_pass",
    ]:
        raise ValueError("STRIDE-CertGuard evaluation order changed")
    if config.get("high_load_role") != (
        "post_hoc_development_only_because_it_motivated_the_architecture"
    ) or config.get("fresh_map_role") != "first_unseen_end_to_end_evidence":
        raise ValueError("STRIDE-CertGuard evidence roles changed")
    forbidden = set(map(str, config.get("forbidden_inputs") or ()))
    if forbidden != {
        "repair_runtime",
        "time_to_feasible",
        "future_repair_rounds",
        "cost_to_go",
        "receding_q",
        "controller_outcome",
        "high_load_labels_for_training_or_threshold_calibration",
        "legacy_validation_labels_for_threshold_calibration",
        "test_data",
        "formal_ood_data",
    }:
        raise ValueError("STRIDE-CertGuard forbidden inputs changed")


def validate_certguard_label_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != LABEL_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_certainty_label_build"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_artifact_id") != LABEL_ARTIFACT_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("feature_dimension", -1)) != BASE_FEATURE_DIMENSION
        or config.get("registered_output") != "build/stride-certguard-labels-v1"
    ):
        raise ValueError("STRIDE-CertGuard label config identity changed")
    if tuple(map(int, config.get("trial_indices") or ())) != tuple(range(16)):
        raise ValueError("STRIDE-CertGuard labels require trials 0-15")
    if dict(config.get("robust_pair_rule") or {}) != _robust_rule():
        raise ValueError("STRIDE-CertGuard robust-pair rule changed")
    if (
        config.get("target")
        != "robust_current_step_winner_exists_for_unordered_candidate_pair"
        or config.get("pair_orientation") != "lexicographic_candidate_id"
        or config.get("sample_weighting") != "equal_total_weight_per_state"
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
    ):
        raise ValueError("STRIDE-CertGuard label construction changed")
    sources = list(config.get("sources") or ())
    if [str(row.get("id")) for row in sources] != [
        "stride-repairability-collection-v2",
        "stride-mapbase-v1",
    ] or [int(row.get("expected_state_count", -1)) for row in sources] != [240, 63]:
        raise ValueError("STRIDE-CertGuard label sources changed")
    for row in sources:
        if any(not _valid_sha256(row.get(field)) for field in ("trials_sha256", "audit_sha256")):
            raise ValueError("STRIDE-CertGuard source hash is invalid")
    reference = dict(config.get("maprank_reference") or {})
    if (
        reference.get("label_summary")
        != "build/stride-maprank-labels-v1/label_build_summary.json"
        or int(reference.get("expected_possible_pair_count", -1)) != 47471
        or int(reference.get("expected_robust_pair_count", -1)) != 21033
        or int(reference.get("expected_uncertain_pair_count", -1)) != 26438
        or not _valid_sha256(reference.get("label_summary_sha256"))
    ):
        raise ValueError("STRIDE-CertGuard MapRank reference changed")


def validate_certguard_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != TRAINING_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_certainty_label_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("base_feature_dimension", -1)) != BASE_FEATURE_DIMENSION
        or int(config.get("uncertainty_input_dimension", -1)) != PAIR_FEATURE_DIMENSION
        or config.get("uncertainty_pair_representation")
        != "absolute_candidate_delta_plus_pair_mean"
        or config.get("design") != "configs/stride_certguard_design.json"
        or config.get("label_config") != "configs/stride_certguard_labels.json"
        or config.get("uncertainty_labels")
        != "build/stride-certguard-labels-v1/uncertainty_pairs.jsonl"
        or config.get("maprank_labels") != "build/stride-maprank-labels-v1"
        or config.get("maprank_selection")
        != "build/stride-maprank-selection-v1/state_selection.jsonl"
        or config.get("maprank_bundle")
        != "build/stride-maprank-training-v1/stride-maprank-v1"
        or config.get("maprank_training_report")
        != "build/stride-maprank-training-v1/maprank_training_report.json"
    ):
        raise ValueError("STRIDE-CertGuard training identity changed")
    if dict(config.get("frozen_direction") or {}) != {
        "controller_id": "stride-maprank-v1",
        "label_schema": "lns2.stride.repairability_conflict_only_ablation.v1",
        "runtime_thresholds": {"base": 0.6, "boundary_only": 1.01},
        "nested_reconstruction_required": True,
        "deployment_model_copied_without_retraining": True,
    }:
        raise ValueError("STRIDE-CertGuard frozen direction model changed")
    if dict(config.get("model_parameters") or {}) != {
        "early_stopping": False,
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "random_state": 20260805,
    } or config.get("model_class") != "sklearn.ensemble.HistGradientBoostingClassifier":
        raise ValueError("STRIDE-CertGuard model protocol changed")
    if (
        int(config.get("outer_map_folds", -1)) != 4
        or int(config.get("inner_map_folds", -1)) != 3
        or list(map(float, config.get("certainty_threshold_grid") or ()))
        != [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
        or config.get("calibration_objective")
        != "minimize_normalized_regret_subject_to_v2_exact_top3_noninferiority"
    ):
        raise ValueError("STRIDE-CertGuard nested calibration changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_uncertainty_roc_auc": 0.60,
        "minimum_weighted_accuracy_gain_over_majority": 0.02,
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.02,
        "maximum_absolute_normalized_regret_degradation_vs_maprank": 0.005,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "maximum_uncertain_override_fraction": 0.40,
        "minimum_directionally_safe_override_precision": 0.50,
        "minimum_maprank_override_retention_fraction": 0.20,
        "maximum_topology_group_normalized_regret_degradation_vs_v2": 0.03,
    }:
        raise ValueError("STRIDE-CertGuard offline gates changed")
    if (
        config.get("training_split") != "train"
        or config.get("legacy_validation_split") != "validation"
        or config.get("split_unit") != "map"
        or config.get("sample_weighting") != "equal_total_weight_per_state"
        or set(map(str, config.get("forbidden_calibration_splits") or ()))
        != {"validation", "high_load", "test", "formal_ood"}
        or config.get("legacy_validation_is_descriptive_only") is not True
        or config.get("high_load_is_development_only") is not True
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE-CertGuard evidence boundary changed")
    for field in (
        "design_sha256",
        "label_config_sha256",
        "maprank_label_summary_sha256",
        "maprank_candidate_aggregates_sha256",
        "maprank_direction_pairs_sha256",
        "maprank_selection_sha256",
        "maprank_manifest_sha256",
        "maprank_training_report_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            raise ValueError(f"STRIDE-CertGuard registration hash is invalid: {field}")


def _verify_sources(config: dict[str, Any], project_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    supported = _supported_collection_audit_schemas()
    paths: list[Path] = []
    audit_rows: list[dict[str, Any]] = []
    for source in config["sources"]:
        trial_path = (project_root / str(source["trials"])).resolve()
        audit_path = (project_root / str(source["audit"])).resolve()
        if sha256_file(trial_path) != str(source["trials_sha256"]):
            raise ValueError(f"CertGuard trial source differs: {trial_path}")
        if sha256_file(audit_path) != str(source["audit_sha256"]):
            raise ValueError(f"CertGuard audit source differs: {audit_path}")
        audit = _read_json(audit_path)
        if (
            audit.get("schema") not in supported
            or audit.get("passed") is not True
            or int(audit.get("state_count", -1)) != int(source["expected_state_count"])
            or str(dict(audit.get("sha256") or {}).get("repair_trials", ""))
            != str(source["trials_sha256"])
        ):
            raise ValueError(f"CertGuard collection audit is invalid: {audit_path}")
        paths.append(trial_path)
        audit_rows.append(
            {
                "id": str(source["id"]),
                "trials": str(trial_path),
                "trials_sha256": sha256_file(trial_path),
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "audit_schema": str(audit["schema"]),
                "run_fingerprint": str(audit["run_fingerprint"]),
                "state_count": int(audit["state_count"]),
            }
        )
    return paths, audit_rows


def build_certguard_labels(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    project_root = path.parents[1]
    config = _read_json(path)
    validate_certguard_label_config(config)
    registered_output = (project_root / str(config["registered_output"])).resolve()
    output_path = Path(output).resolve()
    if output_path != registered_output:
        raise ValueError("STRIDE-CertGuard label output differs from registration")
    reference = dict(config["maprank_reference"])
    reference_path = (project_root / str(reference["label_summary"])).resolve()
    if sha256_file(reference_path) != str(reference["label_summary_sha256"]):
        raise ValueError("STRIDE-CertGuard MapRank label summary differs")
    reference_summary = _read_json(reference_path)
    trial_paths, audits = _verify_sources(config, project_root)
    indices = tuple(map(int, config["trial_indices"]))
    states, metadata, _, _ = _load_trials(trial_paths, indices)
    robust = dict(config["robust_pair_rule"])
    rows: list[dict[str, Any]] = []
    counts_by_split: Counter[str] = Counter()
    positive_by_split: Counter[str] = Counter()
    state_counts_by_split: Counter[str] = Counter()
    for state_id, candidates in sorted(states.items()):
        state = metadata[state_id]
        before_conflicts = int(state["before_conflicts"])
        scores = {
            candidate_id: [
                (before_conflicts - int(outcomes[index]["conflicts_after"]))
                / max(1, before_conflicts)
                for index in indices
            ]
            for candidate_id, outcomes in candidates.items()
        }
        pairs = list(combinations(sorted(candidates), 2))
        weight = 1.0 / len(pairs)
        split = str(state["split"])
        state_counts_by_split[split] += 1
        for left_id, right_id in pairs:
            direction, diagnostics = _robust_winner(
                scores[left_id],
                scores[right_id],
                minimum_win_fraction=float(robust["minimum_paired_win_fraction"]),
                minimum_effect=float(robust["minimum_absolute_mean_effect"]),
                tie_epsilon=float(robust["tie_epsilon"]),
            )
            winner = left_id if direction > 0 else right_id if direction < 0 else None
            label = int(direction != 0)
            rows.append(
                {
                    "schema": LABEL_SCHEMA,
                    "state_id": state_id,
                    "map_id": str(state["map_id"]),
                    "split": split,
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "label": label,
                    "robust_winner_candidate_id": winner,
                    "sample_weight": weight,
                    "mean_left_minus_right_effect": float(diagnostics["mean_effect"]),
                    "absolute_mean_effect": abs(float(diagnostics["mean_effect"])),
                    "first_half_mean_effect": float(diagnostics["first_half_mean_effect"]),
                    "second_half_mean_effect": float(diagnostics["second_half_mean_effect"]),
                    "paired_win_fraction": float(diagnostics["paired_win_fraction"]),
                }
            )
            counts_by_split[split] += 1
            positive_by_split[split] += label
    expected_pairs = int(reference["expected_possible_pair_count"])
    expected_positive = int(reference["expected_robust_pair_count"])
    expected_negative = int(reference["expected_uncertain_pair_count"])
    if (
        len(states) != int(reference_summary["state_count"])
        or len(rows) != expected_pairs
        or sum(int(row["label"]) for row in rows) != expected_positive
        or sum(int(row["label"]) == 0 for row in rows) != expected_negative
        or expected_pairs != int(reference_summary["possible_pair_count"])
        or expected_positive != int(reference_summary["conflict_only_robust_pair_count"])
        or expected_negative != int(reference_summary["conflict_only_uncertain_pair_count"])
    ):
        raise ValueError("STRIDE-CertGuard certainty labels differ from MapRank reference")
    state_weights: Counter[str] = Counter()
    for row in rows:
        state_weights[str(row["state_id"])] += float(row["sample_weight"])
    if any(not math.isclose(value, 1.0, abs_tol=1e-9) for value in state_weights.values()):
        raise ValueError("STRIDE-CertGuard state weights are not equal")
    output_path.mkdir(parents=True, exist_ok=True)
    label_path = output_path / "uncertainty_pairs.jsonl"
    _write_jsonl(label_path, rows)
    report = {
        "schema": LABEL_REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "label_artifact_id": LABEL_ARTIFACT_ID,
        "label_schema": LABEL_SCHEMA,
        "scientific_status": "train_and_legacy_validation_labels_built",
        "state_count": len(states),
        "map_count": len({str(row["map_id"]) for row in metadata.values()}),
        "state_count_by_split": dict(sorted(state_counts_by_split.items())),
        "pair_count": len(rows),
        "pair_count_by_split": dict(sorted(counts_by_split.items())),
        "robust_pair_count": expected_positive,
        "uncertain_pair_count": expected_negative,
        "robust_pair_fraction": expected_positive / len(rows),
        "robust_pair_count_by_split": dict(sorted(positive_by_split.items())),
        "feature_schema": FEATURE_SCHEMA,
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "uncertainty_pair_representation": "absolute_candidate_delta_plus_pair_mean",
        "uncertainty_input_dimension": PAIR_FEATURE_DIMENSION,
        "trials_per_candidate": len(indices),
        "paired_pp_seeds_within_state_and_trial_index": True,
        "sample_weighting": config["sample_weighting"],
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
        "high_load_data_read": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "mean_absolute_pair_effect": statistics.fmean(
            float(row["absolute_mean_effect"]) for row in rows
        ),
        "config_sha256": sha256_file(path),
        "maprank_reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "audited_sources": audits,
        "artifacts": {
            "uncertainty_pairs": label_path.name,
            "uncertainty_pairs_sha256": sha256_file(label_path),
        },
    }
    _write_json(output_path / "certguard_label_report.json", report)
    return report


__all__ = [
    "CONTROLLER_ID",
    "LABEL_SCHEMA",
    "build_certguard_labels",
    "validate_certguard_design",
    "validate_certguard_label_config",
    "validate_certguard_training_config",
]
