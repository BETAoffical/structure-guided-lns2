from __future__ import annotations

from collections import Counter
from pathlib import Path
from string import hexdigits
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import _agent_band, _conflict_band


DESIGN_SCHEMA = "lns2.stride.maprank_design.v1"
SELECTION_SCHEMA = "lns2.stride.maprank_selection.v1"
CONTROLLER_ID = "stride-maprank-v1"
TRAINING_CONFIG_SCHEMA = "lns2.stride.maprank_training_config.v1"
TRAINING_REPORT_SCHEMA = "lns2.stride.maprank_training.v1"
PREDICTION_SCHEMA = "lns2.stride.maprank_prediction.v1"
STRATEGY_SCHEMA = "lns2.stride.maprank_strategy.v1"
LABEL_REPORT_SCHEMA = "lns2.stride.maprank_label_build.v1"
EVALUATION_CONFIG_SCHEMA = "lns2.stride.maprank_evaluation_config.v1"


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in hexdigits for character in text)


def validate_maprank_design(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "registered_before_mapbase_collection_summary_and_label_analysis"
        or config.get("design_id") != "stride-maprank-design-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_artifact_id") != "stride-maprank-labels-v1"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
    ):
        raise ValueError("STRIDE-MapRank design identity changed")
    sources = list(config.get("source_collections") or ())
    if [str(row.get("id")) for row in sources] != [
        "stride-repairability-collection-v2",
        "stride-mapbase-v1",
    ] or [int(row.get("expected_state_count", -1)) for row in sources] != [240, 63]:
        raise ValueError("STRIDE-MapRank collection sources changed")
    selection_sources = dict(config.get("selection_sources") or {})
    if set(selection_sources) != {
        "legacy_selection",
        "mapbase_selection",
        "mapbase_dataset_manifest",
    } or any(
        not _valid_sha256(dict(artifact).get("sha256"))
        for artifact in selection_sources.values()
    ):
        raise ValueError("STRIDE-MapRank selection registration changed")
    if dict(config.get("expected_data_coverage") or {}) != {
        "state_count": 303,
        "train_state_count": 237,
        "legacy_validation_state_count": 66,
        "train_map_count": 24,
        "legacy_validation_map_count": 6,
        "map_level_split_required": True,
        "new_mapbase_maps_are_train_only": True,
    }:
        raise ValueError("STRIDE-MapRank data coverage changed")
    if dict(config.get("label_design") or {}) != {
        "schema": "lns2.stride.repairability_conflict_only_ablation.v1",
        "meaning": "current_step_conflict_reduction_normalized_by_before_conflicts",
        "trial_indices": list(range(16)),
        "paired_pp_seeds_within_state_and_trial_index": True,
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "sample_weighting": "equal_total_weight_per_state",
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
    }:
        raise ValueError("STRIDE-MapRank label semantics changed")
    if dict(config.get("candidate_design") or {}) != {
        "base_families": ["target", "collision", "random"],
        "base_sizes": [4, 8, 16],
        "topology_boundary_runtime_status": "optional_abstaining_augmentation",
        "mapbase_training_rows": "base_only_after_fresh_map_boundary_coverage_failure",
        "frozen_v2_anchor": "artifacts/initlns-closed-loop-controller-v2",
    }:
        raise ValueError("STRIDE-MapRank candidate design changed")
    ranking = dict(config.get("ranking_design") or {})
    if (
        ranking.get("architecture")
        != "frozen_v2_anchor_plus_conflict_only_challenger"
        or ranking.get("feature_schema") != "lns2.realized_features.v2"
        or int(ranking.get("base_feature_dimension", -1)) != 124
        or ranking.get("pair_representation") != "candidate_feature_delta"
        or ranking.get("model_class")
        != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(ranking.get("model_parameters") or {})
        != {
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "random_state": 20260804,
        }
        or int(ranking.get("outer_train_map_folds", -1)) != 4
        or int(ranking.get("inner_train_map_folds", -1)) != 3
        or list(ranking.get("threshold_grid") or ())
        != [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
        or list(ranking.get("threshold_kinds") or ()) != ["base", "boundary_only"]
        or ranking.get("legacy_validation_is_descriptive_only") is not True
        or ranking.get("fresh_development_validation_required") is not True
    ):
        raise ValueError("STRIDE-MapRank ranking design changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "absolute_normalized_regret_improvement_alternative": 0.02,
        "top3_hit_rate_noninferiority_tolerance": 0.01,
        "exact_best_rate_noninferiority_tolerance": 0.01,
        "maximum_topology_group_normalized_regret_degradation": 0.03,
        "minimum_pairwise_accuracy_gain_over_weighted_majority": 0.03,
    }:
        raise ValueError("STRIDE-MapRank offline gates changed")
    if list(config.get("evaluation_order") or ()) != [
        "nested_train_map_oof",
        "legacy_validation_descriptive",
        "fresh_map_shadow",
        "paired_high_load_raw_ttf",
        "paired_fresh_map_raw_ttf",
    ] or config.get("runtime_primary_metric") != "mean_raw_run_to_completion_wall_ttf":
        raise ValueError("STRIDE-MapRank evaluation order changed")
    forbidden = set(map(str, config.get("forbidden_inputs") or ()))
    if forbidden != {
        "repair_runtime",
        "time_to_feasible",
        "future_repair_rounds",
        "cost_to_go",
        "receding_q",
        "controller_outcome",
        "validation_labels_for_threshold_calibration",
        "test_data",
        "formal_ood_data",
    }:
        raise ValueError("STRIDE-MapRank forbidden inputs changed")


def validate_maprank_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != TRAINING_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "registered_before_mapbase_label_analysis"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("challenger_label_schema")
        != "lns2.stride.repairability_conflict_only_ablation.v1"
        or config.get("feature_schema") != "lns2.realized_features.v2"
        or int(config.get("base_feature_dimension", -1)) != 124
        or config.get("labels") != "build/stride-maprank-labels-v1"
        or config.get("selection")
        != "build/stride-maprank-selection-v1/state_selection.jsonl"
        or config.get("data_design") != "configs/stride_maprank_design.json"
        or config.get("label_design")
        != "configs/stride_repairability_label_design.json"
        or config.get("frozen_v2_bundle")
        != "artifacts/initlns-closed-loop-controller-v2"
        or float(config.get("topology_boundary_threshold", -1.0)) != 0.06
    ):
        raise ValueError("STRIDE-MapRank training identity changed")
    for field in (
        "data_design_sha256",
        "label_design_sha256",
        "frozen_v2_manifest_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            raise ValueError(f"STRIDE-MapRank registration hash is invalid: {field}")
    ranking = {
        "model_class": config.get("model_class"),
        "model_parameters": config.get("model_parameters"),
        "outer_train_map_folds": config.get("outer_map_folds"),
        "inner_train_map_folds": config.get("inner_map_folds"),
        "threshold_grid": config.get("threshold_grid"),
        "threshold_kinds": config.get("threshold_kinds"),
    }
    design_ranking = {
        "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "model_parameters": {
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "random_state": 20260804,
        },
        "outer_train_map_folds": 4,
        "inner_train_map_folds": 3,
        "threshold_grid": [
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.01,
        ],
        "threshold_kinds": ["base", "boundary_only"],
    }
    if ranking != design_ranking or config.get("calibration_objective") != (
        "minimize_normalized_regret_subject_to_v2_exact_and_top3_noninferiority"
    ):
        raise ValueError("STRIDE-MapRank model or calibration protocol changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "absolute_normalized_regret_improvement_alternative": 0.02,
        "top3_hit_rate_noninferiority_tolerance": 0.01,
        "exact_best_rate_noninferiority_tolerance": 0.01,
        "maximum_topology_group_normalized_regret_degradation": 0.03,
        "minimum_pairwise_accuracy_gain_over_weighted_majority": 0.03,
    }:
        raise ValueError("STRIDE-MapRank offline gates changed")
    if (
        config.get("training_split") != "train"
        or config.get("legacy_validation_split") != "validation"
        or set(map(str, config.get("forbidden_calibration_splits") or ()))
        != {"validation", "test", "formal_ood"}
        or config.get("legacy_validation_is_descriptive_only") is not True
        or config.get("fresh_development_validation_required") is not True
        or bool(config.get("formal_ood_data_allowed"))
        or bool(config.get("test_data_allowed"))
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE-MapRank evidence boundary changed")


def validate_maprank_evaluation_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != EVALUATION_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "registered_before_maprank_training_outcomes"
        or config.get("experiment_id") != "stride-maprank-evaluation-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
        or config.get("primary_metric") != "mean_raw_wall_time_to_feasible"
        or config.get("ttf_clock_schema") != "lns2.ttf.reset_inclusive_wall.v1"
        or config.get("stopping_rule") != "run-to-completion"
        or config.get("scientific_time_limit_seconds") is not None
        or config.get("environment_time_limit_seconds") is not None
        or config.get("episode_process_timeout_seconds") is not None
    ):
        raise ValueError("STRIDE-MapRank evaluation identity or raw-TTF clock changed")
    if list(config.get("controllers") or ()) != [
        "v2-full",
        "v2-augmented-pool",
        CONTROLLER_ID,
    ] or config.get("execution_order") != "strict_rotating_triplet_order":
        raise ValueError("STRIDE-MapRank comparator design changed")
    if int(config.get("workers", -1)) != 1 or config.get(
        "deterministic_pp_replay_required"
    ) is not True:
        raise ValueError("STRIDE-MapRank paired execution changed")
    if config.get("offline_gate_required") is not True:
        raise ValueError("STRIDE-MapRank evaluation bypasses its offline gate")
    augmentation = dict(config.get("topology_boundary_augmentation") or {})
    if augmentation != {
        "enabled": True,
        "generator_id": "stride-topoboundary-v1",
        "neighborhood_size": 16,
        "core_budget": 4,
        "maximum_added_candidates": 2,
        "runtime_id": "stride-boundary-static-cache-v1",
        "static_grid_cache": True,
    }:
        raise ValueError("STRIDE-MapRank candidate-pool comparator changed")
    development = dict(config.get("high_load_development") or {})
    if list(development.get("solver_seeds") or ()) != [1, 2, 3, 4] or [
        str(row.get("id")) for row in development.get("cohorts") or ()
    ] != ["maze300", "room500"]:
        raise ValueError("STRIDE-MapRank high-load cohort changed")
    if dict(development.get("gates") or {}) != {
        "minimum_raw_ttf_improvement_vs_v2_full": 0.02,
        "minimum_ranker_raw_ttf_improvement_vs_v2_augmented": 0.0,
        "maximum_cohort_raw_ttf_regression": 0.10,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("STRIDE-MapRank high-load gates changed")
    fresh = dict(config.get("fresh_map_raw_ttf") or {})
    if list(fresh.get("solver_seeds") or ()) != [1, 2, 3] or [
        str(row.get("id")) for row in fresh.get("cohorts") or ()
    ] != ["maze200", "room400", "random500", "warehouse500", "den300", "lak500"]:
        raise ValueError("STRIDE-MapRank fresh-map cohort changed")
    if dict(fresh.get("gates") or {}) != {
        "minimum_raw_ttf_improvement_vs_v2_full": 0.05,
        "minimum_ranker_raw_ttf_improvement_vs_v2_augmented": 0.0,
        "maximum_map_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("STRIDE-MapRank fresh-map gates changed")
    if set(map(str, config.get("required_metrics") or ())) != {
        "success_count",
        "raw_wall_time_to_feasible",
        "repair_iterations",
        "normalized_wall_clock_conflict_auc",
        "pp_replan_seconds",
        "controller_seconds_before_repair",
        "neighborhood_selection_seconds",
        "invalid_action_count",
        "fingerprint_mismatch_count",
    }:
        raise ValueError("STRIDE-MapRank required metrics changed")


def _registered_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"registered MapRank input SHA differs: {artifact['path']}")
    return path


def build_maprank_selection(
    config: dict[str, Any],
    *,
    legacy_rows: list[dict[str, Any]],
    mapbase_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validate_maprank_design(config)
    expected = dict(config["expected_data_coverage"])
    if len(legacy_rows) != 240 or len(mapbase_rows) != 63:
        raise ValueError("STRIDE-MapRank source selection cardinality changed")
    legacy_ids = {str(row["state_id"]) for row in legacy_rows}
    mapbase_ids = {str(row["state_id"]) for row in mapbase_rows}
    if len(legacy_ids) != 240 or len(mapbase_ids) != 63 or legacy_ids & mapbase_ids:
        raise ValueError("STRIDE-MapRank state identities overlap or repeat")
    legacy_maps = {str(row["map_id"]) for row in legacy_rows}
    mapbase_maps = {str(row["map_id"]) for row in mapbase_rows}
    if len(mapbase_maps) != 8 or legacy_maps & mapbase_maps:
        raise ValueError("STRIDE-MapRank fresh training maps overlap legacy maps")
    ratios: dict[str, float] = {}
    for row in dataset_rows:
        map_id = str(row["map_id"])
        ratio = float(dict(row["topology_metrics"])["low_degree_cell_ratio"])
        if map_id in ratios and abs(ratios[map_id] - ratio) > 1e-12:
            raise ValueError(f"MapRank map topology changed across tasks: {map_id}")
        ratios[map_id] = ratio
    if not mapbase_maps.issubset(ratios):
        raise ValueError("STRIDE-MapRank map topology coverage is incomplete")
    added = []
    for source in mapbase_rows:
        conflicts = int(source["before_conflicts"])
        if conflicts <= 0 or str(source.get("research_split")) != "train":
            raise ValueError("STRIDE-MapRank MapBase selection is not active train data")
        map_id = str(source["map_id"])
        ratio = ratios[map_id]
        added.append(
            {
                **source,
                "schema": "lns2.stride.state_selection.v1",
                "split": str(source["source_split"]),
                "source_policy": "initial_state_map_expansion",
                "decision_stage": "initial",
                "conflict_band": _conflict_band(conflicts),
                "agent_band": _agent_band(int(source["agent_count"])),
                "static_low_degree_cell_ratio": ratio,
                "topology_group": (
                    "boundary_relevant" if ratio >= 0.06 else "control"
                ),
            }
        )
    combined = sorted([*legacy_rows, *added], key=lambda row: str(row["state_id"]))
    split_counts = Counter(str(row["research_split"]) for row in combined)
    train_maps = {
        str(row["map_id"])
        for row in combined
        if str(row["research_split"]) == "train"
    }
    validation_maps = {
        str(row["map_id"])
        for row in combined
        if str(row["research_split"]) == "validation"
    }
    if (
        len(combined) != int(expected["state_count"])
        or split_counts != Counter(
            {
                "train": int(expected["train_state_count"]),
                "validation": int(expected["legacy_validation_state_count"]),
            }
        )
        or len(train_maps) != int(expected["train_map_count"])
        or len(validation_maps) != int(expected["legacy_validation_map_count"])
        or train_maps & validation_maps
    ):
        raise ValueError("STRIDE-MapRank combined map-level split changed")
    return combined


def prepare_maprank_selection(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _read_json(path)
    validate_maprank_design(config)
    project_root = path.parents[1]
    sources = {
        name: _registered_path(project_root, dict(artifact))
        for name, artifact in dict(config["selection_sources"]).items()
    }
    rows = build_maprank_selection(
        config,
        legacy_rows=_read_jsonl(sources["legacy_selection"]),
        mapbase_rows=_read_jsonl(sources["mapbase_selection"]),
        dataset_rows=_read_jsonl(sources["mapbase_dataset_manifest"]),
    )
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "state_selection.jsonl"
    _write_jsonl(selection_path, rows)
    split_counts = Counter(str(row["research_split"]) for row in rows)
    report = {
        "schema": SELECTION_SCHEMA,
        "design_id": str(config["design_id"]),
        "state_count": len(rows),
        "split_state_counts": dict(sorted(split_counts.items())),
        "split_map_counts": {
            split: len(
                {
                    str(row["map_id"])
                    for row in rows
                    if str(row["research_split"]) == split
                }
            )
            for split in ("train", "validation")
        },
        "new_map_count": len(
            {
                str(row["map_id"])
                for row in rows
                if str(row.get("source_policy")) == "initial_state_map_expansion"
            }
        ),
        "new_state_count": sum(
            str(row.get("source_policy")) == "initial_state_map_expansion"
            for row in rows
        ),
        "new_topology_group_counts": dict(
            sorted(
                Counter(
                    str(row["topology_group"])
                    for row in rows
                    if str(row.get("source_policy"))
                    == "initial_state_map_expansion"
                ).items()
            )
        ),
        "repair_outcomes_used": False,
        "controller_outcomes_used": False,
        "formal_speed_claim": False,
        "sha256": {
            "design": sha256_file(path),
            "state_selection": sha256_file(selection_path),
            **{
                name: sha256_file(source) for name, source in sorted(sources.items())
            },
        },
    }
    _write_json(output_root / "selection_report.json", report)
    return report


def build_maprank_labels(
    *, training_config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    from experiments.stride_repairability import build_repairability_labels

    training_path = Path(training_config_path).resolve()
    training = _read_json(training_path)
    validate_maprank_training_config(training)
    project_root = training_path.parents[1]
    design_path = (project_root / str(training["data_design"])).resolve()
    label_design_path = (project_root / str(training["label_design"])).resolve()
    if sha256_file(design_path) != str(training["data_design_sha256"]):
        raise ValueError("STRIDE-MapRank design SHA differs")
    if sha256_file(label_design_path) != str(training["label_design_sha256"]):
        raise ValueError("STRIDE-MapRank label design SHA differs")
    design = _read_json(design_path)
    validate_maprank_design(design)
    registered_output = (project_root / str(training["labels"])).resolve()
    if Path(output).resolve() != registered_output:
        raise ValueError("STRIDE-MapRank label output differs from registration")
    sources = list(design["source_collections"])
    trial_paths = [(project_root / str(row["trials"])).resolve() for row in sources]
    audit_paths = [(project_root / str(row["audit"])).resolve() for row in sources]
    if any(not path.is_file() for path in [*trial_paths, *audit_paths]):
        raise ValueError("STRIDE-MapRank audited label sources are incomplete")
    summary = build_repairability_labels(
        config_path=label_design_path,
        trial_paths=trial_paths,
        audit_report_paths=audit_paths,
        output=registered_output,
    )
    expected = dict(design["expected_data_coverage"])
    if (
        int(summary.get("state_count", -1)) != int(expected["state_count"])
        or dict(summary.get("map_count_by_split") or {})
        != {
            "train": int(expected["train_map_count"]),
            "validation": int(expected["legacy_validation_map_count"]),
        }
        or int(summary.get("trials_per_candidate", -1)) != 16
        or bool(summary.get("runtime_used_in_label"))
        or bool(summary.get("future_trajectory_used_in_label"))
    ):
        raise ValueError("STRIDE-MapRank label coverage or semantics changed")
    summary_path = registered_output / "label_build_summary.json"
    report = {
        "schema": LABEL_REPORT_SCHEMA,
        "label_artifact_id": str(design["label_artifact_id"]),
        "controller_id": CONTROLLER_ID,
        "state_count": int(summary["state_count"]),
        "map_count_by_split": dict(summary["map_count_by_split"]),
        "candidate_count": int(summary["candidate_count"]),
        "robust_pair_count": int(summary["robust_pair_count"]),
        "conflict_only_robust_pair_count": int(
            summary["conflict_only_robust_pair_count"]
        ),
        "states_with_pairs": int(summary["states_with_pairs"]),
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
        "formal_speed_claim": False,
        "sha256": {
            "training_config": sha256_file(training_path),
            "design": sha256_file(design_path),
            "label_design": sha256_file(label_design_path),
            "label_summary": sha256_file(summary_path),
            "candidate_aggregates": sha256_file(
                registered_output / "candidate_aggregates.jsonl"
            ),
            "conflict_only_pairs": sha256_file(
                registered_output / "conflict_only_dominance_pairs.jsonl"
            ),
        },
    }
    _write_json(registered_output / "maprank_label_report.json", report)
    return report


def run_maprank_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    from experiments.stride_guardrank import _run_anchor_guard_training

    path = Path(config_path).resolve()
    config = _read_json(path)
    validate_maprank_training_config(config)
    project_root = path.parents[1]
    design_path = (project_root / str(config["data_design"])).resolve()
    label_design_path = (project_root / str(config["label_design"])).resolve()
    if sha256_file(design_path) != str(config["data_design_sha256"]):
        raise ValueError("STRIDE-MapRank design SHA differs")
    if sha256_file(label_design_path) != str(config["label_design_sha256"]):
        raise ValueError("STRIDE-MapRank label design SHA differs")
    validate_maprank_design(_read_json(design_path))
    return _run_anchor_guard_training(
        config_path=path,
        output=output,
        config_validator=validate_maprank_training_config,
        controller_id=CONTROLLER_ID,
        report_schema=TRAINING_REPORT_SCHEMA,
        prediction_schema=PREDICTION_SCHEMA,
        strategy_schema=STRATEGY_SCHEMA,
        report_filename="maprank_training_report.json",
    )


__all__ = [
    "build_maprank_labels",
    "build_maprank_selection",
    "prepare_maprank_selection",
    "run_maprank_training",
    "validate_maprank_design",
    "validate_maprank_evaluation_config",
    "validate_maprank_training_config",
]
