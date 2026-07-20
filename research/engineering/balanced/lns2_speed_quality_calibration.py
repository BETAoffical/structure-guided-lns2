from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.engineering.balanced.balanced_controller import (
    REGISTERED_CONFLICT_THRESHOLDS,
    BalancedControllerConfig,
    write_balanced_controller,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import TRACE_FORMAT_DELTA_GZIP_V2
from experiments.compact_controller_model import load_controller_bundle
from research.engineering.legacy_tradeoff.lns2_tradeoff import (
    _equal_map_mean,
    _mean,
    _relative_improvement,
    _write_csv,
)
from experiments.repair_collection import (
    _dataset_fingerprint,
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)


CALIBRATION_SCHEMA = "lns2.complete_episode_route_calibration.v2"
CALIBRATION_VERSION = 2
CALIBRATION_SPLITS = ("policy_train", "policy_validation")


def _split_config(source: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in CALIBRATION_SPLITS:
        raise ValueError(f"unsupported calibration split: {split}")
    design = dict(source["dataset_design"])
    layout_counts = {
        str(name): int(count)
        for name, count in dict(design["layout_counts"])[split].items()
    }
    qualification = dict(source["qualification"])
    return {
        "schema_version": 1,
        "formal": True,
        "study_role": "route_threshold_calibration",
        "split": split,
        "solver_seeds": list(map(int, source["solver_seeds"])),
        "policies": ["official_adaptive", "realized_dynamic"],
        "dataset_design": {
            "mode": "structured",
            "map_count": sum(layout_counts.values()),
            "tasks_per_map": int(design["tasks_per_map"]),
            "task_variants": list(map(str, design["task_variants"])),
            "layout_counts": layout_counts,
        },
        "environment": dict(source["environment"]),
        "qualification": {
            "mode": "natural_distribution_development",
            "minimum_nonzero_states": int(
                dict(qualification["minimum_nonzero_by_split"])[split]
            ),
            "minimum_nonzero_states_per_layout": int(
                dict(qualification["minimum_nonzero_per_layout"])[split]
            ),
            "minimum_active_maps": int(
                dict(qualification["minimum_active_maps"])[split]
            ),
            "minimum_nonzero_states_per_solver_seed": 0,
        },
        "severity_thresholds": dict(
            source.get(
                "severity_thresholds", {"low_max": 0.001, "medium_max": 0.01}
            )
        ),
        "proposal": dict(source["proposal"]),
        "max_decisions": int(source["max_decisions"]),
        "metric_iteration_budget": int(source["metric_iteration_budget"]),
        "wall_time_budget_seconds": float(source["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            source["episode_process_timeout_seconds"]
        ),
        "workers": 1,
        "frozen_models": source["frozen_models"],
        "model_registration": dict(source["model_registration"]),
        "reference_datasets": list(source.get("reference_datasets", [])),
    }


def _collection_root(
    output_root: Path, split: str, controller: str, threshold: int | None = None
) -> Path:
    name = controller if threshold is None else f"balanced-threshold-{threshold:02d}"
    return output_root / "collections" / split / name


def _run_collection(
    *,
    dataset: Path,
    config_path: Path,
    root: Path,
    controller: str,
    policy: str,
    controller_bundle: Path,
    balanced_config: dict[str, Any] | None,
    feature_backend: str,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    existing = (root / "run_config.json").is_file()
    if existing and not resume:
        raise ValueError(f"calibration collection already exists; pass --resume: {root}")
    run_closed_loop_collection(
        dataset,
        config_path,
        root,
        phase="qualify",
        workers=workers,
        resume=resume or existing,
        trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
        controller=controller,
        feature_backend=feature_backend,
        controller_bundle=controller_bundle,
        balanced_config=balanced_config,
    )
    summary = run_closed_loop_collection(
        dataset,
        config_path,
        root,
        phase=policy,
        workers=workers,
        resume=True,
        trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
        controller=controller,
        feature_backend=feature_backend,
        controller_bundle=controller_bundle,
        balanced_config=balanced_config,
    )
    report = dict(summary.get(policy) or {})
    if int(report.get("error_count", 0)) != 0:
        raise RuntimeError(f"complete-episode calibration failed in {root}")
    return report


def _episode_rows(
    root: Path,
    *,
    split: str,
    controller: str,
    threshold: int | None,
) -> list[dict[str, Any]]:
    policy = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
    manifest = _read_jsonl(root / f"{policy}_manifest.jsonl")
    run_fingerprint = _read_json(root / "run_config.json")["run_fingerprint"]
    errors = [row for row in manifest if str(row.get("status")) not in {"ok", "resumed"}]
    if errors:
        raise ValueError(f"calibration collection contains failed episodes: {root}")
    result = []
    for source in manifest:
        summary = dict(source["summary"])
        result.append(
            {
                "split": split,
                "controller": controller,
                "conflict_threshold": threshold,
                "episode_id": source["episode_id"],
                "map_id": source["map_id"],
                "task_id": source["task_id"],
                "layout_mode": source["layout_mode"],
                "agent_count": int(source["agent_count"]),
                "solver_seed": int(source["solver_seed"]),
                "initial_fingerprint": summary["initial_fingerprint"],
                "initial_conflicts": int(summary["initial_conflicts"]),
                "repairable": bool(summary["repairable"]),
                "success": bool(summary["success"]),
                "fixed_budget_conflict_auc": summary["fixed_budget_conflict_auc"],
                "capped_wall_time_seconds": summary[
                    "capped_wall_time_to_feasible"
                ],
                "controller_seconds": float(
                    dict(summary.get("controller_totals") or {}).get(
                        "controller_seconds_before_repair", 0.0
                    )
                ),
                "repair_wall_seconds": summary.get("repair_wall_seconds"),
                "repair_iterations": int(summary["repair_iterations"]),
                "final_sum_of_costs": int(summary["final_sum_of_costs"]),
                "model_decision_count": int(
                    summary.get("model_decision_count", 0)
                ),
                "official_decision_count": int(
                    summary.get("official_decision_count", 0)
                ),
                "run_fingerprint": run_fingerprint,
            }
        )
    return result


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {
        (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("calibration collection has duplicate task/seed episodes")
    return result


def _complete_episode_metrics(
    official_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    *,
    allowed_maps: set[str] | None = None,
) -> dict[str, Any]:
    official_index = _index(official_rows)
    full_index = _index(full_rows)
    balanced_index = _index(balanced_rows)
    keys = set(official_index) & set(full_index) & set(balanced_index)
    if allowed_maps is not None:
        keys = {
            key
            for key in keys
            if str(official_index[key]["map_id"]) in allowed_maps
        }
    for key in keys:
        starts = {
            str(official_index[key]["initial_fingerprint"]),
            str(full_index[key]["initial_fingerprint"]),
            str(balanced_index[key]["initial_fingerprint"]),
        }
        if len(starts) != 1:
            raise ValueError(f"calibration controllers start differently: {key}")
    repairable_keys = [
        key for key in sorted(keys) if bool(official_index[key]["repairable"])
    ]
    official = [official_index[key] for key in repairable_keys]
    full = [full_index[key] for key in repairable_keys]
    balanced = [balanced_index[key] for key in repairable_keys]
    official_auc = _equal_map_mean(official, "fixed_budget_conflict_auc")
    full_auc = _equal_map_mean(full, "fixed_budget_conflict_auc")
    balanced_auc = _equal_map_mean(balanced, "fixed_budget_conflict_auc")
    full_gain = float(official_auc or 0.0) - float(full_auc or 0.0)
    balanced_gain = float(official_auc or 0.0) - float(balanced_auc or 0.0)
    retention = (
        balanced_gain / full_gain
        if full_gain > 0.0
        else 1.0
        if balanced_gain >= 0.0
        else -math.inf
    )
    official_wall = _equal_map_mean(official, "capped_wall_time_seconds")
    full_wall = _equal_map_mean(full, "capped_wall_time_seconds")
    balanced_wall = _equal_map_mean(balanced, "capped_wall_time_seconds")
    common_success = [
        key
        for key in repairable_keys
        if official_index[key]["success"] and balanced_index[key]["success"]
    ]
    soc_ratios = [
        float(balanced_index[key]["final_sum_of_costs"])
        / float(official_index[key]["final_sum_of_costs"])
        for key in common_success
        if float(official_index[key]["final_sum_of_costs"]) != 0.0
    ]
    model_decisions = sum(int(row["model_decision_count"]) for row in balanced)
    official_decisions = sum(
        int(row["official_decision_count"]) for row in balanced
    )
    total_decisions = model_decisions + official_decisions
    return {
        "episode_count": len(keys),
        "repairable_episode_count": len(repairable_keys),
        "map_count": len({str(row["map_id"]) for row in official}),
        "official_success_count": sum(bool(row["success"]) for row in official),
        "full_model_success_count": sum(bool(row["success"]) for row in full),
        "balanced_success_count": sum(bool(row["success"]) for row in balanced),
        "official_auc": official_auc,
        "full_model_auc": full_auc,
        "balanced_auc": balanced_auc,
        "auc_gain_retention": retention,
        "official_wall_seconds": official_wall,
        "full_model_wall_seconds": full_wall,
        "balanced_wall_seconds": balanced_wall,
        "speedup_over_full": _relative_improvement(full_wall, balanced_wall),
        "wall_ratio_vs_lns2": (
            float(balanced_wall) / float(official_wall)
            if official_wall not in {None, 0.0} and balanced_wall is not None
            else None
        ),
        "common_success_soc_ratio": _mean(soc_ratios),
        "common_success_episode_count": len(common_success),
        "model_decision_count": model_decisions,
        "official_decision_count": official_decisions,
        "model_route_fraction": (
            model_decisions / total_decisions if total_decisions else 0.0
        ),
    }


def _passes(metrics: dict[str, Any]) -> bool:
    return bool(
        int(metrics["balanced_success_count"])
        >= int(metrics["official_success_count"])
        and float(metrics["auc_gain_retention"]) >= 0.50
        and (
            metrics["common_success_soc_ratio"] is None
            or float(metrics["common_success_soc_ratio"]) <= 1.02
        )
        and metrics["speedup_over_full"] is not None
        and float(metrics["speedup_over_full"]) >= 0.10
        and metrics["wall_ratio_vs_lns2"] is not None
        and float(metrics["wall_ratio_vs_lns2"]) <= 1.0
    )


def _layout_balanced_folds(rows: list[dict[str, Any]]) -> dict[int, set[str]]:
    map_layout = {
        str(row["map_id"]): str(row["layout_mode"]) for row in rows
    }
    by_layout: dict[str, list[str]] = defaultdict(list)
    for map_id, layout in map_layout.items():
        by_layout[layout].append(map_id)
    if len(map_layout) != 12 or set(map(len, by_layout.values())) != {4}:
        raise ValueError(
            "policy_train calibration requires 12 maps: four in each layout"
        )
    folds = {fold: set() for fold in range(4)}
    for maps in by_layout.values():
        for fold, map_id in enumerate(sorted(maps)):
            folds[fold].add(map_id)
    if any(len(maps) != 3 for maps in folds.values()):
        raise ValueError("policy_train map folds are not layout balanced")
    return folds


def _threshold_payload(
    threshold: int, calibration_fingerprint: str, split: str
) -> dict[str, Any]:
    return BalancedControllerConfig(
        conflict_threshold=threshold,
        pruner_threshold=None,
        source={
            "study_role": "complete_episode_route_threshold_calibration",
            "calibration_run_fingerprint": calibration_fingerprint,
            "split": split,
            "proposal_pruner": "disabled_locked_validation_failed",
            "candidate_threshold": threshold,
        },
        promoted=False,
    ).payload()


def _load_split_rows(
    output_root: Path, split: str, thresholds: Iterable[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    official = _episode_rows(
        _collection_root(output_root, split, "official_adaptive"),
        split=split,
        controller="official_adaptive",
        threshold=None,
    )
    full = _episode_rows(
        _collection_root(output_root, split, "v2-full"),
        split=split,
        controller="v2-full",
        threshold=None,
    )
    balanced = {
        int(threshold): _episode_rows(
            _collection_root(output_root, split, "v2-balanced", int(threshold)),
            split=split,
            controller="v2-balanced",
            threshold=int(threshold),
        )
        for threshold in thresholds
    }
    return official, full, balanced


def _training_grid(
    official: list[dict[str, Any]],
    full: list[dict[str, Any]],
    balanced: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[int, set[str]]]:
    folds = _layout_balanced_folds(official)
    grid = []
    for threshold in REGISTERED_CONFLICT_THRESHOLDS:
        aggregate = _complete_episode_metrics(
            official, full, balanced[int(threshold)]
        )
        fold_metrics = [
            _complete_episode_metrics(
                official,
                full,
                balanced[int(threshold)],
                allowed_maps=folds[fold],
            )
            for fold in range(4)
        ]
        passing_folds = sum(_passes(row) for row in fold_metrics)
        grid.append(
            {
                "conflict_threshold": int(threshold),
                "pruner_threshold": None,
                **aggregate,
                "map_group_fold_count": 4,
                "passing_fold_count": passing_folds,
                "all_folds_pass": passing_folds == 4,
                "aggregate_pass": _passes(aggregate),
                "eligible": passing_folds == 4 and _passes(aggregate),
                "fold_metrics": fold_metrics,
            }
        )
    return grid, folds


def _select_training_candidate(grid: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    eligible = [row for row in grid if bool(row["eligible"])]
    if eligible:
        return (
            max(
                eligible,
                key=lambda row: (
                    float(row["speedup_over_full"]),
                    int(row["conflict_threshold"]),
                ),
            ),
            "eligible_policy_train_complete_episode_candidate",
        )
    quality_safe = [
        row
        for row in grid
        if int(row["balanced_success_count"])
        >= int(row["official_success_count"])
        and float(row["auc_gain_retention"]) >= 0.50
        and (
            row["common_success_soc_ratio"] is None
            or float(row["common_success_soc_ratio"]) <= 1.02
        )
    ]
    fallback = max(
        quality_safe
        or [row for row in grid if int(row["conflict_threshold"]) == 0],
        key=lambda row: (
            float(row["speedup_over_full"])
            if row["speedup_over_full"] is not None
            else -math.inf,
            -int(row["conflict_threshold"]),
        ),
    )
    return fallback, "fallback_not_promotion_eligible"


def _flat_grid(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "fold_metrics"}
        for row in grid
    ]


def run_complete_episode_calibration(
    dataset: str | Path,
    base_config: str | Path,
    controller_bundle: str | Path,
    output: str | Path,
    *,
    feature_backend: str = "auto",
    workers: int = 1,
    resume: bool = False,
    selection_only: bool = False,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("complete-episode calibration timing requires workers=1")
    dataset_root = Path(dataset).resolve()
    config_path = Path(base_config).resolve()
    controller_root = Path(controller_bundle).resolve()
    output_root = Path(output).resolve()
    source = _read_json(config_path)
    if tuple(map(str, source.get("splits", []))) != CALIBRATION_SPLITS:
        raise ValueError(
            "calibration config must register policy_train and policy_validation"
        )
    bundle = load_controller_bundle(controller_root)
    if bundle.pruner_threshold is not None:
        raise ValueError(
            "complete-episode route calibration requires the proposal pruner disabled"
        )
    split_configs = {
        split: _split_config(source, split) for split in CALIBRATION_SPLITS
    }
    implementation_fingerprint = _fingerprint(
        Path(__file__).read_text(encoding="utf-8")
    )
    run_fingerprint = _fingerprint(
        {
            "schema": CALIBRATION_SCHEMA,
            "dataset_fingerprint": _dataset_fingerprint(dataset_root),
            "source_config": source,
            "controller_bundle": _read_json(
                controller_root / "controller_manifest.json"
            ),
            "feature_backend": feature_backend,
            "workers": workers,
            "conflict_thresholds": list(REGISTERED_CONFLICT_THRESHOLDS),
            "pruner_threshold": None,
            "selection_unit": "complete_episode",
            "implementation_fingerprint": implementation_fingerprint,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("calibration output contains a different run")
        if not resume and not selection_only:
            raise ValueError("calibration output already exists; pass --resume")
    _write_json(
        run_path,
        {
            "schema": CALIBRATION_SCHEMA,
            "schema_version": CALIBRATION_VERSION,
            "run_fingerprint": run_fingerprint,
            "dataset": str(dataset_root),
            "dataset_fingerprint": _dataset_fingerprint(dataset_root),
            "base_config": str(config_path),
            "controller_bundle": str(controller_root),
            "feature_backend": feature_backend,
            "workers": workers,
            "selection_unit": "complete_episode",
            "registered_conflict_thresholds": list(
                REGISTERED_CONFLICT_THRESHOLDS
            ),
            "pruner_threshold": None,
            "policy_validation_used_for_selection": False,
            "implementation_fingerprint": implementation_fingerprint,
        },
    )
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {}
    for split, value in split_configs.items():
        destination = config_dir / f"{split}.json"
        _write_json(destination, value)
        config_paths[split] = destination

    if not selection_only:
        split = "policy_train"
        _run_collection(
            dataset=dataset_root,
            config_path=config_paths[split],
            root=_collection_root(output_root, split, "official_adaptive"),
            controller="v1-full",
            policy="official_adaptive",
            controller_bundle=controller_root,
            balanced_config=None,
            feature_backend=feature_backend,
            workers=workers,
            resume=resume,
        )
        _run_collection(
            dataset=dataset_root,
            config_path=config_paths[split],
            root=_collection_root(output_root, split, "v2-full"),
            controller="v2-full",
            policy="realized_dynamic",
            controller_bundle=controller_root,
            balanced_config=None,
            feature_backend=feature_backend,
            workers=workers,
            resume=resume,
        )
        for threshold in REGISTERED_CONFLICT_THRESHOLDS:
            _run_collection(
                dataset=dataset_root,
                config_path=config_paths[split],
                root=_collection_root(
                    output_root, split, "v2-balanced", int(threshold)
                ),
                controller="v2-balanced",
                policy="realized_dynamic",
                controller_bundle=controller_root,
                balanced_config=_threshold_payload(
                    int(threshold), run_fingerprint, split
                ),
                feature_backend=feature_backend,
                workers=workers,
                resume=resume,
            )

    training_official, training_full, training_balanced = _load_split_rows(
        output_root, "policy_train", REGISTERED_CONFLICT_THRESHOLDS
    )
    grid, folds = _training_grid(
        training_official, training_full, training_balanced
    )
    selected, selection_status = _select_training_candidate(grid)
    selected_threshold = int(selected["conflict_threshold"])
    _write_csv(output_root / "calibration_grid.csv", _flat_grid(grid))
    _write_json(
        output_root / "policy_train_selection.json",
        {
            "selection_status": selection_status,
            "selected": {
                key: value
                for key, value in selected.items()
                if key != "fold_metrics"
            },
            "map_group_folds": {
                str(fold): sorted(maps) for fold, maps in folds.items()
            },
            "policy_validation_used_for_selection": False,
        },
    )

    if not selection_only:
        split = "policy_validation"
        _run_collection(
            dataset=dataset_root,
            config_path=config_paths[split],
            root=_collection_root(output_root, split, "official_adaptive"),
            controller="v1-full",
            policy="official_adaptive",
            controller_bundle=controller_root,
            balanced_config=None,
            feature_backend=feature_backend,
            workers=workers,
            resume=resume,
        )
        _run_collection(
            dataset=dataset_root,
            config_path=config_paths[split],
            root=_collection_root(output_root, split, "v2-full"),
            controller="v2-full",
            policy="realized_dynamic",
            controller_bundle=controller_root,
            balanced_config=None,
            feature_backend=feature_backend,
            workers=workers,
            resume=resume,
        )
        _run_collection(
            dataset=dataset_root,
            config_path=config_paths[split],
            root=_collection_root(
                output_root, split, "v2-balanced", selected_threshold
            ),
            controller="v2-balanced",
            policy="realized_dynamic",
            controller_bundle=controller_root,
            balanced_config=_threshold_payload(
                selected_threshold, run_fingerprint, split
            ),
            feature_backend=feature_backend,
            workers=workers,
            resume=resume,
        )

    validation_official, validation_full, validation_balanced = _load_split_rows(
        output_root, "policy_validation", [selected_threshold]
    )
    validation_metrics = _complete_episode_metrics(
        validation_official,
        validation_full,
        validation_balanced[selected_threshold],
    )
    validation_passed = _passes(validation_metrics)
    configuration = BalancedControllerConfig(
        conflict_threshold=selected_threshold,
        pruner_threshold=None,
        source={
            "study_role": "complete_episode_policy_train_map_group_cross_validation",
            "calibration_run_fingerprint": run_fingerprint,
            "selection_status": selection_status,
            "policy_train_candidate_eligible": bool(selected["eligible"]),
            "policy_validation_locked_passed": validation_passed,
            "policy_validation_used_for_selection": False,
            "proposal_pruner": "disabled_locked_offline_validation_failed",
            "selection_unit": "complete_episode",
        },
        promoted=False,
    )
    write_balanced_controller(
        output_root / "balanced_controller.json", configuration
    )
    report = {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": CALIBRATION_VERSION,
        "run_fingerprint": run_fingerprint,
        "selection_status": selection_status,
        "selected": {
            key: value for key, value in selected.items() if key != "fold_metrics"
        },
        "policy_validation": {
            **validation_metrics,
            "locked_validation_passed": validation_passed,
            "used_for_selection": False,
        },
        "map_group_folds": {
            str(fold): sorted(maps) for fold, maps in folds.items()
        },
        "registered_conflict_thresholds": list(
            REGISTERED_CONFLICT_THRESHOLDS
        ),
        "pruner_search_status": "disabled_locked_offline_validation_failed",
        "selection_unit": "complete_episode",
        "configuration": configuration.payload(),
        "default_controller_changed": False,
    }
    _write_json(output_root / "calibration_report.json", report)
    return report


__all__ = [
    "CALIBRATION_SCHEMA",
    "_complete_episode_metrics",
    "_passes",
    "run_complete_episode_calibration",
]
