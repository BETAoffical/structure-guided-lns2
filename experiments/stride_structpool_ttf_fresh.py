from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import registered_input
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
)
from experiments.stride_structpool_paired_ttf import (
    analyze_structpool_paired_ttf,
    run_structpool_paired_ttf,
)
from lns2_selector.evaluation.episode_statistics import dataset_tasks as _dataset_tasks
from experiments.stride_structpool_ttf_quick import (
    CONTROLLERS,
    structpool_ttf_schedule,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_ttf_fresh_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_ttf_fresh_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_ttf_fresh_report.v1"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="StructPool fresh")


def load_structpool_ttf_fresh_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_fresh_map_raw_ttf_after_four_seed_development_pass_before_fresh_timing"
        or config.get("experiment_id") != "stride-structpool-ttf-fresh-v1"
        or config.get("pre_registration_parent_commit")
        != "77679eedb9b45f184895515efa9865de50b3f3b6"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
    ):
        raise ValueError("StructPool fresh TTF identity changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_ranking_over_exact_base_pool",
        "challenger": "same_frozen_v2_ranking_over_base_plus_structpool",
        "ranker_changed": False,
        "paired_solver_seed_required": True,
        "deterministic_pp_seed_contract": "candidate_bound_native_replay",
        "execution_order": "alternating_pair_order",
        "workers": 1,
    }:
        raise ValueError("StructPool fresh TTF comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_fresh_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
    }:
        raise ValueError("StructPool fresh TTF runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role")
        != "structpool_unseen_cross_layout_raw_ttf_not_used_for_training_or_threshold_selection"
        or cohort.get("dataset") != "build/stride-augcontrol-ood-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2, 3)
        or int(cohort.get("episode_count_per_controller", -1)) != 36
        or [str(row.get("id")) for row in groups]
        != ["maze200", "room400", "random500", "warehouse500", "den300", "lak500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("StructPool fresh TTF cohort changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.05,
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("StructPool fresh TTF gates changed")
    expected_inputs = {
        "confirmation_config",
        "confirmation_report",
        "runtime_config",
        "controller_manifest",
        "dataset_config",
        "fetched_manifest",
        "dataset_summary",
        "dataset_manifest",
        "label_state_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool fresh TTF input registry changed")
    for specification in dict(config["inputs"]).values():
        _registered(root, dict(specification))
    confirmation = _read_json(
        root / str(config["inputs"]["confirmation_report"]["path"])
    )
    if (
        confirmation.get("integrity_passed") is not True
        or confirmation.get("performance_passed") is not True
        or confirmation.get("next_step") != "preregister_fresh_map_raw_ttf"
    ):
        raise ValueError("StructPool confirmation did not authorize fresh-map TTF")
    indexed = _dataset_tasks(
        (root / str(cohort["dataset"])).resolve(), str(cohort["split"])
    )
    expected_tasks = {
        str(task) for group in groups for task in list(group.get("tasks") or ())
    }
    if set(indexed) != expected_tasks:
        raise ValueError("StructPool fresh dataset task coverage changed")
    for group in groups:
        if {
            str(indexed[str(task)]["map_id"]) for task in group["tasks"]
        } != {str(group["map_id"])}:
            raise ValueError("StructPool fresh task-to-map identity changed")
    label_maps = {
        str(row["map_id"])
        for row in _read_jsonl(
            root / str(config["inputs"]["label_state_manifest"]["path"])
        )
    }
    fresh_maps = {str(group["map_id"]) for group in groups}
    if fresh_maps & label_maps:
        raise ValueError("StructPool fresh maps overlap the label cohort")
    confirmation_config = _read_json(
        root / str(config["inputs"]["confirmation_config"]["path"])
    )
    development_maps: set[str] = set()
    for group in confirmation_config["cohort"]["groups"]:
        source_rows = _dataset_tasks(
            (root / str(group["dataset"])).resolve(), str(group["split"])
        )
        development_maps.update(
            str(source_rows[str(task)]["map_id"]) for task in group["tasks"]
        )
    if fresh_maps & development_maps:
        raise ValueError("StructPool fresh maps overlap the development TTF cohort")
    return path, root, config


def structpool_fresh_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    return structpool_ttf_schedule(config)


def run_structpool_ttf_fresh(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_ttf_fresh_config(config_path)
    return run_structpool_paired_ttf(
        path,
        root,
        config,
        output,
        status_schema=STATUS_SCHEMA,
        status_filename="fresh_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="structpool_ttf_fresh_report.json",
        report_scientific_status="fresh_map_cross_layout_raw_ttf_confirmation",
        next_step_on_pass="preregister_independent_replication_and_default_promotion_design",
        next_step_on_failure="stop_structpool_ttf_promotion_and_diagnose_fresh_map_failures",
        performance_claim_field="cross_layout_generalization_supported",
        producer_source_files=(
            "experiments/stride_structpool_ttf_fresh.py",
        ),
        resume=resume,
        dry_run=dry_run,
    )


def analyze_structpool_ttf_fresh(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_ttf_fresh_config(config_path)
    return analyze_structpool_paired_ttf(
        path,
        config,
        output,
        status_filename="fresh_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="structpool_ttf_fresh_report.json",
        report_scientific_status="fresh_map_cross_layout_raw_ttf_confirmation",
        next_step_on_pass="preregister_independent_replication_and_default_promotion_design",
        next_step_on_failure="stop_structpool_ttf_promotion_and_diagnose_fresh_map_failures",
        performance_claim_field="cross_layout_generalization_supported",
        producer_source_files=(
            "experiments/stride_structpool_ttf_fresh.py",
        ),
    )


__all__ = [
    "analyze_structpool_ttf_fresh",
    "load_structpool_ttf_fresh_config",
    "run_structpool_ttf_fresh",
    "structpool_fresh_schedule",
]
