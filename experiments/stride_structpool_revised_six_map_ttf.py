from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import registered_input
from experiments.repair_collection import _read_json
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_structpool_paired_ttf import (
    analyze_structpool_paired_ttf,
    run_structpool_paired_ttf,
)
from experiments.stride_structpool_ttf_quick import (
    CONTROLLERS,
    structpool_ttf_schedule,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_revised_six_map_ttf_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_revised_six_map_ttf_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_revised_six_map_ttf_report.v1"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="revised six-map TTF")


def load_revised_six_map_ttf_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_qualification_conditioned_paired_raw_ttf"
        or config.get("experiment_id")
        != "stride-structpool-revised-six-map-ttf-v1"
        or config.get("pre_registration_parent_commit") != "18ca9bd"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
        or config.get("qualification_source")
        != "build/stride-structpool-revised-six-map-qualification-v1"
    ):
        raise ValueError("revised six-map TTF identity changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_ranking_over_exact_base_pool",
        "challenger": "same_frozen_v2_ranking_over_base_plus_structpool",
        "ranker_changed": False,
        "candidate_pool_only_difference": True,
        "paired_solver_seed_required": True,
        "deterministic_pp_seed_contract": "candidate_bound_native_replay",
        "execution_order": "alternating_pair_order",
        "workers": 1,
    }:
        raise ValueError("revised six-map TTF comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_revised_six_map_ttf_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
    }:
        raise ValueError("revised six-map TTF runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role")
        != "qualification_conditioned_map_factor_diagnostic_not_fresh_ood"
        or cohort.get("dataset")
        != "build/stride-structpool-revised-six-map-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2, 3)
        or int(cohort.get("episode_count_per_controller", -1)) != 36
        or [str(row.get("id")) for row in groups]
        != ["den300", "maze100", "orz600", "random500", "room400", "warehouse600"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("revised six-map TTF cohort changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.05,
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("revised six-map TTF gates changed")
    expected_inputs = {
        "qualification_design",
        "qualification_analysis",
        "qualification_manifest",
        "qualification_report",
        "qualification_run_config",
        "runtime_config",
        "controller_manifest",
        "dataset_summary",
        "dataset_manifest",
        "artifact_registry",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("revised six-map TTF input registry changed")
    registered = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    qualification = _read_json(registered["qualification_analysis"])
    source_report = _read_json(registered["qualification_report"])
    if qualification.get("passed") is not True or source_report.get("passed") is not True:
        raise ValueError("revised six-map reset qualification did not pass")
    indexed = _dataset_tasks(
        (root / str(cohort["dataset"])).resolve(), str(cohort["split"])
    )
    expected_tasks = {
        str(task) for group in groups for task in list(group.get("tasks") or ())
    }
    if set(indexed) != expected_tasks:
        raise ValueError("revised six-map TTF task coverage changed")
    for group in groups:
        if {str(indexed[str(task)]["map_id"]) for task in group["tasks"]} != {
            str(group["map_id"])
        }:
            raise ValueError("revised six-map TTF task-to-map identity changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "new_ranker_trained": False,
        "candidate_pool_only_comparison": True,
        "qualification_conditioned": True,
        "fresh_ood_or_generalization_claim": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "pass_authorizes_only_independent_replication_design": True,
        "failure_stops_current_structpool_promotion": True,
    }:
        raise ValueError("revised six-map TTF claim boundary changed")
    return path, root, config


def revised_six_map_ttf_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    return structpool_ttf_schedule(config)


def run_revised_six_map_ttf(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_revised_six_map_ttf_config(config_path)
    return run_structpool_paired_ttf(
        path,
        root,
        config,
        output,
        status_schema=STATUS_SCHEMA,
        status_filename="evaluation_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="revised_six_map_ttf_report.json",
        report_scientific_status="qualification_conditioned_paired_raw_ttf_diagnostic",
        next_step_on_pass="preregister_independent_structpool_replication",
        next_step_on_failure="stop_current_structpool_promotion_and_diagnose_per_map",
        fixed_report_fields={
            "qualification_conditioned": True,
            "fresh_ood_or_generalization_supported": False,
        },
        producer_source_files=(
            "experiments/stride_structpool_revised_six_map_ttf.py",
            "experiments/stride_augcontrol_evaluation.py",
        ),
        resume=resume,
        dry_run=dry_run,
    )


def analyze_revised_six_map_ttf(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_revised_six_map_ttf_config(config_path)
    return analyze_structpool_paired_ttf(
        path,
        config,
        output,
        status_filename="evaluation_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="revised_six_map_ttf_report.json",
        report_scientific_status="qualification_conditioned_paired_raw_ttf_diagnostic",
        next_step_on_pass="preregister_independent_structpool_replication",
        next_step_on_failure="stop_current_structpool_promotion_and_diagnose_per_map",
        fixed_report_fields={
            "qualification_conditioned": True,
            "fresh_ood_or_generalization_supported": False,
        },
        producer_source_files=(
            "experiments/stride_structpool_revised_six_map_ttf.py",
            "experiments/stride_augcontrol_evaluation.py",
        ),
    )


__all__ = [
    "analyze_revised_six_map_ttf",
    "load_revised_six_map_ttf_config",
    "revised_six_map_ttf_schedule",
    "run_revised_six_map_ttf",
]
