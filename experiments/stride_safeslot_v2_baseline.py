from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.stride_guardpool_maze_regression import _controller_kwargs
from experiments.stride_slotpool_structpool_ttf import (
    _expected_keys,
    load_slotpool_structpool_ttf_config,
)


CONFIG_SCHEMA = "lns2.stride.safeslot_v2_baseline_config.v1"
REPORT_SCHEMA = "lns2.stride.safeslot_v2_baseline_report.v1"
EXPERIMENT_ID = "stride-safeslot-v2-baseline-v1"
REPORT_FILENAME = "safeslot_v2_baseline_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered SafeSlot V2 input changed: {path}")
    return path


def load_safeslot_v2_baseline_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], set[tuple[str, str, int]]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_same_runtime_v2_trajectory_baseline_no_timing_claim"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "f33be7f03837c9f1139e3d5d6cd5f6745b0638b5"
        or config.get("controller") != "v2-full"
        or config.get("controller_bundle")
        != "artifacts/initlns-closed-loop-controller-v2"
        or config.get("qualification_source")
        != "build/stride-slotpool-structpool-ttf-diagnostic-v1/qualification"
    ):
        raise ValueError("SafeSlot V2 baseline identity changed")
    if dict(config.get("cohort") or {}) != {
        "source_config": "configs/stride_slotpool_structpool_ttf_diagnostic_v1.json",
        "paired_key_count": 29,
        "known_tail_exclusion_count": 1,
    }:
        raise ValueError("SafeSlot V2 baseline cohort changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_lean_confirmation_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
        "workers": 1,
    }:
        raise ValueError("SafeSlot V2 baseline runtime changed")
    if dict(config.get("integrity") or {}) != {
        "exact_29_keys_required": True,
        "all_success_required": True,
        "zero_errors_required": True,
        "zero_invalid_actions_required": True,
        "zero_fingerprint_mismatches_required": True,
        "same_registered_qualification_required": True,
    }:
        raise ValueError("SafeSlot V2 baseline integrity contract changed")
    if dict(config.get("claim_boundary") or {}) != {
        "trajectory_completion_only": True,
        "formal_ttf_claim": False,
        "wall_clock_comparison_allowed": False,
        "model_training_allowed": False,
        "threshold_selection_allowed": False,
        "no_additional_result_based_exclusion": True,
        "purpose": "complete_matched_runtime_v2_anchor_counterfactual_for_first_divergence_audit",
    }:
        raise ValueError("SafeSlot V2 baseline claim boundary changed")
    expected_inputs = {
        "cohort_config",
        "runtime_config",
        "controller_manifest",
        "dataset_summary",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "qualification_run_config",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SafeSlot V2 baseline input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    _source_path, _source_root, source = load_slotpool_structpool_ttf_config(
        inputs["cohort_config"]
    )
    expected = _expected_keys(source)
    if len(expected) != 29:
        raise ValueError("SafeSlot V2 baseline requires exactly 29 keys")
    return path, root, config, expected


def safeslot_v2_schedule(
    expected: set[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "group_id": group,
            "task_id": task,
            "solver_seed": seed,
            "controller": "v2-full",
        }
        for group, task, seed in sorted(expected)
    ]


def analyze_safeslot_v2_baseline(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config, expected = load_safeslot_v2_baseline_config(config_path)
    output = Path(output).resolve()
    manifest = output / "realized_dynamic_manifest.jsonl"
    rows = _read_jsonl(manifest)
    by_key = {
        (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
    }
    expected_task_seeds = {(task, seed) for _group, task, seed in expected}
    errors: list[str] = []
    if set(by_key) != expected_task_seeds:
        errors.append("V2 baseline manifest does not contain the exact 29 keys")
    success_count = 0
    invalid_action_count = 0
    fingerprint_mismatch_count = 0
    for task, seed in sorted(expected_task_seeds & set(by_key)):
        row = by_key[(task, seed)]
        summary = dict(row.get("summary") or {})
        if row.get("status") != "ok" or row.get("error") is not None:
            errors.append(f"non-ok V2 baseline row: {task} seed {seed}")
        success_count += int(bool(summary.get("success")))
        invalid_action_count += int(summary.get("invalid_action_count", 0))
        fingerprint_mismatch_count += int(
            summary.get("fingerprint_mismatch_count", 0)
        )
    integrity_passed = (
        not errors
        and len(rows) == 29
        and success_count == 29
        and invalid_action_count == 0
        and fingerprint_mismatch_count == 0
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "same_runtime_v2_trajectory_baseline_complete_no_timing_claim",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "integrity_passed": integrity_passed,
        "errors": errors,
        "completed_key_count": len(rows),
        "success_count": success_count,
        "invalid_action_count": invalid_action_count,
        "fingerprint_mismatch_count": fingerprint_mismatch_count,
        "manifest_sha256": sha256_file(manifest),
        "schedule_sha256": _fingerprint(safeslot_v2_schedule(expected)),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


def run_safeslot_v2_baseline(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _path, root, config, expected = load_safeslot_v2_baseline_config(config_path)
    schedule = safeslot_v2_schedule(expected)
    if dry_run:
        return {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "dry_run": True,
        }
    output = Path(output).resolve()
    manifest = output / "realized_dynamic_manifest.jsonl"
    if manifest.is_file() and not resume:
        raise ValueError("SafeSlot V2 baseline exists; pass --resume")
    source_config = _read_json(
        (root / str(config["cohort"]["source_config"])).resolve()
    )
    dataset = (root / str(source_config["cohort"]["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    qualification = (root / str(config["qualification_source"])).resolve()
    job_keys = {(task, seed) for _group, task, seed in expected}
    run_closed_loop_collection(
        dataset,
        runtime,
        output,
        phase="qualify",
        workers=1,
        resume=output.joinpath("run_config.json").is_file(),
        cohort_job_keys=job_keys,
        job_keys=job_keys,
        qualification_source=qualification,
        **_controller_kwargs(root, config, "v2-full"),
    )
    run_closed_loop_collection(
        dataset,
        runtime,
        output,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        cohort_job_keys=job_keys,
        job_keys=job_keys,
        qualification_source=qualification,
        **_controller_kwargs(root, config, "v2-full"),
    )
    return analyze_safeslot_v2_baseline(config_path, output)


__all__ = [
    "REPORT_FILENAME",
    "analyze_safeslot_v2_baseline",
    "load_safeslot_v2_baseline_config",
    "run_safeslot_v2_baseline",
    "safeslot_v2_schedule",
]
