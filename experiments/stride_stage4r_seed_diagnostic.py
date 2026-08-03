from __future__ import annotations

import copy
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_stage4r_quick import (
    CONTROLLERS,
    TTF_CLOCK_SCHEMA,
    _controller_kwargs,
    _controller_summary,
    _mean,
    _metric,
    _project_path,
)


STRIDE_STAGE4R_SEED_DIAGNOSTIC_REGISTRATION_SCHEMA = (
    "lns2.stride.stage4r_seed_diagnostic_registration.v1"
)
STRIDE_STAGE4R_SEED_DIAGNOSTIC_PROTOCOL_SCHEMA = (
    "lns2.stride.stage4r_seed_diagnostic_protocol.v1"
)
STRIDE_STAGE4R_SEED_DIAGNOSTIC_DATASET_SCHEMA = (
    "lns2.stride.stage4r_seed_diagnostic_dataset.v1"
)
STRIDE_STAGE4R_SEED_DIAGNOSTIC_REPORT_SCHEMA = (
    "lns2.stride.stage4r_seed_diagnostic_report.v1"
)
REGISTERED_SOLVER_SEEDS = (1, 2, 3, 4)
REGISTERED_WALL_BUDGET_SECONDS = 120.0
REGISTERED_PROCESS_TIMEOUT_SECONDS = 180.0
ADVERSE_SEED_THRESHOLD = 3


def _checked_file(
    project_root: Path,
    raw_path: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    path = _project_path(project_root, raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(expected_sha256).lower():
        raise ValueError(f"STRIDE Stage 4R seed diagnostic source hash differs: {label}")
    return path


def validate_seed_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != STRIDE_STAGE4R_SEED_DIAGNOSTIC_REGISTRATION_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4R seed diagnostic registration")
    protocol = dict(config.get("stride_stage4r_seed_diagnostic") or {})
    if protocol.get("schema") != STRIDE_STAGE4R_SEED_DIAGNOSTIC_PROTOCOL_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4R seed diagnostic protocol")
    if (
        protocol.get("scientific_status") != "diagnostic_only"
        or bool(protocol.get("formal_speed_claim"))
        or bool(protocol.get("formal_ood_data_allowed"))
    ):
        raise ValueError("Stage 4R seed diagnostic must remain non-formal")
    if tuple(protocol.get("controllers") or ()) != CONTROLLERS:
        raise ValueError("Stage 4R seed diagnostic controllers differ")
    if tuple(map(int, protocol.get("solver_seeds") or ())) != REGISTERED_SOLVER_SEEDS:
        raise ValueError("Stage 4R seed diagnostic solver seeds differ")
    if protocol.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA:
        raise ValueError("Stage 4R seed diagnostic TTF clock differs")
    if float(protocol.get("wall_time_budget_seconds", 0.0)) != REGISTERED_WALL_BUDGET_SECONDS:
        raise ValueError("Stage 4R seed diagnostic wall budget differs")
    if (
        float(protocol.get("episode_process_timeout_seconds", 0.0))
        != REGISTERED_PROCESS_TIMEOUT_SECONDS
    ):
        raise ValueError("Stage 4R seed diagnostic process timeout differs")
    if int(protocol.get("workers", 0)) != 1:
        raise ValueError("Stage 4R seed diagnostic requires serial execution")
    if protocol.get("execution_order") != "strict_rotating_latin_order":
        raise ValueError("Stage 4R seed diagnostic execution order differs")
    if protocol.get("selection_evidence_kind") != "outcome_informed_quick_v2":
        raise ValueError("Stage 4R seed diagnostic must disclose outcome selection")
    registered = tuple(map(str, protocol.get("registered_task_ids") or ()))
    if len(registered) != 4 or len(set(registered)) != 4:
        raise ValueError("Stage 4R seed diagnostic requires four unique tasks")
    problem_tasks = tuple(map(str, protocol.get("problem_task_ids") or ()))
    if len(problem_tasks) != 2 or not set(problem_tasks) < set(registered):
        raise ValueError("Stage 4R seed diagnostic problem-task registration differs")
    bundles = dict(protocol.get("controller_bundles") or {})
    if set(bundles) != set(CONTROLLERS):
        raise ValueError("Stage 4R seed diagnostic controller bundles differ")
    quick_manifests = dict(protocol.get("quick_v2_controller_manifests") or {})
    if set(quick_manifests) != set(CONTROLLERS):
        raise ValueError("Stage 4R seed diagnostic Quick evidence differs")
    return protocol


def _base_config(
    registration_path: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    project_root = registration_path.parents[1]
    path = _checked_file(
        project_root,
        str(registration["base_config"]),
        str(registration["base_config_sha256"]),
        "base_config",
    )
    return _read_json(path)


def _runtime_config_payload(
    registration_path: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    protocol = validate_seed_diagnostic_config(registration)
    runtime = copy.deepcopy(_base_config(registration_path, registration))
    runtime.pop("stride_stage4r_quick", None)
    runtime["formal"] = False
    runtime["solver_seeds"] = list(REGISTERED_SOLVER_SEEDS)
    runtime["workers"] = 1
    runtime["wall_time_budget_seconds"] = REGISTERED_WALL_BUDGET_SECONDS
    runtime["episode_process_timeout_seconds"] = REGISTERED_PROCESS_TIMEOUT_SECONDS
    runtime["environment"] = {
        **dict(runtime["environment"]),
        "time_limit": REGISTERED_WALL_BUDGET_SECONDS,
        "max_repair_iterations": 0,
    }
    runtime["qualification"] = {
        "enforce_registered_thresholds": True,
        "minimum_nonzero_states": 8,
        "minimum_nonzero_states_per_layout": 0,
        "minimum_active_maps": 2,
        "minimum_nonzero_states_per_solver_seed": 1,
    }
    runtime["dataset_design"] = {
        "mode": "balanced_wall_clock",
        "map_count": 4,
        "instance_count": 4,
        "source_counts": {"generated": 2, "movingai": 2},
        "layout_counts": {
            "compartmentalized": 2,
            "game": 1,
            "maze": 1,
        },
        "historical_map_ids": list(
            runtime.get("dataset_design", {}).get("historical_map_ids", [])
        ),
    }
    runtime["stride_stage4r_seed_diagnostic"] = protocol
    return runtime


def _materialize_runtime_config(
    registration_path: Path, destination: Path
) -> tuple[Path, dict[str, Any]]:
    registration = _read_json(registration_path)
    payload = _runtime_config_payload(registration_path, registration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _read_json(destination) != payload:
            raise ValueError("existing Stage 4R seed diagnostic runtime config differs")
    else:
        _write_json(destination, payload)
    return destination, payload


def _copy_registered_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"existing seed diagnostic dataset file differs: {destination}")
        return
    shutil.copy2(source, destination)


def _no_progress_steps(summary: dict[str, Any]) -> int:
    trajectory = [int(value) for value in summary.get("conflict_trajectory") or []]
    return sum(
        right >= left and left > 0
        for left, right in zip(trajectory, trajectory[1:])
    )


def prepare_seed_diagnostic_dataset(
    config_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    registration = _read_json(config_path)
    protocol = validate_seed_diagnostic_config(registration)
    source_manifest = _checked_file(
        project_root,
        str(protocol["source_manifest"]),
        str(protocol["source_manifest_sha256"]),
        "source_manifest",
    )
    source_summary_path = _checked_file(
        project_root,
        str(protocol["source_dataset_summary"]),
        str(protocol["source_dataset_summary_sha256"]),
        "source_dataset_summary",
    )
    quick_report_path = _checked_file(
        project_root,
        str(protocol["quick_v2_report"]),
        str(protocol["quick_v2_report_sha256"]),
        "quick_v2_report",
    )
    source_summary = _read_json(source_summary_path)
    quick_report = _read_json(quick_report_path)
    if (
        bool(source_summary.get("test_data_read"))
        or bool(source_summary.get("formal_ood_data_read"))
        or source_summary.get("training_map_overlap")
        or source_summary.get("formal_ood_map_overlap")
    ):
        raise ValueError("source Quick dataset does not preserve registered isolation")
    if (
        not bool(quick_report.get("passed"))
        or bool(quick_report.get("formal_speed_claim"))
        or quick_report.get("scientific_status") != "diagnostic_only"
    ):
        raise ValueError("Quick v2 selection evidence is not a passed diagnostic")

    quick_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    quick_hashes: dict[str, str] = {}
    for controller, evidence in dict(protocol["quick_v2_controller_manifests"]).items():
        manifest_path = _checked_file(
            project_root,
            str(evidence["path"]),
            str(evidence["sha256"]),
            f"quick_v2_controller_manifest:{controller}",
        )
        quick_hashes[controller] = sha256_file(manifest_path)
        quick_rows[controller] = {
            (str(row["task_id"]), int(row["solver_seed"])): row
            for row in _read_jsonl(manifest_path)
        }

    source_root = _project_path(project_root, str(protocol["source_dataset"]))
    split = str(protocol["split"])
    registered = list(map(str, protocol["registered_task_ids"]))
    source_rows = {
        str(row["task_id"]): row for row in _read_jsonl(source_manifest)
    }
    if any(task_id not in source_rows for task_id in registered):
        raise ValueError("registered seed diagnostic task is absent from source")
    rows = [source_rows[task_id] for task_id in registered]
    if len({str(row["map_id"]) for row in rows}) != len(rows):
        raise ValueError("seed diagnostic requires one task per map")

    selection_audit = []
    for task_id in registered:
        measurements = {}
        for controller in CONTROLLERS:
            row = quick_rows[controller].get((task_id, 1))
            if (
                row is None
                or row.get("status") != "ok"
                or not isinstance(row.get("summary"), dict)
            ):
                raise ValueError(f"missing Quick v2 outcome evidence: {controller}/{task_id}")
            summary = dict(row["summary"])
            measurements[controller] = {
                "success": bool(summary.get("success")),
                "capped_wall_time_to_feasible": _metric(
                    summary, "capped_wall_time_to_feasible"
                ),
                "repair_iterations": int(summary.get("repair_iterations", 0)),
                "no_progress_steps": _no_progress_steps(summary),
            }
        selection_audit.append(
            {
                "task_id": task_id,
                "role": dict(protocol["task_roles"])[task_id],
                "quick_v2_solver_seed": 1,
                "measurements": measurements,
            }
        )

    destination = Path(destination).resolve()
    split_root = destination / split
    for row in rows:
        for field, value in row.items():
            if field.endswith("_file") and isinstance(value, str):
                _copy_registered_file(
                    source_root / split / value,
                    split_root / value,
                )
    ordered_rows = sorted(rows, key=lambda row: str(row["task_id"]))
    manifest = split_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.is_file():
        if _read_jsonl(manifest) != ordered_rows:
            raise ValueError("existing seed diagnostic dataset manifest differs")
    else:
        _write_jsonl(manifest, ordered_rows)

    report = {
        "schema": STRIDE_STAGE4R_SEED_DIAGNOSTIC_DATASET_SCHEMA,
        "scientific_status": "diagnostic_only",
        "selection_outcomes_read": True,
        "selection_evidence_kind": "outcome_informed_quick_v2",
        "formal_speed_claim": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "task_count": len(rows),
        "map_count": len({str(row["map_id"]) for row in rows}),
        "source_counts": dict(
            sorted(Counter(str(row["source_group"]) for row in rows).items())
        ),
        "layout_counts": dict(
            sorted(Counter(str(row["layout_mode"]) for row in rows).items())
        ),
        "selection_rule": str(protocol["selection_rule"]),
        "selection_audit": selection_audit,
        "inputs": {
            "registration_sha256": sha256_file(config_path),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_dataset_summary_sha256": sha256_file(source_summary_path),
            "quick_v2_report_sha256": sha256_file(quick_report_path),
            "quick_v2_controller_manifest_sha256": quick_hashes,
        },
        "dataset_manifest_sha256": sha256_file(manifest),
    }
    report["registration_fingerprint"] = _fingerprint(report)
    _write_json(destination / "dataset_summary.json", report)
    return report


def seed_diagnostic_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_seed_diagnostic_config(config)
    keys = sorted(
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in REGISTERED_SOLVER_SEEDS
    )
    schedule = []
    ordinal = 0
    for key_index, (task_id, solver_seed) in enumerate(keys):
        offset = key_index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        for position, controller in enumerate(order):
            schedule.append(
                {
                    "ordinal": ordinal,
                    "task_id": task_id,
                    "solver_seed": solver_seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
            ordinal += 1
    return schedule


def run_seed_diagnostic(
    config_path: str | Path,
    dataset: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    registration = _read_json(config_path)
    protocol = validate_seed_diagnostic_config(registration)
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    dataset_report = prepare_seed_diagnostic_dataset(config_path, dataset)
    runtime_config_path, _runtime = _materialize_runtime_config(
        config_path, output / "runtime_config.json"
    )
    schedule = seed_diagnostic_schedule(registration)
    cohort = {
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in REGISTERED_SOLVER_SEEDS
    }
    status = {
        "schema": STRIDE_STAGE4R_SEED_DIAGNOSTIC_PROTOCOL_SCHEMA,
        "scientific_status": "diagnostic_only",
        "formal_speed_claim": False,
        "registration_sha256": sha256_file(config_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "dataset_registration_fingerprint": dataset_report[
            "registration_fingerprint"
        ],
        "schedule_sha256": _fingerprint(schedule),
        "schedule": schedule,
        "complete": False,
    }
    if dry_run:
        status["controller_dry_runs"] = {
            controller: run_closed_loop_collection(
                dataset,
                runtime_config_path,
                output / "dry-run" / controller,
                phase="realized_dynamic",
                workers=1,
                dry_run=True,
                cohort_job_keys=cohort,
                job_keys=cohort,
                **_controller_kwargs(project_root, protocol, controller),
            )
            for controller in CONTROLLERS
        }
        return status

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    status_path = output / "seed_diagnostic_status.json"
    existing_status = _read_json(status_path) if status_path.is_file() else None
    if existing_status is not None:
        if existing_status.get("schedule_sha256") != status["schedule_sha256"]:
            raise ValueError("existing Stage 4R seed diagnostic schedule differs")
        if not resume:
            raise ValueError("Stage 4R seed diagnostic output exists; pass resume")
    _write_json(status_path, status)

    qualification_root = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime_config_path,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=cohort,
        job_keys=cohort,
        **_controller_kwargs(project_root, protocol, "v2-full"),
    )
    for controller in CONTROLLERS:
        controller_root = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime_config_path,
            controller_root,
            phase="qualify",
            workers=1,
            resume=controller_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=cohort,
            job_keys=cohort,
            qualification_source=qualification_root,
            **_controller_kwargs(project_root, protocol, controller),
        )

    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        controller_root = output / "controllers" / controller
        manifest = controller_root / "realized_dynamic_manifest.jsonl"
        done_keys = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done_keys:
            run_closed_loop_collection(
                dataset,
                runtime_config_path,
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=cohort,
                job_keys={key},
                **_controller_kwargs(project_root, protocol, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **status,
                "completed_schedule_entries": completed,
                "total_schedule_entries": len(schedule),
                "current": item,
                "complete": False,
            },
        )

    report = analyze_seed_diagnostic(config_path, output)
    _write_json(
        status_path,
        {
            **status,
            "completed_schedule_entries": len(schedule),
            "total_schedule_entries": len(schedule),
            "complete": True,
            "report_sha256": sha256_file(
                output / "stage4r_seed_diagnostic_report.json"
            ),
        },
    )
    return report


def _extended_controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _controller_summary(rows)
    episodes = [
        dict(row["summary"])
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    capped = [_metric(summary, "capped_wall_time_to_feasible") for summary in episodes]
    result.update(
        {
            "median_capped_wall_time_to_feasible": (
                statistics.median(capped) if capped else 0.0
            ),
            "mean_no_progress_steps": _mean(
                _no_progress_steps(summary) for summary in episodes
            ),
            "total_no_progress_steps": sum(
                _no_progress_steps(summary) for summary in episodes
            ),
        }
    )
    return result


def _paired_comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    challenger: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    pairs = []
    for key in keys:
        left_row = baseline.get(key)
        right_row = challenger.get(key)
        if (
            left_row is None
            or right_row is None
            or left_row.get("status") != "ok"
            or right_row.get("status") != "ok"
            or not isinstance(left_row.get("summary"), dict)
            or not isinstance(right_row.get("summary"), dict)
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        pairs.append((key, dict(left_row["summary"]), dict(right_row["summary"])))

    baseline_capped = [_metric(left, "capped_wall_time_to_feasible") for _, left, _ in pairs]
    challenger_capped = [
        _metric(right, "capped_wall_time_to_feasible") for _, _, right in pairs
    ]
    deltas = [right - left for left, right in zip(baseline_capped, challenger_capped)]
    tolerance = 1e-9
    common_success = [
        (left, right)
        for _key, left, right in pairs
        if bool(left.get("success")) and bool(right.get("success"))
    ]
    baseline_common_ttf = _mean(
        _metric(left, "wall_time_to_feasible") for left, _ in common_success
    )
    challenger_common_ttf = _mean(
        _metric(right, "wall_time_to_feasible") for _, right in common_success
    )
    slower_or_more_rounds = [
        int(delta > tolerance)
        or int(right.get("repair_iterations", 0))
        > int(left.get("repair_iterations", 0))
        for delta, (_key, left, right) in zip(deltas, pairs)
    ]
    baseline_success = sum(bool(left.get("success")) for _key, left, _right in pairs)
    challenger_success = sum(bool(right.get("success")) for _key, _left, right in pairs)
    baseline_mean = _mean(baseline_capped)
    challenger_mean = _mean(challenger_capped)
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_success_count": baseline_success,
        "challenger_success_count": challenger_success,
        "success_noninferior": challenger_success >= baseline_success,
        "baseline_mean_capped_ttf": baseline_mean,
        "challenger_mean_capped_ttf": challenger_mean,
        "mean_capped_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_capped_ttf_relative_improvement": (
            (baseline_mean - challenger_mean) / baseline_mean if baseline_mean else 0.0
        ),
        "baseline_median_capped_ttf": statistics.median(baseline_capped),
        "challenger_median_capped_ttf": statistics.median(challenger_capped),
        "faster_count": sum(delta < -tolerance for delta in deltas),
        "slower_count": sum(delta > tolerance for delta in deltas),
        "tied_count": sum(abs(delta) <= tolerance for delta in deltas),
        "slower_or_more_rounds_count": sum(slower_or_more_rounds),
        "common_success_count": len(common_success),
        "baseline_common_success_mean_ttf": baseline_common_ttf,
        "challenger_common_success_mean_ttf": challenger_common_ttf,
        "common_success_relative_ttf_improvement": (
            (baseline_common_ttf - challenger_common_ttf) / baseline_common_ttf
            if baseline_common_ttf
            else 0.0
        ),
        "mean_repair_iterations_delta": _mean(
            int(right.get("repair_iterations", 0))
            - int(left.get("repair_iterations", 0))
            for _key, left, right in pairs
        ),
        "mean_no_progress_steps_delta": _mean(
            _no_progress_steps(right) - _no_progress_steps(left)
            for _key, left, right in pairs
        ),
        "mean_wall_auc_delta": _mean(
            _metric(right, "normalized_wall_clock_conflict_auc")
            - _metric(left, "normalized_wall_clock_conflict_auc")
            for _key, left, right in pairs
        ),
        "per_seed": [
            {
                "task_id": key[0],
                "solver_seed": key[1],
                "baseline_capped_ttf": left_ttf,
                "challenger_capped_ttf": right_ttf,
                "capped_ttf_delta_seconds": right_ttf - left_ttf,
                "repair_iterations_delta": int(right.get("repair_iterations", 0))
                - int(left.get("repair_iterations", 0)),
                "no_progress_steps_delta": _no_progress_steps(right)
                - _no_progress_steps(left),
            }
            for (key, left, right), left_ttf, right_ttf in zip(
                pairs, baseline_capped, challenger_capped
            )
        ],
    }


def analyze_seed_diagnostic(
    config_path: str | Path,
    collection: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    registration = _read_json(config_path)
    protocol = validate_seed_diagnostic_config(registration)
    collection = Path(collection).resolve()
    expected_keys = {
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in REGISTERED_SOLVER_SEEDS
    }
    by_controller: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    errors = []
    for controller in CONTROLLERS:
        path = collection / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(path)
        indexed = {
            (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
        }
        if len(indexed) != len(rows):
            errors.append(f"{controller}: duplicate task/seed rows")
        if set(indexed) != expected_keys:
            errors.append(f"{controller}: incomplete paired coverage")
        episode_errors = sum(
            row.get("status") != "ok" or not isinstance(row.get("summary"), dict)
            for row in rows
        )
        if episode_errors:
            errors.append(f"{controller}: {episode_errors} episode execution error(s)")
        by_controller[controller] = indexed

    fingerprint_mismatches = 0
    conflict_mismatches = 0
    if not errors:
        for key in sorted(expected_keys):
            summaries = [by_controller[name][key]["summary"] for name in CONTROLLERS]
            fingerprint_mismatches += len(
                {str(summary["initial_fingerprint"]) for summary in summaries}
            ) != 1
            conflict_mismatches += len(
                {int(summary["initial_conflicts"]) for summary in summaries}
            ) != 1

    summaries = {
        controller: _extended_controller_summary(list(indexed.values()))
        for controller, indexed in by_controller.items()
    }
    all_keys = sorted(expected_keys)
    comparisons = {
        challenger: _paired_comparison(
            by_controller["v2-full"], by_controller[challenger], all_keys
        )
        for challenger in CONTROLLERS[1:]
    }
    per_task = {}
    for task_id in protocol["registered_task_ids"]:
        keys = [(str(task_id), seed) for seed in REGISTERED_SOLVER_SEEDS]
        per_task[str(task_id)] = {
            "role": dict(protocol["task_roles"])[str(task_id)],
            "controller_summaries": {
                controller: _extended_controller_summary(
                    [
                        by_controller[controller][key]
                        for key in keys
                        if key in by_controller[controller]
                    ]
                )
                for controller in CONTROLLERS
            },
            "comparisons_vs_v2_full": {
                challenger: _paired_comparison(
                    by_controller["v2-full"], by_controller[challenger], keys
                )
                for challenger in CONTROLLERS[1:]
            },
        }

    problem_evidence = {
        task_id: per_task[task_id]["comparisons_vs_v2_full"]["stride-quality-v1"]
        for task_id in protocol["problem_task_ids"]
    }
    intrinsic_tail_weakness = bool(problem_evidence) and all(
        bool(evidence.get("valid"))
        and int(evidence.get("slower_or_more_rounds_count", 0))
        >= ADVERSE_SEED_THRESHOLD
        for evidence in problem_evidence.values()
    )
    problem_faster = sum(
        int(evidence.get("faster_count", 0)) for evidence in problem_evidence.values()
    )
    problem_slower = sum(
        int(evidence.get("slower_count", 0)) for evidence in problem_evidence.values()
    )
    seed_sensitive = (
        not intrinsic_tail_weakness and problem_faster > 0 and problem_slower > 0
    )

    gates = {
        "complete_paired_coverage": not any("coverage" in error for error in errors),
        "zero_episode_errors": not any("execution error" in error for error in errors),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "zero_invalid_actions": all(
            int(summary.get("invalid_action_count", 0)) == 0
            for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            int(summary.get("fingerprint_mismatch_count", 0)) == 0
            for summary in summaries.values()
        ),
        "ttf_clock_registered": all(
            row.get("status") == "ok"
            and isinstance(row.get("summary"), dict)
            and row["summary"].get("ttf_clock_schema") == TTF_CLOCK_SCHEMA
            for indexed in by_controller.values()
            for row in indexed.values()
        ),
    }
    passed = not errors and all(gates.values())
    quality = comparisons["stride-quality-v1"]
    if not passed:
        next_decision = "repair_execution_or_analysis_before_runtime_conclusion"
    elif intrinsic_tail_weakness:
        next_decision = "revise_one_step_label_for_residual_state_hardness"
    elif seed_sensitive:
        next_decision = "run_same_state_paired_pp_repair_replay_before_label_change"
    elif (
        bool(quality.get("success_noninferior"))
        and float(quality.get("mean_capped_ttf_relative_improvement", 0.0)) > 0.0
    ):
        next_decision = "expand_to_outcome_blind_multiseed_development_cohort"
    else:
        next_decision = "revise_quality_label_before_more_runtime_evaluation"

    eligible = [
        controller
        for controller in CONTROLLERS
        if controller == "v2-full"
        or int(summaries[controller]["success_count"])
        >= int(summaries["v2-full"]["success_count"])
    ] if passed else []
    observed_best = min(
        eligible,
        key=lambda controller: float(
            summaries[controller]["mean_capped_wall_time_to_feasible"]
        ),
    ) if eligible else None
    report = {
        "schema": STRIDE_STAGE4R_SEED_DIAGNOSTIC_REPORT_SCHEMA,
        "scientific_status": "diagnostic_only",
        "selection_evidence_kind": "outcome_informed_quick_v2",
        "formal_speed_claim": False,
        "formal_primary_winner": None,
        "observed_mean_capped_ttf_best_controller": observed_best,
        "primary_metric": "mean_capped_wall_time_to_feasible",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected_keys),
        "solver_seeds": list(REGISTERED_SOLVER_SEEDS),
        "controller_summaries": summaries,
        "comparisons_vs_v2_full": comparisons,
        "per_task": per_task,
        "diagnosis": {
            "adverse_seed_threshold_per_problem_task": ADVERSE_SEED_THRESHOLD,
            "intrinsic_tail_weakness": intrinsic_tail_weakness,
            "seed_sensitive": seed_sensitive,
            "problem_task_evidence": problem_evidence,
        },
        "next_decision": next_decision,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "initial_conflict_mismatch_count": conflict_mismatches,
        "gates": gates,
        "passed": passed,
        "errors": errors,
        "selection_outcomes_read": True,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "registration_sha256": sha256_file(config_path),
            "schedule_sha256": sha256_file(collection / "execution_schedule.jsonl"),
        },
    }
    _write_json(collection / "stage4r_seed_diagnostic_report.json", report)
    return report


__all__ = [
    "ADVERSE_SEED_THRESHOLD",
    "CONTROLLERS",
    "REGISTERED_SOLVER_SEEDS",
    "STRIDE_STAGE4R_SEED_DIAGNOSTIC_REPORT_SCHEMA",
    "TTF_CLOCK_SCHEMA",
    "analyze_seed_diagnostic",
    "prepare_seed_diagnostic_dataset",
    "run_seed_diagnostic",
    "seed_diagnostic_schedule",
    "validate_seed_diagnostic_config",
]
