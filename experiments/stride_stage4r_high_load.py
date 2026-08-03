from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_stage4r_quick import TTF_CLOCK_SCHEMA, _project_path


REGISTRATION_SCHEMA = "lns2.stride.stage4r_high_load_registration.v1"
REPORT_SCHEMA = "lns2.stride.stage4r_high_load_report.v1"
STATUS_SCHEMA = "lns2.stride.stage4r_high_load_status.v1"
CONTROLLERS = ("v2-full", "stride-quality-v1")
SOLVER_SEEDS = (1, 2, 3, 4)
WALL_BUDGET_SECONDS = 180.0
PROCESS_TIMEOUT_SECONDS = 240.0


def _mean(values: Iterable[float | int]) -> float:
    numbers = list(map(float, values))
    return statistics.fmean(numbers) if numbers else 0.0


def _metric(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def _no_progress_steps(summary: dict[str, Any]) -> int:
    trajectory = [int(value) for value in summary.get("conflict_trajectory") or []]
    return sum(
        right >= left and left > 0
        for left, right in zip(trajectory, trajectory[1:])
    )


def validate_high_load_registration(config: dict[str, Any]) -> None:
    if config.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4R high-load registration")
    if (
        config.get("scientific_status") != "development_diagnostic_only"
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("high-load run must remain a development diagnostic")
    if tuple(config.get("controllers") or ()) != CONTROLLERS:
        raise ValueError("high-load controller registration differs")
    if tuple(map(int, config.get("solver_seeds") or ())) != SOLVER_SEEDS:
        raise ValueError("high-load solver seed registration differs")
    if float(config.get("wall_time_budget_seconds", 0.0)) != WALL_BUDGET_SECONDS:
        raise ValueError("high-load wall budget differs")
    if (
        float(config.get("episode_process_timeout_seconds", 0.0))
        != PROCESS_TIMEOUT_SECONDS
    ):
        raise ValueError("high-load process timeout differs")
    if config.get("execution_order") != "paired_rotating_controller_order":
        raise ValueError("high-load execution order differs")
    bundles = dict(config.get("controller_bundles") or {})
    if set(bundles) != set(CONTROLLERS):
        raise ValueError("high-load controller bundles differ")
    cohorts = list(config.get("cohorts") or [])
    if len(cohorts) != 2 or len({str(row.get("id")) for row in cohorts}) != 2:
        raise ValueError("high-load registration requires two unique cohorts")
    tasks = []
    for cohort in cohorts:
        registered = list(map(str, cohort.get("tasks") or []))
        if len(registered) != 2 or len(set(registered)) != 2:
            raise ValueError("each high-load cohort requires two unique tasks")
        tasks.extend(registered)
    if len(set(tasks)) != 4:
        raise ValueError("high-load registration requires four unique tasks")


def _validate_runtime(config: dict[str, Any]) -> None:
    if bool(config.get("formal")):
        raise ValueError("high-load runtime must remain non-formal")
    if tuple(map(int, config.get("solver_seeds") or ())) != SOLVER_SEEDS:
        raise ValueError("high-load runtime solver seeds differ")
    if int(config.get("workers", 0)) != 1:
        raise ValueError("high-load runtime requires serial execution")
    if float(config.get("wall_time_budget_seconds", 0.0)) != WALL_BUDGET_SECONDS:
        raise ValueError("high-load runtime wall budget differs")
    if (
        float(config.get("episode_process_timeout_seconds", 0.0))
        != PROCESS_TIMEOUT_SECONDS
    ):
        raise ValueError("high-load runtime process timeout differs")
    if float(dict(config.get("environment") or {}).get("time_limit", 0.0)) != WALL_BUDGET_SECONDS:
        raise ValueError("high-load environment wall budget differs")
    if list(config.get("policies") or []) != [
        "official_adaptive",
        "realized_dynamic",
    ]:
        raise ValueError("high-load runtime policy registration differs")


def _load_registration(
    registration_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(registration_path).resolve()
    project_root = path.parents[1]
    registration = _read_json(path)
    validate_high_load_registration(registration)
    runtime_path = _project_path(project_root, str(registration["runtime_config"]))
    _validate_runtime(_read_json(runtime_path))
    return project_root, runtime_path, registration


def _cohort_manifest(
    project_root: Path, registration: dict[str, Any], cohort: dict[str, Any]
) -> tuple[Path, Path]:
    dataset = _project_path(project_root, str(cohort["source_dataset"]))
    split = str(cohort["source_split"])
    manifest = dataset / split / "manifest.jsonl"
    rows = _read_jsonl(manifest)
    task_ids = [str(row["task_id"]) for row in rows]
    if task_ids != list(map(str, cohort["tasks"])):
        if sorted(task_ids) != sorted(map(str, cohort["tasks"])):
            raise ValueError(f"high-load dataset tasks differ: {cohort['id']}")
    if any(str(row.get("source_group")) != "movingai" for row in rows):
        raise ValueError("high-load diagnostic requires MovingAI tasks")
    if any(str(row.get("split")) != split for row in rows):
        raise ValueError("high-load dataset split differs")
    return dataset, manifest


def high_load_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_high_load_registration(config)
    keys = sorted(
        (str(cohort["id"]), str(task_id), seed)
        for cohort in config["cohorts"]
        for task_id in cohort["tasks"]
        for seed in SOLVER_SEEDS
    )
    schedule: list[dict[str, Any]] = []
    ordinal = 0
    for key_index, (cohort_id, task_id, seed) in enumerate(keys):
        offset = key_index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        for position, controller in enumerate(order):
            schedule.append(
                {
                    "ordinal": ordinal,
                    "cohort_id": cohort_id,
                    "task_id": task_id,
                    "solver_seed": seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
            ordinal += 1
    return schedule


def _controller_kwargs(
    project_root: Path, registration: dict[str, Any], controller: str
) -> dict[str, Any]:
    bundle = dict(registration["controller_bundles"])[controller]
    return {
        "controller": controller,
        "controller_bundle": str(_project_path(project_root, str(bundle))),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "audit",
        "stopping_rule": "wall-clock",
    }


def _manifest_status(output: Path, schedule: list[dict[str, Any]]) -> dict[str, Any]:
    completed = 0
    errors = 0
    controller_counts = Counter()
    for item in schedule:
        path = (
            output
            / "cohorts"
            / str(item["cohort_id"])
            / "controllers"
            / str(item["controller"])
            / "realized_dynamic_manifest.jsonl"
        )
        if not path.is_file():
            continue
        indexed = {
            (str(row["task_id"]), int(row["solver_seed"])): row
            for row in _read_jsonl(path)
        }
        row = indexed.get((str(item["task_id"]), int(item["solver_seed"])))
        if row is None or row.get("status") not in {"ok", "error"}:
            continue
        completed += 1
        controller_counts[str(item["controller"])] += 1
        errors += row.get("status") != "ok"
    return {
        "completed_schedule_entries": completed,
        "total_schedule_entries": len(schedule),
        "controller_completed_episode_counts": dict(controller_counts),
        "error_count": errors,
    }


def run_high_load_diagnostic(
    registration_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root, runtime_path, registration = _load_registration(registration_path)
    output = Path(output).resolve()
    schedule = high_load_schedule(registration)
    cohort_by_id = {str(row["id"]): dict(row) for row in registration["cohorts"]}
    cohort_context: dict[str, tuple[Path, set[tuple[str, int]], Path]] = {}
    for cohort_id, cohort in cohort_by_id.items():
        dataset, manifest = _cohort_manifest(project_root, registration, cohort)
        keys = {
            (str(task_id), seed)
            for task_id in cohort["tasks"]
            for seed in SOLVER_SEEDS
        }
        cohort_context[cohort_id] = (dataset, keys, manifest)

    status_base = {
        "schema": STATUS_SCHEMA,
        "scientific_status": "development_diagnostic_only",
        "formal_speed_claim": False,
        "registration_sha256": sha256_file(Path(registration_path).resolve()),
        "runtime_config_sha256": sha256_file(runtime_path),
        "dataset_manifest_sha256": {
            cohort_id: sha256_file(context[2])
            for cohort_id, context in cohort_context.items()
        },
        "schedule_sha256": _fingerprint(schedule),
        "complete": False,
    }
    if dry_run:
        status_base["controller_dry_runs"] = {}
        for cohort_id, (dataset, keys, _manifest) in cohort_context.items():
            for controller in CONTROLLERS:
                label = f"{cohort_id}/{controller}"
                status_base["controller_dry_runs"][label] = run_closed_loop_collection(
                    dataset,
                    runtime_path,
                    output / "dry-run" / cohort_id / controller,
                    phase="realized_dynamic",
                    workers=1,
                    dry_run=True,
                    cohort_job_keys=keys,
                    job_keys=keys,
                    **_controller_kwargs(project_root, registration, controller),
                )
        return status_base

    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "execution_schedule.jsonl"
    status_path = output / "high_load_status.json"
    if status_path.is_file():
        existing = _read_json(status_path)
        if existing.get("schedule_sha256") != status_base["schedule_sha256"]:
            raise ValueError("existing high-load schedule differs")
        if not resume:
            raise ValueError("high-load output exists; pass resume")
    _write_jsonl(schedule_path, schedule)
    _write_json(status_path, {**status_base, **_manifest_status(output, schedule)})

    for cohort_id, (dataset, keys, _manifest) in cohort_context.items():
        cohort_root = output / "cohorts" / cohort_id
        qualification_root = cohort_root / "qualification"
        run_closed_loop_collection(
            dataset,
            runtime_path,
            qualification_root,
            phase="qualify",
            workers=1,
            resume=qualification_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs(project_root, registration, "v2-full"),
        )
        for controller in CONTROLLERS:
            controller_root = cohort_root / "controllers" / controller
            run_closed_loop_collection(
                dataset,
                runtime_path,
                controller_root,
                phase="qualify",
                workers=1,
                resume=controller_root.joinpath("run_config.json").is_file(),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification_root,
                **_controller_kwargs(project_root, registration, controller),
            )

    for item in schedule:
        cohort_id = str(item["cohort_id"])
        controller = str(item["controller"])
        dataset, keys, _manifest = cohort_context[cohort_id]
        controller_root = output / "cohorts" / cohort_id / "controllers" / controller
        manifest_path = controller_root / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest_path) if manifest_path.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime_path,
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=keys,
                job_keys={key},
                **_controller_kwargs(project_root, registration, controller),
            )
        current_status = _manifest_status(output, schedule)
        _write_json(
            status_path,
            {**status_base, **current_status, "current": item, "complete": False},
        )

    report = analyze_high_load_diagnostic(registration_path, output)
    _write_json(
        status_path,
        {
            **status_base,
            **_manifest_status(output, schedule),
            "complete": True,
            "report_sha256": sha256_file(output / "high_load_report.json"),
        },
    )
    return report


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        dict(row["summary"])
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    error_rows = [
        row
        for row in rows
        if row.get("status") != "ok" or not isinstance(row.get("summary"), dict)
    ]
    totals = [dict(summary.get("controller_totals") or {}) for summary in completed]
    successes = [summary for summary in completed if bool(summary.get("success"))]
    decisions = sum(int(total.get("model_decision_count", 0)) for total in totals)
    diagnostics = sum(
        int(total.get("selected_feature_diagnostic_count", 0)) for total in totals
    )
    fallback = sum(int(total.get("pruner_fallback_count", 0)) for total in totals)
    ood_fallback = sum(
        int(total.get("pruner_ood_fallback_count", 0)) for total in totals
    )
    outside_sum = sum(
        _metric(total, "selected_feature_outside_fraction_sum") for total in totals
    )
    return {
        "episode_count": len(rows),
        "completed_episode_count": len(completed),
        "execution_error_count": len(error_rows),
        "execution_error_kind_counts": dict(
            sorted(
                Counter(
                    str(row.get("error_kind") or "missing_summary")
                    for row in error_rows
                ).items()
            )
        ),
        "success_count": len(successes),
        "failure_count": len(rows) - len(successes),
        "mean_capped_wall_time_to_feasible": _mean(
            _metric(summary, "capped_wall_time_to_feasible") for summary in completed
        ),
        "median_capped_wall_time_to_feasible": (
            statistics.median(
                _metric(summary, "capped_wall_time_to_feasible")
                for summary in completed
            )
            if completed
            else 0.0
        ),
        "mean_success_wall_time_to_feasible": _mean(
            _metric(summary, "wall_time_to_feasible") for summary in successes
        ),
        "mean_repair_iterations": _mean(
            int(summary.get("repair_iterations", 0)) for summary in completed
        ),
        "mean_no_progress_steps": _mean(
            _no_progress_steps(summary) for summary in completed
        ),
        "mean_normalized_wall_clock_conflict_auc": _mean(
            _metric(summary, "normalized_wall_clock_conflict_auc")
            for summary in completed
        ),
        "mean_pp_replan_seconds": _mean(
            _metric(total, "pp_replan_seconds") for total in totals
        ),
        "mean_controller_seconds": _mean(
            _metric(total, "controller_seconds_before_repair") for total in totals
        ),
        "model_decision_count": decisions,
        "pruner_fallback_count": fallback,
        "pruner_fallback_fraction": fallback / decisions if decisions else 0.0,
        "pruner_ood_fallback_count": ood_fallback,
        "pruner_ood_fallback_fraction": ood_fallback / decisions if decisions else 0.0,
        "mean_selected_feature_outside_fraction": (
            outside_sum / diagnostics if diagnostics else 0.0
        ),
        "invalid_action_count": sum(
            int(summary.get("invalid_action_count", 0)) for summary in completed
        ),
        "fingerprint_mismatch_count": sum(
            int(summary.get("fingerprint_mismatch_count", 0)) for summary in completed
        ),
    }


def _paired_comparison(
    baseline: dict[tuple[str, str, int], dict[str, Any]],
    challenger: dict[tuple[str, str, int], dict[str, Any]],
    keys: list[tuple[str, str, int]],
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
    challenger_capped = [_metric(right, "capped_wall_time_to_feasible") for _, _, right in pairs]
    deltas = [right - left for left, right in zip(baseline_capped, challenger_capped)]
    common = [
        (left, right)
        for _key, left, right in pairs
        if bool(left.get("success")) and bool(right.get("success"))
    ]
    baseline_mean = _mean(baseline_capped)
    challenger_mean = _mean(challenger_capped)
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_success_count": sum(bool(left.get("success")) for _, left, _ in pairs),
        "challenger_success_count": sum(bool(right.get("success")) for _, _, right in pairs),
        "success_noninferior": sum(bool(right.get("success")) for _, _, right in pairs)
        >= sum(bool(left.get("success")) for _, left, _ in pairs),
        "baseline_mean_capped_ttf": baseline_mean,
        "challenger_mean_capped_ttf": challenger_mean,
        "mean_capped_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_capped_ttf_relative_improvement": (
            (baseline_mean - challenger_mean) / baseline_mean if baseline_mean else 0.0
        ),
        "faster_count": sum(delta < -1e-9 for delta in deltas),
        "slower_count": sum(delta > 1e-9 for delta in deltas),
        "tied_count": sum(abs(delta) <= 1e-9 for delta in deltas),
        "common_success_count": len(common),
        "baseline_common_success_mean_ttf": _mean(
            _metric(left, "wall_time_to_feasible") for left, _ in common
        ),
        "challenger_common_success_mean_ttf": _mean(
            _metric(right, "wall_time_to_feasible") for _, right in common
        ),
        "mean_repair_iterations_delta": _mean(
            int(right.get("repair_iterations", 0)) - int(left.get("repair_iterations", 0))
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
        "per_episode": [
            {
                "cohort_id": key[0],
                "task_id": key[1],
                "solver_seed": key[2],
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


def analyze_high_load_diagnostic(
    registration_path: str | Path, collection: str | Path
) -> dict[str, Any]:
    project_root, runtime_path, registration = _load_registration(registration_path)
    collection = Path(collection).resolve()
    expected = {
        (str(cohort["id"]), str(task_id), seed)
        for cohort in registration["cohorts"]
        for task_id in cohort["tasks"]
        for seed in SOLVER_SEEDS
    }
    by_controller: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    manifest_hashes: dict[str, dict[str, str]] = {}
    for controller in CONTROLLERS:
        indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
        manifest_hashes[controller] = {}
        for cohort in registration["cohorts"]:
            cohort_id = str(cohort["id"])
            path = (
                collection
                / "cohorts"
                / cohort_id
                / "controllers"
                / controller
                / "realized_dynamic_manifest.jsonl"
            )
            rows = _read_jsonl(path)
            manifest_hashes[controller][cohort_id] = sha256_file(path)
            for row in rows:
                key = (cohort_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in indexed:
                    errors.append(f"{controller}: duplicate task/seed row")
                indexed[key] = row
        if set(indexed) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        episode_errors = sum(
            row.get("status") != "ok" or not isinstance(row.get("summary"), dict)
            for row in indexed.values()
        )
        if episode_errors:
            errors.append(f"{controller}: {episode_errors} episode execution error(s)")
        by_controller[controller] = indexed

    fingerprint_mismatches = 0
    conflict_mismatches = 0
    if not errors:
        for key in sorted(expected):
            summaries = [by_controller[name][key]["summary"] for name in CONTROLLERS]
            fingerprint_mismatches += len(
                {str(summary["initial_fingerprint"]) for summary in summaries}
            ) != 1
            conflict_mismatches += len(
                {int(summary["initial_conflicts"]) for summary in summaries}
            ) != 1

    summaries = {
        controller: _controller_summary(list(indexed.values()))
        for controller, indexed in by_controller.items()
    }
    all_keys = sorted(expected)
    comparison = _paired_comparison(
        by_controller["v2-full"], by_controller["stride-quality-v1"], all_keys
    )
    per_cohort = {}
    for cohort in registration["cohorts"]:
        cohort_id = str(cohort["id"])
        keys = [key for key in all_keys if key[0] == cohort_id]
        per_cohort[cohort_id] = {
            "controller_summaries": {
                controller: _controller_summary(
                    [by_controller[controller][key] for key in keys if key in by_controller[controller]]
                )
                for controller in CONTROLLERS
            },
            "comparison_vs_v2_full": _paired_comparison(
                by_controller["v2-full"], by_controller["stride-quality-v1"], keys
            ),
        }

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
    quality_summary = summaries["stride-quality-v1"]
    distribution_shift = (
        float(quality_summary["mean_selected_feature_outside_fraction"]) > 0.1
        or float(quality_summary["pruner_ood_fallback_fraction"]) > 0.0
    )
    mixed_seed_direction = bool(comparison.get("faster_count")) and bool(
        comparison.get("slower_count")
    )
    if not passed:
        next_decision = "repair_execution_or_analysis_before_runtime_conclusion"
    elif not bool(comparison.get("success_noninferior")):
        next_decision = (
            "expand_high_load_coverage_and_multiseed_labels"
            if distribution_shift
            else "revise_multiseed_quality_label"
        )
    elif float(comparison.get("mean_capped_ttf_relative_improvement", 0.0)) > 0.0:
        next_decision = "retain_quality_candidate_and_expand_development_evidence"
    elif distribution_shift:
        next_decision = "expand_high_load_coverage_before_retraining_quality_model"
    elif mixed_seed_direction:
        next_decision = "aggregate_multiseed_labels_and_model_action_uncertainty"
    else:
        next_decision = "revise_quality_features_and_multiseed_label"

    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_diagnostic_only",
        "formal_speed_claim": False,
        "primary_metric": "mean_capped_wall_time_to_feasible",
        "success_constraint": "stride_quality_success_count_gte_v2_full",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparison_vs_v2_full": comparison,
        "per_cohort": per_cohort,
        "diagnosis": {
            "selected_feature_distribution_shift": distribution_shift,
            "mixed_seed_direction": mixed_seed_direction,
        },
        "next_decision": next_decision,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "initial_conflict_mismatch_count": conflict_mismatches,
        "gates": gates,
        "passed": passed,
        "errors": errors,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "registration_sha256": sha256_file(Path(registration_path).resolve()),
            "runtime_config_sha256": sha256_file(runtime_path),
            "schedule_sha256": sha256_file(collection / "execution_schedule.jsonl"),
            "controller_manifest_sha256": manifest_hashes,
        },
    }
    _write_json(collection / "high_load_report.json", report)
    return report


__all__ = [
    "CONTROLLERS",
    "REPORT_SCHEMA",
    "analyze_high_load_diagnostic",
    "high_load_schedule",
    "run_high_load_diagnostic",
    "validate_high_load_registration",
]
