from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    sha256_file,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import (
    CONTROLLERS,
    TTF_CLOCK_SCHEMA,
    _controller_kwargs,
    _quick_controller_summary,
    _structpool_trace_counts,
    structpool_ttf_schedule,
)


def run_structpool_paired_ttf(
    path: Path,
    root: Path,
    config: dict[str, Any],
    output: str | Path,
    *,
    status_schema: str,
    status_filename: str,
    report_schema: str,
    report_filename: str,
    report_scientific_status: str,
    next_step_on_pass: str,
    next_step_on_failure: str,
    performance_claim_field: str | None = None,
    fixed_report_fields: dict[str, Any] | None = None,
    producer_source_files: tuple[str, ...] = (),
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    schedule = structpool_ttf_schedule(config)
    if dry_run:
        return {
            "schema": status_schema,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / status_filename
    prepared = prepare_resumable_output(
        output,
        status_filename=status_filename,
        status_schema=status_schema,
        config_path=path,
        schedule=schedule,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_structpool_paired_ttf.py",
                "experiments/stride_structpool_ttf_quick.py",
                "experiments/stride_maprank_raw_ttf.py",
                *producer_source_files,
            ),
        ),
        resume=resume,
        report_filename=report_filename,
        report_schema=report_schema,
        label="StructPool paired TTF",
    )
    base_status = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    cohort = dict(config["cohort"])
    dataset = (root / str(cohort["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    seeds = tuple(map(int, cohort["solver_seeds"]))
    all_keys = {
        (str(task), seed)
        for group in cohort["groups"]
        for task in group["tasks"]
        for seed in seeds
    }
    qualification = output / "qualification"
    registered_source = config.get("qualification_source")
    qualification_source = (
        (root / str(registered_source)).resolve()
        if registered_source is not None
        else None
    )
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification,
        phase="qualify",
        workers=1,
        resume=(
            prepared.resumed
            and qualification.joinpath("run_config.json").is_file()
        ),
        cohort_job_keys=all_keys,
        job_keys=all_keys,
        qualification_source=qualification_source,
        **_controller_kwargs(root, config, "v2-full"),
    )
    for controller in CONTROLLERS:
        controller_root = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime,
            controller_root,
            phase="qualify",
            workers=1,
            resume=(
                prepared.resumed
                and controller_root.joinpath("run_config.json").is_file()
            ),
            cohort_job_keys=all_keys,
            job_keys=all_keys,
            qualification_source=qualification,
            **_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        controller_root = output / "controllers" / controller
        manifest = controller_root / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime,
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=all_keys,
                job_keys={key},
                **_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **base_status,
                "completed_schedule_entries": completed,
                "current": item,
                "complete": False,
            },
        )
    report = analyze_structpool_paired_ttf(
        path,
        config,
        output,
        status_filename=status_filename,
        report_schema=report_schema,
        report_filename=report_filename,
        report_scientific_status=report_scientific_status,
        next_step_on_pass=next_step_on_pass,
        next_step_on_failure=next_step_on_failure,
        performance_claim_field=performance_claim_field,
        fixed_report_fields=fixed_report_fields,
        producer=base_status["producer_identity"],
        producer_source_files=producer_source_files,
    )
    _write_json(
        status_path,
        {
            **base_status,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / report_filename),
        },
    )
    return report


def analyze_structpool_paired_ttf(
    path: Path,
    config: dict[str, Any],
    output: str | Path,
    *,
    status_filename: str,
    report_schema: str,
    report_filename: str,
    report_scientific_status: str,
    next_step_on_pass: str,
    next_step_on_failure: str,
    performance_claim_field: str | None = None,
    fixed_report_fields: dict[str, Any] | None = None,
    producer: dict[str, Any] | None = None,
    producer_source_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=status_filename,
        report_filename=report_filename,
        report_schema=report_schema,
        config_path=path,
    )
    if completed is not None:
        return completed
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=path.parent.parent,
            source_files=(
                "experiments/stride_structpool_paired_ttf.py",
                "experiments/stride_structpool_ttf_quick.py",
                "experiments/stride_maprank_raw_ttf.py",
                *producer_source_files,
            ),
            native_required=False,
        )
    groups = [dict(row) for row in config["cohort"]["groups"]]
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    expected = {
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in seeds
    }
    group_by_task = {
        str(task): str(group["id"])
        for group in groups
        for task in group["tasks"]
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    errors: list[str] = []
    structpool_counts = {"selected": 0, "gate_passed": 0, "added_candidates": 0}
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest)
        manifest_hashes[controller] = sha256_file(manifest)
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            task = str(row["task_id"])
            key = (group_by_task.get(task, ""), task, int(row["solver_seed"]))
            if key in rows_by_key:
                errors.append(f"{controller}: duplicate episode {key}")
            rows_by_key[key] = row
            if controller == "v2-plus-structpool" and row.get("status") == "ok":
                observed = _structpool_trace_counts(collection, row)
                for name, value in observed.items():
                    structpool_counts[name] += value
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped_values = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += (
            len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        )
        conflict_mismatches += (
            len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries
        )
        capped_values += sum(
            row.get("capped_wall_time_to_feasible") is not None for row in summaries
        )
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    keys = sorted(expected)
    comparison = _paired_comparison(
        indexed["v2-full"], indexed["v2-plus-structpool"], keys
    )
    per_group = {
        str(group["id"]): _paired_comparison(
            indexed["v2-full"],
            indexed["v2-plus-structpool"],
            [key for key in keys if key[0] == str(group["id"])],
        )
        for group in groups
    }
    qualification_report = _read_json(
        output / "qualification" / "qualification_report.json"
    )
    integrity = {
        "formal_qualification_passed": qualification_report.get("passed") is True,
        "complete_paired_coverage": not any("coverage" in error for error in errors),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for values in indexed.values()
            for row in values.values()
        ),
        "all_controllers_succeeded": all(
            summary["success_count"] == len(expected)
            for summary in summaries.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0
            for summary in summaries.values()
        ),
        "structpool_activated_and_selected": structpool_counts["gate_passed"] > 0
        and structpool_counts["selected"] > 0,
    }
    gates = dict(config["performance_gates"])
    maximum_regression = max(
        (
            -float(row.get("mean_raw_ttf_relative_improvement", 0.0))
            for row in per_group.values()
        ),
        default=0.0,
    )
    performance = {
        "mean_raw_ttf_improvement": float(
            comparison.get("mean_raw_ttf_relative_improvement", -1.0)
        )
        >= float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
        "maximum_group_regression": maximum_regression
        <= float(gates["maximum_group_raw_ttf_regression"]),
        "paired_faster_fraction": float(
            comparison.get("paired_faster_fraction", 0.0)
        )
        >= float(gates["minimum_paired_faster_fraction"]),
        "repair_iterations_noninferior": float(
            comparison.get("mean_repair_iterations_delta", 1.0)
        )
        <= 0.0,
        "success_count_noninferior": summaries["v2-plus-structpool"]["success_count"]
        >= summaries["v2-full"]["success_count"],
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": report_schema,
        "scientific_status": report_scientific_status,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "producer_identity": producer,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparison": comparison,
        "per_group": per_group,
        "structpool_runtime_counts": structpool_counts,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "next_step": next_step_on_pass if performance_passed else next_step_on_failure,
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "qualification_report_sha256": sha256_file(
                output / "qualification" / "qualification_report.json"
            ),
            "controller_manifest_sha256": manifest_hashes,
        },
    }
    if performance_claim_field is not None:
        report[performance_claim_field] = performance_passed
    if fixed_report_fields:
        overlap = set(report) & set(fixed_report_fields)
        if overlap:
            raise ValueError(f"fixed StructPool report fields overlap: {sorted(overlap)}")
        report.update(fixed_report_fields)
    _write_json(output / report_filename, report)
    return report


__all__ = ["analyze_structpool_paired_ttf", "run_structpool_paired_ttf"]
