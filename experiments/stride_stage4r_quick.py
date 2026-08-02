from __future__ import annotations

import math
import shutil
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


STRIDE_STAGE4R_QUICK_SCHEMA = "lns2.stride.stage4r_quick_protocol.v1"
STRIDE_STAGE4R_QUICK_DATASET_SCHEMA = "lns2.stride.stage4r_quick_dataset.v1"
STRIDE_STAGE4R_QUICK_REPORT_SCHEMA = "lns2.stride.stage4r_quick_report.v1"
TTF_CLOCK_SCHEMA = "lns2.ttf.reset_inclusive_wall.v1"
CONTROLLERS = ("v2-full", "stride-control-v1", "stride-quality-v1")


def _project_path(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _mean(values: Iterable[float | int]) -> float:
    numbers = list(map(float, values))
    return statistics.fmean(numbers) if numbers else 0.0


def _checked_source(
    project_root: Path, protocol: dict[str, Any], path_key: str, hash_key: str
) -> Path:
    path = _project_path(project_root, str(protocol[path_key]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(protocol[hash_key]).lower():
        raise ValueError(f"Stage 4R Quick source hash differs: {path_key}")
    return path


def _protocol(config: dict[str, Any]) -> dict[str, Any]:
    protocol = dict(config.get("stride_stage4r_quick") or {})
    if protocol.get("schema") != STRIDE_STAGE4R_QUICK_SCHEMA:
        raise ValueError("unexpected STRIDE Stage 4R Quick protocol schema")
    if bool(config.get("formal")) or protocol.get("scientific_status") != "diagnostic_only":
        raise ValueError("Stage 4R Quick must remain diagnostic-only")
    if bool(protocol.get("formal_ood_data_allowed")):
        raise ValueError("Stage 4R Quick cannot read formal OOD outcomes")
    if tuple(protocol.get("controllers") or ()) != CONTROLLERS:
        raise ValueError("Stage 4R Quick controllers differ from registration")
    if protocol.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA:
        raise ValueError("Stage 4R Quick TTF clock differs from registration")
    if float(config.get("wall_time_budget_seconds", 0.0)) != 300.0:
        raise ValueError("Stage 4R Quick requires the registered 300-second budget")
    if int(config.get("workers", 0)) != 1:
        raise ValueError("Stage 4R Quick requires strict single-episode execution")
    if list(config.get("policies") or []) != [
        "official_adaptive",
        "realized_dynamic",
    ]:
        raise ValueError("Stage 4R Quick policy registration differs")
    if list(config.get("solver_seeds") or []) != [1]:
        raise ValueError("Stage 4R Quick solver seed differs from registration")
    return protocol


def _copy_registered_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"existing Quick dataset file differs: {destination}")
        return
    shutil.copy2(source, destination)


def prepare_stage4r_quick_dataset(
    config_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    protocol = _protocol(config)
    source_manifest = _checked_source(
        project_root, protocol, "source_manifest", "source_manifest_sha256"
    )
    selection_path = _checked_source(
        project_root, protocol, "selection_evidence", "selection_evidence_sha256"
    )
    training_path = _checked_source(
        project_root, protocol, "training_map_source", "training_map_source_sha256"
    )
    formal_path = _checked_source(
        project_root,
        protocol,
        "formal_ood_map_source",
        "formal_ood_map_source_sha256",
    )
    source_root = _project_path(project_root, str(protocol["source_dataset"]))
    split = str(config["split"])
    registered = list(map(str, protocol["registered_task_ids"]))
    if len(registered) != 11 or len(set(registered)) != len(registered):
        raise ValueError("Stage 4R Quick requires 11 unique registered tasks")

    source_rows = {
        str(row["task_id"]): row for row in _read_jsonl(source_manifest)
    }
    if any(task_id not in source_rows for task_id in registered):
        raise ValueError("Stage 4R Quick task is absent from the source manifest")
    rows = [source_rows[task_id] for task_id in registered]
    map_ids = [str(row["map_id"]) for row in rows]
    if len(set(map_ids)) != len(rows):
        raise ValueError("Stage 4R Quick requires one task per map")
    training_maps = {str(row["map_id"]) for row in _read_jsonl(training_path)}
    formal_maps = {str(row["map_id"]) for row in _read_jsonl(formal_path)}
    training_overlap = sorted(set(map_ids) & training_maps)
    formal_overlap = sorted(set(map_ids) & formal_maps)
    if training_overlap or formal_overlap:
        raise ValueError("Stage 4R Quick cohort overlaps forbidden maps")

    selection_rows = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(selection_path)
    }
    selection = []
    for row in rows:
        key = (str(row["task_id"]), 1)
        evidence = selection_rows.get(key)
        if evidence is None:
            raise ValueError(f"missing registered selection evidence: {key}")
        if evidence.get("status") != "ok" or int(evidence["initial_conflicts"]) <= 0:
            raise ValueError(f"registered Quick task was not repairable: {key}")
        selection.append(
            {
                "task_id": key[0],
                "solver_seed": key[1],
                "map_id": str(row["map_id"]),
                "layout_mode": str(row["layout_mode"]),
                "agent_count": int(row["agent_count"]),
                "initial_conflicts": int(evidence["initial_conflicts"]),
                "state_fingerprint": str(evidence["state_fingerprint"]),
            }
        )

    destination = Path(destination).resolve()
    split_root = destination / split
    for row in rows:
        for field, value in row.items():
            if not field.endswith("_file") or not isinstance(value, str):
                continue
            _copy_registered_file(
                source_root / split / value,
                split_root / value,
            )
    ordered_rows = sorted(rows, key=lambda row: str(row["task_id"]))
    manifest = split_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.is_file():
        existing = _read_jsonl(manifest)
        if existing != ordered_rows:
            raise ValueError("existing Quick dataset manifest differs")
    else:
        _write_jsonl(manifest, ordered_rows)

    source_counts = Counter(str(row["source_group"]) for row in rows)
    layout_counts = Counter(str(row["layout_mode"]) for row in rows)
    report = {
        "schema": STRIDE_STAGE4R_QUICK_DATASET_SCHEMA,
        "scientific_status": "diagnostic_only",
        "selection_outcomes_read": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "formal_ood_registration_read": True,
        "task_count": len(rows),
        "map_count": len(set(map_ids)),
        "source_counts": dict(sorted(source_counts.items())),
        "layout_counts": dict(sorted(layout_counts.items())),
        "training_map_overlap": training_overlap,
        "formal_ood_map_overlap": formal_overlap,
        "registered_selection": selection,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "source_manifest_sha256": sha256_file(source_manifest),
            "selection_evidence_sha256": sha256_file(selection_path),
            "training_map_source_sha256": sha256_file(training_path),
            "formal_ood_map_source_sha256": sha256_file(formal_path),
        },
        "dataset_manifest_sha256": sha256_file(manifest),
    }
    report["registration_fingerprint"] = _fingerprint(report)
    _write_json(destination / "dataset_summary.json", report)
    return report


def stage4r_quick_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = _protocol(config)
    keys = sorted(
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in config["solver_seeds"]
    )
    schedule = []
    ordinal = 0
    for key_index, (task_id, seed) in enumerate(keys):
        offset = key_index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        for position, controller in enumerate(order):
            schedule.append(
                {
                    "ordinal": ordinal,
                    "task_id": task_id,
                    "solver_seed": seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
            ordinal += 1
    return schedule


def _controller_kwargs(
    project_root: Path, protocol: dict[str, Any], controller: str
) -> dict[str, Any]:
    bundles = dict(protocol["controller_bundles"])
    return {
        "controller": controller,
        "controller_bundle": str(_project_path(project_root, bundles[controller])),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "audit",
        "stopping_rule": "wall-clock",
    }


def run_stage4r_quick(
    config_path: str | Path,
    dataset: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    protocol = _protocol(config)
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    dataset_report = prepare_stage4r_quick_dataset(config_path, dataset)
    schedule = stage4r_quick_schedule(config)
    cohort = {
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in config["solver_seeds"]
    }
    registration = {
        "schema": STRIDE_STAGE4R_QUICK_SCHEMA,
        "scientific_status": "diagnostic_only",
        "config_sha256": sha256_file(config_path),
        "dataset_registration_fingerprint": dataset_report["registration_fingerprint"],
        "schedule_sha256": _fingerprint(schedule),
        "schedule": schedule,
        "complete": False,
    }
    if dry_run:
        registration["controller_dry_runs"] = {
            controller: run_closed_loop_collection(
                dataset,
                config_path,
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
        return registration
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    status_path = output / "quick_status.json"
    existing_status = _read_json(status_path) if status_path.is_file() else None
    if existing_status is not None:
        if existing_status.get("schedule_sha256") != registration["schedule_sha256"]:
            raise ValueError("existing Stage 4R Quick schedule differs")
        if not resume:
            raise ValueError("Stage 4R Quick output exists; pass resume")
    _write_json(status_path, registration)

    qualification_root = output / "qualification"
    qualification_kwargs = _controller_kwargs(project_root, protocol, "v2-full")
    run_closed_loop_collection(
        dataset,
        config_path,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=cohort,
        job_keys=cohort,
        **qualification_kwargs,
    )
    for controller in CONTROLLERS:
        controller_root = output / "controllers" / controller
        kwargs = _controller_kwargs(project_root, protocol, controller)
        run_closed_loop_collection(
            dataset,
            config_path,
            controller_root,
            phase="qualify",
            workers=1,
            resume=controller_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=cohort,
            job_keys=cohort,
            qualification_source=qualification_root,
            **kwargs,
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
                config_path,
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
                **registration,
                "completed_schedule_entries": completed,
                "total_schedule_entries": len(schedule),
                "current": item,
                "complete": False,
            },
        )

    report = analyze_stage4r_quick(config_path, output)
    _write_json(
        status_path,
        {
            **registration,
            "completed_schedule_entries": len(schedule),
            "total_schedule_entries": len(schedule),
            "complete": True,
            "report_sha256": sha256_file(output / "stage4r_quick_report.json"),
        },
    )
    return report


def _metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [dict(row["summary"]) for row in rows]
    totals = [dict(summary.get("controller_totals") or {}) for summary in episodes]
    successes = [summary for summary in episodes if bool(summary.get("success"))]
    return {
        "episode_count": len(episodes),
        "success_count": len(successes),
        "failure_count": len(episodes) - len(successes),
        "mean_capped_wall_time_to_feasible": _mean(
            _metric(summary, "capped_wall_time_to_feasible") for summary in episodes
        ),
        "mean_success_wall_time_to_feasible": _mean(
            _metric(summary, "wall_time_to_feasible") for summary in successes
        ),
        "mean_repair_iterations": _mean(
            int(summary.get("repair_iterations", 0)) for summary in episodes
        ),
        "mean_normalized_fixed_budget_conflict_auc": _mean(
            _metric(summary, "normalized_fixed_budget_conflict_auc")
            for summary in episodes
        ),
        "mean_normalized_wall_clock_conflict_auc": _mean(
            _metric(summary, "normalized_wall_clock_conflict_auc")
            for summary in episodes
        ),
        "mean_pp_replan_seconds": _mean(
            _metric(total, "pp_replan_seconds") for total in totals
        ),
        "mean_repair_wall_seconds": _mean(
            _metric(summary, "repair_wall_seconds") for summary in episodes
        ),
        "mean_controller_seconds": _mean(
            _metric(total, "controller_seconds_before_repair") for total in totals
        ),
        "mean_proposal_seconds": _mean(
            _metric(total, "proposal_seconds") for total in totals
        ),
        "mean_feature_seconds": _mean(
            _metric(total, "feature_seconds") for total in totals
        ),
        "mean_inference_seconds": _mean(
            _metric(total, "inference_seconds") for total in totals
        ),
        "mean_state_export_seconds": _mean(
            _metric(total, "state_export_seconds") for total in totals
        ),
        "invalid_action_count": sum(int(summary.get("invalid_action_count", 0)) for summary in episodes),
        "fingerprint_mismatch_count": sum(
            int(summary.get("fingerprint_mismatch_count", 0)) for summary in episodes
        ),
        "fallback_count": sum(
            int(total.get("pruner_fallback_count", 0)) for total in totals
        ),
    }


def analyze_stage4r_quick(
    config_path: str | Path,
    collection: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    protocol = _protocol(config)
    collection = Path(collection).resolve()
    expected_keys = {
        (str(task_id), int(seed))
        for task_id in protocol["registered_task_ids"]
        for seed in config["solver_seeds"]
    }
    by_controller: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    errors = []
    for controller in CONTROLLERS:
        path = collection / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(path)
        indexed = {
            (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
        }
        if set(indexed) != expected_keys:
            errors.append(f"{controller}: incomplete paired coverage")
        if any(row.get("status") != "ok" for row in rows):
            errors.append(f"{controller}: episode execution error")
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
        controller: _controller_summary(list(indexed.values()))
        for controller, indexed in by_controller.items()
    }
    baseline = summaries.get("v2-full", {})
    comparisons = {}
    for challenger in CONTROLLERS[1:]:
        common = []
        if not errors:
            for key in sorted(expected_keys):
                left = by_controller["v2-full"][key]["summary"]
                right = by_controller[challenger][key]["summary"]
                if bool(left["success"]) and bool(right["success"]):
                    common.append(
                        (
                            float(left["wall_time_to_feasible"]),
                            float(right["wall_time_to_feasible"]),
                        )
                    )
        base_ttf = _mean(left for left, _right in common)
        challenger_ttf = _mean(right for _left, right in common)
        success_noninferior = int(summaries.get(challenger, {}).get("success_count", 0)) >= int(
            baseline.get("success_count", 0)
        )
        capped = float(summaries.get(challenger, {}).get("mean_capped_wall_time_to_feasible", 0.0))
        base_capped = float(baseline.get("mean_capped_wall_time_to_feasible", 0.0))
        comparisons[challenger] = {
            "success_noninferior": success_noninferior,
            "common_success_count": len(common),
            "baseline_common_success_mean_ttf": base_ttf,
            "challenger_common_success_mean_ttf": challenger_ttf,
            "common_success_relative_ttf_improvement": (
                (base_ttf - challenger_ttf) / base_ttf if base_ttf else 0.0
            ),
            "mean_capped_ttf_improvement_seconds": base_capped - capped,
            "mean_capped_ttf_relative_improvement": (
                (base_capped - capped) / base_capped if base_capped else 0.0
            ),
            "faster_under_primary_metric": success_noninferior and capped < base_capped,
        }

    eligible = [
        controller
        for controller in CONTROLLERS
        if controller == "v2-full"
        or int(summaries.get(controller, {}).get("success_count", 0))
        >= int(baseline.get("success_count", 0))
    ]
    primary_winner = min(
        eligible,
        key=lambda controller: float(
            summaries[controller]["mean_capped_wall_time_to_feasible"]
        ),
    ) if eligible and not errors else None
    if primary_winner == "stride-quality-v1":
        next_decision = "quality_candidate_warrants_larger_development_quick"
    elif primary_winner == "stride-control-v1":
        next_decision = "data_only_control_warrants_larger_development_quick"
    else:
        next_decision = "revise_quality_label_before_more_runtime_evaluation"

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
            row["summary"].get("ttf_clock_schema") == TTF_CLOCK_SCHEMA
            for indexed in by_controller.values()
            for row in indexed.values()
        ),
    }
    passed = not errors and all(gates.values())
    report = {
        "schema": STRIDE_STAGE4R_QUICK_REPORT_SCHEMA,
        "scientific_status": "diagnostic_only",
        "formal_speed_claim": False,
        "primary_metric": "mean_capped_wall_time_to_feasible",
        "success_constraint": "candidate_success_count_gte_v2_full",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected_keys),
        "controller_summaries": summaries,
        "comparisons_vs_v2_full": comparisons,
        "primary_eligible_controllers": eligible,
        "primary_winner": primary_winner,
        "next_decision": next_decision,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "initial_conflict_mismatch_count": conflict_mismatches,
        "gates": gates,
        "passed": passed,
        "errors": errors,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "schedule_sha256": sha256_file(collection / "execution_schedule.jsonl"),
        },
    }
    _write_json(collection / "stage4r_quick_report.json", report)
    return report


__all__ = [
    "CONTROLLERS",
    "STRIDE_STAGE4R_QUICK_REPORT_SCHEMA",
    "STRIDE_STAGE4R_QUICK_SCHEMA",
    "TTF_CLOCK_SCHEMA",
    "analyze_stage4r_quick",
    "prepare_stage4r_quick_dataset",
    "run_stage4r_quick",
    "stage4r_quick_schedule",
]
