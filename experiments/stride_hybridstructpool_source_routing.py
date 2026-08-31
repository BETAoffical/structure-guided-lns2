from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.evaluation.episode_statistics import (
    dataset_tasks as _dataset_tasks,
    mean as _mean,
    metric as _metric,
    paired_raw_ttf_comparison as _paired_comparison,
    raw_ttf_controller_summary as _controller_summary,
)
from lns2_selector.runtime.hybridstructpool import (
    hybridstructpool_runtime_augmentation,
    validate_hybridstructpool_augmentation,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    ROUTED_SOURCE_MODES,
    routed_hybridstructpool_augmentation,
    validate_routed_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_source_routing_config.v1"
STATUS_SCHEMA = "lns2.stride.hybridstructpool_source_routing_status.v1"
AUDIT_SCHEMA = "lns2.stride.hybridstructpool_source_routing_zero_solver_audit.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_source_routing_report.v1"
EXPERIMENT_ID = "stride-hybridstructpool-routed-v1"
PRE_REGISTRATION_PARENT = "5a6d728"
CONTROLLERS = (
    "v2_only",
    "structshell_only",
    "causal_only",
    "full_v8",
    "routed_structshell",
)
CHALLENGERS = tuple(name for name in CONTROLLERS if name != "v2_only")
STATUS_FILENAME = "collection_status.json"
AUDIT_FILENAME = "zero_solver_audit.json"
REPORT_FILENAME = "source_routing_report.json"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="Hybrid source routing")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_source_ablation_before_raw_ttf"
        or config.get("experiment_id") != EXPERIMENT_ID
        or str(config.get("pre_registration_parent_commit"))
        != PRE_REGISTRATION_PARENT
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Hybrid source-routing registration changed")
    comparison = dict(config.get("comparison") or {})
    if comparison != {
        "quality_anchor": "v2_only",
        "execution_order": "rotating_strict_five_controller_serial",
        "paired_solver_seed_required": True,
        "workers_for_timed_episodes": 1,
        "workers_for_qualification": 16,
    }:
        raise ValueError("Hybrid source-routing comparison changed")
    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "config": "configs/stride_stage4r_high_load_runtime.json",
        "stopping_rule": "run-to-completion",
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "episode_process_timeout_seconds": 900.0,
    }:
        raise ValueError("Hybrid source-routing runtime changed")
    full = dict(config.get("full_v8_augmentation") or {})
    if validate_hybridstructpool_augmentation(full) != hybridstructpool_runtime_augmentation():
        raise ValueError("Hybrid source-routing V8 identity changed")
    routed = dict(config.get("routed_augmentations") or {})
    if set(routed) != set(ROUTED_SOURCE_MODES):
        raise ValueError("Hybrid source-routing modes changed")
    for mode in ROUTED_SOURCE_MODES:
        value = dict(routed[mode])
        if (
            validate_routed_hybridstructpool_augmentation(value)
            != routed_hybridstructpool_augmentation(mode)
        ):
            raise ValueError(f"Hybrid source-routing identity changed: {mode}")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "development_quick_not_formal_or_result_blind"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2)
        or int(cohort.get("paired_key_count", -1)) != 8
        or int(cohort.get("episode_count_per_controller", -1)) != 8
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("Hybrid source-routing cohort changed")
    gates = dict(config.get("performance_gates") or {})
    if gates != {
        "success_count": 8,
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.0,
        "minimum_paired_faster_fraction_vs_v2": 0.5,
        "maximum_group_raw_ttf_regression": 0.05,
        "repair_iterations_noninferior_to_v2": True,
        "near_tie_fraction": 0.01,
    }:
        raise ValueError("Hybrid source-routing gates changed")
    inputs = dict(config.get("inputs") or {})
    for specification in inputs.values():
        _registered(root, specification)
    engineering = _read_json(root / str(inputs["hybrid_runtime_report"]["path"]))
    if engineering.get("all_gates_passed") is not True:
        raise ValueError("Hybrid V8 engineering gate did not pass")
    for group in groups:
        tasks = _dataset_tasks((root / str(group["dataset"])).resolve(), str(group["split"]))
        if set(map(str, group["tasks"])) - set(tasks):
            raise ValueError("Hybrid source-routing task is absent from its dataset")
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key_index = 0
    for group in config["cohort"]["groups"]:
        group_id = str(group["id"])
        for task in group["tasks"]:
            task_id = str(task)
            for raw_seed in config["cohort"]["solver_seeds"]:
                seed = int(raw_seed)
                offset = key_index % len(CONTROLLERS)
                for position in range(len(CONTROLLERS)):
                    rows.append(
                        {
                            "group_id": group_id,
                            "task_id": task_id,
                            "solver_seed": seed,
                            "controller": CONTROLLERS[(offset + position) % len(CONTROLLERS)],
                            "within_key_position": position,
                        }
                    )
                key_index += 1
    return rows


def zero_solver_audit(
    config_path: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    _path, root, config = load_config(config_path)
    source_path = root / str(config["inputs"]["hybrid_audit_states"]["path"])
    rows = _read_jsonl(source_path)
    errors: list[str] = []
    maps: set[str] = set()
    fingerprints: set[str] = set()
    totals = defaultdict(int)
    for row in rows:
        maps.add(str(row.get("map_id")))
        fingerprint = str(row.get("state_fingerprint") or "")
        if not fingerprint or fingerprint in fingerprints:
            errors.append(f"duplicate or missing state fingerprint: {fingerprint}")
        fingerprints.add(fingerprint)
        provenance = {
            str(identity): set(map(str, values))
            for identity, values in dict(row.get("provenance_by_candidate_id") or {}).items()
        }
        all_ids = set(map(str, row.get("candidate_ids") or ()))
        base = {identity for identity, values in provenance.items() if "v2_base" in values}
        structural = {
            identity
            for identity, values in provenance.items()
            if values & {"v2_base", "structshell_equal_four_size"}
        }
        causal = {
            identity
            for identity, values in provenance.items()
            if values & {"v2_base", "causalclosure_v2"}
        }
        if set(provenance) != all_ids:
            errors.append(f"candidate/provenance mismatch: {fingerprint}")
        if not base or not base <= structural or not base <= causal or not base <= all_ids:
            errors.append(f"V2 base preservation failed: {fingerprint}")
        if structural != {
            identity
            for identity in all_ids
            if provenance[identity] & {"v2_base", "structshell_equal_four_size"}
        }:
            errors.append(f"StructShell source mask failed: {fingerprint}")
        if causal != {
            identity
            for identity in all_ids
            if provenance[identity] & {"v2_base", "causalclosure_v2"}
        }:
            errors.append(f"Causal source mask failed: {fingerprint}")
        totals["v2_only"] += len(base)
        totals["structshell_only"] += len(structural)
        totals["causal_only"] += len(causal)
        totals["full_v8"] += len(all_ids)
        totals["routed_structshell"] += len(structural)
    integrity = {
        "state_count_is_78": len(rows) == 78,
        "three_maps_present": len(maps) == 3,
        "unique_state_fingerprints": len(fingerprints) == len(rows),
        "all_source_masks_preserve_v2": not any(
            "V2 base preservation" in error for error in errors
        ),
        "source_membership_exact": not any(
            "source mask" in error or "candidate/provenance" in error
            for error in errors
        ),
        "no_solver_or_pp_called": True,
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "state_count": len(rows),
        "map_ids": sorted(maps),
        "candidate_membership_totals": dict(totals),
        "integrity_gates": integrity,
        "integrity_passed": not errors and all(integrity.values()),
        "errors": errors,
        "inputs": {"hybrid_audit_states_sha256": sha256_file(source_path)},
    }
    if output is not None:
        destination = Path(output).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / AUDIT_FILENAME, report)
    return report


def _controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    result = {
        "stopping_rule": "run-to-completion",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if controller == "v2_only":
        return result
    if controller == "full_v8":
        result["hybridstructpool_augmentation"] = dict(config["full_v8_augmentation"])
        return result
    if controller in ROUTED_SOURCE_MODES:
        result["hybridstructpool_augmentation"] = dict(
            config["routed_augmentations"][controller]
        )
        return result
    raise ValueError(f"unknown Hybrid source-routing controller: {controller}")


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "groups"
        / str(item["group_id"])
        / "controllers"
        / str(item["controller"])
        / "realized_dynamic_manifest.jsonl"
    )


def _manifest(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("Hybrid source-routing manifest is ambiguous")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    _path, root, config = load_config(job["config_path"])
    item = dict(job["item"])
    group = next(
        dict(row)
        for row in config["cohort"]["groups"]
        if str(row["id"]) == str(item["group_id"])
    )
    dataset = (root / str(group["dataset"])).resolve()
    keys = {
        (str(task), int(seed))
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    key = (str(item["task_id"]), int(item["solver_seed"]))
    collection = (
        Path(job["output_root"]).resolve()
        / "groups"
        / str(item["group_id"])
        / "controllers"
        / str(item["controller"])
    )
    kwargs = _controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        dataset,
        root / str(config["runtime"]["config"]),
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=Path(job["qualification_source"]).resolve(),
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        dataset,
        root / str(config["runtime"]["config"]),
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        cohort_job_keys=keys,
        job_keys={key},
        qualification_source=Path(job["qualification_source"]).resolve(),
        use_global_collection_lock=False,
        **kwargs,
    )
    manifest = _manifest(Path(job["output_root"]).resolve(), item)
    if manifest is None:
        raise RuntimeError("Hybrid source-routing episode completed without manifest")
    status = str(manifest.get("status"))
    return {
        **item,
        "status": status if status in {"error", "timeout"} else "ok",
        "manifest_status": status,
        "error": manifest.get("error"),
    }


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base_status: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    present = [_manifest(output, item) for item in items]
    completed = [row for row in present if row is not None]
    return {
        **dict(base_status),
        "completed_jobs": len(completed),
        "total_jobs": len(items),
        "completed_by_controller": {
            controller: sum(
                _manifest(output, item) is not None
                for item in items
                if str(item["controller"]) == controller
            )
            for controller in CONTROLLERS
        },
        "error_jobs": sum(str(row.get("status")) == "error" for row in completed),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in completed),
        "complete": bool(complete and len(completed) == len(items)),
    }


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    items = schedule(config)
    if dry_run:
        audit = zero_solver_audit(path)
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": len(items) // len(CONTROLLERS),
            "schedule_entry_count": len(items),
            "schedule_sha256": _fingerprint(items),
            "zero_solver_integrity_passed": audit["integrity_passed"],
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=items,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_hybridstructpool_source_routing.py",
                "scripts/run_stride_hybridstructpool_source_routing.py",
                "experiments/closed_loop_confirmation.py",
                "lns2_selector/evaluation/episode_statistics.py",
                "lns2_selector/runtime/hybridstructpool.py",
                "lns2_selector/runtime/hybridstructpool_routed.py",
                "lns2_selector/runtime/causalclosurepool.py",
                "lns2_selector/runtime/topology_candidates.py",
            ),
        ),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="HybridStructPool source routing",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    audit = zero_solver_audit(path, output)
    if not audit["integrity_passed"]:
        status = _status(output, items, prepared.base_status)
        status["terminal_failure"] = "zero_solver_audit_failed"
        _write_json(output / STATUS_FILENAME, status)
        return status
    _write_jsonl(output / "execution_schedule.jsonl", items)
    for group in config["cohort"]["groups"]:
        group_id = str(group["id"])
        keys = {
            (str(task), int(seed))
            for task in group["tasks"]
            for seed in config["cohort"]["solver_seeds"]
        }
        qualification = output / "groups" / group_id / "qualification"
        run_closed_loop_collection(
            root / str(group["dataset"]),
            root / str(config["runtime"]["config"]),
            qualification,
            phase="qualify",
            workers=int(config["comparison"]["workers_for_qualification"]),
            resume=prepared.resumed and qualification.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            controller="v2-full",
            controller_bundle=str((root / str(config["controller_bundle"])).resolve()),
            feature_backend="native",
            controller_runtime="optimized",
            verification_profile="deployment",
            stopping_rule="run-to-completion",
            repair_seed_policy="episode_stream",
            deterministic_pp_replay=False,
        )
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "qualification_source": str(
                output / "groups" / str(item["group_id"]) / "qualification"
            ),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase="hybridstructpool-source-routing",
            output_root=output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=float(config["runtime"]["episode_process_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row.get("status") in {"error", "timeout"}]
        if terminal:
            status = _status(output, items, prepared.base_status)
            status["terminal_failure"] = terminal[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    status = _status(output, items, prepared.base_status)
    _write_json(output / STATUS_FILENAME, status)
    report = analyze(path, output, producer=prepared.base_status["producer_identity"])
    final_status = _status(output, items, prepared.base_status, complete=True)
    final_status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
    _write_json(output / STATUS_FILENAME, final_status)
    return report


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _controller_summary(rows)
    totals = [
        dict(dict(row.get("summary") or {}).get("controller_totals") or {})
        for row in rows
        if row.get("status") == "ok"
    ]
    for name in (
        "candidate_generation_seconds",
        "feature_seconds",
        "inference_seconds",
        "controller_seconds_before_repair",
        "pp_replan_seconds",
        "hybridstructpool_structural_generation_seconds",
        "hybridstructpool_causal_generation_seconds",
        "hybridstructpool_selected_structural_count",
        "hybridstructpool_selected_causal_count",
    ):
        result[f"mean_{name}"] = _mean([_metric(row, name) for row in totals])
    return result


def _select_promoted(
    eligible: list[str], summaries: Mapping[str, Mapping[str, Any]],
    maximum_group_regressions: Mapping[str, float], near_tie_fraction: float,
) -> str | None:
    if not eligible:
        return None
    best_ttf = min(float(summaries[name]["mean_raw_wall_time_to_feasible"]) for name in eligible)
    near = [
        name
        for name in eligible
        if float(summaries[name]["mean_raw_wall_time_to_feasible"])
        <= best_ttf * (1.0 + near_tie_fraction)
    ]
    return min(
        near,
        key=lambda name: (
            float(maximum_group_regressions[name]),
            float(summaries[name]["mean_neighborhood_selection_seconds"]),
            name,
        ),
    )


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_hybridstructpool_source_routing.py",
                "scripts/run_stride_hybridstructpool_source_routing.py",
                "lns2_selector/evaluation/episode_statistics.py",
            ),
            native_required=False,
        )
    expected = {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for controller in CONTROLLERS:
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in config["cohort"]["groups"]:
            item = {"group_id": group["id"], "controller": controller}
            manifest = _manifest_path(output, item)
            if not manifest.is_file():
                errors.append(f"{controller}/{group['id']}: missing manifest")
                continue
            hashes[controller][str(group["id"])] = sha256_file(manifest)
            for row in _read_jsonl(manifest):
                key = (str(group["id"]), str(row["task_id"]), int(row["solver_seed"]))
                if key in rows_by_key:
                    errors.append(f"{controller}: duplicate {key}")
                rows_by_key[key] = dict(row)
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped = 0
    for key in sorted(expected):
        rows = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries_for_key = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len(
            {str(row.get("initial_fingerprint")) for row in summaries_for_key}
        ) != 1
        conflict_mismatches += len(
            {int(row.get("initial_conflicts", -1)) for row in summaries_for_key}
        ) != 1
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries_for_key
        )
        capped += sum(
            row.get("capped_wall_time_to_feasible") is not None
            for row in summaries_for_key
        )
    summaries = {name: _summary(list(indexed[name].values())) for name in CONTROLLERS}
    keys = sorted(expected)
    comparisons = {
        name: _paired_comparison(indexed["v2_only"], indexed[name], keys)
        for name in CHALLENGERS
    }
    per_group = {
        str(group["id"]): {
            name: _paired_comparison(
                indexed["v2_only"],
                indexed[name],
                [key for key in keys if key[0] == str(group["id"])],
            )
            for name in CHALLENGERS
        }
        for group in config["cohort"]["groups"]
    }
    audit = _read_json(output / AUDIT_FILENAME)
    integrity = {
        "zero_solver_audit_passed": audit.get("integrity_passed") is True,
        "complete_paired_coverage": not any(
            "coverage" in error or "missing" in error for error in errors
        ),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for values in indexed.values()
            for row in values.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped == 0,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0 for summary in summaries.values()
        ),
    }
    integrity_passed = not errors and all(integrity.values())
    gates = dict(config["performance_gates"])
    controller_gates: dict[str, dict[str, bool]] = {}
    maximum_group_regressions: dict[str, float] = {}
    for name in CHALLENGERS:
        comparison = comparisons[name]
        maximum_group_regression = max(
            (
                -float(per_group[group_id][name].get("mean_raw_ttf_relative_improvement", 0.0))
                for group_id in per_group
            ),
            default=0.0,
        )
        maximum_group_regressions[name] = maximum_group_regression
        controller_gates[name] = {
            "success_count_is_8": summaries[name]["success_count"]
            == int(gates["success_count"]),
            "mean_raw_ttf_strictly_better_than_v2": bool(comparison.get("valid"))
            and float(comparison["mean_raw_ttf_relative_improvement"])
            > float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
            "paired_faster_fraction_at_least_half": bool(comparison.get("valid"))
            and float(comparison["paired_faster_fraction"])
            >= float(gates["minimum_paired_faster_fraction_vs_v2"]),
            "maximum_group_regression_at_most_five_percent": maximum_group_regression
            <= float(gates["maximum_group_raw_ttf_regression"]),
            "repair_iterations_noninferior_to_v2": bool(comparison.get("valid"))
            and float(comparison["mean_repair_iterations_delta"]) <= 0.0,
        }
    eligible = [
        name
        for name in CHALLENGERS
        if integrity_passed and all(controller_gates[name].values())
    ]
    promoted = _select_promoted(
        eligible,
        summaries,
        maximum_group_regressions,
        float(gates["near_tie_fraction"]),
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_source_ablation",
        "primary_metric": "mean_run_to_completion_reset_inclusive_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparisons_vs_v2": comparisons,
        "per_group": per_group,
        "integrity_gates": integrity,
        "controller_performance_gates": controller_gates,
        "integrity_passed": integrity_passed,
        "eligible_controllers": eligible,
        "selected_controller": promoted,
        "development_gate_passed": promoted is not None,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "next_step": (
            "preregister_result_blind_60_key_confirmation"
            if promoted is not None
            else "stop_hybrid_runtime_tuning_keep_v2_default"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "zero_solver_audit_sha256": sha256_file(output / AUDIT_FILENAME),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "CONTROLLERS",
    "analyze",
    "load_config",
    "run",
    "schedule",
    "zero_solver_audit",
]
