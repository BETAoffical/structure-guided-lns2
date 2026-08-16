from __future__ import annotations

import math
import random
import statistics
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
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_hybridstructpool_source_routing import _summary
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.runtime.hybridstructpool_routed import (
    validate_routed_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_routed_confirmation_config.v1"
STATUS_SCHEMA = "lns2.stride.hybridstructpool_routed_confirmation_status.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_routed_confirmation_report.v1"
EXPERIMENT_ID = "stride-hybridstructpool-routed-confirmation-v1"
CONTROLLERS = ("official_adaptive", "v2_only", "structshell_only")
POOL_CONTROLLERS = ("v2_only", "structshell_only")
_IDENTITIES = {
    CONFIG_SCHEMA: {
        "experiment_id": EXPERIMENT_ID,
        "pre_registration_parent_commit": "e9cc1ec",
        "episode_process_timeout_seconds": 300.0,
        "status_schema": STATUS_SCHEMA,
        "report_schema": REPORT_SCHEMA,
        "controllers": CONTROLLERS,
        "solver_seeds": (1, 2, 3),
        "comparison": {
            "primary_baseline": "official_adaptive",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_three_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
    },
    "lns2.stride.hybridstructpool_routed_confirmation_config.v2": {
        "experiment_id": "stride-hybridstructpool-routed-confirmation-v2",
        "pre_registration_parent_commit": "87742ae",
        "episode_process_timeout_seconds": 900.0,
        "status_schema": "lns2.stride.hybridstructpool_routed_confirmation_status.v2",
        "report_schema": "lns2.stride.hybridstructpool_routed_confirmation_report.v2",
        "controllers": CONTROLLERS,
        "solver_seeds": (1, 2, 3),
        "comparison": {
            "primary_baseline": "official_adaptive",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_three_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
    },
    "lns2.stride.hybridstructpool_routed_confirmation_config.v3": {
        "experiment_id": "stride-structshell-v2-paired-confirmation-v1",
        "pre_registration_parent_commit": "5d89c4f",
        "episode_process_timeout_seconds": 900.0,
        "status_schema": "lns2.stride.structshell_v2_confirmation_status.v1",
        "report_schema": "lns2.stride.structshell_v2_confirmation_report.v1",
        "controllers": POOL_CONTROLLERS,
        "solver_seeds": (4, 5, 6),
        "comparison": {
            "primary_baseline": "v2_only",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_two_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
    },
    "lns2.stride.hybridstructpool_routed_confirmation_config.v4": {
        "experiment_id": "stride-structshell-v2-official-bounded-confirmation-v1",
        "pre_registration_parent_commit": "114dc63",
        "episode_process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "stopping_rule": "wall-clock",
        "scientific_status": "preregistered_result_blind_paired_bounded_ttf_confirmation",
        "status_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_status.v1",
        "report_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_report.v1",
        "controllers": CONTROLLERS,
        "solver_seeds": (7, 8, 9),
        "comparison": {
            "primary_baseline": "v2_only",
            "secondary_baseline": "official_adaptive",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_three_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
    },
    "lns2.stride.hybridstructpool_routed_confirmation_config.v5": {
        "experiment_id": "stride-structshell-v2-official-bounded-confirmation-v2",
        "pre_registration_parent_commit": "a832132",
        "episode_process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "stopping_rule": "wall-clock",
        "scientific_status": "preregistered_result_blind_paired_bounded_ttf_confirmation",
        "status_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_status.v2",
        "report_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_report.v2",
        "controllers": CONTROLLERS,
        "solver_seeds": (10, 11, 12),
        "comparison": {
            "primary_baseline": "v2_only",
            "secondary_baseline": "official_adaptive",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_three_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
        "cohort_repair": {
            "predecessor_experiment_id": "stride-structshell-v2-official-bounded-confirmation-v1",
            "predecessor_run_fingerprint": "173b89c247d6c7adaa84bf196f6786e0309b9884e118353fd49f990bafc3610c",
            "replaced_group_id": "warehouse-10-20-10-2-2",
            "replacement_group_id": "warehouse-20-40-10-2-2-congestion",
            "selection_basis": "registered_metadata_forced_congestion_and_maximum_agent_load",
            "controller_outcomes_consulted": False,
            "fresh_solver_seeds_required": True,
            "old_formal_episode_count": 0,
        },
    },
    "lns2.stride.hybridstructpool_routed_confirmation_config.v6": {
        "experiment_id": "stride-structshell-v2-official-bounded-confirmation-v3",
        "pre_registration_parent_commit": "4d44225",
        "episode_process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "stopping_rule": "wall-clock",
        "scientific_status": "preregistered_result_blind_paired_bounded_ttf_confirmation",
        "status_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_status.v3",
        "report_schema": "lns2.stride.structshell_v2_official_bounded_confirmation_report.v3",
        "controllers": CONTROLLERS,
        "solver_seeds": (13, 14, 15),
        "comparison": {
            "primary_baseline": "v2_only",
            "secondary_baseline": "official_adaptive",
            "quality_anchor": "v2_only",
            "challenger": "structshell_only",
            "execution_order": "rotating_strict_three_controller_serial",
            "paired_solver_seed_required": True,
            "workers_for_timed_episodes": 1,
            "workers_for_qualification": 16,
        },
        "cohort_repair": {
            "predecessor_experiment_id": "stride-structshell-v2-official-bounded-confirmation-v2",
            "predecessor_run_fingerprint": "dd793fb94255bd6f1adaba3f3441a0a90fe591ea02ed61e02a2774796aa11e90",
            "replaced_group_ids": [
                "random-32-32-10",
                "warehouse-20-40-10-2-2-congestion",
            ],
            "replacement_group_ids": [
                "random-32-32-20-high-load",
                "warehouse-10-20-10-2-1-congestion",
            ],
            "selection_basis": "registered_metadata_smaller_denser_topology_and_higher_agent_load",
            "controller_outcomes_consulted": False,
            "fresh_solver_seeds_required": True,
            "old_formal_episode_count": 0,
            "repair_round": 2,
            "maximum_repair_rounds": 2,
            "final_replacement_attempt": True,
        },
    },
}
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "confirmation_report.json"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="Hybrid routed confirmation")


def _identity(config: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(config.get("schema") or "")
    try:
        return dict(_IDENTITIES[schema])
    except KeyError as error:
        raise ValueError("unsupported Hybrid routed confirmation schema") from error


def _controllers(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(map(str, _identity(config)["controllers"]))


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    identity = _identity(config)
    if (
        config.get("scientific_status")
        != identity.get(
            "scientific_status",
            "preregistered_result_blind_paired_raw_ttf_confirmation",
        )
        or config.get("experiment_id") != identity["experiment_id"]
        or str(config.get("pre_registration_parent_commit"))
        != identity["pre_registration_parent_commit"]
        or tuple(map(str, config.get("controllers") or ()))
        != tuple(identity["controllers"])
    ):
        raise ValueError("Hybrid routed confirmation identity changed")
    comparison = dict(config.get("comparison") or {})
    if comparison != identity["comparison"]:
        raise ValueError("Hybrid routed confirmation comparison changed")
    if "cohort_repair" in identity and dict(config.get("cohort_repair") or {}) != dict(
        identity["cohort_repair"]
    ):
        raise ValueError("Hybrid routed confirmation cohort repair changed")
    runtime = dict(config.get("runtime") or {})
    expected_runtime = {
        "stopping_rule": str(identity.get("stopping_rule", "run-to-completion")),
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "episode_process_timeout_seconds": identity[
            "episode_process_timeout_seconds"
        ],
    }
    if "wall_time_budget_seconds" in identity:
        expected_runtime.update(
            {
                "wall_time_budget_seconds": identity["wall_time_budget_seconds"],
                "environment_time_limit_seconds": identity[
                    "environment_time_limit_seconds"
                ],
                "outer_job_timeout_seconds": identity["outer_job_timeout_seconds"],
            }
        )
    if runtime != expected_runtime:
        raise ValueError("Hybrid routed confirmation runtime changed")
    augmentation = validate_routed_hybridstructpool_augmentation(
        dict(config["challenger_augmentation"])
    )
    if augmentation.get("source_mode") != "structshell_only":
        raise ValueError("confirmation challenger is not frozen StructShell-only")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "result_blind_map_disjoint_confirmation"
        or tuple(map(int, cohort.get("solver_seeds") or ()))
        != tuple(identity["solver_seeds"])
        or int(cohort.get("paired_key_count", -1)) != 60
        or int(cohort.get("episode_count_per_controller", -1)) != 60
        or cohort.get("result_based_filtering") is not False
        or len(groups) != 10
        or any(len(list(group.get("tasks") or ())) != 2 for group in groups)
    ):
        raise ValueError("Hybrid routed confirmation cohort changed")
    map_ids = {str(group["id"]) for group in groups}
    if map_ids & set(map(str, cohort["development_map_ids"])):
        raise ValueError("confirmation map overlaps development")
    families = {str(group["family"]) for group in groups}
    if not {"maze", "room", "warehouse", "game", "dao"} <= families:
        raise ValueError("confirmation does not cover every registered map family")
    for specification in dict(config.get("inputs") or {}).values():
        _registered(root, specification)
    development = _read_json(root / str(config["inputs"]["development_report"]["path"]))
    if (
        development.get("development_gate_passed") is not True
        or development.get("selected_controller") != "structshell_only"
    ):
        raise ValueError("development ablation did not freeze StructShell-only")
    for group in groups:
        tasks = _dataset_tasks((root / str(group["dataset"])).resolve(), str(group["split"]))
        if set(map(str, group["tasks"])) - set(tasks):
            raise ValueError(f"confirmation task absent for map {group['id']}")
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controllers = _controllers(config)
    key_index = 0
    for group in config["cohort"]["groups"]:
        for task in group["tasks"]:
            for seed in config["cohort"]["solver_seeds"]:
                offset = key_index % len(controllers)
                for position in range(len(controllers)):
                    controller_index = (offset + position) % len(controllers)
                    rows.append(
                        {
                            "group_id": str(group["id"]),
                            "family": str(group["family"]),
                            "task_id": str(task),
                            "solver_seed": int(seed),
                            "controller": controllers[controller_index],
                            "within_key_position": position,
                        }
                    )
                key_index += 1
    return rows


def _controller_kwargs(root: Path, config: Mapping[str, Any], name: str) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    common = {
        "stopping_rule": str(runtime["stopping_rule"]),
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
    }
    if common["stopping_rule"] != "run-to-completion":
        common.update(
            {
                "wall_time_budget_seconds": float(
                    runtime["wall_time_budget_seconds"]
                ),
                "episode_process_timeout_seconds": float(
                    runtime["episode_process_timeout_seconds"]
                ),
                "environment_time_limit_seconds": float(
                    runtime["environment_time_limit_seconds"]
                ),
            }
        )
    if name == "official_adaptive":
        return {
            **common,
            "controller": "official_adaptive",
            "feature_backend": "auto",
            "controller_runtime": "reference",
            "verification_profile": "audit",
        }
    result = {
        **common,
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if name == "structshell_only":
        result["hybridstructpool_augmentation"] = dict(config["challenger_augmentation"])
    elif name != "v2_only":
        raise ValueError(f"unknown confirmation controller: {name}")
    return result


def _qualification_controller_kwargs(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Use the V2 arm's exact reset-time protocol for qualification evidence."""

    return _controller_kwargs(root, config, "v2_only")


def _group(config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        dict(group)
        for group in config["cohort"]["groups"]
        if str(group["id"]) == group_id
    )


def _runtime_config_path(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Path:
    source = (root / str(group["runtime_config"])).resolve()
    schema = str(config.get("schema"))
    if schema not in {
        "lns2.stride.hybridstructpool_routed_confirmation_config.v3",
        "lns2.stride.hybridstructpool_routed_confirmation_config.v4",
        "lns2.stride.hybridstructpool_routed_confirmation_config.v5",
        "lns2.stride.hybridstructpool_routed_confirmation_config.v6",
    }:
        return source
    payload = _read_json(source)
    solver_seeds = list(map(int, config["cohort"]["solver_seeds"]))
    payload["solver_seeds"] = solver_seeds
    seed_suffix = "_".join(map(str, solver_seeds))
    destination = (
        output
        / "runtime_configs"
        / f"{str(group['id'])}__solver_seeds_{seed_suffix}.json"
    )
    _write_json(destination, payload)
    return destination


def _controller_dir(output: Path, item: Mapping[str, Any]) -> Path:
    return output / "maps" / str(item["group_id"]) / str(item["controller"])


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    filename = (
        "official_adaptive_manifest.jsonl"
        if str(item["controller"]) == "official_adaptive"
        else "realized_dynamic_manifest.jsonl"
    )
    return _controller_dir(output, item) / filename


def _manifest(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("confirmation manifest is ambiguous")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    _path, root, config = load_config(job["config_path"])
    item = dict(job["item"])
    group = _group(config, str(item["group_id"]))
    dataset = root / str(group["dataset"])
    output = Path(job["output_root"]).resolve()
    runtime = _runtime_config_path(root, output, config, group)
    keys = {
        (str(task), int(seed))
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    collection = _controller_dir(output, item)
    kwargs = _controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        dataset,
        runtime,
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
        runtime,
        collection,
        phase=("official_adaptive" if item["controller"] == "official_adaptive" else "realized_dynamic"),
        workers=1,
        resume=True,
        cohort_job_keys=keys,
        job_keys={(str(item["task_id"]), int(item["solver_seed"]))},
        qualification_source=Path(job["qualification_source"]).resolve(),
        use_global_collection_lock=False,
        **kwargs,
    )
    row = _manifest(output, item)
    if row is None:
        raise RuntimeError("confirmation episode completed without manifest")
    status = str(row.get("status"))
    return {**item, "status": status if status in {"error", "timeout"} else "ok"}


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    rows = [_manifest(output, item) for item in items]
    present = [row for row in rows if row is not None]
    controllers = tuple(dict.fromkeys(str(item["controller"]) for item in items))
    return {
        **dict(base),
        "completed_jobs": len(present),
        "total_jobs": len(items),
        "completed_by_controller": {
            name: sum(
                _manifest(output, item) is not None
                for item in items
                if item["controller"] == name
            )
            for name in controllers
        },
        "error_jobs": sum(row.get("status") == "error" for row in present),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in present),
        "complete": bool(complete and len(present) == len(items)),
    }


def _qualification_summary(
    output: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for group in config["cohort"]["groups"]:
        expected = len(group["tasks"]) * len(config["cohort"]["solver_seeds"])
        path = (
            output
            / "maps"
            / str(group["id"])
            / "qualification"
            / "qualification_report.json"
        )
        if not path.is_file():
            details.append(
                {
                    "group_id": str(group["id"]),
                    "expected_count": expected,
                    "valid_count": 0,
                    "passed": False,
                    "decision": "missing_qualification_report",
                }
            )
            continue
        report = _read_json(path)
        valid = int(report.get("valid_count", -1))
        incomplete = int(report.get("incomplete_reset_count", -1))
        passed = bool(
            report.get("passed") is True
            and valid == expected
            and incomplete == 0
        )
        details.append(
            {
                "group_id": str(group["id"]),
                "expected_count": expected,
                "valid_count": valid,
                "incomplete_reset_count": incomplete,
                "passed": passed,
                "decision": str(report.get("decision") or ""),
                "nonzero_state_count": int(report.get("nonzero_state_count", 0)),
                "initial_feasible_count": int(
                    report.get("initial_feasible_count", 0)
                ),
            }
        )
    return {
        "completed_map_count": sum(
            row["valid_count"] == row["expected_count"] for row in details
        ),
        "passed_map_count": sum(row["passed"] for row in details),
        "total_map_count": len(details),
        "all_maps_passed": bool(details and all(row["passed"] for row in details)),
        "details": details,
    }


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    identity = _identity(config)
    items = schedule(config)
    if dry_run:
        controllers = _controllers(config)
        return {
            "schema": identity["status_schema"],
            "map_count": len(config["cohort"]["groups"]),
            "paired_key_count": len(items) // len(controllers),
            "schedule_entry_count": len(items),
            "schedule_sha256": _fingerprint(items),
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=str(identity["status_schema"]),
        config_path=path,
        schedule=items,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_hybridstructpool_routed_confirmation.py",
                "experiments/closed_loop_confirmation.py",
                "lns2_selector/runtime/hybridstructpool.py",
                "lns2_selector/runtime/hybridstructpool_routed.py",
                "lns2_selector/runtime/topology_candidates.py",
            ),
        ),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=str(identity["report_schema"]),
        label="HybridStructPool routed result-blind confirmation",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    _write_jsonl(output / "execution_schedule.jsonl", items)
    for group in config["cohort"]["groups"]:
        keys = {
            (str(task), int(seed))
            for task in group["tasks"]
            for seed in config["cohort"]["solver_seeds"]
        }
        qualification = output / "maps" / str(group["id"]) / "qualification"
        runtime = _runtime_config_path(root, output, config, group)
        run_closed_loop_collection(
            root / str(group["dataset"]),
            runtime,
            qualification,
            phase="qualify",
            workers=int(config["comparison"]["workers_for_qualification"]),
            resume=prepared.resumed and qualification.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            **_qualification_controller_kwargs(root, config),
        )
    qualification = _qualification_summary(output, config)
    if not qualification["all_maps_passed"]:
        status = _status(output, items, prepared.base_status)
        status["qualification"] = qualification
        status["terminal_failure"] = {
            "status": "qualification_failed",
            "failed_group_ids": [
                row["group_id"]
                for row in qualification["details"]
                if not row["passed"]
            ],
            "error": "not all preregistered maps passed qualification",
        }
        _write_json(output / STATUS_FILENAME, status)
        return status
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "collection_path": str(
                _controller_dir(output, item)
            ),
            "qualification_source": str(
                output / "maps" / str(item["group_id"]) / "qualification"
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
            phase="hybridstructpool-routed-confirmation",
            output_root=output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=float(
                config["runtime"].get(
                    "outer_job_timeout_seconds",
                    config["runtime"]["episode_process_timeout_seconds"],
                )
            ),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        failures = [row for row in results if row.get("status") in {"error", "timeout"}]
        if failures:
            status = _status(output, items, prepared.base_status)
            status["terminal_failure"] = failures[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    report = analyze(path, output, producer=prepared.base_status["producer_identity"])
    status = _status(output, items, prepared.base_status, complete=True)
    status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
    _write_json(output / STATUS_FILENAME, status)
    return report


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _repair_tail(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(row["summary"]["repair_iterations"]) for row in rows if row.get("status") == "ok"]
    return {"p95": _quantile(values, 0.95), "maximum": max(values) if values else None}


def _bounded_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _summary(rows)
    episodes = [
        dict(row.get("summary") or {})
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    successes = [row for row in episodes if row.get("success") is True]
    capped = [float(row["capped_wall_time_to_feasible"]) for row in episodes]
    result.update(
        {
            "success_rate": len(successes) / len(episodes) if episodes else 0.0,
            "right_censored_count": sum(
                str(row.get("stop_reason")) == "wall_timeout" for row in episodes
            ),
            "mean_restricted_wall_time_to_feasible": (
                statistics.fmean(capped) if capped else None
            ),
            "mean_common_success_raw_ttf": (
                statistics.fmean(
                    float(row["wall_time_to_feasible"]) for row in successes
                )
                if successes
                else None
            ),
            "mean_normalized_wall_clock_conflict_auc": (
                statistics.fmean(
                    float(row["normalized_wall_clock_conflict_auc"])
                    for row in episodes
                )
                if episodes
                else None
            ),
        }
    )
    return result


def _bounded_paired_comparison(
    baseline: Mapping[tuple[str, str, int], dict[str, Any]],
    challenger: Mapping[tuple[str, str, int], dict[str, Any]],
    keys: list[tuple[str, str, int]],
) -> dict[str, Any]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key in keys:
        left = baseline.get(key)
        right = challenger.get(key)
        if (
            left is None
            or right is None
            or left.get("status") != "ok"
            or right.get("status") != "ok"
            or not isinstance(left.get("summary"), dict)
            or not isinstance(right.get("summary"), dict)
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        pairs.append((dict(left["summary"]), dict(right["summary"])))
    baseline_capped = [
        float(left["capped_wall_time_to_feasible"]) for left, _right in pairs
    ]
    challenger_capped = [
        float(right["capped_wall_time_to_feasible"]) for _left, right in pairs
    ]
    baseline_mean = statistics.fmean(baseline_capped) if pairs else 0.0
    challenger_mean = statistics.fmean(challenger_capped) if pairs else 0.0
    deltas = [
        right - left for left, right in zip(baseline_capped, challenger_capped)
    ]
    common_success = [
        (float(left["wall_time_to_feasible"]), float(right["wall_time_to_feasible"]))
        for left, right in pairs
        if left.get("success") is True and right.get("success") is True
    ]
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_success_count": sum(left.get("success") is True for left, _ in pairs),
        "challenger_success_count": sum(right.get("success") is True for _, right in pairs),
        "baseline_mean_restricted_ttf": baseline_mean,
        "challenger_mean_restricted_ttf": challenger_mean,
        "mean_restricted_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_restricted_ttf_relative_improvement": (
            (baseline_mean - challenger_mean) / baseline_mean
            if baseline_mean
            else 0.0
        ),
        "faster_count": sum(value < -1e-9 for value in deltas),
        "slower_count": sum(value > 1e-9 for value in deltas),
        "tied_count": sum(abs(value) <= 1e-9 for value in deltas),
        "paired_faster_fraction": (
            sum(value < -1e-9 for value in deltas) / len(deltas) if deltas else 0.0
        ),
        "common_success_count": len(common_success),
        "baseline_common_success_mean_raw_ttf": (
            statistics.fmean(left for left, _right in common_success)
            if common_success
            else None
        ),
        "challenger_common_success_mean_raw_ttf": (
            statistics.fmean(right for _left, right in common_success)
            if common_success
            else None
        ),
        "mean_normalized_wall_auc_delta": (
            statistics.fmean(
                float(right["normalized_wall_clock_conflict_auc"])
                - float(left["normalized_wall_clock_conflict_auc"])
                for left, right in pairs
            )
            if pairs
            else None
        ),
    }


def _bootstrap_improvement(
    baseline: Mapping[tuple[str, str, int], dict[str, Any]],
    challenger: Mapping[tuple[str, str, int], dict[str, Any]],
    keys: list[tuple[str, str, int]],
    replicates: int,
    *,
    bounded: bool = False,
) -> dict[str, Any]:
    metric = (
        "capped_wall_time_to_feasible" if bounded else "wall_time_to_feasible"
    )
    pairs = [
        (
            float(baseline[key]["summary"][metric]),
            float(challenger[key]["summary"][metric]),
        )
        for key in keys
        if baseline[key].get("status") == "ok"
        and challenger[key].get("status") == "ok"
        and (
            bounded
            or (
                baseline[key]["summary"].get("success") is True
                and challenger[key]["summary"].get("success") is True
            )
        )
    ]
    if not pairs:
        return {"pair_count": 0, "relative_improvement": None, "ci95_lower": None, "ci95_upper": None}
    def improvement(sample: list[tuple[float, float]]) -> float:
        baseline_mean = statistics.fmean(value[0] for value in sample)
        return (baseline_mean - statistics.fmean(value[1] for value in sample)) / baseline_mean
    rng = random.Random(20260816)
    estimates = [
        improvement([pairs[rng.randrange(len(pairs))] for _ in pairs])
        for _ in range(replicates)
    ]
    return {
        "pair_count": len(pairs),
        "relative_improvement": improvement(pairs),
        "ci95_lower": _quantile(estimates, 0.025),
        "ci95_upper": _quantile(estimates, 0.975),
        "replicates": replicates,
        "seed": 20260816,
        "metric": metric,
    }


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    identity = _identity(config)
    controllers = _controllers(config)
    bounded = str(config["runtime"]["stopping_rule"]) == "wall-clock"
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=str(identity["status_schema"]),
        report_schema=str(identity["report_schema"]),
        config_path=path,
    )
    if completed is not None:
        return completed
    expected = {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for controller in controllers:
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in config["cohort"]["groups"]:
            item = {"group_id": group["id"], "controller": controller}
            manifest = _manifest_path(output, item)
            if not manifest.is_file():
                errors.append(f"{controller}/{group['id']}: missing manifest")
                continue
            hashes[controller][str(group["id"])] = sha256_file(manifest)
            for row in _read_jsonl(manifest):
                key = (str(group["id"]), str(row["task_id"]), int(row["solver_seed"]))
                if key in rows:
                    errors.append(f"{controller}: duplicate {key}")
                rows[key] = dict(row)
        if set(rows) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped = 0
    for key in sorted(expected):
        rows = [indexed[name].get(key) for name in controllers]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len({row.get("initial_fingerprint") for row in summaries}) != 1
        conflict_mismatches += len({row.get("initial_conflicts") for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        capped += sum(row.get("capped_wall_time_to_feasible") is not None for row in summaries)
    summarize = _bounded_summary if bounded else _summary
    summaries = {name: summarize(list(indexed[name].values())) for name in controllers}
    keys = sorted(expected)
    compare = _bounded_paired_comparison if bounded else _paired_comparison
    comparisons = {
        "challenger_vs_v2": compare(
            indexed["v2_only"], indexed["structshell_only"], keys
        ),
    }
    if "official_adaptive" in controllers:
        comparisons.update(
            {
                "challenger_vs_official": compare(
                    indexed["official_adaptive"], indexed["structshell_only"], keys
                ),
                "v2_vs_official": compare(
                    indexed["official_adaptive"], indexed["v2_only"], keys
                ),
            }
        )
    per_map = {
        str(group["id"]): compare(
            indexed["v2_only"],
            indexed["structshell_only"],
            [key for key in keys if key[0] == str(group["id"])],
        )
        for group in config["cohort"]["groups"]
    }
    tails = {name: _repair_tail(list(indexed[name].values())) for name in controllers}
    bootstrap = _bootstrap_improvement(
        indexed["v2_only"],
        indexed["structshell_only"],
        keys,
        int(config["performance_gates"]["paired_bootstrap_replicates"]),
        bounded=bounded,
    )
    accepted_bounded_stop_reasons = {
        "success",
        "wall_timeout",
        "controller_stalled",
        "native_terminal",
    }
    integrity = {
        "complete_paired_coverage": not any("coverage" in error or "missing" in error for error in errors),
        "zero_execution_errors_or_process_timeouts": all(
            row.get("status") == "ok" for values in indexed.values() for row in values.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        (
            "bounded_ttf_values_complete"
            if bounded
            else "no_capped_ttf_values"
        ): (
            capped == len(expected) * len(controllers)
            if bounded
            else capped == 0
        ),
        **(
            {
                "valid_bounded_stop_reasons": all(
                    str(dict(row.get("summary") or {}).get("stop_reason"))
                    in accepted_bounded_stop_reasons
                    for values in indexed.values()
                    for row in values.values()
                    if row.get("status") == "ok"
                )
            }
            if bounded
            else {}
        ),
        "zero_invalid_actions": all(summary["invalid_action_count"] == 0 for summary in summaries.values()),
        "zero_semantic_mismatches": all(summary["fingerprint_mismatch_count"] == 0 for summary in summaries.values()),
    }
    gates = dict(config["performance_gates"])
    comparison = comparisons["challenger_vs_v2"]
    improvement_field = (
        "mean_restricted_ttf_relative_improvement"
        if bounded
        else "mean_raw_ttf_relative_improvement"
    )
    map_regressions = {
        name: -float(value.get(improvement_field, -math.inf))
        for name, value in per_map.items()
        if value.get("valid")
    }
    if bounded:
        performance = {
            "success_noninferior_to_v2": summaries["structshell_only"]["success_count"]
            >= summaries["v2_only"]["success_count"],
            "mean_restricted_ttf_lower_than_v2": bool(comparison.get("valid"))
            and float(comparison[improvement_field])
            > float(gates["minimum_mean_restricted_ttf_improvement_vs_v2"]),
            "paired_faster_fraction_at_least_half": bool(comparison.get("valid"))
            and float(comparison["paired_faster_fraction"])
            >= float(gates["minimum_paired_faster_fraction_vs_v2"]),
            "bootstrap_supports_positive_restricted_gain": bootstrap["ci95_lower"]
            is not None
            and float(bootstrap["ci95_lower"])
            > float(gates["paired_bootstrap_relative_improvement_lower_bound"]),
            "every_map_within_restricted_ttf_regression_limit": len(map_regressions)
            == len(per_map)
            and max(map_regressions.values(), default=math.inf)
            <= float(gates["maximum_map_restricted_ttf_regression"]),
            "normalized_wall_auc_noninferior_to_v2": float(
                summaries["structshell_only"][
                    "mean_normalized_wall_clock_conflict_auc"
                ]
            )
            <= float(
                summaries["v2_only"]["mean_normalized_wall_clock_conflict_auc"]
            ),
        }
    else:
        performance = {
            "success_noninferior_to_v2": summaries["structshell_only"]["success_count"]
            >= summaries["v2_only"]["success_count"],
            "mean_raw_ttf_lower_than_v2": bool(comparison.get("valid"))
            and float(comparison[improvement_field])
            > float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
            "paired_faster_fraction_at_least_half": bool(comparison.get("valid"))
            and float(comparison["paired_faster_fraction"])
            >= float(gates["minimum_paired_faster_fraction_vs_v2"]),
            "bootstrap_supports_positive_gain": bootstrap["ci95_lower"] is not None
            and float(bootstrap["ci95_lower"])
            > float(gates["paired_bootstrap_relative_improvement_lower_bound"]),
            "every_map_within_regression_limit": len(map_regressions) == len(per_map)
            and max(map_regressions.values(), default=math.inf)
            <= float(gates["maximum_map_raw_ttf_regression"]),
            "p95_repair_iterations_noninferior_to_v2": tails["structshell_only"]["p95"]
            is not None
            and tails["v2_only"]["p95"] is not None
            and float(tails["structshell_only"]["p95"])
            <= float(tails["v2_only"]["p95"]),
            "maximum_repair_iterations_noninferior_to_v2": tails["structshell_only"][
                "maximum"
            ]
            is not None
            and tails["v2_only"]["maximum"] is not None
            and float(tails["structshell_only"]["maximum"])
            <= float(tails["v2_only"]["maximum"]),
        }
    integrity_passed = not errors and all(integrity.values())
    passed = integrity_passed and all(performance.values())
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=("experiments/stride_hybridstructpool_routed_confirmation.py",),
            native_required=False,
        )
    report = {
        "schema": identity["report_schema"],
        "scientific_status": (
            "result_blind_paired_bounded_ttf_confirmation"
            if bounded
            else "result_blind_paired_raw_ttf_confirmation"
        ),
        "stopping_rule": str(config["runtime"]["stopping_rule"]),
        "wall_time_budget_seconds": config["runtime"].get(
            "wall_time_budget_seconds"
        ),
        "map_count": len(config["cohort"]["groups"]),
        "paired_key_count": len(expected),
        "episode_count": sum(len(rows) for rows in indexed.values()),
        "controller_summaries": summaries,
        "comparisons": comparisons,
        "per_map": per_map,
        "repair_iteration_tails": tails,
        "paired_bootstrap_relative_ttf_improvement": bootstrap,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "confirmation_passed": passed,
        "formal_speed_claim": bool(passed and not bounded),
        "default_replacement_allowed": bool(passed and not bounded),
        "bounded_runtime_replacement_allowed": bool(passed and bounded),
        "official_adaptive_is_secondary_comparator": bounded
        and "official_adaptive" in controllers,
        "next_step": (
            "freeze_structshell_only_for_bounded_runtime_then_validate_rescue"
            if passed and bounded
            else "freeze_structshell_only_runtime"
            if passed
            else "keep_v2_default_stop_hybrid_runtime_tuning"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = ["CONTROLLERS", "analyze", "load_config", "run", "schedule"]
