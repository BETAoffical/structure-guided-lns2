from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    read_json,
    read_jsonl,
    registered_input,
    sha256_file,
    write_json,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.run_output_guard import prepare_resumable_output
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from lns2_selector.runtime.structshell_dual16 import (
    structshell_dual16_augmentation,
    structshell_dual16_plateau_augmentation,
    validate_structshell_dual16_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.structshell_maze32_n300_fourmap_quick_config.v1"
STATUS_SCHEMA = "lns2.stride.structshell_maze32_n300_fourmap_quick_status.v1"
REPORT_SCHEMA = "lns2.stride.structshell_maze32_n300_fourmap_quick_report.v1"
EXPERIMENT_ID = "stride-structshell-maze32-n300-fourmap-quick-v1"
CONTROLLERS = (
    "official_adaptive",
    "v2_only",
    "dual16",
    "dual16_plateau",
)
SOLVER_SEED = 25
WALL_TIME_SECONDS = 90.0
PROCESS_FUSE_SECONDS = 105.0
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "quick_report.json"
EXPECTED_GROUPS = (
    (
        "maze-32-32-4-n300",
        "maze-32-32-4",
        "maze",
        "build/stride-structshell-maze32-n300-dataset-v1",
        "balanced_wall_clock",
        "maze-32-32-4__random_04__agents_0300",
    ),
    (
        "random-32-32-20-high-load",
        "random-32-32-20",
        "random",
        "build/stride-stage2-movingai-v1",
        "balanced_wall_clock",
        "random-32-32-20__random_01__agents_0400",
    ),
    (
        "room-64-64-16",
        "room-64-64-16",
        "room",
        "build/initlns-movingai-ood-dataset-v1",
        "movingai_ood",
        "room-64-64-16__random_04__agents_0600",
    ),
    (
        "warehouse-w1020a-opposite-exchange",
        "warehouse-10-20-10-2-1",
        "warehouse",
        "build/wh-f16-v2-r2/dataset",
        "balanced_wall_clock",
        "w1020a__oe__t0233__n0600",
    ),
)
SEED_IDENTITY_AUDIT = {
    "candidate_seed": SOLVER_SEED,
    "selected_solver_seed": SOLVER_SEED,
    "audit_scope": (
        "four_exact_task_ids_controller_episode_paths_and_manifest_rows_only"
    ),
    "global_freshness_scan": False,
    "outcome_fields_read": False,
    "exact_task_count": 4,
    "controller_result_match_count": 0,
    "episode_path_match_count": 0,
    "completed_before_runner_registration": True,
}


def _contract_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "pool_id",
        "runtime_id",
        "source_mode",
        "runtime_structural_family_sizes",
        "maximum_added_candidates",
        "maximum_total_candidates",
    )
    result = {key: value[key] for key in keys}
    if "runtime_filter_id" in value:
        result["runtime_filter_id"] = value["runtime_filter_id"]
    if "stall_guard" in value:
        result["stall_guard"] = value["stall_guard"]
    return result


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Maze32 N300 four-map quick config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Maze32 N300 four-map quick identity changed")

    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "environment_time_limit_seconds": WALL_TIME_SECONDS,
        "episode_process_timeout_seconds": PROCESS_FUSE_SECONDS,
        "workers_for_reset_anchor": 1,
        "workers_for_timed_episodes": 1,
        "execution_order": "rotating_strict_four_controller_serial",
        "timing_boundary": "reset_inclusive_ttf",
    }:
        raise ValueError("Maze32 N300 four-map runtime contract changed")

    contract = dict(config.get("controller_contract") or {})
    dual = validate_structshell_dual16_augmentation(
        structshell_dual16_augmentation()
    )
    plateau = validate_structshell_dual16_augmentation(
        structshell_dual16_plateau_augmentation()
    )
    assert dual is not None and plateau is not None
    if (
        contract.get("v2_base")
        != "frozen_v2_full_native_features_optimized_copeland"
        or dict(contract.get("dual16") or {}) != _contract_projection(dual)
        or dict(contract.get("dual16_plateau") or {})
        != _contract_projection(plateau)
        or contract.get("frozen_copeland") is not True
        or contract.get("full_v2_pool_retained") is not True
        or contract.get("no_router_new_ranker_retry_or_rescue") is not True
    ):
        raise ValueError("Maze32 N300 four-map controller contract changed")

    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    observed_groups = tuple(
        (
            str(group.get("id")),
            str(group.get("map_id")),
            str(group.get("family")),
            str(group.get("dataset")),
            str(group.get("split")),
            str(group.get("task")),
        )
        for group in groups
    )
    if (
        int(cohort.get("solver_seed", -1)) != SOLVER_SEED
        or int(cohort.get("paired_key_count", -1)) != 4
        or int(cohort.get("episode_count", -1)) != 16
        or observed_groups != EXPECTED_GROUPS
        or dict(config.get("solver_seed_identity_audit") or {})
        != SEED_IDENTITY_AUDIT
    ):
        raise ValueError("Maze32 N300 four-map cohort changed")

    report_contract = dict(config.get("report_contract") or {})
    if (
        report_contract.get("selection_gate") is not None
        or report_contract.get("bootstrap") is not False
        or report_contract.get("auc_gate") is not False
        or report_contract.get("formal_speed_claim") is not False
        or report_contract.get("default_replacement_allowed") is not False
    ):
        raise ValueError("Maze32 N300 four-map report contract changed")

    inputs = dict(config.get("inputs") or {})
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="Maze32 N300 four-map V2 controller manifest",
    )
    if controller_manifest.parent != (
        root / str(config["controller_bundle"])
    ).resolve():
        raise ValueError("Maze32 N300 four-map controller bundle changed")
    for group in groups:
        runtime_path = registered_input(
            root,
            dict(inputs.get(str(group["runtime_config"])) or {}),
            label=f"{group['id']} runtime config",
        )
        manifest_path = registered_input(
            root,
            dict(inputs.get(str(group["manifest"])) or {}),
            label=f"{group['id']} dataset manifest",
        )
        task_path = registered_input(
            root,
            dict(inputs.get(str(group["task_input"])) or {}),
            label=f"{group['id']} exact task",
        )
        dataset = (root / str(group["dataset"])).resolve()
        split = str(group["split"])
        task_id = str(group["task"])
        if manifest_path != (dataset / split / "manifest.jsonl").resolve():
            raise ValueError(f"{group['id']} manifest location changed")
        if task_path != (dataset / split / "tasks" / f"{task_id}.json").resolve():
            raise ValueError(f"{group['id']} task location changed")
        task = _dataset_tasks(dataset, split).get(task_id)
        if task is None or str(task.get("map_id")) != str(group["map_id"]):
            raise ValueError(f"{group['id']} exact task identity changed")
        group["_runtime_path"] = str(runtime_path)
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["cohort"]["solver_seed"])
    rows: list[dict[str, Any]] = []
    for key_index, group in enumerate(config["cohort"]["groups"]):
        order = CONTROLLERS[key_index:] + CONTROLLERS[:key_index]
        for position, controller in enumerate(order):
            rows.append(
                {
                    "key_index": key_index,
                    "key_id": f"{group['id']}@{seed}",
                    "within_key_position": position,
                    "group_id": str(group["id"]),
                    "map_id": str(group["map_id"]),
                    "family": str(group["family"]),
                    "task_id": str(group["task"]),
                    "solver_seed": seed,
                    "controller": controller,
                }
            )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    rows = schedule(config)
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "map_count": 4,
        "task_count": 4,
        "paired_key_count": 4,
        "controller_count": 4,
        "timed_episode_count": 16,
        "paired_reset_anchor_count": 4,
        "solver_seed": SOLVER_SEED,
        "controllers": list(CONTROLLERS),
        "keys": [
            str(row["key_id"])
            for row in rows
            if int(row["within_key_position"]) == 0
        ],
        "first_position_rotation": [
            str(row["controller"])
            for row in rows
            if int(row["within_key_position"]) == 0
        ],
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "episode_process_fuse_seconds": PROCESS_FUSE_SECONDS,
        "maximum_registered_timed_seconds": 16 * WALL_TIME_SECONDS,
        "maximum_timed_process_fuse_seconds": 16 * PROCESS_FUSE_SECONDS,
        "maximum_reset_anchor_process_fuse_seconds": 4 * PROCESS_FUSE_SECONDS,
        "maximum_reset_plus_timed_process_fuse_seconds": 20
        * PROCESS_FUSE_SECONDS,
        "strict_serial_timing": True,
        "reset_inclusive_ttf": True,
        "seed_identity_audit": dict(SEED_IDENTITY_AUDIT),
        "solver_or_controller_invoked": False,
        "map_generation": False,
        "global_freshness_scan": False,
        "q0_geometry_audit": False,
        "bootstrap": False,
        "auc_gate": False,
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "environment_time_limit_seconds": WALL_TIME_SECONDS,
        "episode_process_timeout_seconds": PROCESS_FUSE_SECONDS,
    }
    if controller == "official_adaptive":
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
        "controller_bundle": str(
            (root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if controller == "v2_only":
        return result
    if controller == "dual16":
        augmentation = structshell_dual16_augmentation()
    elif controller == "dual16_plateau":
        augmentation = structshell_dual16_plateau_augmentation()
    else:
        raise ValueError(f"unknown Maze32 N300 four-map controller: {controller}")
    result["hybridstructpool_augmentation"] = (
        validate_structshell_dual16_augmentation(augmentation)
    )
    return result


def _group(config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        dict(group)
        for group in config["cohort"]["groups"]
        if str(group["id"]) == group_id
    )


def _runtime_config_path(
    output: Path, group: Mapping[str, Any], solver_seed: int
) -> Path:
    payload = read_json(Path(str(group["_runtime_path"])).resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"{group['id']} runtime config must be an object")
    payload["split"] = str(group["split"])
    payload["solver_seeds"] = [solver_seed]
    payload["wall_time_budget_seconds"] = WALL_TIME_SECONDS
    payload["episode_process_timeout_seconds"] = PROCESS_FUSE_SECONDS
    payload["workers"] = 1
    environment = dict(payload.get("environment") or {})
    environment["time_limit"] = WALL_TIME_SECONDS
    payload["environment"] = environment
    destination = (
        output / "runtime_configs" / f"{group['id']}__seed_{solver_seed:04d}.json"
    )
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError(f"{group['id']} materialized runtime changed")
    else:
        write_json(destination, payload)
    return destination


def _controller_root(output: Path, item: Mapping[str, Any]) -> Path:
    return output / "maps" / str(item["group_id"]) / str(item["controller"])


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    name = (
        "official_adaptive_manifest.jsonl"
        if item["controller"] == "official_adaptive"
        else "realized_dynamic_manifest.jsonl"
    )
    return _controller_root(output, item) / name


def _manifest_row(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        dict(row)
        for row in (read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("Maze32 N300 four-map manifest is ambiguous")
    return matches[0] if matches else None


def _status(
    output: Path,
    rows: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = sum(_manifest_row(output, item) is not None for item in rows)
    result = {
        **dict(base),
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
    }
    if terminal_failure is not None:
        result["terminal_failure"] = dict(terminal_failure)
    return result


def _qualify_group(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    task_id = str(group["task"])
    seed = int(config["cohort"]["solver_seed"])
    key = {(task_id, seed)}
    qualification = output / "maps" / str(group["id"]) / "reset_anchor"
    run_closed_loop_collection(
        root / str(group["dataset"]),
        _runtime_config_path(output, group, seed),
        qualification,
        phase="qualify",
        workers=1,
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        task_ids=[task_id],
        cohort_job_keys=key,
        job_keys=key,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "v2_only"),
    )
    report = read_json(qualification / "qualification_report.json")
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise RuntimeError(f"{group['id']} paired reset anchor failed")
    return qualification


def _run_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    qualification: Path,
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    key = {(task_id, seed)}
    collection = _controller_root(output, item)
    kwargs = controller_kwargs(root, config, str(item["controller"]))
    common = {
        "workers": 1,
        "task_ids": [task_id],
        "cohort_job_keys": key,
        "job_keys": key,
        "qualification_source": qualification,
        "qualification_process_timeout_seconds": PROCESS_FUSE_SECONDS,
        "use_global_collection_lock": False,
    }
    runtime = _runtime_config_path(output, group, seed)
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        resume=collection.joinpath("run_config.json").is_file(),
        **common,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase=(
            "official_adaptive"
            if item["controller"] == "official_adaptive"
            else "realized_dynamic"
        ),
        resume=True,
        **common,
        **kwargs,
    )
    row = _manifest_row(output, item)
    if row is None:
        raise RuntimeError("Maze32 N300 four-map episode produced no manifest")
    return row


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output_path = Path(output).resolve()
    rows = schedule(config)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for item in rows:
        key = (str(item["group_id"]), str(item["controller"]))
        row = _manifest_row(output_path, item)
        if row is None:
            errors.append(f"missing manifest: {key}")
            continue
        summary = row.get("summary")
        if row.get("status") != "ok" or not isinstance(summary, dict):
            errors.append(f"invalid episode: {key}")
            continue
        if (
            float(summary.get("wall_time_budget_seconds", -1.0))
            != WALL_TIME_SECONDS
            or summary.get("ttf_clock_schema")
            != "lns2.ttf.reset_inclusive_wall.v1"
            or summary.get("capped_wall_time_to_feasible") is None
            or int(summary.get("invalid_action_count", -1)) != 0
            or int(summary.get("fingerprint_mismatch_count", -1)) != 0
        ):
            errors.append(f"invalid bounded summary: {key}")
            continue
        indexed[key] = row

    per_map: dict[str, dict[str, Any]] = {}
    controller_summaries: dict[str, dict[str, Any]] = {}

    def diagnostics(summary: Mapping[str, Any]) -> dict[str, float | int]:
        totals = dict(summary.get("controller_totals") or {})
        return {
            "candidate_generation_seconds": float(
                totals.get("candidate_generation_seconds", 0.0)
            ),
            "neighborhood_selection_seconds": float(
                totals.get("neighborhood_selection_seconds", 0.0)
            ),
            "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
            "hybridstructpool_stall_guard_active_decision_count": int(
                totals.get(
                    "hybridstructpool_stall_guard_active_decision_count", 0
                )
            ),
            "hybridstructpool_stall_guard_trigger_count": int(
                totals.get("hybridstructpool_stall_guard_trigger_count", 0)
            ),
            "hybridstructpool_stall_guard_release_count": int(
                totals.get("hybridstructpool_stall_guard_release_count", 0)
            ),
        }

    for group in config["cohort"]["groups"]:
        group_id = str(group["id"])
        selected = [
            dict(indexed[(group_id, controller)]["summary"])
            for controller in CONTROLLERS
            if (group_id, controller) in indexed
        ]
        if len(selected) == len(CONTROLLERS):
            if (
                len({str(row["initial_fingerprint"]) for row in selected}) != 1
                or len({int(row["initial_conflicts"]) for row in selected}) != 1
            ):
                errors.append(f"paired reset mismatch: {group_id}")
            per_map[group_id] = {
                controller: {
                    "success": bool(
                        indexed[(group_id, controller)]["summary"]["success"]
                    ),
                    "restricted_ttf": float(
                        indexed[(group_id, controller)]["summary"][
                            "capped_wall_time_to_feasible"
                        ]
                    ),
                    **diagnostics(
                        indexed[(group_id, controller)]["summary"]
                    ),
                }
                for controller in CONTROLLERS
            }
    for controller in CONTROLLERS:
        selected = [
            dict(indexed[(str(group["id"]), controller)]["summary"])
            for group in config["cohort"]["groups"]
            if (str(group["id"]), controller) in indexed
        ]
        if len(selected) == 4:
            selected_diagnostics = [diagnostics(row) for row in selected]
            controller_summaries[controller] = {
                "episode_count": 4,
                "success_count": sum(bool(row["success"]) for row in selected),
                "mean_restricted_ttf": statistics.fmean(
                    float(row["capped_wall_time_to_feasible"]) for row in selected
                ),
                "mean_candidate_generation_seconds": statistics.fmean(
                    float(row["candidate_generation_seconds"])
                    for row in selected_diagnostics
                ),
                "mean_neighborhood_selection_seconds": statistics.fmean(
                    float(row["neighborhood_selection_seconds"])
                    for row in selected_diagnostics
                ),
                "mean_pp_replan_seconds": statistics.fmean(
                    float(row["pp_replan_seconds"])
                    for row in selected_diagnostics
                ),
                "hybridstructpool_stall_guard_active_decision_count": sum(
                    int(row["hybridstructpool_stall_guard_active_decision_count"])
                    for row in selected_diagnostics
                ),
                "hybridstructpool_stall_guard_trigger_count": sum(
                    int(row["hybridstructpool_stall_guard_trigger_count"])
                    for row in selected_diagnostics
                ),
                "hybridstructpool_stall_guard_release_count": sum(
                    int(row["hybridstructpool_stall_guard_release_count"])
                    for row in selected_diagnostics
                ),
            }

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "exploratory_single_seed_runtime_triage",
        "integrity_passed": not errors,
        "errors": errors,
        "map_count": 4,
        "paired_key_count": 4,
        "episode_count": len(indexed),
        "solver_seed": SOLVER_SEED,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "per_map": per_map,
        "controller_summaries": controller_summaries,
        "selection_gate": None,
        "bootstrap": False,
        "auc_gate": False,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "inputs": {"config_sha256": sha256_file(path)},
        "producer_identity": dict(producer) if producer is not None else None,
    }
    write_json(output_path / REPORT_FILENAME, report)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return plan(config_path)
    path, root, config = load_config(config_path)
    rows = schedule(config)
    output_path = Path(output).resolve()
    producer = closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_structshell_maze32_n300_fourmap_quick.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/structshell_dual16.py",
            "lns2_selector/runtime/hybridstructpool.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
    )
    prepared = prepare_resumable_output(
        output_path,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=rows,
        producer=producer,
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="Maze32 N300 four-map StructShell quick triage",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report

    anchors: dict[str, Path] = {}
    for group in config["cohort"]["groups"]:
        try:
            anchors[str(group["id"])] = _qualify_group(
                root, output_path, config, group, resume=prepared.resumed
            )
        except Exception as error:
            status = _status(
                output_path,
                rows,
                prepared.base_status,
                terminal_failure={
                    "phase": "paired_reset_anchor",
                    "group_id": str(group["id"]),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            write_json(output_path / STATUS_FILENAME, status)
            return status

    for item in rows:
        if _manifest_row(output_path, item) is not None:
            continue
        try:
            manifest = _run_episode(
                root,
                output_path,
                config,
                item,
                anchors[str(item["group_id"])],
            )
        except Exception as error:
            status = _status(
                output_path,
                rows,
                prepared.base_status,
                terminal_failure={
                    "phase": "timed_episode",
                    "item": dict(item),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            write_json(output_path / STATUS_FILENAME, status)
            return status
        if manifest.get("status") in {"error", "timeout"}:
            status = _status(
                output_path,
                rows,
                prepared.base_status,
                terminal_failure={
                    "phase": "timed_episode",
                    "item": dict(item),
                    "error": str(manifest.get("error") or manifest.get("status")),
                },
            )
            write_json(output_path / STATUS_FILENAME, status)
            return status
        write_json(
            output_path / STATUS_FILENAME,
            _status(output_path, rows, prepared.base_status),
        )

    report = analyze(path, output_path, producer=producer)
    status = _status(output_path, rows, prepared.base_status, complete=True)
    status["report_sha256"] = sha256_file(output_path / REPORT_FILENAME)
    write_json(output_path / STATUS_FILENAME, status)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "PROCESS_FUSE_SECONDS",
    "REPORT_SCHEMA",
    "SOLVER_SEED",
    "STATUS_SCHEMA",
    "WALL_TIME_SECONDS",
    "analyze",
    "controller_kwargs",
    "load_config",
    "plan",
    "run",
    "schedule",
]
