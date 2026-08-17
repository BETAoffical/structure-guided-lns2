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
from lns2_selector.runtime.structshell_single_family import (
    structshell_single_family_augmentation,
    validate_structshell_single_family_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.structshell_crossmap_quick_screen_config.v1"
STATUS_SCHEMA = "lns2.stride.structshell_crossmap_quick_screen_status.v1"
REPORT_SCHEMA = "lns2.stride.structshell_crossmap_quick_screen_report.v1"
EXPERIMENT_ID = "stride-structshell-crossmap-quick-screen-v1"
CONTROLLERS = ("v2_only", "component16", "hotspot16")
CHALLENGERS = CONTROLLERS[1:]
PROFILES = {
    "component16": "conflict_component",
    "hotspot16": "hotspot",
}
SOLVER_SEED = 17
WALL_TIME_SECONDS = 30.0
PROCESS_FUSE_SECONDS = 45.0
EXPECTED_GROUPS = (
    (
        "room-64-64-16",
        "room-64-64-16",
        "room",
        "build/initlns-movingai-ood-dataset-v1",
        "movingai_ood",
        "room-64-64-16__random_04__agents_0600",
    ),
    (
        "maze-32-32-4",
        "maze-32-32-4",
        "maze",
        "build/initlns-movingai-ood-dataset-v1",
        "movingai_ood",
        "maze-32-32-4__random_04__agents_0200",
    ),
    (
        "den020d",
        "den020d",
        "dao",
        "build/stride-topoboundary-fresh-preflight-dataset-v1",
        "balanced_wall_clock",
        "den020d__derived_opposite_exchange__task_seed_0157__agents_0620",
    ),
    (
        "random-32-32-20-high-load",
        "random-32-32-20",
        "random",
        "build/stride-stage2-movingai-v1",
        "balanced_wall_clock",
        "random-32-32-20__random_01__agents_0400",
    ),
)
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "quick_screen_report.json"


def load_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("cross-map quick-screen config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("cross-map quick-screen identity changed")

    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "environment_time_limit_seconds": WALL_TIME_SECONDS,
        "episode_process_timeout_seconds": PROCESS_FUSE_SECONDS,
        "workers_for_qualification": 1,
        "workers_for_timed_episodes": 1,
        "execution_order": "rotating_strict_three_controller_serial",
    }:
        raise ValueError("cross-map quick-screen runtime contract changed")

    contract = dict(config.get("controller_contract") or {})
    if (
        int(contract.get("nominal_size", -1)) != 16
        or int(
            contract.get(
                "single_family_maximum_added_candidates_per_decision", -1
            )
        )
        != 1
        or dict(contract.get("profiles") or {}) != PROFILES
        or "official_adaptive" not in set(map(str, contract.get("excluded") or ()))
    ):
        raise ValueError("cross-map fixed16 controller contract changed")

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
        or int(cohort.get("episode_count", -1)) != 12
        or observed_groups != EXPECTED_GROUPS
    ):
        raise ValueError("cross-map quick-screen cohort changed")

    decision = dict(config.get("screen_decision") or {})
    if (
        decision.get("baseline") != "v2_only"
        or decision.get("success_count_not_lower_than_v2") is not True
        or float(
            decision.get(
                "minimum_overall_relative_restricted_ttf_improvement", -1.0
            )
        )
        != 0.05
        or int(decision.get("minimum_paired_non_worse_count", -1)) != 3
        or decision.get("paired_non_worse_definition")
        != "challenger_restricted_ttf_less_than_or_equal_to_v2"
    ):
        raise ValueError("cross-map quick-screen selection gates changed")

    inputs = dict(config.get("inputs") or {})
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="cross-map quick-screen V2 controller manifest",
    )
    expected_bundle = (root / str(config.get("controller_bundle"))).resolve()
    if controller_manifest.parent != expected_bundle:
        raise ValueError("cross-map quick-screen controller bundle changed")

    for group in groups:
        runtime_key = str(group.get("runtime_config"))
        manifest_key = str(group.get("manifest"))
        runtime_path = registered_input(
            root,
            dict(inputs.get(runtime_key) or {}),
            label=f"{group['id']} runtime config",
        )
        manifest_path = registered_input(
            root,
            dict(inputs.get(manifest_key) or {}),
            label=f"{group['id']} dataset manifest",
        )
        dataset = (root / str(group["dataset"])).resolve()
        split = str(group["split"])
        if manifest_path != (dataset / split / "manifest.jsonl").resolve():
            raise ValueError(f"{group['id']} dataset manifest location changed")
        tasks = _dataset_tasks(dataset, split)
        task_id = str(group["task"])
        if task_id not in tasks:
            raise ValueError(f"{group['id']} quick-screen task is absent")
        row = tasks[task_id]
        if str(row.get("map_id")) != str(group["map_id"]):
            raise ValueError(f"{group['id']} quick-screen task map changed")
        group["_runtime_path"] = str(runtime_path)
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controllers = tuple(map(str, config["controllers"]))
    for key_index, group in enumerate(config["cohort"]["groups"]):
        offset = key_index % len(controllers)
        for position in range(len(controllers)):
            controller = controllers[(offset + position) % len(controllers)]
            rows.append(
                {
                    "key_index": key_index,
                    "within_key_position": position,
                    "group_id": str(group["id"]),
                    "map_id": str(group["map_id"]),
                    "family": str(group["family"]),
                    "task_id": str(group["task"]),
                    "solver_seed": int(config["cohort"]["solver_seed"]),
                    "controller": controller,
                }
            )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    rows = schedule(config)
    first_positions = [
        row["controller"] for row in rows if row["within_key_position"] == 0
    ]
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "map_count": 4,
        "paired_key_count": 4,
        "controller_count": 3,
        "timed_episode_count": len(rows),
        "qualification_reset_count": 4,
        "solver_seed": SOLVER_SEED,
        "controllers": list(CONTROLLERS),
        "groups": [str(group["id"]) for group in config["cohort"]["groups"]],
        "first_position_rotation": first_positions,
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "episode_process_fuse_seconds": PROCESS_FUSE_SECONDS,
        "maximum_registered_timed_seconds": len(rows) * WALL_TIME_SECONDS,
        "maximum_timed_process_fuse_seconds": len(rows) * PROCESS_FUSE_SECONDS,
        "maximum_qualification_process_fuse_seconds": 4
        * PROCESS_FUSE_SECONDS,
        "maximum_qualification_plus_timed_process_fuse_seconds": (
            len(rows) + 4
        )
        * PROCESS_FUSE_SECONDS,
        "strict_serial_timing": True,
        "solver_or_controller_invoked": False,
        "map_generation": False,
        "q0_geometry_audit": False,
        "official_adaptive": False,
        "bootstrap": False,
        "auc_gate": False,
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    result: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str(
            (root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
    }
    if controller == "v2_only":
        return result
    try:
        profile = PROFILES[controller]
    except KeyError as error:
        raise ValueError(f"unknown cross-map quick-screen controller: {controller}") from error
    augmentation = validate_structshell_single_family_augmentation(
        structshell_single_family_augmentation(profile, 16)
    )
    assert augmentation is not None
    result["hybridstructpool_augmentation"] = augmentation
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
    source = Path(str(group["_runtime_path"])).resolve()
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"{group['id']} runtime config must be an object")
    payload["solver_seeds"] = [solver_seed]
    destination = output / "runtime_configs" / f"{group['id']}__seed_{solver_seed}.json"
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError(f"{group['id']} materialized runtime config changed")
    else:
        write_json(destination, payload)
    return destination


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "maps"
        / str(item["group_id"])
        / str(item["controller"])
        / "realized_dynamic_manifest.jsonl"
    )


def _manifest_row(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        row
        for row in (read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("cross-map quick-screen manifest is ambiguous")
    return dict(matches[0]) if matches else None


def _status(
    output: Path,
    rows: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    present = [_manifest_row(output, item) for item in rows]
    result = {
        **dict(base),
        "completed_schedule_entries": sum(row is not None for row in present),
        "complete": bool(complete and all(row is not None for row in present)),
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
    seed = int(config["cohort"]["solver_seed"])
    task_id = str(group["task"])
    key = {(task_id, seed)}
    runtime = _runtime_config_path(output, group, seed)
    qualification = output / "maps" / str(group["id"]) / "qualification"
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
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
        raise RuntimeError(f"{group['id']} quick-screen reset qualification failed")
    return qualification


def _run_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    qualification: Path,
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    seed = int(item["solver_seed"])
    task_id = str(item["task_id"])
    key = {(task_id, seed)}
    runtime = _runtime_config_path(output, group, seed)
    collection = output / "maps" / str(item["group_id"]) / str(item["controller"])
    kwargs = controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        task_ids=[task_id],
        cohort_job_keys=key,
        job_keys=key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        task_ids=[task_id],
        cohort_job_keys=key,
        job_keys=key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=PROCESS_FUSE_SECONDS,
        use_global_collection_lock=False,
        **kwargs,
    )
    row = _manifest_row(output, item)
    if row is None:
        raise RuntimeError("cross-map quick-screen episode produced no manifest")
    return row


def select_stage_b_candidate(
    controller_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = dict(controller_summaries["v2_only"])
    baseline_mean = float(baseline["mean_restricted_ttf"])
    if baseline_mean <= 0.0:
        raise ValueError("V2 mean restricted TTF must be positive")
    baseline_by_group = {
        str(group): float(value)
        for group, value in dict(baseline["restricted_ttf_by_group"]).items()
    }
    if len(baseline_by_group) != 4:
        raise ValueError("quick-screen selection requires four paired V2 keys")
    eligible: list[str] = []
    reasons: dict[str, dict[str, bool]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for controller in CHALLENGERS:
        summary = dict(controller_summaries[controller])
        challenger_mean = float(summary["mean_restricted_ttf"])
        challenger_by_group = {
            str(group): float(value)
            for group, value in dict(summary["restricted_ttf_by_group"]).items()
        }
        if set(challenger_by_group) != set(baseline_by_group):
            raise ValueError("quick-screen selection lacks paired map coverage")
        relative_improvement = (baseline_mean - challenger_mean) / baseline_mean
        paired_non_worse_count = sum(
            challenger_by_group[group] <= baseline_by_group[group]
            for group in baseline_by_group
        )
        gates = {
            "success_noninferior": int(summary["success_count"])
            >= int(baseline["success_count"]),
            "overall_relative_restricted_ttf_improvement_at_least_five_percent": (
                relative_improvement >= 0.05
            ),
            "paired_non_worse_at_least_three_of_four": paired_non_worse_count >= 3,
        }
        reasons[controller] = gates
        metrics[controller] = {
            "relative_restricted_ttf_improvement": relative_improvement,
            "paired_non_worse_count": paired_non_worse_count,
            "paired_key_count": 4,
        }
        if all(gates.values()):
            eligible.append(controller)
    winner = min(
        eligible,
        key=lambda name: (float(controller_summaries[name]["mean_restricted_ttf"]), name),
        default=None,
    )
    return {
        "eligible_controllers": eligible,
        "gates": reasons,
        "metrics": metrics,
        "selected_controller": winner,
        "decision": (
            "advance_one_family_to_second_key_screen"
            if winner is not None
            else "stop_no_fixed16_family_passed_quick_screen"
        ),
        "promotion_or_speed_claim_allowed": False,
    }


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
        row = _manifest_row(output_path, item)
        key = (str(item["group_id"]), str(item["controller"]))
        if row is None:
            errors.append(f"missing manifest: {key}")
            continue
        if row.get("status") != "ok" or not isinstance(row.get("summary"), dict):
            errors.append(f"invalid episode status: {key}")
            continue
        indexed[key] = row

    allowed_stops = {"success", "wall_timeout", "controller_stalled", "native_terminal"}
    for key, row in indexed.items():
        summary = dict(row["summary"])
        if (
            float(summary.get("wall_time_budget_seconds", -1.0))
            != WALL_TIME_SECONDS
            or str(summary.get("stop_reason")) not in allowed_stops
            or int(summary.get("invalid_action_count", -1)) != 0
            or int(summary.get("fingerprint_mismatch_count", -1)) != 0
            or summary.get("capped_wall_time_to_feasible") is None
        ):
            errors.append(f"invalid bounded episode: {key}")

    for group in config["cohort"]["groups"]:
        group_id = str(group["id"])
        task_id = str(group["task"])
        qualification_path = (
            output_path
            / "maps"
            / group_id
            / "qualification"
            / "qualification_manifest.jsonl"
        )
        qualification_matches = [
            row
            for row in (
                read_jsonl(qualification_path) if qualification_path.is_file() else []
            )
            if str(row.get("task_id")) == task_id
            and int(row.get("solver_seed", -1)) == SOLVER_SEED
        ]
        if len(qualification_matches) != 1:
            errors.append(f"missing or ambiguous qualification anchor: {group_id}")
            qualification = None
        else:
            qualification = qualification_matches[0]
        summaries = [
            dict(indexed[(group_id, controller)]["summary"])
            for controller in CONTROLLERS
            if (group_id, controller) in indexed
        ]
        if len(summaries) == len(CONTROLLERS) and (
            len({str(row["initial_fingerprint"]) for row in summaries}) != 1
            or len({int(row["initial_conflicts"]) for row in summaries}) != 1
        ):
            errors.append(f"paired reset mismatch: {group_id}")
        if qualification is not None and any(
            str(summary["initial_fingerprint"])
            != str(qualification.get("state_fingerprint"))
            or int(summary["initial_conflicts"])
            != int(qualification.get("initial_conflicts", -1))
            for summary in summaries
        ):
            errors.append(f"qualification anchor mismatch: {group_id}")

    controller_summaries: dict[str, dict[str, Any]] = {}
    for controller in CONTROLLERS:
        restricted_by_group = {
            str(group["id"]): float(
                indexed[(str(group["id"]), controller)]["summary"][
                    "capped_wall_time_to_feasible"
                ]
            )
            for group in config["cohort"]["groups"]
            if (str(group["id"]), controller) in indexed
        }
        selected = [
            dict(indexed[(str(group["id"]), controller)]["summary"])
            for group in config["cohort"]["groups"]
            if (str(group["id"]), controller) in indexed
        ]
        if len(selected) != 4:
            continue
        controller_summaries[controller] = {
            "episode_count": 4,
            "success_count": sum(bool(row["success"]) for row in selected),
            "mean_restricted_ttf": statistics.fmean(
                float(row["capped_wall_time_to_feasible"]) for row in selected
            ),
            "restricted_ttf_by_group": restricted_by_group,
            "mean_raw_ttf_on_success": (
                statistics.fmean(
                    float(row["wall_time_to_feasible"])
                    for row in selected
                    if bool(row["success"])
                )
                if any(bool(row["success"]) for row in selected)
                else None
            ),
            "mean_repair_iterations": statistics.fmean(
                int(row["repair_iterations"]) for row in selected
            ),
            "mean_candidate_generation_seconds": statistics.fmean(
                float(dict(row["controller_totals"]).get("candidate_generation_seconds", 0.0))
                for row in selected
            ),
            "mean_neighborhood_selection_seconds": statistics.fmean(
                float(dict(row["controller_totals"]).get("neighborhood_selection_seconds", 0.0))
                for row in selected
            ),
            "mean_pp_replan_seconds": statistics.fmean(
                float(dict(row["controller_totals"]).get("pp_replan_seconds", 0.0))
                for row in selected
            ),
        }

    per_map: dict[str, dict[str, Any]] = {}
    if len(controller_summaries) == len(CONTROLLERS):
        for group in config["cohort"]["groups"]:
            group_id = str(group["id"])
            baseline = dict(indexed[(group_id, "v2_only")]["summary"])
            per_map[group_id] = {
                controller: {
                    "success": bool(indexed[(group_id, controller)]["summary"]["success"]),
                    "restricted_ttf": float(
                        indexed[(group_id, controller)]["summary"][
                            "capped_wall_time_to_feasible"
                        ]
                    ),
                    "delta_vs_v2_seconds": float(
                        indexed[(group_id, controller)]["summary"][
                            "capped_wall_time_to_feasible"
                        ]
                    )
                    - float(baseline["capped_wall_time_to_feasible"]),
                }
                for controller in CONTROLLERS
            }

    selection = (
        select_stage_b_candidate(controller_summaries)
        if len(controller_summaries) == len(CONTROLLERS) and not errors
        else {
            "eligible_controllers": [],
            "gates": {},
            "metrics": {},
            "selected_controller": None,
            "decision": "invalid_or_incomplete_quick_screen",
            "promotion_or_speed_claim_allowed": False,
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "exploratory_crossmap_development_triage",
        "integrity_passed": not errors,
        "errors": errors,
        "map_count": 4,
        "paired_key_count": 4,
        "episode_count": len(indexed),
        "wall_time_budget_seconds": WALL_TIME_SECONDS,
        "controller_summaries": controller_summaries,
        "per_map": per_map,
        "stage_b_selection": selection,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "omitted": list(config["explicitly_omitted"]),
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
            "experiments/stride_structshell_crossmap_quick_screen.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/structshell_single_family.py",
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
        label="StructShell cross-map quick screen",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report

    qualifications: dict[str, Path] = {}
    for group in config["cohort"]["groups"]:
        try:
            qualifications[str(group["id"])] = _qualify_group(
                root,
                output_path,
                config,
                group,
                resume=prepared.resumed,
            )
        except Exception as error:
            status = _status(
                output_path,
                rows,
                prepared.base_status,
                terminal_failure={
                    "phase": "qualification",
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
                qualifications[str(item["group_id"])],
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
    "CHALLENGERS",
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
    "select_stage_b_candidate",
]
