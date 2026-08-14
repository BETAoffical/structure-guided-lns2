from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    mean,
    quantile,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import _paired_action
from experiments.stride_maze_tail_state_collection import _fused_controller_kwargs
from experiments.stride_multivalue_collection import _base_anchor
from experiments.stride_platformentry_order import (
    _kaplan_meier_restricted_mean,
    detect_persistent_platform,
)
from experiments.stride_pretail_forced_continuation import (
    _episode_summary,
    _source_collection,
    load_pretail_forced_continuation_config,
    prepare_cases,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.trace_replay import decision_rows, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.platformentry_frontier_registration.v1"
STATUS_SCHEMA = "lns2.stride.platformentry_frontier_status.v1"
REPORT_SCHEMA = "lns2.stride.platformentry_frontier_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.platformentry_frontier_override.v1"
EXPERIMENT_ID = "stride-platformentry-frontier-v1"
PRE_REGISTRATION_PARENT = "2f97cfc0f7b1e3a74fb900b72a4aac9cc06b5834"
EXECUTION_AMENDMENT_PARENT = "06cd24964494282585f2bf24280c954260e3cf79"
INITIAL_TRIALS = (0, 1, 2, 3)
EXTENSION_TRIALS = (4, 5, 6, 7)
DETERMINISTIC_ROLES = (
    "deterministic_compact_augment",
    "deterministic_same_size_exchange",
)
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "platformentry_frontier_report.json"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="Platform-entry frontier")


def load_platformentry_frontier_config(
    path: str | Path,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[str, Path],
    dict[str, Path],
    dict[str, Any],
]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_bounded_first_action_frontier_set_mechanism"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != PRE_REGISTRATION_PARENT
        or config.get("execution_amendment_parent_commit")
        != EXECUTION_AMENDMENT_PARENT
        or config.get("execution_amendment_reason")
        != "the_registered_same_set_order_qualification_contains_17_keys_but_the_full_45_case_frontier_cohort_contains_19_keys_so_build_one_full_protocol_identical_qualification_before_any_candidate_PP"
    ):
        raise ValueError("Platform-entry frontier registration identity changed")
    if dict(config.get("execution_amendment_recovery") or {}) != {
        "superseded_output": "build/stride-platformentry-frontier-v1",
        "replacement_output": "build/stride-platformentry-frontier-v1-r2",
        "completed_candidate_episodes_imported": 0,
        "restart_entire_initial_schedule": True,
        "preserve_superseded_artifacts": True,
    }:
        raise ValueError("Platform-entry frontier recovery identity changed")
    expected_inputs = {
        "pretail_registration",
        "frontier_cohort",
        "frontier_report",
        "runtime_config",
        "qualification_report",
        "qualification_manifest",
        "qualification_run_config",
        "qualification_summary",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("Platform-entry frontier input registry changed")
    inputs = {
        name: _registered(root, specification)
        for name, specification in dict(config["inputs"]).items()
    }
    (
        _pretail_path,
        pretail_root,
        _pretail_config,
        pretail_inputs,
        parent,
    ) = load_pretail_forced_continuation_config(inputs["pretail_registration"])
    if pretail_root != root:
        raise ValueError("Platform-entry frontier parent root changed")
    if dict(config.get("cohort") or {}) != {
        "case_count": 45,
        "checkpoint_kind": "first_structural_selection",
        "selection_rule": "all frozen platform witnesses without outcome-based exclusion",
        "frontier_candidate_count": 235,
        "unique_forced_action_count": 325,
        "outcome_enriched_mechanism_cohort": True,
        "generalization_claim": False,
    }:
        raise ValueError("Platform-entry frontier cohort changed")
    execution = dict(config.get("execution") or {})
    required_execution = {
        "initial_trial_indices": list(INITIAL_TRIALS),
        "extension_trial_indices": list(EXTENSION_TRIALS),
        "paired_first_action_pp_seed": True,
        "native_pp_order_only": True,
        "first_action_forced_exactly_once": True,
        "continuation_controller": "matching_frozen_structpool_or_slotpool",
        "registered_qualification_protocol_reused": True,
        "full_45_case_qualification_built_once": True,
        "qualification_worker_count": 16,
        "worker_count": 16,
        "per_episode_maximum_repair_decisions": 64,
        "fixed_metric_horizon": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_timeout": True,
        "right_censoring_is_valid_state_evidence": True,
        "external_process_timeout_is_execution_error": True,
        "uniform_all_action_extension_only": True,
    }
    if execution != required_execution:
        raise ValueError("Platform-entry frontier execution contract changed")
    runtime = _read_json(inputs["runtime_config"])
    if (
        runtime.get("experiment_runtime_id") != "stride-platformentry-order-v1"
        or runtime.get("max_decisions") != 64
        or runtime.get("metric_iteration_budget") != 64
        or runtime.get("wall_time_budget_seconds") != 180.0
        or runtime.get("episode_process_timeout_seconds") != 240.0
        or runtime.get("workers") != 1
        or runtime.get("deterministic_pp_replay") is not True
    ):
        raise ValueError("Platform-entry frontier frozen runtime changed")
    qualification_paths = [
        inputs["qualification_report"],
        inputs["qualification_manifest"],
        inputs["qualification_run_config"],
        inputs["qualification_summary"],
    ]
    if len({entry.parent for entry in qualification_paths}) != 1:
        raise ValueError("Platform-entry frontier qualification roots diverged")
    qualification_report = _read_json(inputs["qualification_report"])
    qualification_summary = _read_json(inputs["qualification_summary"])
    if (
        qualification_report.get("passed") is not True
        or dict(qualification_summary.get("qualification") or {}).get("passed")
        is not True
    ):
        raise ValueError("Platform-entry frontier qualification is incomplete")
    frontier_report = _read_json(inputs["frontier_report"])
    if frontier_report.get("static_readiness_passed") is not True:
        raise ValueError("FrontierDependencyPool static readiness is not passed")
    return path, root, config, inputs, pretail_inputs, parent


def _candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return (
        str(candidate["candidate_id"]),
        tuple(sorted(map(int, candidate["agents"]))),
    )


def prepare_frontier_cases(
    config_path: str | Path,
) -> tuple[
    tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Path], dict[str, Any]],
    list[dict[str, Any]],
]:
    loaded = load_platformentry_frontier_config(config_path)
    _path, _root, config, inputs, pretail_inputs, _parent = loaded
    cases = prepare_cases(
        _read_jsonl(pretail_inputs["root_checkpoints"]),
        _read_jsonl(pretail_inputs["logical_checkpoint_results"]),
        _read_jsonl(pretail_inputs["candidate_aggregates"]),
    )
    cohort = _read_jsonl(inputs["frontier_cohort"])
    by_case = {str(row["case_id"]): dict(row) for row in cohort}
    if len(by_case) != int(config["cohort"]["case_count"]):
        raise ValueError("Platform-entry frontier cohort case ids changed")
    prepared: list[dict[str, Any]] = []
    for case in cases:
        checkpoint = dict(case["checkpoint"])
        case_id = str(checkpoint["case_id"])
        frontier = by_case.get(case_id)
        if frontier is None:
            raise ValueError(f"Platform-entry frontier row missing: {case_id}")
        actual = dict(case["roles"]["actual_selected"])
        if (
            _candidate_identity(actual)
            != _candidate_identity(dict(frontier["base_candidate"]))
            or int(checkpoint["decision_index"]) != int(frontier["decision_index"])
            or str(checkpoint["state_fingerprint"])
            != str(frontier["state_fingerprint"])
        ):
            raise ValueError(f"Platform-entry frontier base identity changed: {case_id}")
        anchor = _base_anchor(checkpoint)
        actions: dict[str, dict[str, Any]] = {}

        def add(candidate: Mapping[str, Any], role: str) -> None:
            row = dict(candidate)
            candidate_id, agents = _candidate_identity(row)
            previous = actions.get(candidate_id)
            if previous is not None and tuple(previous["agents"]) != agents:
                raise ValueError(f"Candidate id collision: {case_id}/{candidate_id}")
            if previous is None:
                actions[candidate_id] = {
                    **row,
                    "candidate_id": candidate_id,
                    "agents": list(agents),
                    "logical_roles": [role],
                }
            elif role not in previous["logical_roles"]:
                previous["logical_roles"].append(role)

        add(actual, "historical_actual")
        add(anchor, "v2_anchor")
        for candidate in frontier["frontierdependency_candidates"]:
            variant = str(candidate["frontierdependency_variant"])
            add(candidate, f"frontier_candidate:{variant}")
        deterministic_ids: dict[str, str] = {}
        for variant, role in (
            ("compact-augment", "deterministic_compact_augment"),
            ("same-size-exchange", "deterministic_same_size_exchange"),
        ):
            selection = dict(frontier["deterministic_selections"][variant])
            candidate_id = str(selection["selected_candidate_id"])
            if candidate_id not in actions:
                raise ValueError(f"Deterministic candidate is absent: {case_id}/{variant}")
            add(actions[candidate_id], role)
            deterministic_ids[role] = candidate_id
        prepared.append(
            {
                **case,
                "frontier": frontier,
                "actions": sorted(actions.values(), key=lambda row: str(row["candidate_id"])),
                "role_candidate_ids": {
                    "historical_actual": str(actual["candidate_id"]),
                    "v2_anchor": str(anchor["candidate_id"]),
                    **deterministic_ids,
                },
            }
        )
    if len(prepared) != 45 or len({str(row["checkpoint"]["case_id"]) for row in prepared}) != 45:
        raise ValueError("Platform-entry frontier prepared cohort changed")
    frontier_count = sum(
        len(row["frontier"]["frontierdependency_candidates"]) for row in prepared
    )
    action_count = sum(len(row["actions"]) for row in prepared)
    if (
        frontier_count != int(config["cohort"]["frontier_candidate_count"])
        or action_count != int(config["cohort"]["unique_forced_action_count"])
    ):
        raise ValueError("Platform-entry frontier action count changed")
    return loaded, prepared


def _paired_first_action_seed(case_id: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "purpose": "platform-entry-frontier-first-action",
                "case_id": str(case_id),
                "trial_index": int(trial_index),
            }
        )[:8],
        16,
    ) & 0x7FFFFFFF


def frontier_schedule(
    cases: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = dict(case["checkpoint"])
        for trial_index in map(int, trial_indices):
            seed = _paired_first_action_seed(str(checkpoint["case_id"]), trial_index)
            for action_position, candidate in enumerate(case["actions"]):
                candidate_id = str(candidate["candidate_id"])
                schedule.append(
                    {
                        "job_id": _fingerprint(
                            {
                                "case_id": str(checkpoint["case_id"]),
                                "trial_index": trial_index,
                                "candidate_id": candidate_id,
                            }
                        ),
                        "case_id": str(checkpoint["case_id"]),
                        "state_id": str(checkpoint["state_id"]),
                        "task_id": str(checkpoint["task_id"]),
                        "solver_seed": int(checkpoint["solver_seed"]),
                        "map_id": str(checkpoint["map_id"]),
                        "challenger": str(checkpoint["challenger"]),
                        "treatment_policy": str(checkpoint["treatment_policy"]),
                        "trial_index": trial_index,
                        "candidate_id": candidate_id,
                        "candidate_agents": list(map(int, candidate["agents"])),
                        "candidate_size": len(candidate["agents"]),
                        "logical_roles": sorted(map(str, candidate["logical_roles"])),
                        "frontier_variant": candidate.get("frontierdependency_variant"),
                        "first_action_pp_seed": seed,
                        "expected_source_state_fingerprint": str(
                            checkpoint["state_fingerprint"]
                        ),
                        "expected_initial_conflicts": int(checkpoint["conflict_pair_count"]),
                        "case_position": case_position,
                        "action_position": action_position,
                    }
                )
    return schedule


def _collection_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "episodes"
        / str(item["case_id"])[:20]
        / f"trial_{int(item['trial_index']):02d}"
        / str(item["candidate_id"])
    )


def _manifest_for_item(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _collection_path(output, item) / "realized_dynamic_manifest.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Platform-entry frontier manifest is ambiguous: {path}")
    return matches[0]


def _case_restore(root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = dict(case["checkpoint"])
    source_root = _source_collection(root, checkpoint)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError(f"Platform-entry frontier source manifest is invalid: {source_root}")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(checkpoint["state_fingerprint"]),
    )
    repair_fingerprint = repair_structure_fingerprint(state)
    return {
        "collection_root": str(source_root),
        "manifest": manifest,
        "decision_index": int(checkpoint["decision_index"]),
        "expected_fingerprint": str(checkpoint["state_fingerprint"]),
        "repair_structure_fingerprint": repair_fingerprint,
        "expected_conflicts": int(checkpoint["conflict_pair_count"]),
        "restore_seed": repairability_restore_seed(repair_fingerprint),
    }


def _episode_override(
    item: Mapping[str, Any], restore: Mapping[str, Any]
) -> dict[str, Any]:
    action = _paired_action(
        list(map(int, item["candidate_agents"])), int(item["first_action_pp_seed"])
    )
    if "repair_order" in action:
        raise ValueError("Platform-entry frontier must use native PP order")
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(item["state_id"]),
        "initial_restore": dict(restore),
        "forced_first_action": action,
        "forced_candidate_id": str(item["candidate_id"]),
        "forced_candidate_role": "frontier_native_order",
        "forced_selection_families": [
            "platformentry-frontier:native-order",
            *map(str, item.get("logical_roles") or ()),
        ],
    }


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_platformentry_frontier.py",
            "experiments/stride_frontierdependencypool.py",
            "lns2_selector/runtime/frontierdependencypool.py",
            "experiments/stride_pretail_forced_continuation.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
        ),
        native_required=native_required,
    )


def _run_fingerprint(
    path: Path, producer: Mapping[str, Any], schedule: list[dict[str, Any]]
) -> str:
    return _fingerprint(
        {
            "registration_sha256": sha256_file(path),
            "producer": dict(producer),
            "schedule": schedule,
        }
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    item = dict(job["item"])
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {
        (str(value[0]), int(value[1])) for value in job["cohort_job_keys"]
    }
    override = dict(job["override"])
    kwargs = dict(job["controller_kwargs"])
    run_closed_loop_collection(
        Path(str(job["dataset"])).resolve(),
        Path(str(job["runtime_config"])).resolve(),
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        qualification_source=Path(str(job["qualification_source"])).resolve(),
        cohort_job_keys=all_keys,
        job_keys=all_keys,
        episode_overrides={key: override},
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        Path(str(job["dataset"])).resolve(),
        Path(str(job["runtime_config"])).resolve(),
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        cohort_job_keys=all_keys,
        job_keys={key},
        episode_overrides={key: override},
        use_global_collection_lock=False,
        **kwargs,
    )
    manifest = _manifest_for_item(Path(str(job["output_root"])), item)
    if manifest is None:
        raise RuntimeError("Platform-entry frontier episode completed without a manifest")
    manifest_status = str(manifest.get("status"))
    status = manifest_status if manifest_status in {"error", "timeout"} else "ok"
    return {
        **item,
        "status": status,
        "manifest_status": manifest_status,
        "error": manifest.get("error"),
        "collection_path": str(collection),
        "state_count": 1 if status == "ok" else 0,
        "outcome_count": 1 if status == "ok" else 0,
    }


def _failed_episode_job(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    return {
        **dict(job["item"]),
        "status": status,
        "manifest_status": status,
        "error": message,
        "collection_path": str(job["collection_path"]),
        "state_count": 0,
        "outcome_count": 0,
    }


def _status(
    *,
    phase: str,
    schedule: list[dict[str, Any]],
    output: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    manifests = [_manifest_for_item(output, item) for item in schedule]
    complete = [row for row in manifests if row is not None]
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "run_fingerprint": run_fingerprint,
        "completed_jobs": len(complete),
        "total_jobs": len(schedule),
        "error_jobs": sum(str(row.get("status")) == "error" for row in complete),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in complete),
        "active_jobs": [],
        "worker_count": 16,
        "complete": len(complete) == len(schedule),
    }


def run_frontier_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    resume: bool = False,
    dry_run: bool = False,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    loaded, all_cases = prepare_frontier_cases(config_path)
    path, root, config, inputs, pretail_inputs, parent = loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("Platform-entry frontier phase must be initial or extension")
    if limit_cases is not None and not dry_run:
        raise ValueError("Formal Platform-entry frontier collection cannot screen cases")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file():
            raise ValueError("Platform-entry frontier extension requires initial analysis")
        if _read_json(report_path).get("extension_allowed") is not True:
            raise ValueError("Platform-entry frontier initial gate forbids extension")
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = frontier_schedule(cases, trials)
    producer = _producer(root, native_required=not dry_run)
    run_fingerprint = _run_fingerprint(path, producer, schedule)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "case_count": len(cases),
            "unique_action_count": sum(len(case["actions"]) for case in cases),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "run_fingerprint": run_fingerprint,
            "worker_count": int(config["execution"]["worker_count"]),
            "native_pp_order_only": True,
            "maximum_repair_decisions": 64,
            "wall_time_fuse_seconds": 180.0,
            "process_timeout_seconds": 240.0,
        }
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / f"{phase}_schedule.jsonl"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("Platform-entry frontier output contains another schedule")
        if not resume:
            raise ValueError("Platform-entry frontier output exists; pass --resume")
    else:
        _write_jsonl(schedule_path, schedule)
    identity = {
        "schema": CONFIG_SCHEMA,
        "phase": phase,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "producer": producer,
        "schedule_sha256": _fingerprint(schedule),
        "run_fingerprint": run_fingerprint,
    }
    identity_path = output / f"{phase}_run_config.json"
    if identity_path.is_file() and _read_json(identity_path) != identity:
        raise ValueError("Platform-entry frontier output contains another run identity")
    _write_json(identity_path, identity)
    existing = [_manifest_for_item(output, item) for item in schedule]
    failures = [
        row
        for row in existing
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("Platform-entry frontier output contains an execution failure")
    pending = [item for item, manifest in zip(schedule, existing) if manifest is None]
    all_keys = sorted(
        {
            (str(row["checkpoint"]["task_id"]), int(row["checkpoint"]["solver_seed"]))
            for row in all_cases
        }
    )
    qualification_root = output / "qualification"
    run_closed_loop_collection(
        (root / str(parent["cohort"]["dataset"])).resolve(),
        pretail_inputs["runtime_config"],
        qualification_root,
        phase="qualify",
        workers=int(config["execution"]["qualification_worker_count"]),
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=set(all_keys),
        job_keys=set(all_keys),
        use_global_collection_lock=False,
        **_fused_controller_kwargs(root, parent, "v2-full"),
    )
    restores = {
        str(case["checkpoint"]["case_id"]): _case_restore(root, case)
        for case in cases
    }
    controller_kwargs = {
        challenger: _fused_controller_kwargs(root, parent, challenger)
        for challenger in sorted({str(item["challenger"]) for item in schedule})
    }
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    jobs = [
        {
            "job_id": str(item["job_id"]),
            "item": item,
            "output_root": str(output),
            "collection_path": str(_collection_path(output, item)),
            "dataset": str(dataset),
            "runtime_config": str(inputs["runtime_config"]),
            "qualification_source": str(qualification_root),
            "cohort_job_keys": all_keys,
            "override": _episode_override(item, restores[str(item["case_id"])]),
            "controller_kwargs": controller_kwargs[str(item["challenger"])],
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            int(config["execution"]["worker_count"]),
            phase=f"platformentry-frontier-{phase}",
            output_root=output,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["execution"]["outer_job_timeout_seconds"]),
            failure_result=_failed_episode_job,
            stop_on_failure=True,
        )
        failures = [
            row for row in results if str(row.get("status")) in {"error", "timeout"}
        ]
        if failures:
            status = _status(
                phase=phase,
                schedule=schedule,
                output=output,
                run_fingerprint=run_fingerprint,
            )
            status["terminal_failure"] = failures[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    status = _status(
        phase=phase,
        schedule=schedule,
        output=output,
        run_fingerprint=run_fingerprint,
    )
    _write_json(output / STATUS_FILENAME, status)
    return status


def _role_summary(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["logical_role"]) == role]
    return {
        "episode_count": len(selected),
        "platform_count": sum(bool(row["entered_platform"]) for row in selected),
        "platform_rate": mean(bool(row["entered_platform"]) for row in selected),
        "success_count": sum(bool(row["success"]) for row in selected),
        "success_rate": mean(bool(row["success"]) for row in selected),
        "right_censored_count": sum(
            str(row["stop_reason"]) in {"repair_limit", "wall_timeout"}
            for row in selected
        ),
        "mean_normalized_fixed_auc": mean(
            float(row["normalized_fixed_auc"]) for row in selected
        ),
        "kaplan_meier_restricted_mean_repair_decisions": (
            _kaplan_meier_restricted_mean(selected, horizon=64)
        ),
        "mean_candidate_size": mean(float(row["candidate_size"]) for row in selected),
    }


def _paired_case_bootstrap(
    logical_rows: list[dict[str, Any]], *, role: str, replicates: int
) -> dict[str, float]:
    by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in logical_rows:
        if str(row["logical_role"]) in {"historical_actual", role}:
            by_case[str(row["case_id"])].append(row)
    case_ids = sorted(by_case)
    if not case_ids:
        return {"point": 0.0, "lower_95": 0.0, "upper_95": 0.0}

    def difference(sample: list[str]) -> float:
        reference: list[bool] = []
        treatment: list[bool] = []
        for case_id in sample:
            reference.extend(
                bool(row["entered_platform"])
                for row in by_case[case_id]
                if row["logical_role"] == "historical_actual"
            )
            treatment.extend(
                bool(row["entered_platform"])
                for row in by_case[case_id]
                if row["logical_role"] == role
            )
        return mean(treatment) - mean(reference)

    point = difference(case_ids)
    rng = random.Random(0x46524F4E)
    samples = [
        difference([rng.choice(case_ids) for _ in case_ids])
        for _ in range(int(replicates))
    ]
    return {
        "point": point,
        "lower_95": quantile(samples, 0.025),
        "upper_95": quantile(samples, 0.975),
    }


def _size_band(size: int) -> str:
    if size <= 24:
        return "le24"
    if size <= 32:
        return "25to32"
    return "33to40"


def _frontier_diagnostics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    frontier = [
        row
        for row in episodes
        if any(
            str(role).startswith("frontier_candidate:")
            for role in row.get("logical_roles") or ()
        )
    ]
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("compact-augment", "same-size-exchange"):
        rows = [row for row in frontier if str(row.get("frontier_variant")) == variant]
        by_variant[variant] = {
            "episode_count": len(rows),
            "platform_rate": mean(bool(row["entered_platform"]) for row in rows),
            "success_rate": mean(bool(row["success"]) for row in rows),
            "mean_normalized_fixed_auc": mean(
                float(row["normalized_fixed_auc"]) for row in rows
            ),
            "mean_candidate_size": mean(float(row["candidate_size"]) for row in rows),
        }
    by_size_band: dict[str, dict[str, Any]] = {}
    for band in ("le24", "25to32", "33to40"):
        rows = [row for row in frontier if _size_band(int(row["candidate_size"])) == band]
        by_size_band[band] = {
            "episode_count": len(rows),
            "platform_rate": mean(bool(row["entered_platform"]) for row in rows),
            "success_rate": mean(bool(row["success"]) for row in rows),
            "mean_normalized_fixed_auc": mean(
                float(row["normalized_fixed_auc"]) for row in rows
            ),
            "mean_candidate_size": mean(float(row["candidate_size"]) for row in rows),
        }
    opportunity: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in frontier:
        grouped[(str(row["case_id"]), int(row["trial_index"]))].append(row)
    for (case_id, trial_index), rows in sorted(grouped.items()):
        opportunity.append(
            {
                "case_id": case_id,
                "trial_index": trial_index,
                "candidate_count": len(rows),
                "any_avoids_platform": any(not row["entered_platform"] for row in rows),
                "any_success": any(row["success"] for row in rows),
                "minimum_normalized_fixed_auc": min(
                    float(row["normalized_fixed_auc"]) for row in rows
                ),
            }
        )
    return {
        "claim_boundary": "posthoc pool upper bound and confounded size diagnostic only",
        "frontier_episode_count": len(frontier),
        "by_variant": by_variant,
        "by_size_band": by_size_band,
        "case_trial_groups_with_candidates": len(opportunity),
        "fraction_with_any_candidate_avoiding_platform": mean(
            bool(row["any_avoids_platform"]) for row in opportunity
        ),
        "fraction_with_any_successful_candidate": mean(
            bool(row["any_success"]) for row in opportunity
        ),
        "case_trial_opportunity": opportunity,
    }


def _initial_fingerprint_integrity(
    complete_groups: list[dict[str, dict[str, Any]]],
    expected_repair_fingerprint: dict[str, str],
) -> dict[str, bool]:
    """Validate initial identity without comparing incompatible hash domains."""
    return {
        "same_initial_runtime_fingerprint_across_actions": all(
            len(
                {
                    str(row["observed_first_action"]["before_fingerprint"])
                    for row in value.values()
                }
            )
            == 1
            for value in complete_groups
        ),
        "registered_initial_repair_fingerprint": all(
            str(row["observed_first_action"]["before_repair_fingerprint"])
            == expected_repair_fingerprint[str(row["case_id"])]
            for value in complete_groups
            for row in value.values()
        ),
    }


def analyze_frontier_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    expected_cases: int = 45,
) -> dict[str, Any]:
    loaded, all_cases = prepare_frontier_cases(config_path)
    _path, root, config, inputs, _pretail_inputs, _parent = loaded
    output = Path(output).resolve()
    if phase not in {"initial", "extended"}:
        raise ValueError("Platform-entry frontier analysis phase must be initial or extended")
    if expected_cases != 45:
        raise ValueError("Formal Platform-entry frontier analysis requires all 45 cases")
    cases = all_cases
    trials = INITIAL_TRIALS if phase == "initial" else (*INITIAL_TRIALS, *EXTENSION_TRIALS)
    schedule = frontier_schedule(cases, trials)
    case_by_id = {str(case["checkpoint"]["case_id"]): case for case in cases}
    expected_repair_fingerprint = {
        case_id: str(_case_restore(root, case)["repair_structure_fingerprint"])
        for case_id, case in case_by_id.items()
    }
    report_name = "initial_report.json" if phase == "initial" else REPORT_FILENAME
    report_path = output / report_name
    cached = _read_json(report_path) if report_path.is_file() else None
    cached_episodes = list(cached.get("episodes") or ()) if cached else []
    cache_valid = (
        bool(cached)
        and cached.get("schema") == REPORT_SCHEMA
        and cached.get("phase") == phase
        and len(cached_episodes) == len(schedule)
        and all(
            str(row.get("job_id")) == str(item["job_id"])
            for row, item in zip(cached_episodes, schedule)
        )
        and dict(cached.get("artifact_sha256") or {}).get("frontier_cohort")
        == sha256_file(inputs["frontier_cohort"])
        and dict(cached.get("artifact_sha256") or {}).get("runtime_config")
        == sha256_file(inputs["runtime_config"])
    )
    episodes: list[dict[str, Any]] = []
    if cache_valid:
        episodes = [dict(row) for row in cached_episodes]
    else:
        for item in schedule:
            manifest = _manifest_for_item(output, item)
            if manifest is None:
                episodes.append({**item, "missing": True})
                continue
            row = {**item, "manifest_status": str(manifest.get("status"))}
            row.update(_episode_summary(manifest))
            if manifest.get("status") == "ok":
                decisions, _events = decision_rows(_collection_path(output, item), manifest)
                row.update(detect_persistent_platform(decisions))
                first = decisions[0] if decisions else None
                row["observed_first_action"] = (
                    {
                        "neighborhood": list(
                            map(int, first["actual_metrics"].get("neighborhood") or ())
                        ),
                        "requested_pp_seed": int(
                            first["actual_metrics"].get("requested_pp_random_seed", -1)
                        ),
                        "before_repair_fingerprint": str(
                            first["before_repair_fingerprint"]
                        ),
                        "before_fingerprint": str(first["before_fingerprint"]),
                        "explicit_repair_order_requested": "repair_order"
                        in dict(first["actual_action"]),
                    }
                    if first is not None
                    else None
                )
            episodes.append(row)
    complete = [row for row in episodes if not row.get("missing")]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in complete:
        grouped[(str(row["case_id"]), int(row["trial_index"]))][
            str(row["candidate_id"])
        ] = row
    logical_rows: list[dict[str, Any]] = []
    for row in complete:
        for role in row["logical_roles"]:
            if role in {
                "historical_actual",
                "v2_anchor",
                *DETERMINISTIC_ROLES,
            }:
                logical_rows.append({**row, "logical_role": role})
    roles = ("historical_actual", "v2_anchor", *DETERMINISTIC_ROLES)
    role_summary = {role: _role_summary(logical_rows, role) for role in roles}
    reference = role_summary["historical_actual"]
    bootstrap = {
        role: _paired_case_bootstrap(
            logical_rows,
            role=role,
            replicates=int(
                config["final_gate_per_deterministic_variant"][
                    "paired_case_cluster_bootstrap_replicates"
                ]
            ),
        )
        for role in DETERMINISTIC_ROLES
    }
    map_summary: dict[str, dict[str, Any]] = {}
    for map_id in sorted({str(row["map_id"]) for row in complete}):
        rows = [row for row in logical_rows if str(row["map_id"]) == map_id]
        summaries = {role: _role_summary(rows, role) for role in roles}
        map_summary[map_id] = {
            **summaries,
            "platform_rate_difference_vs_actual": {
                role: float(summaries[role]["platform_rate"])
                - float(summaries["historical_actual"]["platform_rate"])
                for role in ("v2_anchor", *DETERMINISTIC_ROLES)
            },
        }
    initial_conditions: dict[str, dict[str, bool]] = {}
    final_conditions: dict[str, dict[str, bool]] = {}
    for role in DETERMINISTIC_ROLES:
        treatment = role_summary[role]
        map_ok = all(
            float(row["platform_rate_difference_vs_actual"][role]) <= 0.05
            for row in map_summary.values()
        )
        initial_conditions[role] = {
            "platform_direction_improved": float(bootstrap[role]["point"]) < 0.0,
            "success_not_lower": float(treatment["success_rate"])
            - float(reference["success_rate"])
            >= 0.0,
            "no_map_platform_worsening_over_five_points": map_ok,
        }
        final_conditions[role] = {
            "platform_bootstrap_upper_below_zero": float(bootstrap[role]["upper_95"])
            < 0.0,
            "success_not_lower": float(treatment["success_rate"])
            - float(reference["success_rate"])
            >= 0.0,
            "normalized_auc_lower": float(treatment["mean_normalized_fixed_auc"])
            - float(reference["mean_normalized_fixed_auc"])
            < 0.0,
            "restricted_mean_repair_decisions_lower": float(
                treatment["kaplan_meier_restricted_mean_repair_decisions"]
            )
            - float(reference["kaplan_meier_restricted_mean_repair_decisions"])
            < 0.0,
            "no_map_platform_worsening_over_five_points": map_ok,
        }
    expected_groups = expected_cases * len(trials)
    expected_candidate_ids = {
        str(case["checkpoint"]["case_id"]): {
            str(action["candidate_id"]) for action in case["actions"]
        }
        for case in cases
    }
    complete_groups = [value for value in grouped.values() if value]
    integrity = {
        "case_count": len(cases) == expected_cases,
        "unique_action_count": sum(len(case["actions"]) for case in cases)
        == int(config["cohort"]["unique_forced_action_count"]),
        "schedule_count": len(schedule)
        == int(config["cohort"]["unique_forced_action_count"]) * len(trials),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "allowed_stop_reasons": all(
            str(row.get("stop_reason")) in {"success", "repair_limit", "wall_timeout"}
            for row in complete
        ),
        "forced_first_action_once": all(
            int(row.get("forced_first_action_count", -1)) == 1 for row in complete
        ),
        "no_invalid_actions": all(
            int(row.get("invalid_action_count", -1)) == 0 for row in complete
        ),
        "no_fingerprint_mismatch": all(
            int(row.get("fingerprint_mismatch_count", -1)) == 0 for row in complete
        ),
        "complete_candidate_sets": len(grouped) == expected_groups
        and all(
            set(value) == expected_candidate_ids[case_id]
            for (case_id, _trial), value in grouped.items()
        ),
        "paired_first_action_seed": all(
            len(
                {
                    int(row["observed_first_action"]["requested_pp_seed"])
                    for row in value.values()
                }
            )
            == 1
            and next(iter(value.values()))["observed_first_action"]["requested_pp_seed"]
            == next(iter(value.values()))["first_action_pp_seed"]
            for value in complete_groups
        ),
        "scheduled_neighborhood_applied": all(
            sorted(row["observed_first_action"]["neighborhood"])
            == sorted(row["candidate_agents"])
            for row in complete
        ),
        **_initial_fingerprint_integrity(
            complete_groups, expected_repair_fingerprint
        ),
        "native_pp_order_only": all(
            row["observed_first_action"]["explicit_repair_order_requested"] is False
            for row in complete
        ),
    }
    integrity_passed = all(integrity.values())
    qualifying_initial_roles = [
        role for role, conditions in initial_conditions.items() if all(conditions.values())
    ]
    passing_final_roles = [
        role for role, conditions in final_conditions.items() if all(conditions.values())
    ]
    extension_allowed = (
        phase == "initial" and integrity_passed and bool(qualifying_initial_roles)
    )
    mechanism_passed = (
        phase == "extended" and integrity_passed and bool(passing_final_roles)
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "completed_initial_frontier_set_mechanism"
            if phase == "initial"
            else "completed_extended_frontier_set_mechanism"
        ),
        "phase": phase,
        "case_count": len(cases),
        "unique_action_count": sum(len(case["actions"]) for case in cases),
        "episode_count": len(complete),
        "role_summary": role_summary,
        "paired_platform_risk_difference_vs_actual": bootstrap,
        "map_summary": map_summary,
        "initial_extension_gate": {
            "per_variant_conditions": initial_conditions,
            "qualifying_roles": qualifying_initial_roles,
            "passed": bool(qualifying_initial_roles),
            "extension_scope": "all frozen states and all frozen actions",
        },
        "final_gate": {
            "per_variant_conditions": final_conditions,
            "passing_roles": passing_final_roles,
            "passed": mechanism_passed,
        },
        "frontier_pool_diagnostic": _frontier_diagnostics(complete),
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "extension_allowed": extension_allowed,
        "mechanism_passed": mechanism_passed,
        "analysis_episode_cache_reused": cache_valid,
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_sha256": {
            "frontier_cohort": sha256_file(inputs["frontier_cohort"]),
            "frontier_report": sha256_file(inputs["frontier_report"]),
            "runtime_config": sha256_file(inputs["runtime_config"]),
            "qualification_report": sha256_file(inputs["qualification_report"]),
        },
        "episodes": episodes,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "DETERMINISTIC_ROLES",
    "EXTENSION_TRIALS",
    "INITIAL_TRIALS",
    "analyze_frontier_collection",
    "frontier_schedule",
    "load_platformentry_frontier_config",
    "prepare_frontier_cases",
    "run_frontier_collection",
]
