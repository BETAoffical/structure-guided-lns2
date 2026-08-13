from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any, Iterable

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
from experiments.stride_pretail_forced_continuation import (
    _episode_summary,
    _source_collection,
    load_pretail_forced_continuation_config,
    prepare_cases,
)
from experiments.stride_repairability_causal_audit import conflict_priority_order
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.trace_replay import decision_rows, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.platformentry_order_registration.v1"
STATUS_SCHEMA = "lns2.stride.platformentry_order_status.v1"
REPORT_SCHEMA = "lns2.stride.platformentry_order_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.platformentry_order_override.v1"
ARMS = ("native_order", "conflict_priority_order")
INITIAL_TRIALS = (0, 1, 2, 3)
EXTENSION_TRIALS = (4, 5, 6, 7)
PLATFORM_STREAK = 3
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "platformentry_order_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="Platform-entry order")


def load_platformentry_order_config(
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
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_bounded_first_action_order_mechanism"
        or config.get("experiment_id") != "stride-platformentry-order-v1"
        or config.get("pre_registration_parent_commit")
        != "bbebdb44f4acad14d69e5523c95da657a177417d"
        or config.get("execution_amendment_parent_commit")
        != "d53dea7ead078a8f302c576251a1eaef89acfd6e"
        or config.get("execution_amendment_reason")
        != "enforce_the_existing_native_PP_remaining_time_budget_inside_each_single_agent_search_after_the_first_formal_run_exposed_an_unbounded_low_level_overrun_before_any_outcome_analysis"
    ):
        raise ValueError("Platform-entry order registration identity changed")
    if dict(config.get("execution_amendment_recovery") or {}) != {
        "superseded_output": "build/stride-platformentry-order-v1",
        "replacement_output": "build/stride-platformentry-order-v1-r2",
        "completed_old_episodes_imported": 0,
        "restart_entire_initial_schedule": True,
        "preserve_superseded_artifacts": True,
    }:
        raise ValueError("Platform-entry order recovery identity changed")
    if set(config.get("inputs") or {}) != {
        "pretail_registration",
        "platform_entry_witnesses",
        "runtime_config",
    }:
        raise ValueError("Platform-entry order input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    (
        _pretail_path,
        pretail_root,
        _pretail,
        pretail_inputs,
        parent,
    ) = load_pretail_forced_continuation_config(inputs["pretail_registration"])
    if pretail_root != root:
        raise ValueError("Platform-entry order parent root changed")
    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "source_case_count": 45,
        "escape_classification": "same_set_order_escape",
        "case_count": 36,
        "checkpoint_kind": "first_structural_selection",
        "selection_rule": "all frozen cases with observed same-set order escape",
        "outcome_enriched_mechanism_cohort": True,
        "generalization_claim": False,
        "no_case_exclusion_after_registration": True,
    }:
        raise ValueError("Platform-entry order cohort changed")
    if tuple(map(str, config.get("arms") or ())) != ARMS:
        raise ValueError("Platform-entry order arms changed")
    execution = dict(config.get("execution") or {})
    if execution != {
        "initial_trial_indices": list(INITIAL_TRIALS),
        "extension_trial_indices": list(EXTENSION_TRIALS),
        "paired_first_action_pp_seed": True,
        "first_action_forced_exactly_once": True,
        "continuation_controller": "matching_frozen_structpool_or_slotpool",
        "qualification_mode": "build_current_protocol_v2_full_reset_set_once",
        "qualification_worker_count": 16,
        "qualification_process_timeout_seconds": 360.0,
        "worker_count": 16,
        "per_episode_maximum_repair_decisions": 64,
        "fixed_metric_horizon": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_timeout": True,
        "right_censoring_is_valid_state_evidence": True,
        "right_censoring_is_not_an_execution_error": True,
        "external_process_timeout_is_execution_error": True,
        "uniform_all_case_extension_only": True,
    }:
        raise ValueError("Platform-entry order execution bounds changed")
    if dict(config.get("platform_definition") or {}) != {
        "minimum_consecutive_exact_rollbacks": PLATFORM_STREAK,
        "replan_success_required": False,
        "repair_structure_fingerprint_unchanged": True,
        "conflict_edge_set_unchanged": True,
        "same_repair_structure_fingerprint_across_streak": True,
        "maximum_observation_decisions": 64,
    }:
        raise ValueError("Platform-entry order platform definition changed")
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
        raise ValueError("Platform-entry order runtime changed")
    return path, root, config, inputs, pretail_inputs, parent


def _paired_first_action_seed(case_id: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "purpose": "platform-entry-order-first-action",
                "case_id": str(case_id),
                "trial_index": int(trial_index),
            }
        )[:8],
        16,
    ) & 0x7FFFFFFF


def prepare_order_cases(
    config_path: str | Path,
) -> tuple[
    tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Path], dict[str, Any]],
    list[dict[str, Any]],
]:
    loaded = load_platformentry_order_config(config_path)
    _path, _root, config, inputs, pretail_inputs, _parent = loaded
    cases = prepare_cases(
        _read_jsonl(pretail_inputs["root_checkpoints"]),
        _read_jsonl(pretail_inputs["logical_checkpoint_results"]),
        _read_jsonl(pretail_inputs["candidate_aggregates"]),
    )
    witnesses = _read_jsonl(inputs["platform_entry_witnesses"])
    if len(witnesses) != int(config["cohort"]["source_case_count"]):
        raise ValueError("Platform-entry witness source case count changed")
    witness_by_case = {str(row["case_id"]): dict(row) for row in witnesses}
    if len(witness_by_case) != len(witnesses):
        raise ValueError("Platform-entry witness case ids are not unique")
    selected: list[dict[str, Any]] = []
    for case in cases:
        checkpoint = dict(case["checkpoint"])
        case_id = str(checkpoint["case_id"])
        witness = witness_by_case.get(case_id)
        if witness is None:
            raise ValueError(f"Platform-entry witness is missing: {case_id}")
        if str(witness["escape_classification"]) != "same_set_order_escape":
            continue
        candidate = dict(case["roles"]["actual_selected"])
        first_action = dict(witness["first_structural_action"])
        if (
            int(checkpoint["decision_index"]) != int(first_action["decision_index"])
            or str(checkpoint["state_fingerprint"])
            != str(first_action["pre_state_fingerprints"]["state"])
            or str(checkpoint["selected_candidate_id"])
            != str(first_action["candidate_id"])
            or sorted(map(int, candidate["agents"]))
            != sorted(map(int, first_action["agents"]))
        ):
            raise ValueError(f"Platform-entry first structural action changed: {case_id}")
        selected.append({**case, "witness": witness})
    expected = int(config["cohort"]["case_count"])
    if len(selected) != expected or len(
        {str(row["checkpoint"]["case_id"]) for row in selected}
    ) != expected:
        raise ValueError("Platform-entry same-set cohort changed")
    return loaded, selected


def order_schedule(
    cases: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = dict(case["checkpoint"])
        candidate = dict(case["roles"]["actual_selected"])
        for trial_index in map(int, trial_indices):
            for arm_position, arm in enumerate(ARMS):
                schedule.append(
                    {
                        "job_id": _fingerprint(
                            {
                                "case_id": str(checkpoint["case_id"]),
                                "trial_index": trial_index,
                                "arm": arm,
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
                        "arm": arm,
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_agents": list(map(int, candidate["agents"])),
                        "first_action_pp_seed": _paired_first_action_seed(
                            str(checkpoint["case_id"]), trial_index
                        ),
                        "expected_source_state_fingerprint": str(
                            checkpoint["state_fingerprint"]
                        ),
                        "expected_initial_conflicts": int(
                            checkpoint["conflict_pair_count"]
                        ),
                        "case_position": case_position,
                        "arm_position": arm_position,
                    }
                )
    return schedule


def _collection_path(output: Path, item: dict[str, Any]) -> Path:
    return (
        output
        / "episodes"
        / str(item["case_id"])[:20]
        / f"trial_{int(item['trial_index']):02d}"
        / str(item["arm"])
    )


def _manifest_for_item(output: Path, item: dict[str, Any]) -> dict[str, Any] | None:
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
        raise ValueError(f"Platform-entry episode manifest is ambiguous: {path}")
    return matches[0]


def _episode_override(
    root: Path,
    case: dict[str, Any],
    *,
    trial_index: int,
    arm: str,
) -> dict[str, Any]:
    checkpoint = dict(case["checkpoint"])
    candidate = dict(case["roles"]["actual_selected"])
    source_root = _source_collection(root, checkpoint)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError(f"Platform-entry source manifest is invalid: {source_root}")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(checkpoint["state_fingerprint"]),
    )
    repair_fingerprint = repair_structure_fingerprint(state)
    pp_seed = _paired_first_action_seed(str(checkpoint["case_id"]), trial_index)
    agents = list(map(int, candidate["agents"]))
    action = _paired_action(agents, pp_seed)
    if arm == "conflict_priority_order":
        action["repair_order"] = conflict_priority_order(state, agents)
    elif arm != "native_order":
        raise ValueError(f"unknown Platform-entry order arm: {arm}")
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(checkpoint["state_id"]),
        "initial_restore": {
            "collection_root": str(source_root),
            "manifest": manifest,
            "decision_index": int(checkpoint["decision_index"]),
            "expected_fingerprint": str(checkpoint["state_fingerprint"]),
            "repair_structure_fingerprint": repair_fingerprint,
            "expected_conflicts": int(checkpoint["conflict_pair_count"]),
            "restore_seed": repairability_restore_seed(repair_fingerprint),
        },
        "forced_first_action": action,
        "forced_candidate_id": str(candidate["candidate_id"]),
        "forced_candidate_role": arm,
        "forced_selection_families": [
            f"platformentry-order:{arm}",
            *map(str, candidate.get("selection_families") or ()),
        ],
    }


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_platformentry_order.py",
            "experiments/stride_pretail_forced_continuation.py",
            "experiments/stride_repairability_causal_audit.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
        ),
        native_required=native_required,
    )


def _phase_schedule_path(output: Path, phase: str) -> Path:
    return output / f"{phase}_schedule.jsonl"


def _run_fingerprint(
    path: Path, producer: dict[str, Any], schedule: list[dict[str, Any]]
) -> str:
    return _fingerprint(
        {
            "registration_sha256": sha256_file(path),
            "producer": producer,
            "schedule": schedule,
        }
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_order_cases(job["config_path"])
    _path, root, _config, inputs, _pretail_inputs, parent = loaded
    item = dict(job["item"])
    case_by_id = {str(row["checkpoint"]["case_id"]): row for row in cases}
    case = case_by_id[str(item["case_id"])]
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {
        (str(row["checkpoint"]["task_id"]), int(row["checkpoint"]["solver_seed"]))
        for row in cases
    }
    override = _episode_override(
        root,
        case,
        trial_index=int(item["trial_index"]),
        arm=str(item["arm"]),
    )
    kwargs = _fused_controller_kwargs(root, parent, str(item["challenger"]))
    common = {
        "cohort_job_keys": all_keys,
        "job_keys": {key},
        "episode_overrides": {key: override},
        "use_global_collection_lock": False,
        **kwargs,
    }
    run_closed_loop_collection(
        (root / str(parent["cohort"]["dataset"])).resolve(),
        inputs["runtime_config"],
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
        (root / str(parent["cohort"]["dataset"])).resolve(),
        inputs["runtime_config"],
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        **common,
    )
    manifest = _manifest_for_item(Path(str(job["output_root"])), item)
    if manifest is None:
        raise RuntimeError("Platform-entry episode completed without a manifest")
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
        "experiment_id": "stride-platformentry-order-v1",
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


def run_order_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    resume: bool = False,
    dry_run: bool = False,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    loaded, all_cases = prepare_order_cases(config_path)
    path, root, config, _inputs, _pretail_inputs, _parent = loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("Platform-entry order phase must be initial or extension")
    if phase == "extension":
        initial_report_path = Path(output).resolve() / "initial_report.json"
        if not initial_report_path.is_file():
            raise ValueError("Platform-entry extension requires initial analysis")
        initial_report = _read_json(initial_report_path)
        if initial_report.get("extension_allowed") is not True:
            raise ValueError("Platform-entry initial gate forbids extension")
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = order_schedule(cases, trials)
    producer = _producer(root, native_required=not dry_run)
    run_fingerprint = _run_fingerprint(path, producer, schedule)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "experiment_id": "stride-platformentry-order-v1",
            "phase": phase,
            "case_count": len(cases),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "run_fingerprint": run_fingerprint,
            "worker_count": int(config["execution"]["worker_count"]),
            "maximum_repair_decisions": 64,
            "wall_time_fuse_seconds": 180.0,
            "process_timeout_seconds": 240.0,
        }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = _phase_schedule_path(output, phase)
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("Platform-entry output contains a different schedule")
        if not resume:
            raise ValueError("Platform-entry output exists; pass --resume")
    else:
        _write_jsonl(schedule_path, schedule)
    identity_path = output / f"{phase}_run_config.json"
    identity = {
        "schema": CONFIG_SCHEMA,
        "phase": phase,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "producer": producer,
        "schedule_sha256": _fingerprint(schedule),
        "run_fingerprint": run_fingerprint,
    }
    if identity_path.is_file() and _read_json(identity_path) != identity:
        raise ValueError("Platform-entry output contains a different run identity")
    _write_json(identity_path, identity)
    pending = [item for item in schedule if _manifest_for_item(output, item) is None]
    existing_failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if existing_failures:
        raise RuntimeError("Platform-entry output contains an execution failure")
    all_keys = {
        (str(row["checkpoint"]["task_id"]), int(row["checkpoint"]["solver_seed"]))
        for row in all_cases
    }
    qualification_root = output / "qualification"
    run_closed_loop_collection(
        (root / str(_parent["cohort"]["dataset"])).resolve(),
        _pretail_inputs["runtime_config"],
        qualification_root,
        phase="qualify",
        workers=int(config["execution"]["qualification_worker_count"]),
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=all_keys,
        job_keys=all_keys,
        use_global_collection_lock=False,
        **_fused_controller_kwargs(root, _parent, "v2-full"),
    )
    jobs = [
        {
            "job_id": str(item["job_id"]),
            "config_path": str(path),
            "output_root": str(output),
            "collection_path": str(_collection_path(output, item)),
            "qualification_source": str(qualification_root),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            int(config["execution"]["worker_count"]),
            phase=f"platformentry-order-{phase}",
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


def detect_persistent_platform(
    rows: list[dict[str, Any]], *, minimum_streak: int = PLATFORM_STREAK
) -> dict[str, Any]:
    streak = 0
    streak_fingerprint: str | None = None
    start_decision: int | None = None
    maximum_streak = 0
    first_entry_decision: int | None = None
    for row in rows:
        metrics = dict(row.get("actual_metrics") or {})
        before = str(row["before_repair_fingerprint"])
        after = str(row["after_repair_fingerprint"])
        exact_rollback = (
            metrics.get("replan_success") is False
            and before == after
            and not bool(row["repair_state_changed"])
            and int(row["before_conflicts"])
            == int(row["actual_lns2"]["outcome"]["conflicts_after"])
        )
        if exact_rollback:
            if streak_fingerprint == before:
                streak += 1
            else:
                streak = 1
                streak_fingerprint = before
                start_decision = int(row["decision_index"])
            maximum_streak = max(maximum_streak, streak)
            if streak >= minimum_streak and first_entry_decision is None:
                first_entry_decision = start_decision
        else:
            streak = 0
            streak_fingerprint = None
            start_decision = None
    return {
        "entered_platform": first_entry_decision is not None,
        "first_entry_decision": first_entry_decision,
        "maximum_consecutive_exact_rollbacks": maximum_streak,
        "minimum_required_streak": minimum_streak,
    }


def _kaplan_meier_restricted_mean(
    rows: list[dict[str, Any]], *, horizon: int
) -> float:
    observations = [
        (
            min(max(int(row["repair_iterations"]), 0), horizon),
            bool(row["success"]),
        )
        for row in rows
    ]
    if not observations:
        return 0.0
    survival = 1.0
    area = 0.0
    previous = 0
    for time in sorted({time for time, event in observations if event and time <= horizon}):
        area += survival * max(0, time - previous)
        at_risk = sum(duration >= time for duration, _event in observations)
        events = sum(duration == time and event for duration, event in observations)
        if at_risk:
            survival *= 1.0 - events / at_risk
        previous = time
    area += survival * max(0, horizon - previous)
    return area


def _paired_case_bootstrap(
    episodes: list[dict[str, Any]], *, replicates: int
) -> dict[str, float]:
    by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in episodes:
        by_case[str(row["case_id"])].append(row)
    case_ids = sorted(by_case)
    if not case_ids:
        return {"point": 0.0, "lower_95": 0.0, "upper_95": 0.0}

    def difference(sample: list[str]) -> float:
        native: list[bool] = []
        order: list[bool] = []
        for case_id in sample:
            native.extend(
                bool(row["entered_platform"])
                for row in by_case[case_id]
                if row["arm"] == "native_order"
            )
            order.extend(
                bool(row["entered_platform"])
                for row in by_case[case_id]
                if row["arm"] == "conflict_priority_order"
            )
        return mean(order) - mean(native)

    point = difference(case_ids)
    rng = random.Random(0x53545249)
    samples = [
        difference([rng.choice(case_ids) for _ in case_ids])
        for _ in range(int(replicates))
    ]
    return {
        "point": point,
        "lower_95": quantile(samples, 0.025),
        "upper_95": quantile(samples, 0.975),
    }


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["arm"]) == arm]
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
    }


def analyze_order_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    expected_cases: int = 36,
) -> dict[str, Any]:
    loaded, all_cases = prepare_order_cases(config_path)
    _path, root, config, inputs, _pretail_inputs, _parent = loaded
    output = Path(output).resolve()
    if phase not in {"initial", "extended"}:
        raise ValueError("Platform-entry analysis phase must be initial or extended")
    cases = all_cases[: int(expected_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else (*INITIAL_TRIALS, *EXTENSION_TRIALS)
    schedule = order_schedule(cases, trials)
    registered_overrides = {
        str(case["checkpoint"]["case_id"]): _episode_override(
            root,
            case,
            trial_index=0,
            arm="conflict_priority_order",
        )
        for case in cases
    }
    expected_priority_order = {
        case_id: list(map(int, row["forced_first_action"]["repair_order"]))
        for case_id, row in registered_overrides.items()
    }
    expected_repair_fingerprint = {
        case_id: str(row["initial_restore"]["repair_structure_fingerprint"])
        for case_id, row in registered_overrides.items()
    }
    episodes: list[dict[str, Any]] = []
    for item in schedule:
        manifest = _manifest_for_item(output, item)
        if manifest is None:
            episodes.append({**item, "missing": True})
            continue
        row = {**item, "manifest_status": str(manifest.get("status"))}
        row.update(_episode_summary(manifest))
        if manifest.get("status") == "ok":
            decisions, _events = decision_rows(_collection_path(output, item), manifest)
            platform = detect_persistent_platform(decisions)
            row.update(platform)
            first = decisions[0] if decisions else None
            row["observed_first_action"] = (
                {
                    "neighborhood": list(
                        map(int, first["actual_metrics"].get("neighborhood") or ())
                    ),
                    "repair_order": list(
                        map(int, first["actual_metrics"].get("repair_order") or ())
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
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in complete:
        by_pair[(str(row["case_id"]), int(row["trial_index"]))][
            str(row["arm"])
        ] = row
    complete_pairs = [value for value in by_pair.values() if set(value) == set(ARMS)]
    arm_summary = {arm: _arm_summary(complete, arm) for arm in ARMS}
    native = arm_summary["native_order"]
    order = arm_summary["conflict_priority_order"]
    bootstrap = _paired_case_bootstrap(
        complete,
        replicates=int(config["final_gate"]["paired_case_cluster_bootstrap_replicates"]),
    )
    map_summary: dict[str, dict[str, Any]] = {}
    for map_id in sorted({str(row["map_id"]) for row in complete}):
        rows = [row for row in complete if str(row["map_id"]) == map_id]
        native_map = _arm_summary(rows, "native_order")
        order_map = _arm_summary(rows, "conflict_priority_order")
        map_summary[map_id] = {
            "native_order": native_map,
            "conflict_priority_order": order_map,
            "platform_rate_difference_order_minus_native": (
                float(order_map["platform_rate"]) - float(native_map["platform_rate"])
            ),
        }
    initial_gate_conditions = {
        "platform_direction_improved": float(bootstrap["point"]) < 0.0,
        "success_not_lower": (
            float(order["success_rate"]) - float(native["success_rate"])
        )
        >= 0.0,
        "no_map_platform_worsening_over_five_points": all(
            float(row["platform_rate_difference_order_minus_native"]) <= 0.05
            for row in map_summary.values()
        ),
    }
    final_gate_conditions = {
        "platform_bootstrap_upper_below_zero": float(bootstrap["upper_95"]) < 0.0,
        "success_not_lower": (
            float(order["success_rate"]) - float(native["success_rate"])
        )
        >= 0.0,
        "normalized_auc_lower": (
            float(order["mean_normalized_fixed_auc"])
            - float(native["mean_normalized_fixed_auc"])
        )
        < 0.0,
        "restricted_mean_repair_decisions_lower": (
            float(order["kaplan_meier_restricted_mean_repair_decisions"])
            - float(native["kaplan_meier_restricted_mean_repair_decisions"])
        )
        < 0.0,
        "no_map_platform_worsening_over_five_points": all(
            float(row["platform_rate_difference_order_minus_native"]) <= 0.05
            for row in map_summary.values()
        ),
    }
    expected_trials = len(trials)
    integrity = {
        "case_count": len(cases) == expected_cases,
        "schedule_count": len(schedule) == expected_cases * expected_trials * 2,
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
        "complete_paired_arms": len(by_pair) == expected_cases * expected_trials
        and all(set(value) == set(ARMS) for value in by_pair.values()),
        "paired_first_action_seed": all(
            int(value["native_order"]["observed_first_action"]["requested_pp_seed"])
            == int(
                value["conflict_priority_order"]["observed_first_action"][
                    "requested_pp_seed"
                ]
            )
            == int(value["native_order"]["first_action_pp_seed"])
            for value in complete_pairs
        ),
        "same_forced_agent_set": all(
            sorted(value["native_order"]["observed_first_action"]["neighborhood"])
            == sorted(
                value["conflict_priority_order"]["observed_first_action"][
                    "neighborhood"
                ]
            )
            == sorted(value["native_order"]["candidate_agents"])
            for value in complete_pairs
        ),
        "same_initial_repair_fingerprint": all(
            str(
                value["native_order"]["observed_first_action"][
                    "before_repair_fingerprint"
                ]
            )
            == str(
                value["conflict_priority_order"]["observed_first_action"][
                    "before_repair_fingerprint"
                ]
            )
            for value in complete_pairs
        ),
        "registered_initial_repair_fingerprint": all(
            str(value[arm]["observed_first_action"]["before_repair_fingerprint"])
            == expected_repair_fingerprint[str(value[arm]["case_id"])]
            for value in complete_pairs
            for arm in ARMS
        ),
        "only_priority_arm_requests_explicit_order": all(
            value["native_order"]["observed_first_action"][
                "explicit_repair_order_requested"
            ]
            is False
            and value["conflict_priority_order"]["observed_first_action"][
                "explicit_repair_order_requested"
            ]
            is True
            for value in complete_pairs
        ),
        "priority_order_applied_exactly": all(
            value["conflict_priority_order"]["observed_first_action"]["repair_order"]
            == expected_priority_order[
                str(value["conflict_priority_order"]["case_id"])
            ]
            for value in complete_pairs
        ),
    }
    integrity_passed = all(integrity.values())
    extension_allowed = (
        phase == "initial"
        and integrity_passed
        and all(initial_gate_conditions.values())
    )
    final_passed = (
        phase == "extended"
        and integrity_passed
        and all(final_gate_conditions.values())
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "completed_initial_order_mechanism"
            if phase == "initial"
            else "completed_extended_order_mechanism"
        ),
        "phase": phase,
        "case_count": len(cases),
        "episode_count": len(complete),
        "arm_summary": arm_summary,
        "paired_platform_risk_difference_order_minus_native": bootstrap,
        "success_rate_difference_order_minus_native": (
            float(order["success_rate"]) - float(native["success_rate"])
        ),
        "normalized_fixed_auc_difference_order_minus_native": (
            float(order["mean_normalized_fixed_auc"])
            - float(native["mean_normalized_fixed_auc"])
        ),
        "restricted_mean_repair_decision_difference_order_minus_native": (
            float(order["kaplan_meier_restricted_mean_repair_decisions"])
            - float(native["kaplan_meier_restricted_mean_repair_decisions"])
        ),
        "map_summary": map_summary,
        "initial_extension_gate": {
            "conditions": initial_gate_conditions,
            "passed": all(initial_gate_conditions.values()),
        },
        "final_gate": {
            "conditions": final_gate_conditions,
            "passed": final_passed,
        },
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "extension_allowed": extension_allowed,
        "mechanism_passed": final_passed,
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_sha256": {
            "platform_entry_witnesses": sha256_file(
                inputs["platform_entry_witnesses"]
            ),
            "runtime_config": sha256_file(inputs["runtime_config"]),
        },
        "episodes": episodes,
    }
    report_name = "initial_report.json" if phase == "initial" else REPORT_FILENAME
    _write_json(output / report_name, report)
    return report


__all__ = [
    "ARMS",
    "EXTENSION_TRIALS",
    "INITIAL_TRIALS",
    "analyze_order_collection",
    "detect_persistent_platform",
    "load_platformentry_order_config",
    "order_schedule",
    "prepare_order_cases",
    "run_order_collection",
]
