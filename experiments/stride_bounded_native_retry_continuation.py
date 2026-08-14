from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    contained_file,
    mean,
    quantile,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_state_blob, read_trace_events
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
from experiments.stride_nativeorder_transactionalrepair import (
    build_nativeorder_cohort,
    retry_pp_seed,
)
from experiments.stride_platformentry_order import _kaplan_meier_restricted_mean
from experiments.stride_pretail_forced_continuation import (
    _episode_summary,
    load_pretail_forced_continuation_config,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.bounded_native_retry_continuation_registration.v1"
STATUS_SCHEMA = "lns2.stride.bounded_native_retry_continuation_status.v1"
REPORT_SCHEMA = "lns2.stride.bounded_native_retry_continuation_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.bounded_native_retry_continuation_override.v1"
EXPERIMENT_ID = "stride-bounded-native-retry-continuation-v1"
ARMS = ("single_native_attempt", "bounded_same_set_native_retry")
INITIAL_TRIALS = (0, 1, 2, 3)
EXTENSION_TRIALS = (4, 5, 6, 7)
STATUS_FILENAME = "collection_status.json"


def load_registration(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any], dict[str, Path], dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "a8763617e46c97b213ec62f57b1713ff38161423"
        or config.get("scientific_status")
        != "preregistered_bounded_native_retry_trajectory_mechanism"
    ):
        raise ValueError("bounded native retry continuation registration changed")
    inputs = {
        name: registered_input(root, row, label=f"bounded retry {name}")
        for name, row in dict(config["inputs"]).items()
    }
    if tuple(map(str, config["arms"])) != ARMS:
        raise ValueError("bounded native retry arms changed")
    execution = dict(config["execution"])
    required_execution = {
        "initial_trial_indices": list(INITIAL_TRIALS),
        "extension_trial_indices": list(EXTENSION_TRIALS),
        "worker_count": 16,
        "qualification_worker_count": 16,
        "qualification_mode": "build_current_protocol_reset_only_per_controller_identity",
        "qualification_process_timeout_seconds": 360.0,
        "per_episode_maximum_repair_decisions": 64,
        "fixed_metric_horizon": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "minimum_consecutive_exact_rollbacks": 3,
        "initial_repeat_count": 2,
        "maximum_interventions_per_episode": 3,
        "maximum_interventions_per_platform_signature": 1,
        "retry_after_failure_reason": "conflict_bound_exceeded",
        "retry_after_time_limit": False,
        "native_pp_order_only": True,
        "first_transaction_matches_nativeorder_artifacts": True,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if execution != required_execution:
        raise ValueError("bounded native retry execution protocol changed")
    native_metadata, cohort = build_nativeorder_cohort(
        inputs["nativeorder_registration"]
    )
    if len(cohort) != 45 or len({str(row["map_id"]) for row in cohort}) != 3:
        raise ValueError("bounded native retry cohort changed")
    native_report = _read_json(inputs["nativeorder_report"])
    if (
        native_report.get("integrity_passed") is not True
        or dict(native_report["final_readiness_gates"])[
            "same_set_native_retry"
        ].get("paired_risk_upper_below_zero")
        is not True
    ):
        raise ValueError("registered NativeOrder mechanism evidence changed")
    (
        _pretail_path,
        pretail_root,
        _pretail_config,
        pretail_inputs,
        parent,
    ) = load_pretail_forced_continuation_config(inputs["pretail_registration"])
    if pretail_root != root:
        raise ValueError("bounded native retry project root changed")
    return path, root, config, inputs, native_metadata, pretail_inputs, parent


def prepare_cases(
    config_path: str | Path,
) -> tuple[
    tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any], dict[str, Path], dict[str, Any]],
    list[dict[str, Any]],
]:
    loaded = load_registration(config_path)
    _path, _root, _config, inputs, _metadata, _pretail_inputs, _parent = loaded
    _native_metadata, cohort = build_nativeorder_cohort(
        inputs["nativeorder_registration"]
    )
    frozen = [dict(row) for row in cohort]
    frozen.sort(key=lambda row: str(row["state_fingerprint"]))
    return loaded, frozen


def continuation_schedule(
    cases: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = dict(case["logical_checkpoint"])
        candidate = dict(case["selected_candidate"])
        repair_fp = repair_structure_fingerprint(
            _read_restored_state(case)
        )
        if repair_fp != str(case["state_context"].get("repair_structure_fingerprint", repair_fp)):
            # The context field is optional in old frozen blobs; the actual
            # registered state remains authoritative.
            raise ValueError("bounded retry repair context changed")
        for trial_index in map(int, trial_indices):
            first_seed = repairability_pp_seed(repair_fp, trial_index)
            for arm_position, arm in enumerate(ARMS):
                rows.append(
                    {
                        "job_id": _fingerprint(
                            {
                                "state_fingerprint": case["state_fingerprint"],
                                "trial_index": trial_index,
                                "arm": arm,
                            }
                        ),
                        "case_id": str(checkpoint["case_id"]),
                        "state_id": str(case["state_fingerprint"]),
                        "state_fingerprint": str(case["state_fingerprint"]),
                        "task_id": str(case["task_id"]),
                        "solver_seed": int(case["solver_seed"]),
                        "map_id": str(case["map_id"]),
                        "challenger": str(checkpoint["challenger"]),
                        "trial_index": trial_index,
                        "arm": arm,
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_agents": list(map(int, candidate["agents"])),
                        "first_pp_seed": first_seed,
                        "first_retry_pp_seed": retry_pp_seed(repair_fp, trial_index),
                        "case_position": case_position,
                        "arm_position": arm_position,
                    }
                )
    return rows


def _read_restored_state(case: Mapping[str, Any]) -> dict[str, Any]:
    state = read_state_blob(Path(str(case["state_blob"])))
    state["context"] = dict(case["state_context"])
    return state


def _source_collection(case: Mapping[str, Any]) -> Path:
    path = Path(str(case["source_run_config"])).resolve()
    if sha256_file(path) != str(case["source_run_config_sha256"]):
        raise ValueError("bounded retry source run config changed")
    return path.parent


def _episode_override(
    case: Mapping[str, Any], *, trial_index: int, arm: str
) -> dict[str, Any]:
    checkpoint = dict(case["logical_checkpoint"])
    candidate = dict(case["selected_candidate"])
    source_root = _source_collection(case)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError(f"bounded retry source manifest invalid: {source_root}")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(case["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(state)
    if repair_fp != repair_structure_fingerprint(_read_restored_state(case)):
        raise ValueError("bounded retry source state differs from causal blob")
    if arm not in ARMS:
        raise ValueError(f"unknown bounded retry arm: {arm}")
    first_seed = repairability_pp_seed(repair_fp, int(trial_index))
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(case["state_fingerprint"]),
        "initial_restore": {
            "collection_root": str(source_root),
            "manifest": manifest,
            "decision_index": int(checkpoint["decision_index"]),
            "expected_fingerprint": str(case["state_fingerprint"]),
            "repair_structure_fingerprint": repair_fp,
            "expected_conflicts": int(checkpoint["before_conflicts"]),
            "restore_seed": repairability_restore_seed(repair_fp),
        },
        "forced_first_action": _paired_action(
            list(map(int, candidate["agents"])), first_seed
        ),
        "forced_candidate_id": str(candidate["candidate_id"]),
        "forced_candidate_role": arm,
        "forced_selection_families": [
            f"bounded-native-retry:{arm}",
            *map(str, candidate.get("selection_families") or ()),
        ],
        "bounded_native_retry": {
            "enabled": arm == "bounded_same_set_native_retry",
            "minimum_consecutive_rollbacks": 3,
            "maximum_interventions": 3,
            "initial_repeat_count": 2,
            "seed_namespace": "stride-bounded-native-retry-continuation-v1",
            "episode_key": str(case["state_fingerprint"]),
            "trial_index": int(trial_index),
            "first_retry_seed": retry_pp_seed(repair_fp, int(trial_index)),
        },
    }


def _collection_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "episodes"
        / str(item["state_fingerprint"])[:20]
        / f"trial_{int(item['trial_index']):02d}"
        / str(item["arm"])
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
    if len(matches) > 1:
        raise ValueError(f"bounded retry manifest ambiguous: {path}")
    return matches[0] if matches else None


def _audit_decision_rows(
    collection_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Read validated transitions without pretending a transaction is replayable.

    A bounded retry contains two native PP calls in one controller decision.
    ``trace_replay.decision_rows`` intentionally requires one replay action per
    transition, so it is the wrong abstraction for this report-only audit.
    """
    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="trace_file",
    )
    events = read_trace_events(trace_path)
    if (
        len(events) < 2
        or events[0].get("event") != "initial"
        or events[-1].get("event") != "finish"
    ):
        raise ValueError("bounded retry trace event boundaries changed")
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        if event.get("event") != "transition":
            raise ValueError("bounded retry trace contains a non-transition event")
        action = event.get("action")
        metrics = event.get("metrics")
        if not isinstance(action, dict) or not isinstance(metrics, dict):
            raise ValueError("bounded retry transition lacks action or metrics")
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "actual_action": dict(action),
                "actual_metrics": dict(metrics),
            }
        )
    return rows


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_bounded_native_retry_continuation.py",
            "scripts/run_stride_bounded_native_retry_continuation.py",
            "lns2_selector/runtime/bounded_native_retry.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/stride_nativeorder_transactionalrepair.py",
            "experiments/trace_replay.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _run_fingerprint(path: Path, producer: Mapping[str, Any], schedule: list[dict[str, Any]]) -> str:
    return _fingerprint(
        {
            "registration_sha256": sha256_file(path),
            "producer": producer,
            "schedule": schedule,
        }
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_cases(job["config_path"])
    _path, root, _config, inputs, _metadata, _pretail_inputs, parent = loaded
    item = dict(job["item"])
    case_by_id = {str(row["state_fingerprint"]): row for row in cases}
    case = case_by_id[str(item["state_fingerprint"])]
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in cases}
    override = _episode_override(
        case, trial_index=int(item["trial_index"]), arm=str(item["arm"])
    )
    kwargs = _fused_controller_kwargs(root, parent, str(item["challenger"]))
    common = {
        "cohort_job_keys": all_keys,
        "job_keys": {key},
        "episode_overrides": {key: override},
        "use_global_collection_lock": False,
        **kwargs,
    }
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    run_closed_loop_collection(
        dataset,
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
        dataset,
        inputs["runtime_config"],
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        **common,
    )
    manifest = _manifest_for_item(Path(str(job["output_root"])), item)
    if manifest is None:
        raise RuntimeError("bounded retry episode completed without manifest")
    manifest_status = str(manifest.get("status"))
    return {
        **item,
        "status": manifest_status if manifest_status in {"error", "timeout"} else "ok",
        "manifest_status": manifest_status,
        "error": manifest.get("error"),
        "collection_path": str(collection),
        "state_count": int(manifest_status == "ok"),
        "outcome_count": int(manifest_status == "ok"),
    }


def _failed_job(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        **dict(job["item"]),
        "status": status,
        "manifest_status": status,
        "error": message,
        "collection_path": str(job["collection_path"]),
        "state_count": 0,
        "outcome_count": 0,
    }


def _status(phase: str, schedule: list[dict[str, Any]], output: Path, run_fp: str) -> dict[str, Any]:
    manifests = [_manifest_for_item(output, item) for item in schedule]
    complete = [row for row in manifests if row is not None]
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "run_fingerprint": run_fp,
        "completed_jobs": len(complete),
        "total_jobs": len(schedule),
        "error_jobs": sum(str(row.get("status")) == "error" for row in complete),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in complete),
        "active_jobs": [],
        "worker_count": 16,
        "complete": len(complete) == len(schedule),
    }


def run_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    resume: bool = False,
    dry_run: bool = False,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    loaded, all_cases = prepare_cases(config_path)
    path, root, config, inputs, _metadata, _pretail_inputs, _parent = loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("bounded retry phase must be initial or extension")
    output = Path(output).resolve()
    if phase == "extension":
        initial_report_path = output / "initial_report.json"
        if not initial_report_path.is_file() or _read_json(initial_report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("bounded retry initial gate forbids extension")
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = continuation_schedule(cases, trials)
    producer = _producer(root, native_required=not dry_run)
    run_fp = _run_fingerprint(path, producer, schedule)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "phase": phase,
            "case_count": len(cases),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "run_fingerprint": run_fp,
            "worker_count": 16,
        }
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / f"{phase}_schedule.jsonl"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("bounded retry output contains a different schedule")
        if not resume:
            raise ValueError("bounded retry output exists; pass --resume")
    else:
        _write_jsonl(schedule_path, schedule)
    identity = {
        "schema": CONFIG_SCHEMA,
        "phase": phase,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "producer": producer,
        "schedule_sha256": _fingerprint(schedule),
        "run_fingerprint": run_fp,
    }
    identity_path = output / f"{phase}_run_config.json"
    if identity_path.is_file() and _read_json(identity_path) != identity:
        raise ValueError("bounded retry output contains a different run identity")
    _write_json(identity_path, identity)
    pending = [item for item in schedule if _manifest_for_item(output, item) is None]
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("bounded retry output contains an execution failure")
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in all_cases}
    # Qualification reuse deliberately binds the controller implementation.
    # StructPool and SlotPool load different augmentation code, even though
    # qualification itself is reset-only, so build one current-producer source
    # per frozen controller identity instead of weakening that identity gate.
    challengers = sorted({str(item["challenger"]) for item in schedule})
    qualification_roots = {
        challenger: output / "qualification" / challenger
        for challenger in challengers
    }
    for challenger, qualification_root in qualification_roots.items():
        run_closed_loop_collection(
            (root / str(_parent["cohort"]["dataset"])).resolve(),
            inputs["runtime_config"],
            qualification_root,
            phase="qualify",
            workers=int(config["execution"]["qualification_worker_count"]),
            qualification_process_timeout_seconds=float(
                config["execution"]["qualification_process_timeout_seconds"]
            ),
            resume=qualification_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=all_keys,
            job_keys=all_keys,
            use_global_collection_lock=False,
            **_fused_controller_kwargs(root, _parent, challenger),
        )
    jobs = [
        {
            "job_id": str(item["job_id"]),
            "config_path": str(path),
            "output_root": str(output),
            "collection_path": str(_collection_path(output, item)),
            "qualification_source": str(
                qualification_roots[str(item["challenger"])]
            ),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            int(config["execution"]["worker_count"]),
            phase=f"bounded-native-retry-{phase}",
            output_root=output,
            run_fingerprint=run_fp,
            timeout_seconds=float(config["execution"]["outer_job_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row["status"] in {"error", "timeout"}]
        if terminal:
            status = _status(phase, schedule, output, run_fp)
            status["terminal_failure"] = terminal[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    status = _status(phase, schedule, output, run_fp)
    _write_json(output / STATUS_FILENAME, status)
    return status


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    return {
        "episode_count": len(selected),
        "persistent_platform_count": sum(bool(row["entered_persistent_platform"]) for row in selected),
        "persistent_platform_rate": mean(bool(row["entered_persistent_platform"]) for row in selected),
        "success_count": sum(bool(row["success"]) for row in selected),
        "success_rate": mean(bool(row["success"]) for row in selected),
        "right_censored_count": sum(row["stop_reason"] in {"repair_limit", "wall_timeout"} for row in selected),
        "mean_normalized_fixed_auc": mean(float(row["normalized_fixed_auc"]) for row in selected),
        "restricted_mean_repair_decisions": _kaplan_meier_restricted_mean(selected, horizon=64),
        "mean_repair_wall_seconds": mean(float(row["repair_wall_seconds"]) for row in selected),
        "mean_intervention_count": mean(float(row["intervention_count"]) for row in selected),
    }


def _paired_bootstrap(rows: list[dict[str, Any]], *, replicates: int) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_fingerprint"])].append(row)
    states = sorted(by_state)

    def difference(sample: list[str]) -> float:
        baseline: list[bool] = []
        treatment: list[bool] = []
        for state in sample:
            baseline.extend(bool(row["entered_persistent_platform"]) for row in by_state[state] if row["arm"] == ARMS[0])
            treatment.extend(bool(row["entered_persistent_platform"]) for row in by_state[state] if row["arm"] == ARMS[1])
        return mean(treatment) - mean(baseline)

    point = difference(states)
    rng = random.Random(0x42524E43)
    samples = [difference([rng.choice(states) for _ in states]) for _ in range(int(replicates))]
    return {"point": point, "lower_95": quantile(samples, 0.025), "upper_95": quantile(samples, 0.975)}


def _legacy_artifact(root: Path, state: str, trial: int, policy: str) -> dict[str, Any]:
    path = (
        root
        / "build/stride-nativeorder-transactionalrepair-v1/initial/trials"
        / state
        / f"trial_{trial:02d}__{policy}.json"
    )
    return _read_json(path)


def _snapshot_matches_legacy(snapshot: Mapping[str, Any], attempt: Mapping[str, Any]) -> bool:
    return (
        list(map(int, snapshot["neighborhood"])) == list(map(int, attempt["agents"]))
        and list(map(int, snapshot["repair_order"])) == list(map(int, attempt["repair_order"]))
        and bool(snapshot["replan_success"]) == bool(attempt["replan_success"])
        and str(snapshot["failure_reason"]) == str(attempt["failure_reason"])
        and int(snapshot["attempted_agent_count"]) == int(attempt["attempted_agent_count"])
        and int(snapshot["inserted_agent_count"]) == int(attempt["inserted_agent_count"])
        and int(snapshot["failed_agent"]) == int(attempt["failed_agent"])
        and int(snapshot["failed_order_index"]) == int(attempt["failed_order_index"])
        and bool(snapshot["rolled_back"]) == bool(attempt["rolled_back"])
        and int(snapshot["conflicts_after"]) == int(attempt["conflicts_after"])
    )


def analyze_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    loaded, all_cases = prepare_cases(config_path)
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    path, root, config, _inputs, _metadata, _pretail_inputs, _parent = loaded
    if phase not in {"initial", "extended"}:
        raise ValueError("bounded retry analysis phase must be initial or extended")
    trials = INITIAL_TRIALS if phase == "initial" else (*INITIAL_TRIALS, *EXTENSION_TRIALS)
    schedule = continuation_schedule(cases, trials)
    output = Path(output).resolve()
    rows: list[dict[str, Any]] = []
    for item in schedule:
        manifest = _manifest_for_item(output, item)
        if manifest is None:
            rows.append({**item, "missing": True})
            continue
        row = {**item, "manifest_status": str(manifest.get("status"))}
        row.update(_episode_summary(manifest))
        row["repair_wall_seconds"] = float(
            dict(manifest.get("summary") or {}).get("repair_wall_seconds", 0.0)
        )
        if manifest.get("status") == "ok":
            decisions = _audit_decision_rows(_collection_path(output, item), manifest)
            first = decisions[0]
            first_record = dict(first["actual_metrics"]["bounded_native_retry"])
            summary = dict(manifest["summary"]["bounded_native_retry"])
            row.update(
                {
                    "entered_persistent_platform": bool(summary["persistent_platform_seen"]),
                    "persistent_platform_decisions": int(summary["persistent_platform_decision_count"]),
                    "intervention_count": int(summary["intervention_count"]),
                    "used_signature_count": int(summary["used_platform_signature_count"]),
                    "first_retry_record": first_record,
                    "explicit_repair_order_requested": any(
                        "repair_order" in dict(decision["actual_action"])
                        for decision in decisions
                    ),
                    "retry_after_time_limit": any(
                        dict(decision["actual_metrics"].get("bounded_native_retry") or {}).get("triggered")
                        and dict(dict(decision["actual_metrics"]["bounded_native_retry"])["first_attempt"])["failure_reason"] == "time_limit"
                        for decision in decisions
                    ),
                    "retry_signatures": [
                        str(record["platform_signature"])
                        for decision in decisions
                        for record in [dict(decision["actual_metrics"].get("bounded_native_retry") or {})]
                        if record.get("triggered")
                    ],
                }
            )
        rows.append(row)
    complete = [row for row in rows if not row.get("missing")]
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in complete:
        paired[(row["state_fingerprint"], int(row["trial_index"]))][row["arm"]] = row
    parity: list[bool] = []
    legacy_first: list[bool] = []
    legacy_retry: list[bool] = []
    for pair in paired.values():
        if set(pair) != set(ARMS):
            continue
        base_record = dict(pair[ARMS[0]]["first_retry_record"])
        treatment_record = dict(pair[ARMS[1]]["first_retry_record"])
        parity.append(base_record["first_attempt"] == treatment_record["first_attempt"])
        state = str(pair[ARMS[0]]["state_fingerprint"])
        trial = int(pair[ARMS[0]]["trial_index"])
        legacy = _legacy_artifact(root, state, trial, "same_set_native_retry")
        attempts = list(legacy["policy"]["attempts"])
        legacy_first.append(
            int(base_record["first_attempt"]["requested_pp_random_seed"])
            == int(legacy["first_pp_seed"])
            and _snapshot_matches_legacy(base_record["first_attempt"], attempts[0])
        )
        if treatment_record["triggered"]:
            legacy_retry.append(
                len(attempts) == 2
                and int(treatment_record["retry_seed"]) == int(legacy["retry_pp_seed"])
                and _snapshot_matches_legacy(treatment_record["retry_attempt"], attempts[1])
            )
        else:
            legacy_retry.append(len(attempts) == 1)
    summaries = {arm: _arm_summary(complete, arm) for arm in ARMS}
    bootstrap = _paired_bootstrap(
        complete,
        replicates=int(config["final_gate"]["paired_state_cluster_bootstrap_replicates"]),
    )
    maps: dict[str, Any] = {}
    for map_id in sorted({row["map_id"] for row in complete}):
        selected = [row for row in complete if row["map_id"] == map_id]
        base = _arm_summary(selected, ARMS[0])
        treatment = _arm_summary(selected, ARMS[1])
        maps[map_id] = {
            ARMS[0]: base,
            ARMS[1]: treatment,
            "platform_rate_difference": treatment["persistent_platform_rate"] - base["persistent_platform_rate"],
        }
    base = summaries[ARMS[0]]
    treatment = summaries[ARMS[1]]
    initial_gate = {
        "platform_direction_improved": bootstrap["point"] < 0.0,
        "success_not_lower": treatment["success_rate"] >= base["success_rate"],
        "no_map_worse_over_five_points": all(row["platform_rate_difference"] <= 0.05 for row in maps.values()),
    }
    final_gate = {
        "platform_bootstrap_upper_below_zero": bootstrap["upper_95"] < 0.0,
        "success_not_lower": treatment["success_rate"] >= base["success_rate"],
        "normalized_auc_lower": treatment["mean_normalized_fixed_auc"] < base["mean_normalized_fixed_auc"],
        "restricted_mean_decisions_lower": treatment["restricted_mean_repair_decisions"] < base["restricted_mean_repair_decisions"],
        "no_map_worse_over_five_points": all(row["platform_rate_difference"] <= 0.05 for row in maps.values()),
    }
    integrity = {
        "case_count": len(cases) == (45 if limit_cases is None else int(limit_cases)),
        "schedule_count": len(schedule) == len(cases) * len(trials) * 2,
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "paired_arms_complete": len(paired) == len(cases) * len(trials) and all(set(value) == set(ARMS) for value in paired.values()),
        "first_attempt_paired": bool(parity) and all(parity),
        "legacy_first_attempt_parity": bool(legacy_first) and all(legacy_first),
        "legacy_retry_parity": bool(legacy_retry) and all(legacy_retry),
        "native_order_only": all(not row.get("explicit_repair_order_requested") for row in complete),
        "maximum_three_interventions": all(int(row.get("intervention_count", 0)) <= 3 for row in complete),
        "one_intervention_per_signature": all(len(row.get("retry_signatures", ())) == len(set(row.get("retry_signatures", ()))) for row in complete),
        "no_retry_after_time_limit": all(not row.get("retry_after_time_limit") for row in complete),
        "zero_errors_and_process_timeouts": all(row.get("manifest_status") == "ok" for row in complete),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "config_sha256": sha256_file(path),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "arm_summaries": summaries,
        "paired_state_cluster_bootstrap": bootstrap,
        "by_map": maps,
        "initial_gate": initial_gate,
        "final_gate": final_gate,
        "extension_allowed": phase == "initial" and all(integrity.values()) and all(initial_gate.values()),
        "mechanism_passed": phase == "extended" and all(integrity.values()) and all(final_gate.values()),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    filename = "initial_report.json" if phase == "initial" else "extended_report.json"
    _write_json(output / filename, report)
    return report


__all__ = [
    "analyze_collection",
    "continuation_schedule",
    "load_registration",
    "prepare_cases",
    "run_collection",
]
