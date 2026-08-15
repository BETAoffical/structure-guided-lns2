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
from experiments.stride_bounded_native_retry_continuation import (
    _collection_path,
    _failed_job,
    _read_restored_state,
    prepare_cases as prepare_bounded_cases,
)
from experiments.stride_collection import _paired_action
from experiments.stride_maze_tail_state_collection import _fused_controller_kwargs
from experiments.stride_platformentry_order import _kaplan_meier_restricted_mean
from experiments.stride_pretail_forced_continuation import _episode_summary
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import (
    decision_rows as replay_decision_rows,
    target_state_from_trace,
)
from lns2_selector.runtime.failure_informed_rescue import (
    BLOCKER_AUGMENTED_MODE,
    CONTROL_MODE,
    SAME_SET_MODE,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.failure_informed_rescue_continuation_registration.v1"
STATUS_SCHEMA = "lns2.stride.failure_informed_rescue_continuation_status.v1"
REPORT_SCHEMA = "lns2.stride.failure_informed_rescue_continuation_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.failure_informed_rescue_continuation_override.v1"
EXPERIMENT_ID = "stride-failure-informed-rescue-continuation-v1"
ARMS = (CONTROL_MODE, SAME_SET_MODE, BLOCKER_AUGMENTED_MODE)
INITIAL_TRIALS = (0, 1, 2, 3)
EXTENSION_TRIALS = (4, 5, 6, 7)
STATUS_FILENAME = "collection_status.json"


def load_registration(
    path: str | Path,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[str, Path],
    tuple[Any, ...],
    list[dict[str, Any]],
]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "819e1d168e62eab813451dd9a54a191e1643f325"
        or config.get("scientific_status")
        != "preregistered_failure_informed_next_decision_mechanism"
        or config.get("protocol_revision")
        != "r2_post_trigger_persistence_corrected_after_protocol_smoke_before_formal_collection"
    ):
        raise ValueError("failure-informed rescue registration changed")
    inputs = {
        name: registered_input(root, row, label=f"failure rescue {name}")
        for name, row in dict(config["inputs"]).items()
    }
    if tuple(map(str, config["arms"])) != ARMS:
        raise ValueError("failure-informed rescue arms changed")
    required_execution = {
        "initial_trial_indices": list(INITIAL_TRIALS),
        "extension_trial_indices": list(EXTENSION_TRIALS),
        "worker_count": 16,
        "qualification_worker_count": 16,
        "qualification_process_timeout_seconds": 360.0,
        "per_episode_maximum_repair_decisions": 64,
        "fixed_metric_horizon": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "initial_repeat_count": 2,
        "maximum_added_external_blockers": 8,
        "rescue_decision_offset": 1,
        "maximum_rescues_per_episode": 1,
        "trigger_failure_reason": "conflict_bound_exceeded",
        "rescue_after_time_limit": False,
        "native_pp_order_only": True,
        "collect_diagnostics_all_decisions_all_arms": True,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if dict(config["execution"]) != required_execution:
        raise ValueError("failure-informed rescue execution protocol changed")
    bounded_loaded, cases = prepare_bounded_cases(inputs["bounded_registration"])
    bounded_report = _read_json(inputs["bounded_result_report"])
    if (
        bounded_report.get("integrity_passed") is not False
        or bounded_report.get("extension_allowed") is not False
        or len(cases) != 45
        or len({str(row["map_id"]) for row in cases}) != 3
    ):
        raise ValueError("registered bounded-retry stop evidence changed")
    return path, root, config, inputs, bounded_loaded, cases


def prepare_cases(
    config_path: str | Path,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    loaded = load_registration(config_path)
    return loaded, [dict(row) for row in loaded[-1]]


def _source_collection(case: Mapping[str, Any]) -> Path:
    path = Path(str(case["source_run_config"])).resolve()
    if sha256_file(path) != str(case["source_run_config_sha256"]):
        raise ValueError("failure-informed rescue source run config changed")
    return path.parent


def continuation_schedule(
    cases: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = dict(case["logical_checkpoint"])
        candidate = dict(case["selected_candidate"])
        repair_fp = repair_structure_fingerprint(_read_restored_state(case))
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
                        "case_position": case_position,
                        "arm_position": arm_position,
                    }
                )
    return rows


def _episode_override(
    case: Mapping[str, Any], *, trial_index: int, arm: str
) -> dict[str, Any]:
    checkpoint = dict(case["logical_checkpoint"])
    candidate = dict(case["selected_candidate"])
    source_root = _source_collection(case)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("failure-informed rescue source manifest invalid")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(case["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(state)
    if repair_fp != repair_structure_fingerprint(_read_restored_state(case)):
        raise ValueError("failure-informed rescue source state changed")
    if arm not in ARMS:
        raise ValueError(f"unknown failure-informed rescue arm: {arm}")
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
        "forced_candidate_role": "shared_diagnostic_first_action",
        "forced_selection_families": [
            "failure-informed-rescue:shared-first-action",
            *map(str, candidate.get("selection_families") or ()),
        ],
        "failure_informed_rescue": {
            "mode": arm,
            "maximum_added_blockers": 8,
            "seed_namespace": EXPERIMENT_ID,
            "episode_key": str(case["state_fingerprint"]),
            "trial_index": int(trial_index),
            "initial_repeat_count": 2,
        },
    }


def _manifest_for_item(
    output: Path, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _collection_path(output, item) / "realized_dynamic_manifest.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("failure-informed rescue manifest is ambiguous")
    return matches[0] if matches else None


def _decision_rows(
    collection_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows, events = replay_decision_rows(collection_root, dict(manifest))
    transitions = events[1:-1]
    if len(rows) != len(transitions):
        raise ValueError("failure-informed rescue replay row count changed")
    return [
        {
            "decision_index": int(row["decision_index"]),
            "before_platform_signature": str(row["before_repair_fingerprint"]),
            "after_platform_signature": str(row["after_repair_fingerprint"]),
            "before_conflicts": int(row["before_conflicts"]),
            "after_conflicts": int(
                row["actual_lns2"]["outcome"]["conflicts_after"]
            ),
            "actual_action": dict(row["actual_action"]),
            "actual_metrics": dict(row["actual_metrics"]),
            "controller": dict(event["controller"]),
        }
        for row, event in zip(rows, transitions, strict=True)
    ]


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_failure_informed_rescue_continuation.py",
            "scripts/run_stride_failure_informed_rescue_continuation.py",
            "lns2_selector/runtime/failure_informed_rescue.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_cases(job["config_path"])
    _path, root, _config, _inputs, bounded_loaded, _frozen = loaded
    _bp, _br, _bc, bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    item = dict(job["item"])
    case = {str(row["state_fingerprint"]): row for row in cases}[
        str(item["state_fingerprint"])
    ]
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in cases}
    override = _episode_override(
        case, trial_index=int(item["trial_index"]), arm=str(item["arm"])
    )
    kwargs = _fused_controller_kwargs(root, parent, str(item["challenger"]))
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    run_closed_loop_collection(
        dataset,
        bounded_inputs["runtime_config"],
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
        bounded_inputs["runtime_config"],
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
        raise RuntimeError("failure-informed rescue completed without manifest")
    status = str(manifest.get("status"))
    return {
        **item,
        "status": status if status in {"error", "timeout"} else "ok",
        "manifest_status": status,
        "error": manifest.get("error"),
        "collection_path": str(collection),
        "state_count": int(status == "ok"),
        "outcome_count": int(status == "ok"),
    }


def _status(
    phase: str, schedule: list[dict[str, Any]], output: Path, run_fp: str
) -> dict[str, Any]:
    complete = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None
    ]
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
    path, root, config, _inputs, bounded_loaded, _frozen = loaded
    _bp, _br, _bc, bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("failure-informed rescue phase must be initial or extension")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file() or _read_json(report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("failure-informed rescue initial gate forbids extension")
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = continuation_schedule(cases, trials)
    producer = _producer(root, native_required=not dry_run)
    run_fp = _fingerprint(
        {
            "registration_sha256": sha256_file(path),
            "producer": producer,
            "schedule": schedule,
        }
    )
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
            raise ValueError("failure-informed rescue schedule changed")
        if not resume:
            raise ValueError("failure-informed rescue output exists; pass --resume")
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
        raise ValueError("failure-informed rescue run identity changed")
    _write_json(identity_path, identity)
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("failure-informed rescue contains terminal failure")
    pending = [item for item in schedule if _manifest_for_item(output, item) is None]
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in all_cases}
    qualification_roots = {
        challenger: output / "qualification" / challenger
        for challenger in sorted({str(item["challenger"]) for item in schedule})
    }
    for challenger, qualification_root in qualification_roots.items():
        run_closed_loop_collection(
            (root / str(parent["cohort"]["dataset"])).resolve(),
            bounded_inputs["runtime_config"],
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
            **_fused_controller_kwargs(root, parent, challenger),
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
            phase=f"failure-informed-rescue-{phase}",
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


def _entered_platform(decisions: list[dict[str, Any]]) -> tuple[bool, int]:
    streak = 2
    signature: str | None = (
        decisions[0]["before_platform_signature"] if decisions else None
    )
    persistent = 0
    seen = False
    for row in decisions:
        metrics = dict(row["actual_metrics"])
        exact = bool(
            metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
            and metrics.get("replan_success") is False
            and metrics.get("pp_rolled_back") is True
            and row["before_platform_signature"]
            == row["after_platform_signature"]
        )
        if exact:
            if signature == row["before_platform_signature"]:
                streak += 1
            else:
                signature = row["before_platform_signature"]
                streak = 1
        else:
            signature = row["after_platform_signature"]
            streak = 0
        if streak >= 3:
            seen = True
            persistent += 1
    return seen, persistent


def _unresolved_after_next_decision(
    decisions: list[dict[str, Any]], first_record: Mapping[str, Any]
) -> bool | None:
    if not bool(first_record["trigger_eligible"]):
        return None
    if len(decisions) < 2:
        raise ValueError("eligible failure-informed event lacks decision 1")
    return bool(
        decisions[1]["after_platform_signature"]
        == decisions[1]["before_platform_signature"]
    )


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    eligible = [row for row in selected if row.get("trigger_eligible")]
    return {
        "episode_count": len(selected),
        "platform_count": sum(bool(row["entered_platform"]) for row in selected),
        "platform_rate": mean(bool(row["entered_platform"]) for row in selected),
        "eligible_event_count": len(eligible),
        "unresolved_after_next_decision_count": sum(
            bool(row["unresolved_after_next_decision"]) for row in eligible
        ),
        "unresolved_after_next_decision_rate": (
            mean(bool(row["unresolved_after_next_decision"]) for row in eligible)
            if eligible
            else 0.0
        ),
        "next_decision_escape_rate": (
            mean(not bool(row["unresolved_after_next_decision"]) for row in eligible)
            if eligible
            else 0.0
        ),
        "success_count": sum(bool(row["success"]) for row in selected),
        "success_rate": mean(bool(row["success"]) for row in selected),
        "right_censored_count": sum(
            row["stop_reason"] in {"repair_limit", "wall_timeout"}
            for row in selected
        ),
        "mean_normalized_fixed_auc": mean(
            float(row["normalized_fixed_auc"]) for row in selected
        ),
        "restricted_mean_repair_decisions": _kaplan_meier_restricted_mean(
            selected, horizon=64
        ),
        "mean_repair_wall_seconds": mean(
            float(row["repair_wall_seconds"]) for row in selected
        ),
        "trigger_count": len(eligible),
        "mean_observed_external_blocker_count": mean(
            float(row.get("observed_external_blocker_count", 0))
            for row in selected
        ),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]], baseline_arm: str, treatment_arm: str, replicates: int
) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_fingerprint"])].append(row)
    states = [
        state
        for state in sorted(by_state)
        if any(
            row["arm"] == baseline_arm and row.get("trigger_eligible")
            for row in by_state[state]
        )
        and any(
            row["arm"] == treatment_arm and row.get("trigger_eligible")
            for row in by_state[state]
        )
    ]
    if not states:
        raise ValueError("failure-informed rescue has no paired eligible states")

    def difference(sample: list[str]) -> float:
        baseline: list[bool] = []
        treatment: list[bool] = []
        for state in sample:
            baseline.extend(
                bool(row["unresolved_after_next_decision"])
                for row in by_state[state]
                if row["arm"] == baseline_arm and row.get("trigger_eligible")
            )
            treatment.extend(
                bool(row["unresolved_after_next_decision"])
                for row in by_state[state]
                if row["arm"] == treatment_arm and row.get("trigger_eligible")
            )
        return mean(treatment) - mean(baseline)

    point = difference(states)
    rng = random.Random(0x46495243)
    samples = [
        difference([rng.choice(states) for _ in states])
        for _ in range(int(replicates))
    ]
    return {
        "point": point,
        "lower_95": quantile(samples, 0.025),
        "upper_95": quantile(samples, 0.975),
    }


def analyze_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    loaded, all_cases = prepare_cases(config_path)
    path, _root, config, _inputs, _bounded_loaded, _frozen = loaded
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    if phase not in {"initial", "extended"}:
        raise ValueError("failure-informed rescue analysis phase changed")
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
            decisions = _decision_rows(_collection_path(output, item), manifest)
            entered, persistent = _entered_platform(decisions)
            first_record = dict(
                decisions[0]["actual_metrics"]["failure_informed_rescue"]
            )
            final_rescue = next(
                (
                    dict(decision["actual_metrics"]["failure_informed_rescue"])
                    for decision in decisions
                    if dict(
                        decision["actual_metrics"].get(
                            "failure_informed_rescue"
                        )
                        or {}
                    ).get("rescue_attempt")
                ),
                first_record,
            )
            trigger_eligible = bool(first_record["trigger_eligible"])
            unresolved_after_next_decision = (
                _unresolved_after_next_decision(decisions, first_record)
            )
            row.update(
                {
                    "entered_platform": entered,
                    "persistent_platform_decisions": persistent,
                    "first_record": first_record,
                    "trigger_eligible": trigger_eligible,
                    "triggered": bool(first_record["triggered"]),
                    "unresolved_after_next_decision": (
                        unresolved_after_next_decision
                    ),
                    "observed_external_blocker_count": len(
                        first_record["observed_external_blockers"]
                    ),
                    "selected_blocker_count": len(
                        first_record["selected_blockers"]
                    ),
                    "resolved_by_rescue": bool(
                        final_rescue["resolved_by_rescue"]
                    ),
                    "explicit_repair_order_requested": any(
                        "repair_order" in decision["actual_action"]
                        for decision in decisions
                    ),
                    "rescue_action_count": sum(
                        bool(
                            decision["controller"].get(
                                "failure_informed_rescue_action"
                            )
                        )
                        for decision in decisions
                    ),
                    "all_decisions_diagnostic": all(
                        bool(
                            decision["actual_metrics"].get(
                                "requested_collect_pp_diagnostics"
                            )
                        )
                        for decision in decisions
                    ),
                }
            )
        rows.append(row)
    complete = [row for row in rows if not row.get("missing")]
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in complete:
        paired[(row["state_fingerprint"], int(row["trial_index"]))][
            row["arm"]
        ] = row
    first_parity = [
        len({
            _fingerprint(value[arm]["first_record"]["first_attempt"])
            for arm in ARMS
        }) == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    eligibility_parity = [
        len({bool(value[arm]["trigger_eligible"]) for arm in ARMS}) == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    summaries = {arm: _arm_summary(complete, arm) for arm in ARMS}
    replicates = int(config["final_gate"]["paired_state_cluster_bootstrap_replicates"])
    blocker_vs_control = _paired_bootstrap(
        complete, CONTROL_MODE, BLOCKER_AUGMENTED_MODE, replicates
    )
    blocker_vs_same = _paired_bootstrap(
        complete, SAME_SET_MODE, BLOCKER_AUGMENTED_MODE, replicates
    )
    maps: dict[str, Any] = {}
    for map_id in sorted({row["map_id"] for row in complete}):
        selected = [row for row in complete if row["map_id"] == map_id]
        arm_rows = {arm: _arm_summary(selected, arm) for arm in ARMS}
        maps[map_id] = {
            **arm_rows,
            "blocker_vs_control_unresolved_difference": (
                arm_rows[BLOCKER_AUGMENTED_MODE][
                    "unresolved_after_next_decision_rate"
                ]
                - arm_rows[CONTROL_MODE]["unresolved_after_next_decision_rate"]
            ),
        }
    control = summaries[CONTROL_MODE]
    same = summaries[SAME_SET_MODE]
    blocker = summaries[BLOCKER_AUGMENTED_MODE]
    initial_gate = {
        "blocker_unresolved_direction_improved": blocker_vs_control["point"] < 0.0,
        "blocker_success_not_lower": blocker["success_rate"] >= control["success_rate"],
        "blocker_escape_not_lower_than_same_set": (
            blocker["next_decision_escape_rate"]
            >= same["next_decision_escape_rate"]
        ),
        "no_map_worse_over_five_points": all(
            row["blocker_vs_control_unresolved_difference"] <= 0.05
            for row in maps.values()
        ),
    }
    final_gate = {
        "blocker_unresolved_bootstrap_upper_below_zero": blocker_vs_control[
            "upper_95"
        ]
        < 0.0,
        "blocker_success_not_lower": blocker["success_rate"] >= control["success_rate"],
        "blocker_normalized_auc_lower": blocker["mean_normalized_fixed_auc"]
        < control["mean_normalized_fixed_auc"],
        "blocker_restricted_mean_decisions_lower": blocker[
            "restricted_mean_repair_decisions"
        ]
        < control["restricted_mean_repair_decisions"],
        "blocker_not_worse_than_same_set_unresolved": blocker_vs_same["point"] <= 0.0,
        "blocker_escape_not_lower_than_same_set": (
            blocker["next_decision_escape_rate"]
            >= same["next_decision_escape_rate"]
        ),
        "no_map_worse_over_five_points": all(
            row["blocker_vs_control_unresolved_difference"] <= 0.05
            for row in maps.values()
        ),
    }
    integrity = {
        "case_count": len(cases) == (45 if limit_cases is None else int(limit_cases)),
        "schedule_count": len(schedule) == len(cases) * len(trials) * len(ARMS),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "paired_arms_complete": len(paired) == len(cases) * len(trials)
        and all(set(value) == set(ARMS) for value in paired.values()),
        "shared_first_attempt_paired": bool(first_parity) and all(first_parity),
        "trigger_eligibility_paired": bool(eligibility_parity)
        and all(eligibility_parity),
        "eligible_next_decision_observed": all(
            row.get("unresolved_after_next_decision") is not None
            for row in complete
            if row.get("trigger_eligible")
        ),
        "native_order_only": all(
            not row.get("explicit_repair_order_requested") for row in complete
        ),
        "diagnostics_all_decisions": all(
            row.get("all_decisions_diagnostic") for row in complete
        ),
        "at_most_one_rescue": all(
            int(row.get("rescue_action_count", 0)) <= 1 for row in complete
        ),
        "control_never_rescues": all(
            int(row.get("rescue_action_count", 0)) == 0
            for row in complete
            if row["arm"] == CONTROL_MODE
        ),
        "treatments_rescue_iff_triggered": all(
            int(row.get("rescue_action_count", 0)) == int(bool(row.get("triggered")))
            for row in complete
            if row["arm"] != CONTROL_MODE
        ),
        "blocker_cap_respected": all(
            int(row.get("selected_blocker_count", 0)) <= 8 for row in complete
        ),
        "zero_errors_and_process_timeouts": all(
            row.get("manifest_status") == "ok" for row in complete
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "config_sha256": sha256_file(path),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "arm_summaries": summaries,
        "blocker_vs_control_unresolved_bootstrap": blocker_vs_control,
        "blocker_vs_same_set_unresolved_bootstrap": blocker_vs_same,
        "by_map": maps,
        "initial_gate": initial_gate,
        "final_gate": final_gate,
        "extension_allowed": phase == "initial"
        and all(integrity.values())
        and all(initial_gate.values()),
        "mechanism_passed": phase == "extended"
        and all(integrity.values())
        and all(final_gate.values()),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(
        output / ("initial_report.json" if phase == "initial" else "extended_report.json"),
        report,
    )
    return report


__all__ = [
    "analyze_collection",
    "continuation_schedule",
    "load_registration",
    "prepare_cases",
    "run_collection",
]
