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
)
from experiments.stride_collection import _paired_action
from experiments.stride_failure_informed_rescue_continuation import (
    _decision_rows,
    _entered_platform,
    _manifest_for_item,
    _source_collection,
    _unresolved_after_next_decision,
    prepare_cases as prepare_failure_cases,
)
from experiments.stride_maze_tail_state_collection import _fused_controller_kwargs
from experiments.stride_platformentry_order import _kaplan_meier_restricted_mean
from experiments.stride_pretail_forced_continuation import _episode_summary
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import target_state_from_trace
from lns2_selector.runtime.failure_informed_rescue import (
    BLOCKER_AUGMENTED_MODE,
    COMPACT_BLOCKER_AUGMENTED_MODE,
    COMPACT_SAME_SET_MODE,
    CONTROL_MODE,
    SAME_SET_MODE,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.semantic_compaction import COMPACTION_RULE_ID


CONFIG_SCHEMA = "lns2.stride.compact_blocker_rescue_continuation_registration.v1"
STATUS_SCHEMA = "lns2.stride.compact_blocker_rescue_continuation_status.v1"
REPORT_SCHEMA = "lns2.stride.compact_blocker_rescue_continuation_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.compact_blocker_rescue_continuation_override.v1"
EXPERIMENT_ID = "stride-compact-blocker-rescue-continuation-v1"
ARMS = (
    CONTROL_MODE,
    SAME_SET_MODE,
    COMPACT_SAME_SET_MODE,
    BLOCKER_AUGMENTED_MODE,
    COMPACT_BLOCKER_AUGMENTED_MODE,
)
RESCUE_ARMS = tuple(arm for arm in ARMS if arm != CONTROL_MODE)
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
        != "bb0515921a0c389151f33fbb1b6646eb57537fd2"
        or config.get("protocol_revision")
        != "v1_before_any_compact_blocker_continuation_episode"
        or config.get("scientific_status")
        != "preregistered_outcome_enriched_factorial_interaction_screen"
    ):
        raise ValueError("compact-blocker rescue registration changed")
    inputs = {
        name: registered_input(root, row, label=f"compact-blocker rescue {name}")
        for name, row in dict(config["inputs"]).items()
    }
    if set(inputs) != {
        "failure_rescue_registration",
        "failure_rescue_extended_report",
        "compact_registration",
        "compact_extended_report",
    }:
        raise ValueError("compact-blocker rescue inputs changed")
    parent_report = _read_json(inputs["failure_rescue_extended_report"])
    compact_report = _read_json(inputs["compact_extended_report"])
    if (
        parent_report.get("phase") != "extended"
        or parent_report.get("integrity_passed") is not True
        or parent_report.get("mechanism_passed") is not True
        or compact_report.get("phase") != "extended"
        or compact_report.get("integrity_passed") is not True
        or int(compact_report.get("eligible_state_count", -1)) != 14
    ):
        raise ValueError("compact-blocker parent evidence changed")
    parent_loaded, cases = prepare_failure_cases(inputs["failure_rescue_registration"])
    if (
        len(cases) != 45
        or len({str(row["map_id"]) for row in cases}) != 3
        or tuple(map(str, config["arms"])) != ARMS
    ):
        raise ValueError("compact-blocker rescue cohort or arms changed")
    factorial = dict(config["membership_factorial"])
    compact_rule = dict(factorial["compact_rule"])
    if (
        tuple(factorial["base_membership"]) != ("full", "semantic_compact")
        or tuple(factorial["external_blockers"])
        != ("absent", "up_to_eight_observed")
        or factorial["fresh_seed_shared_by_all_rescue_arms"] is not True
        or compact_rule.get("rule_id") != COMPACTION_RULE_ID
        or compact_rule.get("fixed_target_size") is not None
    ):
        raise ValueError("compact-blocker factorial changed")
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
        "explicit_timed_native_api_required": True,
        "collect_diagnostics_all_decisions_all_arms": True,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if dict(config["execution"]) != required_execution:
        raise ValueError("compact-blocker execution protocol changed")
    return path, root, config, inputs, parent_loaded, [dict(row) for row in cases]


def prepare_factorial_cases(
    config_path: str | Path,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    loaded = load_registration(config_path)
    return loaded, [dict(row) for row in loaded[-1]]


def compact_blocker_schedule(
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
        raise ValueError("compact-blocker source manifest invalid")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(case["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(state)
    if arm not in ARMS:
        raise ValueError(f"unknown compact-blocker arm: {arm}")
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
        "forced_candidate_role": "shared_factorial_first_action",
        "forced_selection_families": [
            "compact-blocker-rescue:shared-first-action",
            *map(str, candidate.get("selection_families") or ()),
        ],
        "failure_informed_rescue": {
            "mode": arm,
            "maximum_added_blockers": 8,
            "seed_namespace": EXPERIMENT_ID,
            "episode_key": str(case["state_fingerprint"]),
            "trial_index": int(trial_index),
            "initial_repeat_count": 2,
            "enable_semantic_compaction_audit": True,
        },
    }


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_compact_blocker_rescue_continuation.py",
            "scripts/run_stride_compact_blocker_rescue_continuation.py",
            "lns2_selector/runtime/failure_informed_rescue.py",
            "lns2_selector/runtime/semantic_compaction.py",
            "lns2_selector/runtime/repairdependencypool.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/state_analysis.py",
            "experiments/trace_replay.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_factorial_cases(job["config_path"])
    _path, root, _config, _inputs, parent_loaded, _frozen = loaded
    _pp, _pr, _pc, _pi, bounded_loaded, _pf = parent_loaded
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
        raise RuntimeError("compact-blocker rescue completed without manifest")
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


def _compact_blocker_status(
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
        "timeout_jobs": sum(
            str(row.get("status")) == "timeout" for row in complete
        ),
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
    loaded, all_cases = prepare_factorial_cases(config_path)
    path, root, config, _inputs, parent_loaded, _frozen = loaded
    _pp, _pr, _pc, _pi, bounded_loaded, _pf = parent_loaded
    _bp, _br, _bc, bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("compact-blocker phase must be initial or extension")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file() or _read_json(report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("compact-blocker initial gate forbids extension")
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = compact_blocker_schedule(cases, trials)
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
            raise ValueError("compact-blocker schedule changed")
        if not resume:
            raise ValueError("compact-blocker output exists; pass --resume")
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
        raise ValueError("compact-blocker run identity changed")
    _write_json(identity_path, identity)
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("compact-blocker output contains terminal failure")
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
            phase=f"compact-blocker-rescue-{phase}",
            output_root=output,
            run_fingerprint=run_fp,
            timeout_seconds=float(config["execution"]["outer_job_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row["status"] in {"error", "timeout"}]
        if terminal:
            status = _compact_blocker_status(phase, schedule, output, run_fp)
            status["terminal_failure"] = terminal[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    status = _compact_blocker_status(phase, schedule, output, run_fp)
    _write_json(output / STATUS_FILENAME, status)
    return status


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    eligible = [row for row in selected if row.get("trigger_eligible")]
    rescues = [row for row in eligible if row.get("rescue_wall_seconds") is not None]
    return {
        "episode_count": len(selected),
        "eligible_event_count": len(eligible),
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
        "success_rate": mean(bool(row["success"]) for row in selected),
        "mean_normalized_fixed_auc": mean(
            float(row["normalized_fixed_auc"]) for row in selected
        ),
        "restricted_mean_repair_decisions": _kaplan_meier_restricted_mean(
            selected, horizon=64
        ),
        "mean_total_repair_wall_seconds": mean(
            float(row["repair_wall_seconds"]) for row in selected
        ),
        "mean_rescue_wall_seconds": (
            mean(float(row["rescue_wall_seconds"]) for row in rescues)
            if rescues
            else 0.0
        ),
        "mean_rescue_size": (
            mean(float(row["rescue_actual_size"]) for row in rescues)
            if rescues
            else 0.0
        ),
        "mean_removed_agent_count": mean(
            float(row.get("removed_agent_count", 0)) for row in selected
        ),
        "right_censored_count": sum(
            row["stop_reason"] in {"repair_limit", "wall_timeout"}
            for row in selected
        ),
    }


def _paired_bootstrap_metric(
    rows: list[dict[str, Any]],
    baseline_arm: str,
    treatment_arm: str,
    *,
    field: str,
    replicates: int,
) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_fingerprint"])].append(row)
    states = [
        state
        for state in sorted(by_state)
        if any(row["arm"] == baseline_arm for row in by_state[state])
        and any(row["arm"] == treatment_arm for row in by_state[state])
    ]
    if not states:
        raise ValueError(f"compact-blocker bootstrap has no paired states for {field}")

    def difference(sample: list[str]) -> float:
        baseline = [
            float(row[field])
            for state in sample
            for row in by_state[state]
            if row["arm"] == baseline_arm
        ]
        treatment = [
            float(row[field])
            for state in sample
            for row in by_state[state]
            if row["arm"] == treatment_arm
        ]
        return mean(treatment) - mean(baseline)

    point = difference(states)
    rng = random.Random(0x43425243 + sum(map(ord, field)))
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
    loaded, all_cases = prepare_factorial_cases(config_path)
    path, _root, config, _inputs, _parent_loaded, _frozen = loaded
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    if phase not in {"initial", "extended"}:
        raise ValueError("compact-blocker analysis phase changed")
    trials = (
        INITIAL_TRIALS
        if phase == "initial"
        else (*INITIAL_TRIALS, *EXTENSION_TRIALS)
    )
    schedule = compact_blocker_schedule(cases, trials)
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
            rescue_decision = next(
                (
                    decision
                    for decision in decisions
                    if decision["controller"].get("failure_informed_rescue_action")
                ),
                None,
            )
            final_rescue = (
                dict(rescue_decision["actual_metrics"]["failure_informed_rescue"])
                if rescue_decision is not None
                else first_record
            )
            compact_plan = dict(first_record["compact_plan"])
            trigger_eligible = bool(first_record["trigger_eligible"])
            row.update(
                {
                    "entered_platform": entered,
                    "persistent_platform_decisions": persistent,
                    "first_record": first_record,
                    "trigger_eligible": trigger_eligible,
                    "compact_eligible": bool(compact_plan["eligible"]),
                    "compact_plan": compact_plan,
                    "triggered": bool(first_record["triggered"]),
                    "unresolved_after_next_decision": _unresolved_after_next_decision(
                        decisions, first_record
                    ),
                    "selected_blocker_count": len(
                        first_record["selected_blockers"]
                    ),
                    "removed_agent_count": len(first_record["removed_agents"]),
                    "planned_agents": list(map(int, first_record["planned_agents"])),
                    "rescue_wall_seconds": (
                        float(rescue_decision["controller"]["repair_wall_seconds"])
                        if rescue_decision is not None
                        else None
                    ),
                    "rescue_actual_size": (
                        len(final_rescue["rescue_attempt"]["neighborhood"])
                        if final_rescue.get("rescue_attempt")
                        else None
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
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for row in complete:
        paired[(row["state_fingerprint"], int(row["trial_index"]))][
            row["arm"]
        ] = row
    first_parity = [
        len(
            {
                _fingerprint(value[arm]["first_record"]["first_attempt"])
                for arm in ARMS
            }
        )
        == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    rescue_seed_parity = [
        len(
            {
                int(value[arm]["first_record"]["rescue_seed"])
                for arm in RESCUE_ARMS
            }
        )
        == 1
        for value in paired.values()
        if set(value) == set(ARMS)
        and value[CONTROL_MODE].get("trigger_eligible")
    ]
    compact_parity = [
        len(
            {
                _fingerprint(value[arm]["compact_plan"])
                for arm in ARMS
            }
        )
        == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    compact_states = {
        str(row["state_fingerprint"])
        for row in complete
        if row.get("compact_eligible")
    }
    compact_rows = [
        row for row in complete if str(row["state_fingerprint"]) in compact_states
    ]
    primary_rows = [
        row
        for row in compact_rows
        if row.get("trigger_eligible")
        and row["arm"] in {
            BLOCKER_AUGMENTED_MODE,
            COMPACT_BLOCKER_AUGMENTED_MODE,
        }
    ]
    wall_rows = [
        row for row in primary_rows if row.get("rescue_wall_seconds") is not None
    ]
    summaries = {arm: _summary(complete, arm) for arm in ARMS}
    compact_summaries = {arm: _summary(compact_rows, arm) for arm in ARMS}
    replicates = int(config["final_gate"]["paired_state_cluster_bootstrap_replicates"])
    combo_unresolved = _paired_bootstrap_metric(
        primary_rows,
        BLOCKER_AUGMENTED_MODE,
        COMPACT_BLOCKER_AUGMENTED_MODE,
        field="unresolved_after_next_decision",
        replicates=replicates,
    )
    combo_wall = _paired_bootstrap_metric(
        wall_rows,
        BLOCKER_AUGMENTED_MODE,
        COMPACT_BLOCKER_AUGMENTED_MODE,
        field="rescue_wall_seconds",
        replicates=replicates,
    )
    blocker_rows = [row for row in complete if row.get("trigger_eligible")]
    blocker_vs_control = _paired_bootstrap_metric(
        [
            row
            for row in blocker_rows
            if row["arm"] in {CONTROL_MODE, BLOCKER_AUGMENTED_MODE}
        ],
        CONTROL_MODE,
        BLOCKER_AUGMENTED_MODE,
        field="unresolved_after_next_decision",
        replicates=replicates,
    )
    by_map: dict[str, Any] = {}
    for map_id in sorted({str(row["map_id"]) for row in compact_rows}):
        selected = [row for row in compact_rows if row["map_id"] == map_id]
        arm_rows = {arm: _summary(selected, arm) for arm in ARMS}
        eligible_count = int(
            arm_rows[BLOCKER_AUGMENTED_MODE]["eligible_event_count"]
        )
        difference = (
            arm_rows[COMPACT_BLOCKER_AUGMENTED_MODE][
                "unresolved_after_next_decision_rate"
            ]
            - arm_rows[BLOCKER_AUGMENTED_MODE][
                "unresolved_after_next_decision_rate"
            ]
            if eligible_count
            else None
        )
        by_map[map_id] = {
            **arm_rows,
            "compact_eligible_event_count": eligible_count,
            "compact_blocker_vs_blocker_unresolved_difference": difference,
        }
    blocker = compact_summaries[BLOCKER_AUGMENTED_MODE]
    combo = compact_summaries[COMPACT_BLOCKER_AUGMENTED_MODE]
    initial_gate = {
        "minimum_compact_eligible_state_count": len(compact_states) >= 10,
        "compact_blocker_unresolved_direction_improved": combo_unresolved["point"]
        < 0.0,
        "compact_blocker_rescue_wall_lower": combo_wall["point"] < 0.0,
        "compact_blocker_success_not_lower": combo["success_rate"]
        >= blocker["success_rate"],
        "compact_blocker_normalized_auc_not_higher": combo[
            "mean_normalized_fixed_auc"
        ]
        <= blocker["mean_normalized_fixed_auc"],
        "compact_blocker_restricted_decisions_not_higher": combo[
            "restricted_mean_repair_decisions"
        ]
        <= blocker["restricted_mean_repair_decisions"],
        "blocker_vs_control_direction_reproduced": blocker_vs_control["point"] < 0.0,
        "no_map_worse_over_five_points": all(
            row["compact_blocker_vs_blocker_unresolved_difference"] is None
            or row["compact_blocker_vs_blocker_unresolved_difference"] <= 0.05
            for row in by_map.values()
        ),
    }
    final_gate = {
        "compact_blocker_unresolved_upper_below_zero": combo_unresolved["upper_95"]
        < 0.0,
        "compact_blocker_wall_upper_below_zero": combo_wall["upper_95"] < 0.0,
        "compact_blocker_success_not_lower": combo["success_rate"]
        >= blocker["success_rate"],
        "compact_blocker_normalized_auc_lower": combo[
            "mean_normalized_fixed_auc"
        ]
        < blocker["mean_normalized_fixed_auc"],
        "compact_blocker_restricted_decisions_lower": combo[
            "restricted_mean_repair_decisions"
        ]
        < blocker["restricted_mean_repair_decisions"],
        "blocker_vs_control_upper_below_zero": blocker_vs_control["upper_95"] < 0.0,
        "no_map_worse_over_five_points": initial_gate[
            "no_map_worse_over_five_points"
        ],
    }
    integrity = {
        "case_count": len(cases) == (45 if limit_cases is None else int(limit_cases)),
        "schedule_count": len(schedule) == len(cases) * len(trials) * len(ARMS),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(
            row.get("manifest_status") == "ok" for row in complete
        ),
        "paired_arms_complete": len(paired) == len(cases) * len(trials)
        and all(set(value) == set(ARMS) for value in paired.values()),
        "shared_first_attempt_paired": bool(first_parity) and all(first_parity),
        "shared_fresh_rescue_seed": bool(rescue_seed_parity)
        and all(rescue_seed_parity),
        "compact_plan_paired": bool(compact_parity) and all(compact_parity),
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
        "compact_is_subset_before_blockers": all(
            set(row["compact_plan"]["compact_agents"]).issubset(
                row["compact_plan"]["base_agents"]
            )
            and row["compact_plan"].get("fixed_target_size") is None
            for row in complete
        ),
        "planned_agents_unique": all(
            len(row.get("planned_agents", ())) == len(set(row.get("planned_agents", ())))
            for row in complete
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
        "compact_eligible_state_count": len(compact_states),
        "compact_eligible_state_fingerprints": sorted(compact_states),
        "arm_summaries": summaries,
        "compact_eligible_arm_summaries": compact_summaries,
        "compact_blocker_vs_blocker_unresolved_bootstrap": combo_unresolved,
        "compact_blocker_vs_blocker_rescue_wall_bootstrap": combo_wall,
        "blocker_vs_control_unresolved_bootstrap": blocker_vs_control,
        "by_map": by_map,
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
    "ARMS",
    "analyze_collection",
    "compact_blocker_schedule",
    "load_registration",
    "prepare_factorial_cases",
    "run_collection",
]
