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
    COMPACT_BLOCKER_AUGMENTED_MODE,
    CONTROL_MODE,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.signature_scoped_rescue_continuation_registration.v1"
STATUS_SCHEMA = "lns2.stride.signature_scoped_rescue_continuation_status.v1"
REPORT_SCHEMA = "lns2.stride.signature_scoped_rescue_continuation_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.signature_scoped_rescue_continuation_override.v1"
EXPERIMENT_ID = "stride-signature-scoped-rescue-continuation-v1"
FROZEN_ARM = "frozen_controller"
ONE_SHOT_ARM = "one_shot_compact_blocker"
SIGNATURE_SCOPED_ARM = "signature_scoped_compact_blocker"
ARMS = (FROZEN_ARM, ONE_SHOT_ARM, SIGNATURE_SCOPED_ARM)
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
        != "fc3b3995b592793384d5c24b5df3490eab32a047"
        or config.get("protocol_revision")
        != "r1_repairability_basin_audit_frozen"
        or tuple(map(str, config.get("arms") or ())) != ARMS
    ):
        raise ValueError("signature-scoped rescue registration changed")
    inputs = {
        name: registered_input(root, row, label=f"signature rescue {name}")
        for name, row in dict(config["inputs"]).items()
    }
    expected_execution = {
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
        "minimum_consecutive_exact_rollbacks": 3,
        "initial_repeat_count": 2,
        "maximum_added_external_blockers": 8,
        "maximum_signature_scoped_rescues_per_episode": 3,
        "same_signature_rescue_limit": 1,
        "rescue_decision_offset": 1,
        "native_pp_order_only": True,
        "one_pp_call_per_decision": True,
        "rescue_after_time_limit": False,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if dict(config["execution"]) != expected_execution:
        raise ValueError("signature-scoped rescue execution protocol changed")
    basin = _read_json(inputs["repairability_basin_report"])
    decision = dict(basin.get("decision") or {})
    if (
        basin.get("integrity_passed") is not True
        or decision.get("accumulated_blocker_rescue_supported") is not False
        or decision.get("new_signature_rescue_supported") is not True
        or decision.get("next_protocol")
        != "bounded_one_rescue_per_new_platform_signature"
    ):
        raise ValueError("repairability basin decision changed")
    compact = _read_json(inputs["compact_blocker_initial_report"])
    if compact.get("integrity_passed") is not True:
        raise ValueError("compact blocker evidence integrity changed")
    failure_loaded, cases = prepare_failure_cases(
        inputs["failure_rescue_registration"]
    )
    if len(cases) != 45 or len({str(row["map_id"]) for row in cases}) != 3:
        raise ValueError("signature-scoped rescue cohort changed")
    return path, root, config, inputs, failure_loaded, cases


def prepare_cases(
    config_path: str | Path,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    loaded = load_registration(config_path)
    return loaded, [dict(row) for row in loaded[-1]]


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
    if arm not in ARMS:
        raise ValueError(f"unknown signature-scoped rescue arm: {arm}")
    checkpoint = dict(case["logical_checkpoint"])
    candidate = dict(case["selected_candidate"])
    source_root = _source_collection(case)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("signature-scoped rescue source manifest invalid")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(case["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(state)
    if repair_fp != repair_structure_fingerprint(_read_restored_state(case)):
        raise ValueError("signature-scoped rescue source state changed")
    first_seed = repairability_pp_seed(repair_fp, int(trial_index))
    override = {
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
            "signature-scoped-rescue:shared-first-action",
            *map(str, candidate.get("selection_families") or ()),
        ],
    }
    if arm in {FROZEN_ARM, ONE_SHOT_ARM}:
        override["failure_informed_rescue"] = {
            "mode": (
                CONTROL_MODE
                if arm == FROZEN_ARM
                else COMPACT_BLOCKER_AUGMENTED_MODE
            ),
            "maximum_added_blockers": 8,
            "seed_namespace": EXPERIMENT_ID,
            "episode_key": str(case["state_fingerprint"]),
            "trial_index": int(trial_index),
            "initial_repeat_count": 2,
            "enable_semantic_compaction_audit": True,
        }
    else:
        override["signature_scoped_rescue"] = {
            "maximum_added_blockers": 8,
            "minimum_consecutive_rollbacks": 3,
            "maximum_interventions": 3,
            "initial_repeat_count": 2,
            "seed_namespace": EXPERIMENT_ID,
            "episode_key": str(case["state_fingerprint"]),
            "trial_index": int(trial_index),
            "enable_semantic_compaction_audit": True,
        }
    return override


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_signature_scoped_rescue_continuation.py",
            "scripts/run_stride_signature_scoped_rescue_continuation.py",
            "lns2_selector/runtime/signature_scoped_rescue.py",
            "lns2_selector/runtime/failure_informed_rescue.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_cases(job["config_path"])
    _path, root, _config, _inputs, failure_loaded, _frozen = loaded
    _fp, _fr, _fc, _fi, bounded_loaded, _failure_cases = failure_loaded
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
        raise RuntimeError("signature-scoped rescue completed without manifest")
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
    path, root, config, _inputs, failure_loaded, _frozen = loaded
    _fp, _fr, _fc, _fi, bounded_loaded, _failure_cases = failure_loaded
    _bp, _br, _bc, bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("signature-scoped rescue phase changed")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file() or _read_json(report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("signature-scoped initial gate forbids extension")
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
            raise ValueError("signature-scoped rescue schedule changed")
        if not resume:
            raise ValueError("signature-scoped rescue output exists; pass --resume")
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
        raise ValueError("signature-scoped rescue run identity changed")
    _write_json(identity_path, identity)
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("signature-scoped rescue contains terminal failure")
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
            phase=f"signature-scoped-rescue-{phase}",
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
        "platform_rate": mean(bool(row["entered_platform"]) for row in selected),
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
        "mean_rescue_action_count": mean(
            float(row["rescue_action_count"]) for row in selected
        ),
        "new_signature_rescue_count": sum(
            int(row["new_signature_rescue_count"]) for row in selected
        ),
    }


def _paired_cluster_bootstrap(
    rows: list[dict[str, Any]], metric: str, replicates: int
) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_fingerprint"])].append(row)
    states = sorted(by_state)

    def difference(sample: list[str]) -> float:
        values: list[float] = []
        for state in sample:
            by_key: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
            for row in by_state[state]:
                by_key[int(row["trial_index"])][str(row["arm"])] = row
            for pair in by_key.values():
                if ONE_SHOT_ARM in pair and SIGNATURE_SCOPED_ARM in pair:
                    values.append(
                        float(pair[SIGNATURE_SCOPED_ARM][metric])
                        - float(pair[ONE_SHOT_ARM][metric])
                    )
        if not values:
            raise ValueError("signature-scoped rescue has no paired rows")
        return mean(values)

    point = difference(states)
    rng = random.Random(0x5349474E)
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
    path, _root, config, _inputs, _failure_loaded, _frozen = loaded
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    if phase not in {"initial", "extended"}:
        raise ValueError("signature-scoped rescue analysis phase changed")
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
        summary = dict(manifest.get("summary") or {})
        row["repair_wall_seconds"] = float(summary.get("repair_wall_seconds", 0.0))
        if manifest.get("status") == "ok":
            decisions = _decision_rows(_collection_path(output, item), manifest)
            entered, persistent = _entered_platform(decisions)
            key = (
                "signature_scoped_rescue"
                if item["arm"] == SIGNATURE_SCOPED_ARM
                else "failure_informed_rescue"
            )
            first_record = dict(decisions[0]["actual_metrics"][key])
            tracker_summary = dict(summary.get(key) or {})
            rescue_action_count = sum(
                bool(
                    decision["controller"].get(
                        "signature_scoped_rescue_action"
                        if item["arm"] == SIGNATURE_SCOPED_ARM
                        else "failure_informed_rescue_action"
                    )
                )
                for decision in decisions
            )
            new_count = (
                int(tracker_summary.get("new_signature_intervention_count", 0))
                if item["arm"] == SIGNATURE_SCOPED_ARM
                else 0
            )
            row.update(
                {
                    "entered_platform": entered,
                    "persistent_platform_decisions": persistent,
                    "first_record": first_record,
                    "first_attempt": dict(first_record["first_attempt"]),
                    "trigger_eligible": bool(first_record["trigger_eligible"]),
                    "rescue_action_count": rescue_action_count,
                    "new_signature_rescue_count": new_count,
                    "explicit_repair_order_requested": any(
                        "repair_order" in decision["actual_action"]
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
        paired[(str(row["state_fingerprint"]), int(row["trial_index"]))][
            str(row["arm"])
        ] = row
    parity = [
        len({_fingerprint(value[arm]["first_attempt"]) for arm in ARMS}) == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    summaries = {arm: _arm_summary(complete, arm) for arm in ARMS}
    replicates = int(config["final_gate"]["paired_state_cluster_bootstrap_replicates"])
    bootstraps = {
        metric: _paired_cluster_bootstrap(complete, metric, replicates)
        for metric in (
            "success",
            "normalized_fixed_auc",
            "repair_iterations",
            "repair_wall_seconds",
        )
    }
    by_map: dict[str, Any] = {}
    for map_id in sorted({str(row["map_id"]) for row in complete}):
        selected = [row for row in complete if str(row["map_id"]) == map_id]
        arm_rows = {arm: _arm_summary(selected, arm) for arm in ARMS}
        by_map[map_id] = {
            **arm_rows,
            "signature_scoped_success_difference": (
                arm_rows[SIGNATURE_SCOPED_ARM]["success_rate"]
                - arm_rows[ONE_SHOT_ARM]["success_rate"]
            ),
        }
    one_shot = summaries[ONE_SHOT_ARM]
    scoped = summaries[SIGNATURE_SCOPED_ARM]
    initial_gate = {
        "new_signature_rescue_executed": scoped["new_signature_rescue_count"] >= 1,
        "success_not_lower": scoped["success_rate"] >= one_shot["success_rate"],
        "normalized_auc_not_higher": scoped["mean_normalized_fixed_auc"]
        <= one_shot["mean_normalized_fixed_auc"],
        "restricted_decisions_not_higher": scoped["restricted_mean_repair_decisions"]
        <= one_shot["restricted_mean_repair_decisions"],
        "repair_wall_not_higher": scoped["mean_repair_wall_seconds"]
        <= one_shot["mean_repair_wall_seconds"],
        "no_map_success_worse_over_five_points": all(
            value["signature_scoped_success_difference"] >= -0.05
            for value in by_map.values()
        ),
    }
    final_gate = {
        "success_bootstrap_lower_nonnegative": bootstraps["success"]["lower_95"] >= 0.0,
        "auc_bootstrap_upper_nonpositive": bootstraps["normalized_fixed_auc"]["upper_95"] <= 0.0,
        "decisions_bootstrap_upper_nonpositive": bootstraps["repair_iterations"]["upper_95"] <= 0.0,
        "repair_wall_bootstrap_upper_nonpositive": bootstraps["repair_wall_seconds"]["upper_95"] <= 0.0,
        "no_map_success_worse_over_five_points": all(
            value["signature_scoped_success_difference"] >= -0.05
            for value in by_map.values()
        ),
    }
    integrity = {
        "case_count": len(cases) == (45 if limit_cases is None else int(limit_cases)),
        "schedule_count": len(schedule) == len(cases) * len(trials) * len(ARMS),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "paired_arms_complete": len(paired) == len(cases) * len(trials)
        and all(set(value) == set(ARMS) for value in paired.values()),
        "shared_first_attempt_paired": bool(parity) and all(parity),
        "native_order_only": all(
            not row.get("explicit_repair_order_requested") for row in complete
        ),
        "diagnostics_all_decisions": all(
            row.get("all_decisions_diagnostic") for row in complete
        ),
        "frozen_never_rescues": all(
            int(row.get("rescue_action_count", 0)) == 0
            for row in complete
            if row["arm"] == FROZEN_ARM
        ),
        "one_shot_at_most_one_rescue": all(
            int(row.get("rescue_action_count", 0)) <= 1
            for row in complete
            if row["arm"] == ONE_SHOT_ARM
        ),
        "signature_scoped_at_most_three_rescues": all(
            int(row.get("rescue_action_count", 0)) <= 3
            for row in complete
            if row["arm"] == SIGNATURE_SCOPED_ARM
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
        "signature_scoped_minus_one_shot_bootstrap": bootstraps,
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
    "continuation_schedule",
    "load_registration",
    "prepare_cases",
    "run_collection",
]
