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
from experiments.stride_failure_informed_rescue_continuation import (
    _decision_rows,
    _entered_platform,
    _source_collection,
    prepare_cases as prepare_failure_cases,
)
from experiments.stride_maze_tail_state_collection import _fused_controller_kwargs
from experiments.stride_platformentry_order import _kaplan_meier_restricted_mean
from experiments.stride_pretail_forced_continuation import _episode_summary
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.trace_replay import target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.exactstate_controller_attribution_registration.v1"
STATUS_SCHEMA = "lns2.stride.exactstate_controller_attribution_status.v1"
REPORT_SCHEMA = "lns2.stride.exactstate_controller_attribution_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.exactstate_controller_attribution_override.v1"
EXPERIMENT_ID = "stride-exactstate-controller-attribution-v1"
FROZEN_ARM = "frozen_controller"
ADAPTIVE_ARM = "official_adaptive_n8"
TARGET_ARM = "fixed_target_n8"
ARMS = (FROZEN_ARM, ADAPTIVE_ARM, TARGET_ARM)
ARM_POLICIES = {
    FROZEN_ARM: "realized_dynamic",
    ADAPTIVE_ARM: "official_adaptive",
    TARGET_ARM: "fixed_target",
}
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
        != "70894a1dab3ada0c572265df0bed6588f66d759c"
        or config.get("protocol_revision")
        != "r1_exact_first_repeat_stall_controller_contrast"
        or config.get("runtime_hash_correction_parent_commit")
        != "64e1abace608eac89d763814d2364dc72f8ccf9d"
        or config.get("runtime_hash_correction_reason")
        != "restore the complete already-registered proposal_dynamic portable model SHA-256 after the first smoke stopped before qualification or any episode"
        or tuple(map(str, config.get("arms") or ())) != ARMS
    ):
        raise ValueError("exact-state controller attribution registration changed")
    inputs = {
        name: registered_input(root, row, label=f"exact-state attribution {name}")
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
        "neighborhood_size": 8,
        "replan_algorithm": "PP",
        "native_pp_order_only": True,
        "deterministic_pp_replay": True,
        "forced_first_action": False,
        "runtime_retry": False,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if dict(config["execution"]) != expected_execution:
        raise ValueError("exact-state controller attribution execution changed")
    runtime = _read_json(inputs["runtime_config"])
    if (
        tuple(map(str, runtime.get("policies") or ()))
        != ("official_adaptive", "fixed_target", "realized_dynamic")
        or int(runtime["environment"]["neighborhood_size"]) != 8
        or str(runtime["environment"]["replan_algorithm"]) != "PP"
        or runtime.get("deterministic_pp_replay") is not True
    ):
        raise ValueError("exact-state attribution runtime changed")
    warm = _read_json(inputs["official_lns2_warm_start_status"])
    if (
        warm.get("status") != "complete"
        or int(warm.get("complete_jobs", -1)) != 2
        or int(warm.get("successful_jobs", -1)) != 0
        or int(warm.get("error_jobs", -1)) != 0
    ):
        raise ValueError("registered official LNS2 warm-start evidence changed")
    failure_loaded, cases = prepare_failure_cases(
        inputs["failure_rescue_registration"]
    )
    if (
        len(cases) != 45
        or len({str(row["map_id"]) for row in cases}) != 3
        or any(
            str(row.get("logical_checkpoint", {}).get("checkpoint_kind"))
            != "first_repeat_stall"
            for row in cases
        )
    ):
        raise ValueError("exact-state controller attribution cohort changed")
    return path, root, config, inputs, failure_loaded, cases


def prepare_cases(
    config_path: str | Path,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    loaded = load_registration(config_path)
    return loaded, [dict(row) for row in loaded[-1]]


def continuation_schedule(
    cases: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = dict(case["logical_checkpoint"])
        for trial_index in map(int, trial_indices):
            for arm_position, arm in enumerate(ARMS):
                schedule.append(
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
                        "policy": ARM_POLICIES[arm],
                        "case_position": case_position,
                        "arm_position": arm_position,
                    }
                )
    return schedule


def _episode_override(
    case: Mapping[str, Any], *, trial_index: int
) -> dict[str, Any]:
    checkpoint = dict(case["logical_checkpoint"])
    source_root = _source_collection(case)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("exact-state attribution source manifest invalid")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(case["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(state)
    if repair_fp != repair_structure_fingerprint(_read_restored_state(case)):
        raise ValueError("exact-state attribution source state changed")
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(case["state_fingerprint"]),
        "pp_replay_seed_salt": f"{EXPERIMENT_ID}:trial:{int(trial_index)}",
        "initial_restore": {
            "collection_root": str(source_root),
            "manifest": manifest,
            "decision_index": int(checkpoint["decision_index"]),
            "expected_fingerprint": str(case["state_fingerprint"]),
            "repair_structure_fingerprint": repair_fp,
            "expected_conflicts": int(checkpoint["before_conflicts"]),
            "restore_seed": repairability_restore_seed(
                _fingerprint(
                    {
                        "repair_fingerprint": repair_fp,
                        "trial_index": int(trial_index),
                        "experiment": EXPERIMENT_ID,
                    }
                )
            ),
        },
    }


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_exactstate_controller_attribution.py",
            "scripts/run_stride_exactstate_controller_attribution.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/runtime/online_selection.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _manifest_for_item(
    output: Path, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    policy = str(item["policy"])
    path = _collection_path(output, item) / f"{policy}_manifest.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError(f"exact-state attribution manifest ambiguous: {path}")
    return matches[0] if matches else None


def _arm_controller_kwargs(
    root: Path,
    parent: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    if str(item["arm"]) == FROZEN_ARM:
        return _fused_controller_kwargs(root, parent, str(item["challenger"]))
    return {
        "controller": "official_adaptive",
        "feature_backend": "auto",
        "controller_runtime": "reference",
        "verification_profile": "audit",
    }


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, cases = prepare_cases(job["config_path"])
    _path, root, _config, inputs, failure_loaded, _frozen = loaded
    _fp, _fr, _fc, _fi, bounded_loaded, _failure_cases = failure_loaded
    _bp, _br, _bc, _bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    item = dict(job["item"])
    case = {str(row["state_fingerprint"]): row for row in cases}[
        str(item["state_fingerprint"])
    ]
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in cases}
    override = _episode_override(case, trial_index=int(item["trial_index"]))
    kwargs = _arm_controller_kwargs(root, parent, item)
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
        phase=str(item["policy"]),
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
        raise RuntimeError("exact-state attribution completed without manifest")
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
    path, root, config, inputs, failure_loaded, _frozen = loaded
    _fp, _fr, _fc, _fi, bounded_loaded, _failure_cases = failure_loaded
    _bp, _br, _bc, _bounded_inputs, _bm, _pretail_inputs, parent = bounded_loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("exact-state attribution phase changed")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file() or _read_json(report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("exact-state attribution initial gate forbids extension")
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
            raise ValueError("exact-state attribution schedule changed")
        if not resume:
            raise ValueError("exact-state attribution output exists; pass --resume")
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
        raise ValueError("exact-state attribution run identity changed")
    _write_json(identity_path, identity)
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("exact-state attribution contains terminal failure")
    pending = [item for item in schedule if _manifest_for_item(output, item) is None]
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in all_cases}
    qualification_roots = {
        challenger: output / "qualification" / challenger
        for challenger in sorted({str(item["challenger"]) for item in schedule})
    }
    for challenger, qualification_root in qualification_roots.items():
        run_closed_loop_collection(
            (root / str(parent["cohort"]["dataset"])).resolve(),
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
            phase=f"exact-state-controller-attribution-{phase}",
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
        "mean_first_strict_progress_decision": mean(
            float(row["first_strict_progress_decision"]) for row in selected
        ),
        "mean_unique_neighborhood_count": mean(
            float(row["unique_neighborhood_count"]) for row in selected
        ),
    }


def _paired_cluster_bootstrap(
    rows: list[dict[str, Any]], challenger_arm: str, metric: str, replicates: int
) -> dict[str, float]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_state[str(row["state_fingerprint"])].append(row)
    states = sorted(by_state)

    def difference(sample: list[str]) -> float:
        values: list[float] = []
        for state in sample:
            paired: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
            for row in by_state[state]:
                paired[int(row["trial_index"])][str(row["arm"])] = row
            for arms in paired.values():
                if FROZEN_ARM in arms and challenger_arm in arms:
                    values.append(
                        float(arms[challenger_arm][metric])
                        - float(arms[FROZEN_ARM][metric])
                    )
        if not values:
            raise ValueError("exact-state attribution has no paired rows")
        return mean(values)

    point = difference(states)
    rng = random.Random(0x45584143 + ARMS.index(challenger_arm))
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
        raise ValueError("exact-state attribution analysis phase changed")
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
            initial_conflicts = int(row["initial_conflicts"])
            first_progress = next(
                (
                    int(decision["decision_index"]) + 1
                    for decision in decisions
                    if int(decision["after_conflicts"]) < initial_conflicts
                ),
                64,
            )
            neighborhoods = {
                tuple(sorted(map(int, decision["actual_metrics"].get("neighborhood", []))))
                for decision in decisions
            }
            requested_seeds = [
                int(decision["actual_metrics"].get("requested_pp_random_seed", -1))
                for decision in decisions
            ]
            row.update(
                {
                    "entered_platform": entered,
                    "persistent_platform_decisions": persistent,
                    "first_strict_progress_decision": first_progress,
                    "unique_neighborhood_count": len(neighborhoods),
                    "first_requested_pp_seed": requested_seeds[0] if requested_seeds else -1,
                    "explicit_repair_order_requested": any(
                        "repair_order" in decision["actual_action"]
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
    seed_parity = [
        len({int(value[arm]["first_requested_pp_seed"]) for arm in ARMS}) == 1
        for value in paired.values()
        if set(value) == set(ARMS)
    ]
    summaries = {arm: _arm_summary(complete, arm) for arm in ARMS}
    replicates = int(config["final_gate"]["paired_state_cluster_bootstrap_replicates"])
    bootstraps = {
        arm: {
            metric: _paired_cluster_bootstrap(complete, arm, metric, replicates)
            for metric in (
                "entered_platform",
                "success",
                "normalized_fixed_auc",
                "repair_iterations",
                "repair_wall_seconds",
            )
        }
        for arm in (ADAPTIVE_ARM, TARGET_ARM)
    }
    by_map: dict[str, Any] = {}
    for map_id in sorted({str(row["map_id"]) for row in complete}):
        selected = [row for row in complete if str(row["map_id"]) == map_id]
        arm_rows = {arm: _arm_summary(selected, arm) for arm in ARMS}
        by_map[map_id] = {
            **arm_rows,
            **{
                f"{arm}_platform_difference": (
                    arm_rows[arm]["platform_rate"]
                    - arm_rows[FROZEN_ARM]["platform_rate"]
                )
                for arm in (ADAPTIVE_ARM, TARGET_ARM)
            },
        }
    eligible_arms = []
    for arm in (ADAPTIVE_ARM, TARGET_ARM):
        if (
            summaries[arm]["platform_rate"] < summaries[FROZEN_ARM]["platform_rate"]
            and summaries[arm]["success_rate"] >= summaries[FROZEN_ARM]["success_rate"]
            and all(
                values[f"{arm}_platform_difference"] <= 0.05
                for values in by_map.values()
            )
        ):
            eligible_arms.append(arm)
    final_gate_by_arm = {
        arm: {
            "platform_bootstrap_upper_strictly_negative": bootstraps[arm]["entered_platform"]["upper_95"] < 0.0,
            "success_bootstrap_lower_nonnegative": bootstraps[arm]["success"]["lower_95"] >= 0.0,
            "auc_bootstrap_upper_nonpositive": bootstraps[arm]["normalized_fixed_auc"]["upper_95"] <= 0.0,
            "decisions_bootstrap_upper_nonpositive": bootstraps[arm]["repair_iterations"]["upper_95"] <= 0.0,
            "no_map_platform_worse_over_five_points": all(
                values[f"{arm}_platform_difference"] <= 0.05
                for values in by_map.values()
            ),
        }
        for arm in (ADAPTIVE_ARM, TARGET_ARM)
    }
    integrity = {
        "case_count": len(cases) == (45 if limit_cases is None else int(limit_cases)),
        "schedule_count": len(schedule) == len(cases) * len(trials) * len(ARMS),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "paired_arms_complete": len(paired) == len(cases) * len(trials)
        and all(set(value) == set(ARMS) for value in paired.values()),
        "first_pp_seed_paired": bool(seed_parity) and all(seed_parity),
        "native_order_only": all(
            not row.get("explicit_repair_order_requested") for row in complete
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
        "official_minus_frozen_bootstrap": bootstraps,
        "by_map": by_map,
        "initial_eligible_arms": eligible_arms,
        "initial_gate": {
            "at_least_one_official_arm_eligible": bool(eligible_arms),
        },
        "final_gate_by_arm": final_gate_by_arm,
        "extension_allowed": phase == "initial"
        and all(integrity.values())
        and bool(eligible_arms),
        "diagnostic_supported_arms": [
            arm for arm, gate in final_gate_by_arm.items() if all(gate.values())
        ] if phase == "extended" else [],
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(
        output / ("initial_report.json" if phase == "initial" else "extended_report.json"),
        report,
    )
    return report


__all__ = [
    "ADAPTIVE_ARM",
    "ARMS",
    "FROZEN_ARM",
    "TARGET_ARM",
    "analyze_collection",
    "continuation_schedule",
    "load_registration",
    "prepare_cases",
    "run_collection",
]
