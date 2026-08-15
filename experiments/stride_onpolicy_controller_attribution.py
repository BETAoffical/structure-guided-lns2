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
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_exactstate_controller_attribution import (
    prepare_cases as prepare_exact_cases,
)
from experiments.stride_failure_informed_rescue_continuation import _decision_rows
from experiments.stride_maze_tail_state_collection import _fused_controller_kwargs
from experiments.stride_platformentry_order import _kaplan_meier_restricted_mean
from experiments.stride_tailswitch import _episode_summary


CONFIG_SCHEMA = "lns2.stride.onpolicy_controller_attribution_registration.v1"
STATUS_SCHEMA = "lns2.stride.onpolicy_controller_attribution_status.v1"
REPORT_SCHEMA = "lns2.stride.onpolicy_controller_attribution_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.onpolicy_controller_attribution_override.v1"
EXPERIMENT_ID = "stride-onpolicy-controller-attribution-v1"
SLOT_ARM = "frozen_slotpool"
STRUCT_ARM = "frozen_structpool"
ADAPTIVE_ARM = "official_adaptive_n8"
TARGET_ARM = "fixed_target_n8"
ARMS = (SLOT_ARM, STRUCT_ARM, ADAPTIVE_ARM, TARGET_ARM)
FROZEN_ARMS = (SLOT_ARM, STRUCT_ARM)
OFFICIAL_ARMS = (ADAPTIVE_ARM, TARGET_ARM)
ARM_POLICIES = {
    SLOT_ARM: "realized_dynamic",
    STRUCT_ARM: "realized_dynamic",
    ADAPTIVE_ARM: "official_adaptive",
    TARGET_ARM: "fixed_target",
}
ARM_CHALLENGERS = {
    SLOT_ARM: "v2-plus-slotpool",
    STRUCT_ARM: "v2-plus-structpool",
    ADAPTIVE_ARM: "v2-plus-slotpool",
    TARGET_ARM: "v2-plus-slotpool",
}
INITIAL_TRIALS = (0, 1, 2, 3)
EXTENSION_TRIALS = (4, 5, 6, 7)
STATUS_FILENAME = "collection_status.json"


def _collapse_tasks(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for case in cases:
        grouped[(str(case["task_id"]), int(case["solver_seed"]))].append(case)
    tasks: list[dict[str, Any]] = []
    for (task_id, solver_seed), rows in sorted(grouped.items()):
        maps = {str(row["map_id"]) for row in rows}
        if len(maps) != 1:
            raise ValueError("on-policy task key spans multiple maps")
        task_fingerprint = _fingerprint(
            {"task_id": task_id, "solver_seed": solver_seed}
        )
        tasks.append(
            {
                "task_fingerprint": task_fingerprint,
                "state_fingerprint": task_fingerprint,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "map_id": next(iter(maps)),
                "source_checkpoint_count": len(rows),
                "source_challengers": sorted(
                    {
                        str(
                            row.get("challenger")
                            or dict(row["logical_checkpoint"])["challenger"]
                        )
                        for row in rows
                    }
                ),
            }
        )
    return tasks


def _qualification_roots(inputs: Mapping[str, Path]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for prefix in ("slot", "struct"):
        artifacts = [
            inputs[f"{prefix}_qualification_{suffix}"]
            for suffix in ("manifest", "report", "run_config", "summary")
        ]
        parents = {path.parent for path in artifacts}
        if len(parents) != 1:
            raise ValueError("registered qualification artifacts changed roots")
        report = _read_json(inputs[f"{prefix}_qualification_report"])
        if (
            report.get("decision") != "eligible_for_closed_loop"
            or any(value is not True for value in dict(report["gates"]).values())
            or int(report.get("expected_reset_count", -1)) != 19
        ):
            raise ValueError("registered qualification is no longer eligible")
        roots[prefix] = next(iter(parents))
    if (
        inputs["slot_qualification_manifest"].read_bytes()
        != inputs["struct_qualification_manifest"].read_bytes()
    ):
        raise ValueError("slot and struct qualifications no longer share resets")
    return roots


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
        or config.get("scientific_status")
        != "preregistered_initial_state_controller_attribution_diagnostic"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "669df94a43eb14d7bc6578a69e5c0a9a0cddd67f"
        or config.get("protocol_revision")
        != "r1_unique_task_seed_initial_solution_four_arm_contrast"
        or tuple(map(str, config.get("arms") or ())) != ARMS
    ):
        raise ValueError("on-policy controller attribution registration changed")
    inputs = {
        name: registered_input(root, row, label=f"on-policy attribution {name}")
        for name, row in dict(config["inputs"]).items()
    }
    expected_execution = {
        "initial_trial_indices": list(INITIAL_TRIALS),
        "extension_trial_indices": list(EXTENSION_TRIALS),
        "worker_count": 16,
        "per_episode_maximum_repair_decisions": 64,
        "fixed_metric_horizon": 64,
        "wall_time_fuse_seconds": 180.0,
        "process_timeout_seconds": 240.0,
        "outer_job_timeout_seconds": 300.0,
        "neighborhood_size": 8,
        "replan_algorithm": "PP",
        "native_pp_order_only": True,
        "deterministic_pp_replay": True,
        "trial_specific_pp_seed_salt": True,
        "initial_state_restore": False,
        "forced_first_action": False,
        "runtime_retry": False,
        "episode_atomic_checkpoints": True,
        "stop_on_first_execution_error_or_process_timeout": True,
    }
    if dict(config["execution"]) != expected_execution:
        raise ValueError("on-policy controller attribution execution changed")
    runtime = _read_json(inputs["runtime_config"])
    if (
        tuple(map(str, runtime.get("policies") or ()))
        != ("official_adaptive", "fixed_target", "realized_dynamic")
        or int(runtime["environment"]["neighborhood_size"]) != 8
        or str(runtime["environment"]["replan_algorithm"]) != "PP"
        or runtime.get("deterministic_pp_replay") is not True
    ):
        raise ValueError("on-policy attribution runtime changed")
    exact_loaded, cases = prepare_exact_cases(inputs["exactstate_registration"])
    exact_report = _read_json(inputs["exactstate_initial_report"])
    if (
        exact_report.get("integrity_passed") is not True
        or exact_report.get("extension_allowed") is not False
        or len(cases) != 45
    ):
        raise ValueError("on-policy source cohort/report changed")
    tasks = _collapse_tasks(cases)
    map_counts = collections.Counter(str(row["map_id"]) for row in tasks)
    if (
        len(tasks) != 19
        or dict(sorted(map_counts.items()))
        != dict(sorted(dict(config["cohort"]["map_counts"]).items()))
        or int(config["cohort"]["source_checkpoint_count"]) != 45
        or int(config["cohort"]["unique_task_solver_key_count"]) != 19
    ):
        raise ValueError("on-policy unique task cohort changed")
    _qualification_roots(inputs)
    return path, root, config, inputs, exact_loaded, tasks


def prepare_tasks(
    config_path: str | Path,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    loaded = load_registration(config_path)
    return loaded, [dict(row) for row in loaded[-1]]


def continuation_schedule(
    tasks: list[dict[str, Any]], trial_indices: Iterable[int]
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for task_position, task in enumerate(tasks):
        for trial_index in map(int, trial_indices):
            for arm_position, arm in enumerate(ARMS):
                item = {
                    **task,
                    "trial_index": trial_index,
                    "arm": arm,
                    "policy": ARM_POLICIES[arm],
                    "challenger": ARM_CHALLENGERS[arm],
                    "task_position": task_position,
                    "arm_position": arm_position,
                }
                item["job_id"] = _fingerprint(
                    {
                        "task_fingerprint": task["task_fingerprint"],
                        "trial_index": trial_index,
                        "arm": arm,
                    }
                )
                schedule.append(item)
    return schedule


def _episode_override(
    task: Mapping[str, Any], *, trial_index: int
) -> dict[str, Any]:
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(task["task_fingerprint"]),
        "pp_replay_seed_salt": f"{EXPERIMENT_ID}:trial:{int(trial_index)}",
    }


def _parent_from_exact(exact_loaded: tuple[Any, ...]) -> Mapping[str, Any]:
    _path, _root, _config, _inputs, failure_loaded, _cases = exact_loaded
    _fp, _fr, _fc, _fi, bounded_loaded, _failure_cases = failure_loaded
    _bp, _br, _bc, _bi, _bm, _pretail_inputs, parent = bounded_loaded
    return parent


def _producer(root: Path, *, native_required: bool) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_onpolicy_controller_attribution.py",
            "scripts/run_stride_onpolicy_controller_attribution.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/online_selection.py",
            "lns2_selector/evaluation/trace_validation.py",
        ),
        native_required=native_required,
    )


def _collection_path(output: Path, item: Mapping[str, Any]) -> Path:
    return (
        output
        / "episodes"
        / str(item["task_fingerprint"])[:20]
        / f"trial_{int(item['trial_index']):02d}"
        / str(item["arm"])
    )


def _manifest_for_item(
    output: Path, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _collection_path(output, item) / f"{item['policy']}_manifest.jsonl"
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError(f"on-policy attribution manifest ambiguous: {path}")
    return matches[0] if matches else None


def _arm_controller_kwargs(
    root: Path, parent: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    if str(item["arm"]) in FROZEN_ARMS:
        return _fused_controller_kwargs(root, parent, str(item["challenger"]))
    return {
        "controller": "official_adaptive",
        "feature_backend": "auto",
        "controller_runtime": "reference",
        "verification_profile": "audit",
    }


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    loaded, tasks = prepare_tasks(job["config_path"])
    _path, root, _config, inputs, exact_loaded, _all_tasks = loaded
    parent = _parent_from_exact(exact_loaded)
    item = dict(job["item"])
    task = {str(row["task_fingerprint"]): row for row in tasks}[
        str(item["task_fingerprint"])
    ]
    collection = Path(str(job["collection_path"])).resolve()
    key = (str(item["task_id"]), int(item["solver_seed"]))
    all_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in tasks}
    override = _episode_override(task, trial_index=int(item["trial_index"]))
    kwargs = _arm_controller_kwargs(root, parent, item)
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    common = {
        "cohort_job_keys": all_keys,
        "job_keys": {key},
        "episode_overrides": {key: override},
        "use_global_collection_lock": False,
        **kwargs,
    }
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
        **common,
    )
    manifest = _manifest_for_item(Path(str(job["output_root"])), item)
    if manifest is None:
        raise RuntimeError("on-policy attribution completed without manifest")
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
        "arm_completed_counts": {
            arm: sum(
                _manifest_for_item(output, item) is not None
                for item in schedule
                if str(item["arm"]) == arm
            )
            for arm in ARMS
        },
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
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    loaded, all_tasks = prepare_tasks(config_path)
    path, root, config, inputs, exact_loaded, _tasks = loaded
    if phase not in {"initial", "extension"}:
        raise ValueError("on-policy attribution phase changed")
    output = Path(output).resolve()
    if phase == "extension":
        report_path = output / "initial_report.json"
        if not report_path.is_file() or _read_json(report_path).get(
            "extension_allowed"
        ) is not True:
            raise ValueError("on-policy attribution initial gate forbids extension")
    tasks = all_tasks if limit_tasks is None else all_tasks[: int(limit_tasks)]
    trials = INITIAL_TRIALS if phase == "initial" else EXTENSION_TRIALS
    schedule = continuation_schedule(tasks, trials)
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
            "task_count": len(tasks),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "run_fingerprint": run_fp,
            "worker_count": 16,
        }
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / f"{phase}_schedule.jsonl"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("on-policy attribution schedule changed")
        if not resume:
            raise ValueError("on-policy attribution output exists; pass --resume")
    else:
        _write_jsonl(schedule_path, schedule)
    identity = {
        "schema": f"{STATUS_SCHEMA}.run_config",
        "phase": phase,
        "registration": str(path),
        "registration_sha256": sha256_file(path),
        "producer": producer,
        "schedule_sha256": _fingerprint(schedule),
        "run_fingerprint": run_fp,
    }
    identity_path = output / f"{phase}_run_config.json"
    if identity_path.is_file() and _read_json(identity_path) != identity:
        raise ValueError("on-policy attribution run identity changed")
    _write_json(identity_path, identity)
    failures = [
        row
        for row in (_manifest_for_item(output, item) for item in schedule)
        if row is not None and str(row.get("status")) in {"error", "timeout"}
    ]
    if failures:
        raise RuntimeError("on-policy attribution contains terminal failure")
    pending = [item for item in schedule if _manifest_for_item(output, item) is None]
    qualification_roots = _qualification_roots(inputs)
    jobs = [
        {
            "job_id": str(item["job_id"]),
            "config_path": str(path),
            "output_root": str(output),
            "collection_path": str(_collection_path(output, item)),
            "qualification_source": str(
                qualification_roots[
                    "struct" if str(item["arm"]) == STRUCT_ARM else "slot"
                ]
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
            phase=f"onpolicy-controller-attribution-{phase}",
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


def _exact_rollback(row: Mapping[str, Any]) -> bool:
    metrics = dict(row["actual_metrics"])
    return bool(
        metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
        and metrics.get("replan_success") is False
        and metrics.get("pp_rolled_back") is True
        and row["before_platform_signature"] == row["after_platform_signature"]
    )


def _first_platform(
    decisions: list[dict[str, Any]], *, offset: int = 0
) -> tuple[int | None, str | None]:
    streak = 0
    signature: str | None = None
    for position, row in enumerate(decisions, start=1 + offset):
        if _exact_rollback(row):
            before = str(row["before_platform_signature"])
            if before == signature:
                streak += 1
            else:
                signature = before
                streak = 1
        else:
            signature = str(row["after_platform_signature"])
            streak = 0
        if streak >= 3:
            return position, signature
    return None, None


def _platform_diagnostics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    entry_position, entry_signature = _first_platform(decisions)
    if entry_position is None or entry_signature is None:
        return {
            "entered_platform": False,
            "first_platform_entry_decision": 64,
            "escaped_first_platform": False,
            "first_platform_unescaped_after_three": False,
            "decisions_to_first_platform_escape": 64,
            "post_escape_platform_reentry": False,
        }
    escape_position = next(
        (
            position
            for position, row in enumerate(decisions[entry_position:], start=entry_position + 1)
            if str(row["after_platform_signature"]) != entry_signature
        ),
        None,
    )
    decisions_to_escape = (
        escape_position - entry_position if escape_position is not None else 64
    )
    reentry = False
    if escape_position is not None:
        later_position, _later_signature = _first_platform(
            decisions[escape_position:], offset=escape_position
        )
        reentry = later_position is not None
    return {
        "entered_platform": True,
        "first_platform_entry_decision": int(entry_position),
        "escaped_first_platform": escape_position is not None,
        "first_platform_unescaped_after_three": escape_position is None
        or decisions_to_escape > 3,
        "decisions_to_first_platform_escape": int(decisions_to_escape),
        "post_escape_platform_reentry": reentry,
    }


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["arm"]) == arm]
    entered = [row for row in selected if bool(row["entered_platform"])]
    escaped = [row for row in entered if bool(row["escaped_first_platform"])]
    return {
        "episode_count": len(selected),
        "platform_entry_count": len(entered),
        "platform_entry_rate": mean(bool(row["entered_platform"]) for row in selected),
        "restricted_mean_decisions_to_platform": mean(
            float(row["first_platform_entry_decision"]) for row in selected
        ),
        "platform_escape_within_three_rate": (
            mean(not bool(row["first_platform_unescaped_after_three"]) for row in entered)
            if entered
            else 0.0
        ),
        "platform_eventual_escape_rate": (
            mean(bool(row["escaped_first_platform"]) for row in entered)
            if entered
            else 0.0
        ),
        "post_escape_platform_reentry_rate": (
            mean(bool(row["post_escape_platform_reentry"]) for row in escaped)
            if escaped
            else 0.0
        ),
        "success_given_platform_rate": (
            mean(bool(row["success"]) for row in entered) if entered else 0.0
        ),
        "success_count": sum(bool(row["success"]) for row in selected),
        "success_rate": mean(bool(row["success"]) for row in selected),
        "right_censored_count": sum(
            row["stop_reason"] in {"repair_limit", "wall_timeout"}
            for row in selected
        ),
        "mean_capped_ttf_seconds": mean(
            float(row["capped_ttf_seconds"]) for row in selected
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
        "mean_unique_neighborhood_count": mean(
            float(row["unique_neighborhood_count"]) for row in selected
        ),
    }


def _paired_cluster_bootstrap(
    rows: list[dict[str, Any]],
    challenger_arm: str,
    baseline_arm: str,
    metric: str,
    replicates: int,
) -> dict[str, float]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_task[str(row["task_fingerprint"])].append(row)
    tasks = sorted(by_task)

    def difference(sample: list[str]) -> float:
        values: list[float] = []
        for task in sample:
            paired: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
            for row in by_task[task]:
                paired[int(row["trial_index"])][str(row["arm"])] = row
            for arms in paired.values():
                if baseline_arm in arms and challenger_arm in arms:
                    values.append(
                        float(arms[challenger_arm][metric])
                        - float(arms[baseline_arm][metric])
                    )
        if not values:
            raise ValueError("on-policy attribution has no paired rows")
        return mean(values)

    point = difference(tasks)
    rng = random.Random(
        0x4F4E504F + ARMS.index(challenger_arm) * 17 + ARMS.index(baseline_arm)
    )
    samples = [
        difference([rng.choice(tasks) for _ in tasks])
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
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    loaded, all_tasks = prepare_tasks(config_path)
    path, _root, config, _inputs, _exact_loaded, _tasks = loaded
    tasks = all_tasks if limit_tasks is None else all_tasks[: int(limit_tasks)]
    if phase not in {"initial", "extended"}:
        raise ValueError("on-policy attribution analysis phase changed")
    trials = INITIAL_TRIALS if phase == "initial" else (*INITIAL_TRIALS, *EXTENSION_TRIALS)
    schedule = continuation_schedule(tasks, trials)
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
        row.update(
            {
                "initial_fingerprint": str(summary.get("initial_fingerprint")),
                "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
                "capped_ttf_seconds": float(
                    summary.get(
                        "capped_wall_time_to_feasible",
                        config["execution"]["wall_time_fuse_seconds"],
                    )
                ),
            }
        )
        if manifest.get("status") == "ok":
            decisions = _decision_rows(_collection_path(output, item), manifest)
            requested_seeds = [
                int(decision["actual_metrics"].get("requested_pp_random_seed", -1))
                for decision in decisions
            ]
            neighborhoods = {
                tuple(sorted(map(int, decision["actual_metrics"].get("neighborhood", []))))
                for decision in decisions
            }
            row.update(
                {
                    **_platform_diagnostics(decisions),
                    "unique_neighborhood_count": len(neighborhoods),
                    "first_requested_pp_seed": requested_seeds[0]
                    if requested_seeds
                    else -1,
                    "explicit_repair_order_requested": any(
                        "repair_order" in decision["actual_action"]
                        for decision in decisions
                    ),
                    "forced_action_observed": any(
                        bool(decision["actual_metrics"].get("forced_first_action"))
                        for decision in decisions
                    ),
                }
            )
        rows.append(row)
    complete = [row for row in rows if not row.get("missing")]
    paired: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in complete:
        paired[(str(row["task_fingerprint"]), int(row["trial_index"]))][
            str(row["arm"])
        ] = row
    complete_pairs = [value for value in paired.values() if set(value) == set(ARMS)]
    seed_parity = [
        len({int(value[arm]["first_requested_pp_seed"]) for arm in ARMS}) == 1
        for value in complete_pairs
    ]
    initial_state_parity = [
        len({str(value[arm]["initial_fingerprint"]) for arm in ARMS}) == 1
        for value in complete_pairs
    ]
    summaries = {arm: _arm_summary(complete, arm) for arm in ARMS}
    by_map: dict[str, Any] = {}
    for map_id in sorted({str(row["map_id"]) for row in complete}):
        selected = [row for row in complete if str(row["map_id"]) == map_id]
        by_map[map_id] = {arm: _arm_summary(selected, arm) for arm in ARMS}
    eligible_arms: list[str] = []
    for arm in OFFICIAL_ARMS:
        platform_better = all(
            summaries[arm]["platform_entry_rate"]
            < summaries[baseline]["platform_entry_rate"]
            for baseline in FROZEN_ARMS
        )
        success_preserved = summaries[arm]["success_rate"] >= max(
            summaries[baseline]["success_rate"] for baseline in FROZEN_ARMS
        )
        ttf_preserved = summaries[arm]["mean_capped_ttf_seconds"] <= min(
            summaries[baseline]["mean_capped_ttf_seconds"]
            for baseline in FROZEN_ARMS
        )
        maps_preserved = all(
            values[arm]["platform_entry_rate"]
            <= min(values[baseline]["platform_entry_rate"] for baseline in FROZEN_ARMS)
            + 0.05
            for values in by_map.values()
        )
        if platform_better and success_preserved and ttf_preserved and maps_preserved:
            eligible_arms.append(arm)
    replicates = int(config["final_gate"]["paired_task_cluster_bootstrap_replicates"])
    bootstraps = {
        arm: {
            baseline: {
                metric: _paired_cluster_bootstrap(
                    complete, arm, baseline, metric, replicates
                )
                for metric in (
                    "entered_platform",
                    "success",
                    "capped_ttf_seconds",
                    "repair_iterations",
                    "normalized_fixed_auc",
                )
            }
            for baseline in FROZEN_ARMS
        }
        for arm in OFFICIAL_ARMS
    }
    final_gate_by_arm: dict[str, dict[str, bool]] = {}
    for arm in OFFICIAL_ARMS:
        final_gate_by_arm[arm] = {
            "platform_bootstrap_upper_negative_against_each_frozen": all(
                bootstraps[arm][baseline]["entered_platform"]["upper_95"] < 0.0
                for baseline in FROZEN_ARMS
            ),
            "success_bootstrap_lower_nonnegative_against_each_frozen": all(
                bootstraps[arm][baseline]["success"]["lower_95"] >= 0.0
                for baseline in FROZEN_ARMS
            ),
            "capped_ttf_bootstrap_upper_nonpositive_against_each_frozen": all(
                bootstraps[arm][baseline]["capped_ttf_seconds"]["upper_95"]
                <= 0.0
                for baseline in FROZEN_ARMS
            ),
            "decisions_bootstrap_upper_nonpositive_against_each_frozen": all(
                bootstraps[arm][baseline]["repair_iterations"]["upper_95"] <= 0.0
                for baseline in FROZEN_ARMS
            ),
            "no_map_platform_worse_over_five_points": all(
                values[arm]["platform_entry_rate"]
                <= min(
                    values[baseline]["platform_entry_rate"]
                    for baseline in FROZEN_ARMS
                )
                + 0.05
                for values in by_map.values()
            ),
        }
    expected_tasks = 19 if limit_tasks is None else int(limit_tasks)
    integrity = {
        "unique_task_count": len(tasks) == expected_tasks,
        "schedule_count": len(schedule) == len(tasks) * len(trials) * len(ARMS),
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "paired_arms_complete": len(complete_pairs) == len(tasks) * len(trials),
        "initial_state_fingerprint_paired": bool(initial_state_parity)
        and all(initial_state_parity),
        "first_pp_seed_paired": bool(seed_parity) and all(seed_parity),
        "native_order_only": all(
            not row.get("explicit_repair_order_requested") for row in complete
        ),
        "no_forced_action": all(not row.get("forced_action_observed") for row in complete),
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
        "initial_gate": {"at_least_one_official_arm_eligible": bool(eligible_arms)},
        "final_gate_by_arm": final_gate_by_arm,
        "extension_allowed": phase == "initial"
        and all(integrity.values())
        and bool(eligible_arms),
        "diagnostic_supported_arms": [
            arm for arm, gate in final_gate_by_arm.items() if all(gate.values())
        ]
        if phase == "extended"
        else [],
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
    "SLOT_ARM",
    "STRUCT_ARM",
    "TARGET_ARM",
    "analyze_collection",
    "continuation_schedule",
    "load_registration",
    "prepare_tasks",
    "run_collection",
]
