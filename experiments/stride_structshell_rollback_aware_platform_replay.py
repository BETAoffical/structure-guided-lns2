from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    contained_file,
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
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_failure_informed_rescue_continuation import _decision_rows
from experiments.stride_hybridstructpool_routed_confirmation import (
    _bounded_paired_comparison,
    _bounded_summary,
)
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.runtime.hybridstructpool_routed import (
    rollback_aware_routed_hybridstructpool_augmentation,
    routed_hybridstructpool_augmentation,
    validate_rollback_aware_routed_hybridstructpool_augmentation,
    validate_routed_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.structshell_rollback_aware_platform_replay_config.v1"
STATUS_SCHEMA = "lns2.stride.structshell_rollback_aware_platform_replay_status.v1"
REPORT_SCHEMA = "lns2.stride.structshell_rollback_aware_platform_replay_report.v1"
EXPERIMENT_ID = "stride-structshell-rollback-aware-platform-replay-v1"
BASELINE_ARM = "structshell_routed_v1"
CHALLENGER_ARM = "structshell_rollback_aware_v2"
ARMS = (BASELINE_ARM, CHALLENGER_ARM)
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "platform_replay_report.json"
_EXPECTED_RUNTIME = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 180.0,
    "environment_time_limit_seconds": 180.0,
    "episode_process_timeout_seconds": 240.0,
    "outer_job_timeout_seconds": 300.0,
    "workers_for_qualification": 16,
    "workers_for_timed_episodes": 1,
    "native_pp_order_only": True,
    "maximum_pp_calls_per_decision": 1,
    "runtime_retry_or_rescue": False,
}


def _registered(root: Path, row: Mapping[str, Any], label: str) -> Path:
    return registered_input(root, dict(row), label=label)


def _successful_manifest(row: Mapping[str, Any]) -> bool:
    """Accept both ordinary and atomically reconstructed episode manifests."""

    return str(row.get("status")) in {"ok", "resumed"}


def _metric_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an atomic recovery for helpers whose historical API says ok."""

    result = dict(row)
    if _successful_manifest(result):
        result["status"] = "ok"
    return result


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "diagnostic_known_platform_mechanism_replay"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "d6a43f314ed9670c3d3435f5a47a7b8286452b6f"
        or tuple(map(str, config.get("arms") or ())) != ARMS
        or dict(config.get("runtime") or {}) != _EXPECTED_RUNTIME
    ):
        raise ValueError("StructShell rollback-aware replay identity changed")
    baseline = validate_routed_hybridstructpool_augmentation(
        dict(config.get("baseline_augmentation") or {})
    )
    challenger = validate_rollback_aware_routed_hybridstructpool_augmentation(
        dict(config.get("challenger_augmentation") or {})
    )
    if baseline != routed_hybridstructpool_augmentation("structshell_only"):
        raise ValueError("replay baseline is not frozen routed StructShell v1")
    if challenger != rollback_aware_routed_hybridstructpool_augmentation():
        raise ValueError("replay challenger is not rollback-aware StructShell v2")
    cohort = dict(config.get("cohort") or {})
    cases = [dict(row) for row in cohort.get("known_platform_cases") or ()]
    if (
        cohort.get("role") != "known_platform_development_mechanism_replay"
        or cohort.get("promotion_evidence") is not False
        or cohort.get("known_failure_selection") is not True
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (14,)
        or int(cohort.get("paired_key_count", -1)) != 2
        or len(cases) != 2
        or {str(row.get("task_id")) for row in cases}
        != {
            "random-32-32-20__random_01__agents_0400",
            "random-32-32-20__random_02__agents_0400",
        }
        or any(int(row.get("solver_seed", -1)) != 14 for row in cases)
    ):
        raise ValueError("StructShell rollback-aware replay cohort changed")
    inputs = {
        name: _registered(root, row, f"rollback replay {name}")
        for name, row in dict(config.get("inputs") or {}).items()
    }
    dataset_tasks = _dataset_tasks(inputs["dataset_manifest"].parents[1], "balanced_wall_clock")
    if {str(row["task_id"]) for row in cases} - set(dataset_tasks):
        raise ValueError("known platform task is absent from the registered dataset")
    source_rows = _read_jsonl(inputs["source_structshell_manifest"])
    source = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): dict(row)
        for row in source_rows
    }
    source_collection = inputs["source_structshell_manifest"].parent.resolve()
    for case in cases:
        key = (str(case["task_id"]), int(case["solver_seed"]))
        row = source.get(key)
        summary = dict((row or {}).get("summary") or {})
        trace_path = contained_file(
            source_collection,
            str((row or {}).get("trace_file") or ""),
            field=f"registered known-platform trace {key}",
        )
        trace_is_registered = bool(
            row is not None
            and sha256_file(trace_path) == str(case["registered_trace_sha256"])
        )
        if (
            row is None
            or row.get("status") != "ok"
            or summary.get("success") is not False
            or summary.get("stop_reason") != "wall_timeout"
            or str(summary.get("initial_fingerprint"))
            != str(case["registered_initial_fingerprint"])
            or int(summary.get("initial_conflicts", -1))
            != int(case["registered_initial_conflicts"])
            or str(row.get("trace_sha256")) != str(case["registered_trace_sha256"])
            or not trace_is_registered
        ):
            raise ValueError(f"registered known platform evidence changed: {key}")
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key_index, case in enumerate(config["cohort"]["known_platform_cases"]):
        for position in range(len(ARMS)):
            arm = ARMS[(key_index + position) % len(ARMS)]
            rows.append(
                {
                    "group_id": "random-32-32-20-high-load",
                    "family": "random",
                    "task_id": str(case["task_id"]),
                    "solver_seed": int(case["solver_seed"]),
                    "arm": arm,
                    "within_key_position": position,
                }
            )
    return rows


def _runtime_config_path(root: Path, output: Path, config: Mapping[str, Any]) -> Path:
    source = _registered(
        root,
        dict(config["inputs"]["runtime_config"]),
        "rollback replay runtime config",
    )
    payload = _read_json(source)
    payload["solver_seeds"] = [14]
    destination = output / "runtime_config_solver_seed_14.json"
    _write_json(destination, payload)
    return destination


def _controller_kwargs(root: Path, config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "wall-clock",
        "wall_time_budget_seconds": 180.0,
        "environment_time_limit_seconds": 180.0,
        "episode_process_timeout_seconds": 240.0,
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
    }
    if arm == BASELINE_ARM:
        kwargs["hybridstructpool_augmentation"] = dict(
            config["baseline_augmentation"]
        )
    elif arm == CHALLENGER_ARM:
        kwargs["hybridstructpool_augmentation"] = dict(
            config["challenger_augmentation"]
        )
    else:
        raise ValueError(f"unknown rollback replay arm: {arm}")
    return kwargs


def _collection_path(output: Path, arm: str) -> Path:
    return output / "arms" / arm


def _manifest_path(output: Path, arm: str) -> Path:
    return _collection_path(output, arm) / "realized_dynamic_manifest.jsonl"


def _manifest(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, str(item["arm"]))
    rows = _read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("rollback replay manifest is ambiguous")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    path, root, config = load_config(job["config_path"])
    del path
    item = dict(job["item"])
    output = Path(job["output_root"]).resolve()
    dataset = (root / str(config["cohort"]["dataset"])).resolve()
    runtime = Path(job["runtime_config_path"]).resolve()
    collection = _collection_path(output, str(item["arm"]))
    keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in config["cohort"]["known_platform_cases"]
    }
    kwargs = _controller_kwargs(root, config, str(item["arm"]))
    common = {
        "cohort_job_keys": keys,
        "qualification_source": Path(job["qualification_source"]).resolve(),
        "use_global_collection_lock": False,
        **kwargs,
    }
    run_closed_loop_collection(
        dataset,
        runtime,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        job_keys=keys,
        **common,
    )
    run_closed_loop_collection(
        dataset,
        runtime,
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        job_keys={(str(item["task_id"]), int(item["solver_seed"]))},
        **common,
    )
    row = _manifest(output, item)
    if row is None:
        raise RuntimeError("rollback replay episode completed without manifest")
    status = str(row.get("status"))
    return {**item, "status": status if status in {"error", "timeout"} else "ok"}


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    manifests = [_manifest(output, item) for item in items]
    present = [row for row in manifests if row is not None]
    return {
        **dict(base),
        "completed_schedule_entries": len(present),
        "completed_jobs": len(present),
        "total_jobs": len(items),
        "completed_by_arm": {
            arm: sum(
                manifest is not None
                for item, manifest in zip(items, manifests, strict=True)
                if str(item["arm"]) == arm
            )
            for arm in ARMS
        },
        "error_jobs": sum(row.get("status") == "error" for row in present),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in present),
        "complete": bool(complete and len(present) == len(items)),
    }


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    items = schedule(config)
    producer = closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_structshell_rollback_aware_platform_replay.py",
            "scripts/run_stride_structshell_rollback_aware_platform_replay.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/hybridstructpool_routed.py",
            "lns2_selector/runtime/rollback_aware_selection.py",
        ),
        native_required=not dry_run,
    )
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": 2,
            "schedule_entry_count": len(items),
            "schedule_sha256": _fingerprint(items),
            "timed_worker_count": 1,
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=items,
        producer=producer,
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="StructShell rollback-aware known-platform replay",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.resumed and prepared.status.get("terminal_failure") is not None:
        # A process error/timeout is terminal by protocol.  Resume is allowed
        # only for clean interruption, never to consume the remaining schedule
        # after a recorded terminal failure.
        return dict(prepared.status)
    existing_terminal_manifests = [
        row
        for item in items
        if (row := _manifest(output, item)) is not None
        and str(row.get("status")) in {"error", "timeout"}
    ]
    if existing_terminal_manifests:
        status = _status(output, items, prepared.base_status)
        status["terminal_failure"] = existing_terminal_manifests[0]
        _write_json(output / STATUS_FILENAME, status)
        return status
    _write_jsonl(output / "execution_schedule.jsonl", items)
    runtime = _runtime_config_path(root, output, config)
    dataset = (root / str(config["cohort"]["dataset"])).resolve()
    keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in config["cohort"]["known_platform_cases"]
    }
    qualification = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification,
        phase="qualify",
        workers=int(config["runtime"]["workers_for_qualification"]),
        resume=prepared.resumed and qualification.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        **_controller_kwargs(root, config, BASELINE_ARM),
    )
    report_path = qualification / "qualification_report.json"
    qualification_report = _read_json(report_path)
    qualification_rows = _read_jsonl(qualification / "qualification_manifest.jsonl")
    qualification_index = {
        (str(row.get("task_id")), int(row.get("solver_seed", -1))): dict(row)
        for row in qualification_rows
    }
    registered_cases = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in config["cohort"]["known_platform_cases"]
    }
    qualification_identity_matches = bool(
        set(qualification_index) == keys
        and all(
            _successful_manifest(qualification_index[key])
            and str(qualification_index[key].get("state_fingerprint"))
            == str(registered_cases[key]["registered_initial_fingerprint"])
            and int(qualification_index[key].get("initial_conflicts", -1))
            == int(registered_cases[key]["registered_initial_conflicts"])
            for key in keys
        )
    )
    if (
        qualification_report.get("passed") is not True
        or int(qualification_report.get("valid_count", -1)) != 2
        or int(qualification_report.get("incomplete_reset_count", -1)) != 0
        or not qualification_identity_matches
    ):
        status = _status(output, items, prepared.base_status)
        status["terminal_failure"] = {
            "status": "qualification_failed",
            "error": (
                "both registered known-platform resets did not qualify with "
                "their registered initial identities"
            ),
        }
        _write_json(output / STATUS_FILENAME, status)
        return status
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "runtime_config_path": str(runtime),
            "qualification_source": str(qualification),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase="structshell-rollback-aware-platform-replay",
            output_root=output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=float(config["runtime"]["outer_job_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        failures = [row for row in results if row.get("status") in {"error", "timeout"}]
        if failures:
            status = _status(output, items, prepared.base_status)
            status["terminal_failure"] = failures[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    report = analyze(path, output, producer=producer)
    status = _status(output, items, prepared.base_status, complete=True)
    status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
    _write_json(output / STATUS_FILENAME, status)
    return report


def _exact_rollback(row: Mapping[str, Any]) -> bool:
    metrics = dict(row["actual_metrics"])
    return bool(
        metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
        and metrics.get("replan_success") is False
        and metrics.get("pp_rolled_back") is True
        and row["before_platform_signature"] == row["after_platform_signature"]
    )


def _selection_class(row: Mapping[str, Any]) -> str:
    controller = dict(row.get("controller") or {})
    guard = dict(controller.get("exact_rollback_candidate_guard") or {})
    selection = dict(guard.get("selection") or {})
    if selection.get("v2_anchor_fallback_used") is True:
        return "v2_anchor_fallback"
    provenance = set(
        map(
            str,
            dict(controller.get("proposal") or {}).get(
                "hybridstructpool_selected_provenance", ()
            ),
        )
    )
    if "structshell_equal_four_size" in provenance and "v2_base" not in provenance:
        return "pure_structshell_bannable"
    return "v2_or_exact_duplicate_nonbannable"


def _longest_exact_streak(
    rows: Iterable[Mapping[str, Any]], *, selection_class: str | None = None
) -> int:
    longest = streak = 0
    previous: tuple[str, str, str] | None = None
    for row in rows:
        current_class = _selection_class(row)
        key = (
            str(row["before_platform_signature"]),
            str(dict(row["controller"]).get("selected_candidate_id")),
            current_class,
        )
        if _exact_rollback(row) and (
            selection_class is None or current_class == selection_class
        ):
            streak = streak + 1 if key == previous else 1
            previous = key
            longest = max(longest, streak)
        else:
            streak = 0
            previous = None
    return longest


def _first_exact_repair_platform(
    rows: list[Mapping[str, Any]], *, minimum_streak: int = 3
) -> dict[str, Any]:
    """Locate the first repeated exact-repair state and its real escape.

    Candidate changes do not end a platform.  Escape requires a changed repair
    fingerprint (or feasibility), so StructShell/V2/anchor label changes cannot
    manufacture a positive result.
    """

    if minimum_streak < 1:
        raise ValueError("platform minimum streak must be positive")
    streak = 0
    streak_signature: str | None = None
    platform_start: int | None = None
    platform_signature: str | None = None
    for index, row in enumerate(rows):
        signature = str(row["before_platform_signature"])
        if _exact_rollback(row):
            if signature == streak_signature:
                streak += 1
            else:
                streak_signature = signature
                streak = 1
            if streak >= minimum_streak:
                platform_start = index - minimum_streak + 1
                platform_signature = signature
                break
        else:
            streak = 0
            streak_signature = None
    if platform_start is None or platform_signature is None:
        return {
            "formed": False,
            "repair_fingerprint": None,
            "start_decision": None,
            "escape_decision": None,
            "escape_latency_decisions": None,
            "escaped": False,
            "longest_consecutive_exact_same_repair": 0,
        }
    longest = current = 0
    escape_decision: int | None = None
    for index in range(platform_start, len(rows)):
        row = rows[index]
        before = str(row["before_platform_signature"])
        after = str(row["after_platform_signature"])
        metrics = dict(row["actual_metrics"])
        if before != platform_signature:
            escape_decision = index - 1
            break
        if after != platform_signature or int(metrics.get("conflicts_after", 1)) == 0:
            escape_decision = index
            break
        if _exact_rollback(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    escaped = escape_decision is not None
    return {
        "formed": True,
        "repair_fingerprint": platform_signature,
        "start_decision": platform_start,
        "escape_decision": escape_decision,
        "escape_latency_decisions": (
            int(escape_decision - platform_start + 1) if escaped else None
        ),
        "escaped": escaped,
        "longest_consecutive_exact_same_repair": longest,
    }


def _exact_repair_platforms(
    rows: list[Mapping[str, Any]], *, minimum_streak: int = 3
) -> list[dict[str, Any]]:
    """Return every contiguous exact-rollback platform, ignoring candidates."""

    if minimum_streak < 1:
        raise ValueError("platform minimum streak must be positive")
    platforms: list[dict[str, Any]] = []
    start: int | None = None
    signature: str | None = None

    def finish(end: int) -> None:
        nonlocal start, signature
        if start is not None and signature is not None:
            length = end - start
            if length >= minimum_streak:
                platforms.append(
                    {
                        "repair_fingerprint": signature,
                        "start_decision": start,
                        "end_decision": end - 1,
                        "exact_rollback_count": length,
                    }
                )
        start = None
        signature = None

    for index, row in enumerate(rows):
        current_signature = str(row["before_platform_signature"])
        if _exact_rollback(row):
            if start is None:
                start = index
                signature = current_signature
            elif current_signature != signature:
                finish(index)
                start = index
                signature = current_signature
        else:
            finish(index)
    finish(len(rows))
    return platforms


def _episode_diagnostic(
    collection: Path, manifest: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    rows = _decision_rows(collection, manifest)
    first_repair_platform = _first_exact_repair_platform(rows)
    repair_platforms = _exact_repair_platforms(rows)
    escape_decision = first_repair_platform.get("escape_decision")
    post_escape_platforms = [
        platform
        for platform in repair_platforms
        if escape_decision is not None
        and int(platform["start_decision"]) > int(escape_decision)
    ]
    summary = dict(manifest.get("summary") or {})
    totals = dict(summary.get("controller_totals") or {})
    exact_by_class: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if _exact_rollback(row):
            exact_by_class[_selection_class(row)] += 1
    guard_errors: list[str] = []
    ban_count = override_count = cache_hit_count = 0
    previous: Mapping[str, Any] | None = None
    first_override = None
    first_cache_hit = None
    for index, row in enumerate(rows):
        controller = dict(row["controller"])
        draw_index = controller.get("repair_seed_draw_index")
        if draw_index != index:
            guard_errors.append(f"decision {index}: PP seed draw index changed")
        guard = dict(controller.get("exact_rollback_candidate_guard") or {})
        if arm == CHALLENGER_ARM:
            selection = dict(guard.get("selection") or {})
            observation = dict(guard.get("observation") or {})
            if not selection or not observation:
                guard_errors.append(f"decision {index}: missing guard trace")
                continue
            if bool(observation.get("exact_conflict_bound_rollback")) != _exact_rollback(row):
                guard_errors.append(f"decision {index}: exact rollback trace mismatch")
            banned = set(map(str, selection.get("banned_candidate_ids") or ()))
            if str(selection.get("v2_anchor_candidate_id")) in banned:
                guard_errors.append(f"decision {index}: V2 anchor was banned")
            if str(selection.get("selected_candidate_id")) in banned:
                guard_errors.append(f"decision {index}: selected candidate was already banned")
            if observation.get("newly_banned") is True:
                ban_count += 1
                if (
                    selection.get("selected_candidate_is_bannable") is not True
                    or int(observation.get("consecutive_exact_rollbacks", -1)) != 3
                ):
                    guard_errors.append(f"decision {index}: invalid third-rollback ban")
            if selection.get("selection_overridden") is True:
                override_count += 1
                if first_override is None:
                    first_override = index
            proposal = dict(controller.get("proposal") or {})
            cache_hit = proposal.get("repair_state_cache_hit") is True
            cache_hit_count += int(cache_hit)
            previous_guard = dict(
                dict((previous or {}).get("controller") or {}).get(
                    "exact_rollback_candidate_guard"
                )
                or {}
            )
            previous_observation = dict(previous_guard.get("observation") or {})
            expected_cache_hit = bool(
                previous is not None
                and previous_observation.get("cache_reuse_allowed") is True
                and previous["after_platform_signature"]
                == row["before_platform_signature"]
            )
            if cache_hit != expected_cache_hit:
                guard_errors.append(
                    f"decision {index}: repair-state cache use did not match guard state"
                )
            if cache_hit:
                if first_cache_hit is None:
                    first_cache_hit = index
                if (
                    previous is None
                    or not _exact_rollback(previous)
                    or previous_observation.get("cache_reuse_allowed") is not True
                    or previous["after_platform_signature"]
                    != row["before_platform_signature"]
                    or float(proposal.get("candidate_generation_seconds", 0.0)) != 0.0
                    or proposal.get("hybridstructpool_gate_evaluated") is not False
                    or float(proposal.get("hybridstructpool_gate_seconds", 0.0))
                    != 0.0
                    or proposal.get("repair_state_cache_candidate_pool_reused")
                    is not True
                    or proposal.get("repair_state_cache_feature_rows_recomputed")
                    is not True
                    or proposal.get("repair_state_cache_scores_recomputed") is not True
                    or proposal.get("repair_state_cache_v2_anchor_refreshed") is not True
                ):
                    guard_errors.append(f"decision {index}: invalid repair-state cache hit")
            if (
                selection.get("v2_anchor_fallback_used") is True
                and _exact_rollback(row)
                and (
                    observation.get("fallback_cycle_reset") is not True
                    or observation.get("banned_candidate_ids") not in ([], ())
                    or observation.get("cache_reuse_allowed") is not False
                )
            ):
                guard_errors.append(
                    f"decision {index}: V2 anchor rollback did not reopen generation"
                )
            if previous is not None:
                previous_guard = dict(
                    dict(previous["controller"]).get("exact_rollback_candidate_guard")
                    or {}
                )
                previous_observation = dict(previous_guard.get("observation") or {})
                if (
                    previous_observation.get("newly_banned") is True
                    and previous["after_platform_signature"]
                    == row["before_platform_signature"]
                    and str(dict(previous["controller"]).get("selected_candidate_id"))
                    == str(controller.get("selected_candidate_id"))
                ):
                    guard_errors.append(f"decision {index}: banned candidate was repeated")
        elif guard:
            guard_errors.append(f"decision {index}: baseline unexpectedly used guard")
        previous = row
    rescue_disabled = all(
        summary.get(name) is None
        for name in (
            "bounded_native_retry",
            "failure_informed_rescue",
            "signature_scoped_rescue",
        )
    )
    return {
        "task_id": str(manifest["task_id"]),
        "arm": arm,
        "success": bool(summary.get("success")),
        "stop_reason": str(summary.get("stop_reason")),
        "initial_fingerprint": str(summary.get("initial_fingerprint")),
        "initial_conflicts": int(summary.get("initial_conflicts", -1)),
        "final_conflicts": int(summary.get("final_conflicts", -1)),
        "repair_iterations": int(summary.get("repair_iterations", -1)),
        "restricted_ttf_seconds": float(summary["capped_wall_time_to_feasible"]),
        "normalized_wall_auc": float(summary["normalized_wall_clock_conflict_auc"]),
        "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
        "candidate_generation_seconds": float(
            totals.get("candidate_generation_seconds", 0.0)
        ),
        "structshell_generation_seconds": float(
            totals.get("hybridstructpool_structural_generation_seconds", 0.0)
        ),
        "feature_seconds": float(totals.get("feature_seconds", 0.0)),
        "inference_seconds": float(totals.get("inference_seconds", 0.0)),
        "exact_rollback_count": sum(exact_by_class.values()),
        "exact_rollback_count_by_selection_class": dict(exact_by_class),
        "longest_exact_same_candidate_streak": _longest_exact_streak(rows),
        "longest_pure_structshell_exact_streak": _longest_exact_streak(
            rows, selection_class="pure_structshell_bannable"
        ),
        "longest_v2_anchor_fallback_exact_streak": _longest_exact_streak(
            rows, selection_class="v2_anchor_fallback"
        ),
        "first_exact_repair_platform": first_repair_platform,
        "exact_repair_platforms": repair_platforms,
        "post_escape_exact_repair_platforms": post_escape_platforms,
        "post_escape_exact_repair_platform_count": len(post_escape_platforms),
        "guard_new_ban_count": ban_count,
        "guard_override_count": override_count,
        "repair_state_cache_hit_count": cache_hit_count,
        "first_repair_state_cache_hit_decision": first_cache_hit,
        "first_guard_override_decision": first_override,
        "single_pp_per_decision": bool(
            len(rows) == int(summary.get("repair_iterations", -1))
            and int(totals.get("model_decision_count", -1)) == len(rows)
            and rescue_disabled
        ),
        "guard_trace_errors": guard_errors,
    }


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    items = schedule(config)
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    diagnostics: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    for arm in ARMS:
        manifest_path = _manifest_path(output, arm)
        rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
        arm_index: dict[tuple[str, int], dict[str, Any]] = {}
        arm_diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
            if key in arm_index:
                errors.append(f"{arm}: duplicate {key}")
                continue
            arm_index[key] = dict(row)
            if _successful_manifest(row):
                arm_diagnostics[key] = _episode_diagnostic(
                    _collection_path(output, arm), row, arm
                )
        indexed[arm] = arm_index
        diagnostics[arm] = arm_diagnostics
    expected = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in config["cohort"]["known_platform_cases"]
    }
    for arm in ARMS:
        if set(indexed[arm]) != expected:
            errors.append(f"{arm}: incomplete paired coverage")
    paired_identity = all(
        _successful_manifest(indexed[BASELINE_ARM].get(key, {}))
        and _successful_manifest(indexed[CHALLENGER_ARM].get(key, {}))
        and dict(indexed[BASELINE_ARM][key]["summary"]).get("initial_fingerprint")
        == dict(indexed[CHALLENGER_ARM][key]["summary"]).get("initial_fingerprint")
        and dict(indexed[BASELINE_ARM][key]["summary"]).get("initial_conflicts")
        == dict(indexed[CHALLENGER_ARM][key]["summary"]).get("initial_conflicts")
        for key in expected
        if key in indexed[BASELINE_ARM] and key in indexed[CHALLENGER_ARM]
    ) and all(key in indexed[BASELINE_ARM] and key in indexed[CHALLENGER_ARM] for key in expected)
    summaries = {
        arm: _bounded_summary(
            [_metric_manifest(row) for row in indexed[arm].values()]
        )
        for arm in ARMS
    }
    # The imported paired helper uses a three-field key; expose local two-field
    # views while retaining the exact same bounded metric semantics.
    baseline_three = {
        ("random-32-32-20-high-load", task, seed): _metric_manifest(row)
        for (task, seed), row in indexed[BASELINE_ARM].items()
    }
    challenger_three = {
        ("random-32-32-20-high-load", task, seed): _metric_manifest(row)
        for (task, seed), row in indexed[CHALLENGER_ARM].items()
    }
    comparison = _bounded_paired_comparison(
        baseline_three, challenger_three, sorted(baseline_three)
    )
    baseline_diag = diagnostics[BASELINE_ARM]
    challenger_diag = diagnostics[CHALLENGER_ARM]
    registered_cases = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in config["cohort"]["known_platform_cases"]
    }
    registered_initial_identity = all(
        dict(indexed[arm][key]["summary"]).get("initial_fingerprint")
        == registered_cases[key]["registered_initial_fingerprint"]
        and int(dict(indexed[arm][key]["summary"]).get("initial_conflicts", -1))
        == int(registered_cases[key]["registered_initial_conflicts"])
        for arm in ARMS
        for key in expected
        if key in indexed[arm] and _successful_manifest(indexed[arm][key])
    ) and all(
        key in indexed[arm] and _successful_manifest(indexed[arm][key])
        for arm in ARMS
        for key in expected
    )
    prefix_equal = True
    for key in sorted(expected):
        if key not in baseline_diag or key not in challenger_diag:
            prefix_equal = False
            continue
        baseline_decisions = _decision_rows(
            _collection_path(output, BASELINE_ARM), indexed[BASELINE_ARM][key]
        )
        challenger_decisions = _decision_rows(
            _collection_path(output, CHALLENGER_ARM), indexed[CHALLENGER_ARM][key]
        )
        intervention_boundaries = [
            value
            for value in (
                challenger_diag[key]["first_repair_state_cache_hit_decision"],
                challenger_diag[key]["first_guard_override_decision"],
            )
            if value is not None
        ]
        if not intervention_boundaries:
            prefix_equal = False
            continue
        first_intervention = min(map(int, intervention_boundaries))
        baseline_prefix = baseline_decisions[:first_intervention]
        challenger_prefix = challenger_decisions[:first_intervention]
        prefix_equal = (
            prefix_equal
            and len(baseline_prefix) == len(challenger_prefix)
            and all(
                left["before_platform_signature"]
                == right["before_platform_signature"]
                and left["actual_action"] == right["actual_action"]
                and left["after_platform_signature"]
                == right["after_platform_signature"]
                and {
                    name: dict(left["actual_metrics"]).get(name)
                    for name in (
                        "replan_success",
                        "pp_rolled_back",
                        "pp_failure_reason",
                        "conflicts_before",
                        "conflicts_after",
                    )
                }
                == {
                    name: dict(right["actual_metrics"]).get(name)
                    for name in (
                        "replan_success",
                        "pp_rolled_back",
                        "pp_failure_reason",
                        "conflicts_before",
                        "conflicts_after",
                    )
                }
                for left, right in zip(baseline_prefix, challenger_prefix)
            )
        )
    integrity = {
        "complete_four_episode_coverage": not any("coverage" in value for value in errors),
        "zero_execution_errors_or_process_timeouts": all(
            _successful_manifest(row)
            for arm_rows in indexed.values()
            for row in arm_rows.values()
        ),
        "paired_initial_fingerprints_and_conflicts": paired_identity,
        "registered_initial_fingerprints_and_conflicts": registered_initial_identity,
        "registered_ttf_clock": all(
            dict(row.get("summary") or {}).get("ttf_clock_schema") == TTF_CLOCK_SCHEMA
            for arm_rows in indexed.values()
            for row in arm_rows.values()
            if _successful_manifest(row)
        ),
        "zero_invalid_actions_or_semantic_mismatches": all(
            int(dict(row.get("summary") or {}).get("invalid_action_count", -1)) == 0
            and int(dict(row.get("summary") or {}).get("fingerprint_mismatch_count", -1)) == 0
            for arm_rows in indexed.values()
            for row in arm_rows.values()
            if _successful_manifest(row)
        ),
        "single_pp_per_decision": all(
            row["single_pp_per_decision"]
            for arm_rows in diagnostics.values()
            for row in arm_rows.values()
        ),
        "guard_trace_semantics_valid": all(
            not row["guard_trace_errors"]
            for row in challenger_diag.values()
        ),
        "paired_prefix_equal_before_first_cache_or_override": prefix_equal,
    }
    mechanism = {
        "baseline_platform_reproduced_in_both_cases": all(
            baseline_diag[key]["longest_pure_structshell_exact_streak"] >= 3
            for key in expected
            if key in baseline_diag
        ) and len(baseline_diag) == 2,
        "both_cases_trigger_ban_and_override": all(
            challenger_diag[key]["guard_new_ban_count"] >= 1
            and challenger_diag[key]["guard_override_count"] >= 1
            for key in expected
            if key in challenger_diag
        ) and len(challenger_diag) == 2,
        "pure_structshell_streak_bounded_to_three": all(
            challenger_diag[key]["longest_pure_structshell_exact_streak"] <= 3
            for key in expected
            if key in challenger_diag
        ) and len(challenger_diag) == 2,
        "both_cases_really_escape_first_repair_platform": all(
            dict(challenger_diag[key]["first_exact_repair_platform"]).get(
                "escaped"
            )
            is True
            for key in expected
            if key in challenger_diag
        ) and len(challenger_diag) == 2,
        "first_platform_escape_latency_lower_in_each_case": all(
            dict(challenger_diag[key]["first_exact_repair_platform"]).get(
                "escaped"
            )
            is True
            and (
                dict(baseline_diag[key]["first_exact_repair_platform"]).get(
                    "escape_latency_decisions"
                )
                is None
                or int(
                    dict(challenger_diag[key]["first_exact_repair_platform"])[
                        "escape_latency_decisions"
                    ]
                )
                < int(
                    dict(baseline_diag[key]["first_exact_repair_platform"])[
                        "escape_latency_decisions"
                    ]
                )
            )
            for key in expected
            if key in baseline_diag and key in challenger_diag
        ) and len(challenger_diag) == len(baseline_diag) == 2,
        "no_post_escape_platform_transfer": all(
            int(challenger_diag[key]["post_escape_exact_repair_platform_count"])
            == 0
            for key in expected
            if key in challenger_diag
        ) and len(challenger_diag) == 2,
        "exact_rollbacks_lower_in_each_case": all(
            challenger_diag[key]["exact_rollback_count"]
            < baseline_diag[key]["exact_rollback_count"]
            for key in expected
            if key in baseline_diag and key in challenger_diag
        ) and len(challenger_diag) == len(baseline_diag) == 2,
        "success_noninferior": summaries[CHALLENGER_ARM]["success_count"]
        >= summaries[BASELINE_ARM]["success_count"],
        "mean_restricted_ttf_noninferior": bool(comparison.get("valid"))
        and float(comparison["challenger_mean_restricted_ttf"])
        <= float(comparison["baseline_mean_restricted_ttf"]),
        "mean_normalized_wall_auc_noninferior": summaries[CHALLENGER_ARM][
            "mean_normalized_wall_clock_conflict_auc"
        ]
        <= summaries[BASELINE_ARM]["mean_normalized_wall_clock_conflict_auc"],
    }
    integrity_passed = not errors and all(integrity.values())
    mechanism_passed = integrity_passed and all(mechanism.values())
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_structshell_rollback_aware_platform_replay.py",
                "experiments/closed_loop_confirmation.py",
                "lns2_selector/runtime/rollback_aware_selection.py",
            ),
            native_required=False,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "diagnostic_known_platform_mechanism_replay",
        "paired_key_count": len(expected),
        "episode_count": sum(len(rows) for rows in indexed.values()),
        "arm_summaries": summaries,
        "paired_bounded_comparison": comparison,
        "per_episode_diagnostics": {
            arm: [diagnostics[arm][key] for key in sorted(diagnostics[arm])]
            for arm in ARMS
        },
        "integrity_gates": integrity,
        "mechanism_gates": mechanism,
        "integrity_passed": integrity_passed,
        "mechanism_replay_passed": mechanism_passed,
        "promotion_evidence": False,
        "runtime_replacement_allowed": False,
        "next_step": (
            "preregister_fresh_result_blind_bounded_confirmation"
            if mechanism_passed
            else "stop_rollback_aware_branch_keep_v2_default"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "arm_manifest_sha256": {
                arm: sha256_file(_manifest_path(output, arm)) for arm in ARMS
            },
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "ARMS",
    "BASELINE_ARM",
    "CHALLENGER_ARM",
    "analyze",
    "load_config",
    "run",
    "schedule",
]
