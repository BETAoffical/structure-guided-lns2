from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
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
from experiments.stride_hybridstructpool_routed_confirmation import (
    _bounded_paired_comparison,
    _controller_dir,
    _manifest_path,
)
from experiments.stride_failure_informed_rescue_continuation import _decision_rows
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from experiments.stride_structshell_rollback_aware_ttf import (
    _extended_summary,
    _metric_manifest,
    _successful_manifest,
    _timing_summary,
    load_config as load_source_config,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    overall_rollback_routed_hybridstructpool_augmentation,
    routed_hybridstructpool_augmentation,
    validate_overall_rollback_routed_hybridstructpool_augmentation,
    validate_routed_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.structshell_overall_rollback_screen_config.v1"
STATUS_SCHEMA = "lns2.stride.structshell_overall_rollback_screen_status.v1"
REPORT_SCHEMA = "lns2.stride.structshell_overall_rollback_screen_report.v1"
EXPERIMENT_ID = "stride-structshell-overall-rollback-screen-v1"
PRE_REGISTRATION_PARENT = "1d16a15489c005cc6b816db5211cac8746628135"
CONTROLLERS = (
    "official_adaptive",
    "v2_only",
    "structshell_routed_v1",
    "structshell_rollback_overall_v3",
)
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "screen_report.json"

_EXPECTED_COMPARISON = {
    "primary_baseline": "v2_only",
    "mechanism_baseline": "structshell_routed_v1",
    "external_baseline": "official_adaptive",
    "challenger": "structshell_rollback_overall_v3",
    "execution_order": "rotating_strict_four_controller_serial",
    "paired_solver_seed_required": True,
    "workers_for_timed_episodes": 1,
    "workers_for_qualification": 16,
}
_EXPECTED_RUNTIME = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 180.0,
    "environment_time_limit_seconds": 180.0,
    "episode_process_timeout_seconds": 240.0,
    "outer_job_timeout_seconds": 300.0,
    "native_pp_order_only": True,
    "maximum_pp_calls_per_decision": 1,
    "runtime_retry_or_rescue": False,
}
_EXPECTED_COHORT = {
    "role": "seen_key_mechanism_and_cost_screen_no_promotion_claim",
    "source_config_input": "rollback_aware_ttf_source_config",
    "source_stage": "screen",
    "solver_seed": 16,
    "task_selection": "first_registered_task_per_group",
    "map_count": 10,
    "paired_key_count": 10,
    "episode_count_per_controller": 10,
    "episode_count": 40,
    "result_based_filtering": False,
}


Key = tuple[str, str, int]


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_structshell_overall_rollback_screen.py",
            "scripts/run_stride_structshell_overall_rollback_screen.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/stride_failure_informed_rescue_continuation.py",
            "experiments/stride_hybridstructpool_routed_confirmation.py",
            "experiments/stride_structpool_ttf_quick.py",
            "experiments/stride_structshell_rollback_aware_ttf.py",
            "lns2_selector/runtime/hybridstructpool_routed.py",
            "lns2_selector/runtime/rollback_aware_selection.py",
            "lns2_selector/runtime/overall_rollback_selection.py",
        ),
        native_required=native_required,
    )


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_seen_key_overall_rollback_mechanism_cost_screen"
        or config.get("experiment_id") != EXPERIMENT_ID
        or str(config.get("pre_registration_parent_commit"))
        != PRE_REGISTRATION_PARENT
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or dict(config.get("comparison") or {}) != _EXPECTED_COMPARISON
        or dict(config.get("runtime") or {}) != _EXPECTED_RUNTIME
        or dict(config.get("cohort") or {}) != _EXPECTED_COHORT
    ):
        raise ValueError("overall-rollback diagnostic registration changed")

    source_specification = dict(
        dict(config.get("inputs") or {}).get("rollback_aware_ttf_source_config")
        or {}
    )
    source_path = registered_input(
        root,
        source_specification,
        label="overall-rollback source screen config",
    )
    _source_path, source_root, source_config = load_source_config(source_path)
    if source_root != root:
        raise ValueError("overall-rollback source config belongs to another repository")

    no_guard = validate_routed_hybridstructpool_augmentation(
        dict(config.get("no_guard_augmentation") or {})
    )
    if no_guard != routed_hybridstructpool_augmentation("routed_structshell"):
        raise ValueError("overall-rollback no-guard identity changed")
    challenger = validate_overall_rollback_routed_hybridstructpool_augmentation(
        dict(config.get("challenger_augmentation") or {})
    )
    if challenger != overall_rollback_routed_hybridstructpool_augmentation():
        raise ValueError("overall-rollback challenger identity changed")

    controller_manifest = dict(
        dict(config.get("inputs") or {}).get("controller_manifest") or {}
    )
    registered_input(
        root,
        controller_manifest,
        label="overall-rollback controller manifest",
    )
    if str(config.get("controller_bundle")) != str(
        source_config.get("controller_bundle")
    ):
        raise ValueError("overall-rollback controller bundle changed")
    if len(screen_keys(source_config)) != 10 or len(schedule(config, source_config)) != 40:
        raise ValueError("overall-rollback fixed ten-key schedule changed")
    return path, root, config, source_config


def screen_keys(source_config: Mapping[str, Any]) -> set[Key]:
    return {
        (str(group["id"]), str(group["tasks"][0]), 16)
        for group in source_config["cohort"]["groups"]
    }


def schedule(
    config: Mapping[str, Any], source_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    del config
    rows: list[dict[str, Any]] = []
    for key_index, group in enumerate(source_config["cohort"]["groups"]):
        offset = key_index % len(CONTROLLERS)
        for position in range(len(CONTROLLERS)):
            rows.append(
                {
                    "group_id": str(group["id"]),
                    "family": str(group["family"]),
                    "task_id": str(group["tasks"][0]),
                    "solver_seed": 16,
                    "controller": CONTROLLERS[(offset + position) % len(CONTROLLERS)],
                    "within_key_position": position,
                }
            )
    return rows


def _group(source_config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        dict(group)
        for group in source_config["cohort"]["groups"]
        if str(group["id"]) == str(group_id)
    )


def _controller_kwargs(
    root: Path, config: Mapping[str, Any], name: str
) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    common = {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
    }
    if name == "official_adaptive":
        return {
            **common,
            "controller": "official_adaptive",
            "feature_backend": "auto",
            "controller_runtime": "reference",
            "verification_profile": "audit",
        }
    result = {
        **common,
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if name == "structshell_routed_v1":
        result["hybridstructpool_augmentation"] = dict(
            config["no_guard_augmentation"]
        )
    elif name == "structshell_rollback_overall_v3":
        result["hybridstructpool_augmentation"] = dict(
            config["challenger_augmentation"]
        )
    elif name != "v2_only":
        raise ValueError(f"unknown overall-rollback screen controller: {name}")
    return result


def _runtime_config_path(
    root: Path,
    output: Path,
    source_config: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Path:
    payload = _read_json((root / str(group["runtime_config"])).resolve())
    payload["solver_seeds"] = list(source_config["cohort"]["solver_seeds"])
    destination = output / "runtime_configs" / f"{group['id']}__qualification.json"
    _write_json(destination, payload)
    return destination


def _manifest(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("overall-rollback screen manifest is ambiguous")
    return matches[0] if matches else None


def _qualification_summary(
    output: Path, source_config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = len(source_config["cohort"]["solver_seeds"]) * 2
    details: list[dict[str, Any]] = []
    for group in source_config["cohort"]["groups"]:
        path = output / "qualification" / str(group["id"]) / "qualification_report.json"
        if not path.is_file():
            details.append(
                {
                    "group_id": str(group["id"]),
                    "expected_count": expected,
                    "valid_count": 0,
                    "passed": False,
                    "decision": "missing_qualification_report",
                }
            )
            continue
        report = _read_json(path)
        valid = int(report.get("valid_count", -1))
        details.append(
            {
                "group_id": str(group["id"]),
                "expected_count": expected,
                "valid_count": valid,
                "passed": bool(
                    report.get("passed") is True
                    and valid == expected
                    and int(report.get("incomplete_reset_count", -1)) == 0
                ),
                "decision": str(report.get("decision") or ""),
            }
        )
    return {
        "completed_map_count": sum(row["valid_count"] == expected for row in details),
        "passed_map_count": sum(row["passed"] for row in details),
        "total_map_count": len(details),
        "all_maps_passed": bool(details) and all(row["passed"] for row in details),
        "details": details,
    }


def _ensure_qualification(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> dict[str, Any]:
    for group in source_config["cohort"]["groups"]:
        keys = {
            (str(task), int(seed))
            for task in group["tasks"]
            for seed in source_config["cohort"]["solver_seeds"]
        }
        destination = output / "qualification" / str(group["id"])
        run_closed_loop_collection(
            root / str(group["dataset"]),
            _runtime_config_path(root, output, source_config, group),
            destination,
            phase="qualify",
            workers=16,
            resume=destination.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs(root, config, "v2_only"),
        )
    return _qualification_summary(output, source_config)


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    _path, root, config, source_config = load_config(job["config_path"])
    output = Path(job["output_root"]).resolve()
    item = dict(job["item"])
    group = _group(source_config, str(item["group_id"]))
    keys = {
        (str(task), int(seed))
        for task in group["tasks"]
        for seed in source_config["cohort"]["solver_seeds"]
    }
    collection = _controller_dir(output, item)
    kwargs = _controller_kwargs(root, config, str(item["controller"]))
    qualification = output / "qualification" / str(group["id"])
    runtime = _runtime_config_path(root, output, source_config, group)
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase=(
            "official_adaptive"
            if item["controller"] == "official_adaptive"
            else "realized_dynamic"
        ),
        workers=1,
        resume=True,
        cohort_job_keys=keys,
        job_keys={(str(item["task_id"]), int(item["solver_seed"]))},
        qualification_source=qualification,
        use_global_collection_lock=False,
        **kwargs,
    )
    row = _manifest(output, item)
    if row is None:
        raise RuntimeError("overall-rollback episode completed without manifest")
    status = str(row.get("status"))
    if status not in {"ok", "resumed", "error", "timeout"}:
        raise RuntimeError(f"unknown overall-rollback manifest status: {status}")
    return {**item, "status": status if status in {"error", "timeout"} else "ok"}


def _failed_episode_job(
    job: dict[str, Any], status: str, message: str
) -> dict[str, Any]:
    return {
        **dict(job["item"]),
        "status": status,
        "manifest_status": status,
        "error": message,
        "state_count": 0,
        "outcome_count": 0,
    }


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    rows = [_manifest(output, item) for item in items]
    present = [row for row in rows if row is not None]
    return {
        **dict(base),
        "completed_jobs": len(present),
        "completed_schedule_entries": len(present),
        "total_jobs": len(items),
        "completed_by_controller": {
            name: sum(
                _manifest(output, item) is not None
                for item in items
                if item["controller"] == name
            )
            for name in CONTROLLERS
        },
        "error_jobs": sum(row.get("status") == "error" for row in present),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in present),
        "active_jobs": 0,
        "complete": bool(complete and len(present) == len(items)),
    }


def _read_index(
    source_config: Mapping[str, Any], output: Path
) -> tuple[dict[str, dict[Key, dict[str, Any]]], list[str], dict[str, dict[str, str]]]:
    indexed = {name: {} for name in CONTROLLERS}
    errors: list[str] = []
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for controller in CONTROLLERS:
        for group in source_config["cohort"]["groups"]:
            item = {"group_id": group["id"], "controller": controller}
            path = _manifest_path(output, item)
            if not path.is_file():
                errors.append(f"{controller}/{group['id']}: missing manifest")
                continue
            hashes[controller][str(group["id"])] = sha256_file(path)
            for raw in _read_jsonl(path):
                row = _metric_manifest(raw)
                key = (str(group["id"]), str(row["task_id"]), int(row["solver_seed"]))
                if key in indexed[controller]:
                    errors.append(f"{controller}: duplicate {key}")
                indexed[controller][key] = row
    return indexed, errors, hashes


def _mechanism_totals(rows: Mapping[Key, Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    selected_sizes: dict[str, int] = defaultdict(int)
    selected_families: dict[str, int] = defaultdict(int)
    for row in rows.values():
        if not _successful_manifest(row):
            continue
        summary = dict(row["summary"])
        controller_totals = dict(summary.get("controller_totals") or {})
        for name, value in controller_totals.items():
            if isinstance(value, (int, float)) and (
                "guard" in str(name)
                or "rollback" in str(name)
                or str(name).startswith("hybridstructpool_")
                or str(name) == "repair_state_cache_hit_count"
            ):
                totals[str(name)] += float(value)
        for name, value in dict(summary.get("selected_size_counts") or {}).items():
            selected_sizes[str(name)] += int(value)
        for name, value in dict(summary.get("selected_family_counts") or {}).items():
            selected_families[str(name)] += int(value)
    return {
        "controller_totals": dict(sorted(totals.items())),
        "selected_size_counts": dict(sorted(selected_sizes.items())),
        "selected_family_counts": dict(sorted(selected_families.items())),
    }


def _state_guard_trace_audit(
    output: Path,
    indexed: Mapping[str, Mapping[Key, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Passively verify the registered per-repair-state guard contract.

    Summary counters are useful for cost attribution but cannot prove that a
    V2 rollback did not reopen StructShell or that a fallback was generated
    fresh.  This audit therefore replays every v3 transition trace and checks
    the serialized selection and observation records directly.
    """

    errors: list[str] = []
    decision_count = 0
    counted_structshell_rollbacks = 0
    newly_suppressed_count = 0
    fresh_v2_decision_count = 0
    v2_exact_rollbacks_while_suppressed = 0
    maximum_count_by_fingerprint = 0
    audited_episode_count = 0

    for key, manifest in indexed["structshell_rollback_overall_v3"].items():
        collection = output / "maps" / key[0] / "structshell_rollback_overall_v3"
        try:
            rows = _decision_rows(collection, manifest)
        except Exception as error:  # trace corruption is an integrity failure
            errors.append(f"{key}: trace replay failed: {error}")
            continue
        audited_episode_count += 1
        expected_decisions = int(dict(manifest.get("summary") or {}).get(
            "repair_iterations", -1
        ))
        if len(rows) != expected_decisions:
            errors.append(
                f"{key}: transition/repair count mismatch {len(rows)} != {expected_decisions}"
            )
        suppressed: set[str] = set()
        counted_by_fingerprint: dict[str, int] = defaultdict(int)
        for row in rows:
            decision_count += 1
            controller = dict(row.get("controller") or {})
            record = dict(controller.get("exact_rollback_state_guard") or {})
            selection = dict(record.get("selection") or {})
            observation = dict(record.get("observation") or {})
            if not selection or not observation:
                errors.append(
                    f"{key}/decision-{row['decision_index']}: missing state-guard trace"
                )
                continue
            fingerprint = str(selection.get("repair_fingerprint") or "")
            if not fingerprint or fingerprint != str(
                row.get("before_platform_signature") or ""
            ):
                errors.append(
                    f"{key}/decision-{row['decision_index']}: repair fingerprint mismatch"
                )
            if int(selection.get("exact_rollback_limit", -1)) != 3:
                errors.append(
                    f"{key}/decision-{row['decision_index']}: rollback limit changed"
                )
            if int(selection.get("state_exact_rollbacks", -1)) != int(
                counted_by_fingerprint[fingerprint]
            ):
                errors.append(
                    f"{key}/decision-{row['decision_index']}: selection rollback count mismatch"
                )

            latched_before = fingerprint in suppressed
            selection_phase = str(selection.get("selection_phase") or "")
            selected_is_structural = bool(
                selection.get("selected_candidate_is_pure_structshell")
            )
            proposal = dict(controller.get("proposal") or {})
            if latched_before:
                if selection.get("structshell_suppressed") is not True:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: latch reopened"
                    )
                if selected_is_structural:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: StructShell selected while latched"
                    )
                if selection_phase != "fresh_v2_only":
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback is not fresh_v2_only"
                    )
                if int(selection.get("pure_structshell_candidate_count", -1)) != 0:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback retained StructShell"
                    )
                if proposal.get("repair_state_cache_hit") is not False:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback reused repair-state cache"
                    )
                if proposal.get("hybridstructpool_gate_evaluated") is not False:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback re-evaluated Hybrid gate"
                    )
                if proposal.get("hybridstructpool_state_bounded_v2_fallback") is not True:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback proposal flag missing"
                    )
                if str(proposal.get("hybridstructpool_gate_reason") or "") != (
                    "state_exact_rollback_budget_exhausted"
                ):
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: fallback gate reason changed"
                    )
                for candidate in list(controller.get("candidate_pool") or ()):
                    provenance = set(
                        map(
                            str,
                            candidate.get("hybridstructpool_provenance") or (),
                        )
                    )
                    if (
                        "structshell_equal_four_size" in provenance
                        and "v2_base" not in provenance
                    ):
                        errors.append(
                            f"{key}/decision-{row['decision_index']}: fallback pool contains pure StructShell"
                        )
                fresh_v2_decision_count += 1
            elif selection_phase == "fresh_v2_only":
                errors.append(
                    f"{key}/decision-{row['decision_index']}: fallback preceded suppression"
                )

            rollback_counted = bool(observation.get("rollback_counted"))
            exact_rollback = bool(
                observation.get("exact_conflict_bound_rollback")
            )
            if rollback_counted != bool(
                selected_is_structural and exact_rollback and not latched_before
            ):
                errors.append(
                    f"{key}/decision-{row['decision_index']}: counted rollback predicate mismatch"
                )
            if rollback_counted:
                counted_by_fingerprint[fingerprint] += 1
                counted_structshell_rollbacks += 1
            if int(observation.get("pure_structshell_exact_rollbacks", -1)) != int(
                counted_by_fingerprint[fingerprint]
            ):
                errors.append(
                    f"{key}/decision-{row['decision_index']}: observation rollback count mismatch"
                )
            maximum_count_by_fingerprint = max(
                maximum_count_by_fingerprint,
                counted_by_fingerprint[fingerprint],
                int(observation.get("pure_structshell_exact_rollbacks", 0)),
            )
            if counted_by_fingerprint[fingerprint] > 3 or int(
                observation.get("pure_structshell_exact_rollbacks", 0)
            ) > 3:
                errors.append(
                    f"{key}/decision-{row['decision_index']}: state rollback budget exceeded"
                )
            if bool(observation.get("newly_suppressed")):
                newly_suppressed_count += 1
                if counted_by_fingerprint[fingerprint] != 3:
                    errors.append(
                        f"{key}/decision-{row['decision_index']}: suppression did not occur at three"
                    )
                suppressed.add(fingerprint)
            elif bool(observation.get("structshell_suppressed")):
                suppressed.add(fingerprint)
            if (
                bool(observation.get("rollback_counted"))
                and counted_by_fingerprint[fingerprint] == 3
                and not (
                    observation.get("newly_suppressed") is True
                    and observation.get("structshell_suppressed") is True
                )
            ):
                errors.append(
                    f"{key}/decision-{row['decision_index']}: third rollback did not latch suppression"
                )

            if (
                latched_before
                and bool(observation.get("exact_conflict_bound_rollback"))
                and not bool(observation.get("selected_candidate_is_pure_structshell"))
            ):
                v2_exact_rollbacks_while_suppressed += 1

    gates = {
        "all_v3_episodes_trace_audited": audited_episode_count
        == len(indexed["structshell_rollback_overall_v3"]),
        "every_transition_has_state_guard_record": not any(
            "missing state-guard trace" in error for error in errors
        ),
        "every_transition_is_one_pp_decision": not any(
            "transition/repair count mismatch" in error for error in errors
        ),
        "pure_structshell_exact_rollbacks_at_most_three_per_fingerprint": not any(
            "budget exceeded" in error for error in errors
        ),
        "suppression_occurs_exactly_at_three": not any(
            "suppression did not occur at three" in error for error in errors
        ),
        "no_structshell_selection_while_latched": not any(
            "StructShell selected while latched" in error for error in errors
        ),
        "latch_never_reopens": not any("latch reopened" in error for error in errors),
        "latched_fallback_is_fresh_v2_only": not any(
            any(
                token in error
                for token in (
                    "fallback is not fresh_v2_only",
                    "fallback retained StructShell",
                    "fallback reused repair-state cache",
                    "fallback re-evaluated Hybrid gate",
                    "fallback proposal flag missing",
                    "fallback gate reason changed",
                    "fallback pool contains pure StructShell",
                    "fallback preceded suppression",
                )
            )
            for error in errors
        ),
        "repair_fingerprints_match_trace": not any(
            "repair fingerprint mismatch" in error for error in errors
        ),
        "registered_limit_is_three": not any(
            "rollback limit changed" in error for error in errors
        ),
        "selection_and_observation_counts_match_trace": not any(
            "rollback count mismatch" in error for error in errors
        ),
        "counted_rollbacks_match_exact_structshell_predicate": not any(
            "counted rollback predicate mismatch" in error for error in errors
        ),
        "third_counted_rollback_immediately_latches": not any(
            "third rollback did not latch suppression" in error for error in errors
        ),
    }
    return {
        "audited_episode_count": audited_episode_count,
        "decision_count": decision_count,
        "counted_pure_structshell_exact_rollbacks": counted_structshell_rollbacks,
        "newly_suppressed_state_count": newly_suppressed_count,
        "fresh_v2_decision_count": fresh_v2_decision_count,
        "v2_exact_rollbacks_while_suppressed": v2_exact_rollbacks_while_suppressed,
        "maximum_counted_rollbacks_per_repair_fingerprint": (
            maximum_count_by_fingerprint
        ),
        "gates": gates,
        "passed": not errors and all(gates.values()),
        "errors": errors,
    }


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config, source_config = load_config(config_path)
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
    expected = screen_keys(source_config)
    indexed, errors, hashes = _read_index(source_config, output)
    for controller in CONTROLLERS:
        if set(indexed[controller]) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
    keys = sorted(expected)
    fingerprint_mismatches = conflict_mismatches = bad_clock = bad_capped = 0
    for key in keys:
        rows = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or not _successful_manifest(row) for row in rows):
            continue
        episode_summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len(
            {row.get("initial_fingerprint") for row in episode_summaries}
        ) != 1
        conflict_mismatches += len(
            {row.get("initial_conflicts") for row in episode_summaries}
        ) != 1
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
            for row in episode_summaries
        )
        bad_capped += sum(
            row.get("capped_wall_time_to_feasible") is None
            for row in episode_summaries
        )
    summaries = {
        name: _extended_summary(indexed[name].values()) for name in CONTROLLERS
    }
    timing = {name: _timing_summary(indexed[name].values()) for name in CONTROLLERS}
    mechanisms = {name: _mechanism_totals(indexed[name]) for name in CONTROLLERS}
    trace_audit = _state_guard_trace_audit(output, indexed)
    contrast_pairs = {
        "v3_vs_v2": ("v2_only", "structshell_rollback_overall_v3"),
        "v3_vs_no_guard": (
            "structshell_routed_v1",
            "structshell_rollback_overall_v3",
        ),
        "no_guard_vs_v2": ("v2_only", "structshell_routed_v1"),
        "v2_vs_official": ("official_adaptive", "v2_only"),
        "v3_vs_official": (
            "official_adaptive",
            "structshell_rollback_overall_v3",
        ),
    }
    comparisons = {
        label: _bounded_paired_comparison(indexed[left], indexed[right], keys)
        for label, (left, right) in contrast_pairs.items()
    }
    per_map: dict[str, dict[str, Any]] = {}
    for key in keys:
        value: dict[str, Any] = {
            "controller_summaries": {
                name: _extended_summary([indexed[name][key]])
                for name in CONTROLLERS
            },
            "timing_accountability": {
                name: _timing_summary([indexed[name][key]])
                for name in CONTROLLERS
            },
        }
        value.update(
            {
                label: _bounded_paired_comparison(
                    indexed[left], indexed[right], [key]
                )
                for label, (left, right) in contrast_pairs.items()
            }
        )
        per_map[key[0]] = value
    integrity = {
        "complete_paired_coverage": not any(
            "coverage" in error or "missing" in error for error in errors
        ),
        "zero_execution_errors_or_process_timeouts": all(
            _successful_manifest(row)
            for controller in indexed.values()
            for row in controller.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "registered_ttf_clock": bad_clock == 0,
        "bounded_ttf_complete": bad_capped == 0,
        "valid_bounded_stop_reasons": all(
            str(dict(row.get("summary") or {}).get("stop_reason"))
            in {"success", "wall_timeout", "controller_stalled", "native_terminal"}
            for controller in indexed.values()
            for row in controller.values()
            if _successful_manifest(row)
        ),
        "zero_invalid_actions": all(
            int(summaries[name].get("invalid_action_count", -1)) == 0
            for name in CONTROLLERS
        ),
        "zero_semantic_mismatches": all(
            int(summaries[name].get("fingerprint_mismatch_count", -1)) == 0
            for name in CONTROLLERS
        ),
        "one_pp_per_decision_without_retry_or_rescue": all(
            int(dict(row["summary"]).get("model_decision_count", -1))
            == int(dict(row["summary"]).get("repair_iterations", -2))
            and all(
                dict(row["summary"]).get(name) is None
                for name in (
                    "bounded_native_retry",
                    "failure_informed_rescue",
                    "signature_scoped_rescue",
                )
            )
            for controller in (
                "structshell_routed_v1",
                "structshell_rollback_overall_v3",
            )
            for row in indexed[controller].values()
            if _successful_manifest(row)
        ),
        "state_guard_trace_contract": bool(trace_audit["passed"]),
    }
    v3 = summaries["structshell_rollback_overall_v3"]
    no_guard = summaries["structshell_routed_v1"]
    v2 = summaries["v2_only"]
    vs_v2 = comparisons["v3_vs_v2"]
    vs_no_guard = comparisons["v3_vs_no_guard"]
    no_double_loss = all(
        not (
            int(value["v3_vs_v2"]["baseline_success_count"])
            > int(value["v3_vs_v2"]["challenger_success_count"])
            and float(value["v3_vs_v2"]["mean_restricted_ttf_delta_seconds"])
            > 0.0
        )
        for value in per_map.values()
    )
    v2_directional = {
        "success_noninferior_to_v2": int(v3["success_count"])
        >= int(v2["success_count"]),
        "mean_restricted_ttf_lower_than_v2": float(
            vs_v2["mean_restricted_ttf_delta_seconds"]
        )
        < 0.0,
        "paired_faster_fraction_at_least_half_vs_v2": float(
            vs_v2["paired_faster_fraction"]
        )
        >= 0.5,
        "normalized_wall_auc_noninferior_to_v2": float(
            v3["mean_normalized_wall_clock_conflict_auc"]
        )
        <= float(v2["mean_normalized_wall_clock_conflict_auc"]),
        "p95_restricted_ttf_noninferior_to_v2": float(
            v3["restricted_ttf"]["p95"]
        )
        <= float(v2["restricted_ttf"]["p95"]),
        "p95_repair_decisions_noninferior_to_v2": float(
            v3["repair_decisions"]["p95"]
        )
        <= float(v2["repair_decisions"]["p95"]),
        "no_map_combines_success_loss_and_worse_restricted_ttf": no_double_loss,
    }
    no_guard_directional = {
        "success_noninferior_to_no_guard": int(v3["success_count"])
        >= int(no_guard["success_count"]),
        "mean_restricted_ttf_lower_than_no_guard": float(
            vs_no_guard["mean_restricted_ttf_delta_seconds"]
        )
        < 0.0,
        "paired_faster_fraction_at_least_half_vs_no_guard": float(
            vs_no_guard["paired_faster_fraction"]
        )
        >= 0.5,
        "normalized_wall_auc_noninferior_to_no_guard": float(
            v3["mean_normalized_wall_clock_conflict_auc"]
        )
        <= float(no_guard["mean_normalized_wall_clock_conflict_auc"]),
        "repair_p95_noninferior_to_no_guard": float(v3["repair_decisions"]["p95"])
        <= float(no_guard["repair_decisions"]["p95"]),
    }
    if producer is None:
        producer = _producer(root)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "seen_key_mechanism_and_cost_screen_no_promotion_claim",
        "experiment_id": EXPERIMENT_ID,
        "map_count": 10,
        "paired_key_count": len(keys),
        "episode_count": sum(len(value) for value in indexed.values()),
        "controller_summaries": summaries,
        "comparisons": comparisons,
        "per_map": per_map,
        "timing_accountability": timing,
        "timing_accountability_semantics": {
            "guard_seconds": (
                "overlapping diagnostic: selection is already included in "
                "selection/controller time and observation is already included "
                "in post-step/iteration wall; do not add it to components"
            ),
            "restricted_ttf": "reset-inclusive min(observed wall, 180 seconds)",
        },
        "mechanism_diagnostics": mechanisms,
        "state_guard_trace_audit": trace_audit,
        "integrity_gates": integrity,
        "v2_directional_diagnostic_gates": v2_directional,
        "no_guard_directional_diagnostic_gates": no_guard_directional,
        "integrity_passed": not errors and all(integrity.values()),
        "directionally_better_than_v2": bool(all(v2_directional.values())),
        "directionally_better_than_no_guard": bool(
            all(no_guard_directional.values())
        ),
        "diagnostic_complete": not errors and all(integrity.values()),
        "promotion_claim": False,
        "runtime_replacement_allowed": False,
        "fresh_confirmation_required": True,
        "next_step": (
            "preregister_fresh_result_blind_confirmation"
            if not errors
            and all(integrity.values())
            and all(v2_directional.values())
            else "keep_v2_default_and_stop_v3"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": hashes,
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config, source_config = load_config(config_path)
    output = Path(output).resolve()
    items = schedule(config, source_config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "qualification_key_count": 60,
            "paired_key_count": 10,
            "schedule_entry_count": 40,
            "schedule_sha256": _fingerprint(items),
            "timed_worker_count": 1,
            "qualification_worker_limit": 16,
            "wall_time_budget_seconds": 180.0,
            "episode_process_timeout_seconds": 240.0,
            "outer_job_timeout_seconds": 300.0,
            "promotion_claim": False,
        }
    producer = _producer(root)
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
        label="overall-rollback StructShell screen",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.resumed and prepared.status.get("terminal_failure") is not None:
        return dict(prepared.status)
    _write_jsonl(output / "execution_schedule.jsonl", items)
    initial_status = _status(output, items, prepared.base_status)
    _write_json(output / STATUS_FILENAME, initial_status)
    existing_terminal = [
        row
        for item in items
        if (row := _manifest(output, item)) is not None
        and str(row.get("status")) in {"error", "timeout"}
    ]
    if existing_terminal:
        initial_status["terminal_failure"] = existing_terminal[0]
        _write_json(output / STATUS_FILENAME, initial_status)
        return initial_status
    qualification = _ensure_qualification(root, output, config, source_config)
    if not qualification["all_maps_passed"]:
        status = _status(output, items, prepared.base_status)
        status["qualification"] = qualification
        status["terminal_failure"] = "qualification_failed"
        _write_json(output / STATUS_FILENAME, status)
        return status
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase="structshell-overall-rollback-screen",
            output_root=output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=float(config["runtime"]["outer_job_timeout_seconds"]),
            failure_result=_failed_episode_job,
            stop_on_failure=True,
        )
        failures = [row for row in results if row.get("status") in {"error", "timeout"}]
        if failures:
            status = _status(output, items, prepared.base_status)
            status["qualification"] = qualification
            status["terminal_failure"] = failures[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    report = analyze(path, output, producer=producer)
    status = _status(output, items, prepared.base_status, complete=True)
    status["qualification"] = qualification
    status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
    _write_json(output / STATUS_FILENAME, status)
    return report


__all__ = [
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "analyze",
    "load_config",
    "run",
    "schedule",
    "screen_keys",
]
