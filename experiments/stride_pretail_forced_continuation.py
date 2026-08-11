from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    mean,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.run_output_guard import prepare_resumable_output
from experiments.stride_collection import _paired_action
from experiments.stride_maze_tail_state_collection import (
    _fused_controller_kwargs,
    load_maze_tail_state_collection_config,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.stride_tailswitch import _episode_summary, classify_adverse_pair
from experiments.trace_replay import target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.topology_candidates import _jaccard


CONFIG_SCHEMA = "lns2.stride.pretail_forced_continuation_registration.v1"
STATUS_SCHEMA = "lns2.stride.pretail_forced_continuation_status.v1"
REPORT_SCHEMA = "lns2.stride.pretail_forced_continuation_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.pretail_forced_continuation_override.v1"
ARMS = ("actual_selected", "one_step_oracle", "coverage_diverse")
STATUS_FILENAME = "pretail_forced_continuation_status.json"
REPORT_FILENAME = "pretail_forced_continuation_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="PreTail forced continuation")


def load_pretail_forced_continuation_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_bounded_forced_first_action_continuation_diagnostic"
        or config.get("experiment_id") != "stride-pretail-forced-continuation-v1"
        or config.get("pre_registration_parent_commit")
        != "0ec302239db793974009e92c39b0d5e35798aa67"
        or config.get("execution_amendment_parent_commit")
        != "d7d416bd1424702dd2060be777574e919979daf2"
        or config.get("execution_amendment_reason")
        != "replace_thread_pool_with_eight_independent_process_shards_after_smoke_showed_native_spawn_serialization_before_formal_collection"
    ):
        raise ValueError("PreTail forced-continuation identity changed")
    expected_inputs = {
        "root_checkpoints",
        "logical_checkpoint_results",
        "candidate_aggregates",
        "tailswitch_registration",
        "tailswitch_status",
        "tailswitch_report",
        "tailswitch_qualification_manifest",
        "tailswitch_qualification_report",
        "tailswitch_qualification_run_config",
        "state_collection_config",
        "runtime_config",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("PreTail forced-continuation input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    _parent_path, parent_root, parent = load_maze_tail_state_collection_config(
        inputs["state_collection_config"]
    )
    if parent_root != root:
        raise ValueError("PreTail parent configuration root changed")
    execution = dict(config.get("execution") or {})
    if execution != {
        "checkpoint_kind": "first_structural_selection",
        "paired_first_action_trial_indices": [0, 1],
        "first_action_forced_exactly_once": True,
        "continuation_controller": "matching_frozen_structpool_or_slotpool",
        "parallel_shard_count": 8,
        "workers_per_shard": 1,
        "maximum_repair_decisions_from_restored_state": 200,
        "fixed_metric_horizon": 200,
        "wall_time_fuse_seconds": 300.0,
        "process_timeout_seconds": 360.0,
        "right_censoring_is_valid_state_evidence": True,
        "right_censoring_is_not_an_execution_error": True,
        "external_process_timeout_is_execution_error": True,
    }:
        raise ValueError("PreTail execution fuse or schedule changed")
    if tuple(map(str, config.get("arms") or ())) != ARMS:
        raise ValueError("PreTail forced-continuation arms changed")
    if dict(config.get("claim_boundary") or {}) != {
        "mechanism_diagnostic_only": True,
        "one_step_oracle_is_result_derived_diagnostic_only": True,
        "model_training_allowed": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
        "censored_ttf_is_not_imputed": True,
    }:
        raise ValueError("PreTail claim boundary changed")
    runtime = _read_json(inputs["runtime_config"])
    if (
        runtime.get("max_decisions") != 200
        or runtime.get("metric_iteration_budget") != 200
        or runtime.get("wall_time_budget_seconds") != 300.0
        or runtime.get("episode_process_timeout_seconds") != 360.0
        or runtime.get("deterministic_pp_replay") is not True
    ):
        raise ValueError("PreTail fused runtime changed")
    return path, root, config, inputs, parent


def _paired_first_action_seed(case_id: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "purpose": "pretail-forced-first-action",
                "case_id": str(case_id),
                "trial_index": int(trial_index),
            }
        )[:8],
        16,
    ) & 0x7FFFFFFF


def _candidate_roles(
    checkpoint: dict[str, Any],
    logical: dict[str, Any],
    aggregates: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pool = {
        str(candidate["candidate_id"]): dict(candidate)
        for candidate in checkpoint["candidate_pool"]
    }
    selected_id = str(checkpoint["selected_candidate_id"])
    oracle_id = str(logical["best_candidate_id"])
    if selected_id not in pool or oracle_id not in pool:
        raise ValueError("PreTail selected or one-step oracle candidate left the pool")
    selected = pool[selected_id]
    diverse = [
        candidate
        for candidate in pool.values()
        if str(candidate["candidate_id"]) != selected_id
        and _jaccard(candidate["agents"], selected["agents"]) <= 0.8
    ]
    if not diverse:
        diverse = [
            candidate
            for candidate in pool.values()
            if str(candidate["candidate_id"]) != selected_id
        ]
    if not diverse:
        diverse = [selected]

    def coverage_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        aggregate = aggregates[
            (str(checkpoint["state_fingerprint"]), str(candidate["candidate_id"]))
        ]
        features = dict(aggregate["features"])
        raw_score = candidate.get("score")
        frozen_score = float(raw_score) if raw_score is not None else -math.inf
        return (
            -float(features["realized.internal_conflict_coverage"]),
            -float(features["realized.incident_conflict_coverage"]),
            -frozen_score,
            str(candidate["candidate_id"]),
        )

    coverage = sorted(diverse, key=coverage_key)[0]
    return {
        "actual_selected": selected,
        "one_step_oracle": pool[oracle_id],
        "coverage_diverse": coverage,
    }


def prepare_cases(
    checkpoints: list[dict[str, Any]],
    logical_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in checkpoints
        if str(row.get("checkpoint_kind")) == "first_structural_selection"
    ]
    logical_by_case = {
        str(row["case_id"]): dict(row)
        for row in logical_rows
        if str(row.get("checkpoint_kind")) == "first_structural_selection"
    }
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): dict(row)
        for row in aggregate_rows
    }
    cases: list[dict[str, Any]] = []
    for checkpoint in selected:
        case_id = str(checkpoint["case_id"])
        logical = logical_by_case.get(case_id)
        if logical is None:
            raise ValueError(f"PreTail logical result is missing: {case_id}")
        roles = _candidate_roles(checkpoint, logical, aggregates)
        cases.append({"checkpoint": checkpoint, "logical": logical, "roles": roles})
    if len(cases) != 45 or len({str(row["checkpoint"]["case_id"]) for row in cases}) != 45:
        raise ValueError("PreTail frozen 45-case cohort changed")
    return cases


def continuation_schedule(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for case_position, case in enumerate(cases):
        checkpoint = case["checkpoint"]
        for trial_index in (0, 1):
            for arm_position, arm in enumerate(ARMS):
                candidate = case["roles"][arm]
                schedule.append(
                    {
                        "case_id": str(checkpoint["case_id"]),
                        "state_id": str(checkpoint["state_id"]),
                        "task_id": str(checkpoint["task_id"]),
                        "solver_seed": int(checkpoint["solver_seed"]),
                        "challenger": str(checkpoint["challenger"]),
                        "treatment_policy": str(checkpoint["treatment_policy"]),
                        "trial_index": trial_index,
                        "arm": arm,
                        "candidate_id": str(candidate["candidate_id"]),
                        "case_position": case_position,
                        "arm_position": arm_position,
                    }
                )
    return schedule


def select_shard(
    cases: list[dict[str, Any]], *, shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("PreTail shard index/count is invalid")
    return [
        case for position, case in enumerate(cases) if position % shard_count == shard_index
    ]


def _state_directory(output: Path, case_id: str) -> Path:
    return output / "states" / _fingerprint({"case_id": case_id})[:20]


def _collection_path(output: Path, item: dict[str, Any]) -> Path:
    return (
        _state_directory(output, str(item["case_id"]))
        / f"trial_{int(item['trial_index']):02d}"
        / str(item["arm"])
    )


def _source_collection(root: Path, checkpoint: dict[str, Any]) -> Path:
    return (
        root
        / "build"
        / "stride-tailswitch-v1"
        / "states"
        / _fingerprint({"state_id": str(checkpoint["state_id"])})[:20]
        / str(checkpoint["treatment_policy"])
    )


def _episode_override(
    root: Path,
    checkpoint: dict[str, Any],
    candidate: dict[str, Any],
    *,
    trial_index: int,
    arm: str,
) -> dict[str, Any]:
    source_root = _source_collection(root, checkpoint)
    manifests = _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError(f"PreTail source manifest is invalid: {source_root}")
    manifest = dict(manifests[0])
    state, _trace = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(checkpoint["decision_index"]),
        expected_fingerprint=str(checkpoint["state_fingerprint"]),
    )
    repair_fingerprint = repair_structure_fingerprint(state)
    first_seed = _paired_first_action_seed(str(checkpoint["case_id"]), trial_index)
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
        "forced_first_action": _paired_action(
            list(map(int, candidate["agents"])), first_seed
        ),
        "forced_candidate_id": str(candidate["candidate_id"]),
        "forced_candidate_role": arm,
        "forced_selection_families": list(
            map(str, candidate.get("selection_families") or ())
        ),
    }


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_pretail_forced_continuation.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/stride_tailswitch.py",
            "experiments/trace_replay.py",
        ),
        native_required=native_required,
    )


def _done(collection: Path) -> bool:
    manifest = collection / "realized_dynamic_manifest.jsonl"
    return manifest.is_file() and any(
        row.get("status") in {"ok", "error", "timeout"} for row in _read_jsonl(manifest)
    )


def run_pretail_forced_continuation(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    limit_cases: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    path, root, config, inputs, parent = load_pretail_forced_continuation_config(
        config_path
    )
    all_cases = prepare_cases(
        _read_jsonl(inputs["root_checkpoints"]),
        _read_jsonl(inputs["logical_checkpoint_results"]),
        _read_jsonl(inputs["candidate_aggregates"]),
    )
    cases = all_cases if limit_cases is None else all_cases[: int(limit_cases)]
    if (shard_index is None) != (shard_count is None):
        raise ValueError("PreTail shard index and count must be provided together")
    if shard_index is not None and shard_count is not None:
        cases = select_shard(
            cases, shard_index=int(shard_index), shard_count=int(shard_count)
        )
    schedule = continuation_schedule(cases)
    if dry_run:
        identical_oracle = sum(
            item["arm"] == "one_step_oracle"
            and item["candidate_id"]
            == next(
                row["candidate_id"]
                for row in schedule
                if row["case_id"] == item["case_id"]
                and row["trial_index"] == item["trial_index"]
                and row["arm"] == "actual_selected"
            )
            for item in schedule
        ) // 2
        return {
            "schema": STATUS_SCHEMA,
            "case_count": len(cases),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "identical_actual_oracle_case_count": identical_oracle,
            "maximum_repair_decisions": 200,
            "wall_time_fuse_seconds": 300.0,
            "process_timeout_seconds": 360.0,
            "shard_index": shard_index,
            "shard_count": shard_count,
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=_producer(root),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="PreTail forced continuation",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    status_base = prepared.base_status
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    runtime = inputs["runtime_config"]
    registered_job_keys = {
        (str(case["checkpoint"]["task_id"]), int(case["checkpoint"]["solver_seed"]))
        for case in all_cases
    }
    qualification_root = output / "qualification"
    qualification_source = inputs["tailswitch_qualification_report"].parent
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=(resume and qualification_root.joinpath("run_config.json").is_file()),
        cohort_job_keys=registered_job_keys,
        job_keys=registered_job_keys,
        qualification_source=qualification_source,
        **_fused_controller_kwargs(root, parent, "v2-full"),
    )
    case_by_id = {str(case["checkpoint"]["case_id"]): case for case in cases}

    def run_item(item: dict[str, Any]) -> dict[str, Any]:
        collection = _collection_path(output, item)
        if _done(collection):
            return item
        case = case_by_id[str(item["case_id"])]
        checkpoint = case["checkpoint"]
        candidate = case["roles"][str(item["arm"])]
        key = (str(item["task_id"]), int(item["solver_seed"]))
        override = _episode_override(
            root,
            checkpoint,
            candidate,
            trial_index=int(item["trial_index"]),
            arm=str(item["arm"]),
        )
        kwargs = _fused_controller_kwargs(root, parent, str(item["challenger"]))
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            workers=1,
            resume=(resume and collection.joinpath("run_config.json").is_file()),
            cohort_job_keys=registered_job_keys,
            job_keys=registered_job_keys,
            qualification_source=qualification_root,
            episode_overrides={key: override},
            **kwargs,
        )
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="realized_dynamic",
            workers=1,
            resume=True,
            cohort_job_keys=registered_job_keys,
            job_keys={key},
            episode_overrides={key: override},
            **kwargs,
        )
        return item

    pending = [item for item in schedule if not _done(_collection_path(output, item))]
    completed = len(schedule) - len(pending)
    for item in pending:
        run_item(item)
        completed += 1
        _write_json(
            output / STATUS_FILENAME,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "total_schedule_entries": len(schedule),
                "active_jobs": int(completed < len(schedule)),
                "current": item,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "complete": False,
            },
        )
    report = analyze_pretail_forced_continuation(
        path, output, expected_cases=len(cases)
    )
    _write_json(
        output / STATUS_FILENAME,
        {
            **status_base,
            "completed_schedule_entries": len(schedule),
            "total_schedule_entries": len(schedule),
            "active_jobs": 0,
            "complete": True,
            "report_sha256": sha256_file(output / REPORT_FILENAME),
        },
    )
    return report


def analyze_pretail_forced_continuation(
    config_path: str | Path,
    output: str | Path,
    *,
    expected_cases: int = 45,
) -> dict[str, Any]:
    _path, _root, config, inputs, _parent = load_pretail_forced_continuation_config(
        config_path
    )
    all_cases = prepare_cases(
        _read_jsonl(inputs["root_checkpoints"]),
        _read_jsonl(inputs["logical_checkpoint_results"]),
        _read_jsonl(inputs["candidate_aggregates"]),
    )
    output = Path(output).resolve()
    registered_schedule_path = output / "execution_schedule.jsonl"
    if registered_schedule_path.is_file():
        registered_schedule = _read_jsonl(registered_schedule_path)
        ordered_case_ids = list(
            dict.fromkeys(str(row["case_id"]) for row in registered_schedule)
        )
        case_by_id = {
            str(case["checkpoint"]["case_id"]): case for case in all_cases
        }
        cases = [case_by_id[case_id] for case_id in ordered_case_ids]
    else:
        cases = all_cases[:expected_cases]
    schedule = continuation_schedule(cases)
    episodes: list[dict[str, Any]] = []
    for item in schedule:
        manifest_path = _collection_path(output, item) / "realized_dynamic_manifest.jsonl"
        manifests = _read_jsonl(manifest_path) if manifest_path.is_file() else []
        if len(manifests) != 1:
            episodes.append({**item, "missing": True})
            continue
        row = dict(manifests[0])
        episodes.append({**item, "manifest_status": row.get("status"), **_episode_summary(row)})
    complete = [row for row in episodes if not row.get("missing")]
    grouped = {
        (str(row["case_id"]), int(row["trial_index"])): {}
        for row in complete
    }
    for row in complete:
        grouped[(str(row["case_id"]), int(row["trial_index"]))][str(row["arm"])] = row
    rule = dict(config["paired_outcome_rule"])
    comparisons: list[dict[str, Any]] = []
    for (case_id, trial_index), arms in sorted(grouped.items()):
        if set(arms) != set(ARMS):
            continue
        reference = arms["actual_selected"]
        for arm in ARMS[1:]:
            comparison = classify_adverse_pair(reference, arms[arm], rule)
            comparisons.append(
                {
                    "case_id": case_id,
                    "trial_index": trial_index,
                    "challenger": str(reference["challenger"]),
                    "task_id": str(reference["task_id"]),
                    "arm": arm,
                    "identical_action": str(reference["candidate_id"])
                    == str(arms[arm]["candidate_id"]),
                    **comparison,
                }
            )
    arm_summary: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [row for row in complete if row["arm"] == arm]
        arm_summary[arm] = {
            "episode_count": len(rows),
            "success_count": sum(bool(row["success"]) for row in rows),
            "right_censored_count": sum(
                str(row["stop_reason"]) in {"repair_limit", "wall_timeout"}
                for row in rows
            ),
            "mean_normalized_fixed_auc": mean(
                float(row["normalized_fixed_auc"]) for row in rows
            ),
            "mean_final_conflicts": mean(float(row["final_conflicts"]) for row in rows),
            "mean_repair_iterations": mean(
                float(row["repair_iterations"]) for row in rows
            ),
        }
    informative = [row for row in comparisons if not row["identical_action"]]
    evidence: dict[str, Any] = {}
    for arm in ARMS[1:]:
        rows = [row for row in informative if row["arm"] == arm]
        beneficial = [row for row in rows if row["classification"] == "beneficial"]
        evidence[arm] = {
            "informative_comparison_count": len(rows),
            "beneficial_count": len(beneficial),
            "beneficial_fraction": len(beneficial) / len(rows) if rows else 0.0,
            "beneficial_challenger_count": len(
                {str(row["challenger"]) for row in beneficial}
            ),
            "beneficial_task_count": len({str(row["task_id"]) for row in beneficial}),
        }
    integrity = {
        "case_count": len(cases) == expected_cases,
        "schedule_count": len(schedule) == expected_cases * 6,
        "episode_count": len(complete) == len(schedule),
        "manifest_status_ok": all(row.get("manifest_status") == "ok" for row in complete),
        "allowed_stop_reasons": all(
            str(row.get("stop_reason")) in {"success", "repair_limit", "wall_timeout"}
            for row in complete
        ),
        "forced_first_action_once": all(
            int(row.get("forced_first_action_count", -1)) == 1 for row in complete
        ),
        "no_invalid_actions": all(int(row.get("invalid_action_count", -1)) == 0 for row in complete),
        "no_fingerprint_mismatch": all(
            int(row.get("fingerprint_mismatch_count", -1)) == 0 for row in complete
        ),
        "complete_paired_arms": len(grouped) == expected_cases * 2
        and all(set(rows) == set(ARMS) for rows in grouped.values()),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_bounded_forced_first_action_continuation_diagnostic",
        "case_count": len(cases),
        "episode_count": len(complete),
        "execution_fuse": dict(config["execution"]),
        "arm_summary": arm_summary,
        "paired_comparisons": comparisons,
        "escape_evidence": evidence,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "claim_boundary": dict(config["claim_boundary"]),
        "artifact_sha256": {
            "root_checkpoints": sha256_file(inputs["root_checkpoints"]),
            "logical_checkpoint_results": sha256_file(inputs["logical_checkpoint_results"]),
            "candidate_aggregates": sha256_file(inputs["candidate_aggregates"]),
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "ARMS",
    "analyze_pretail_forced_continuation",
    "continuation_schedule",
    "load_pretail_forced_continuation_config",
    "prepare_cases",
    "run_pretail_forced_continuation",
    "select_shard",
]
