from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments._common import (
    producer_identity,
    sha256_file,
    strict_nonnegative_int as _strict_nonnegative_int,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.parallel_runtime import candidate_lane_counts
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
)
from experiments.run_output_guard import prepare_run_output
from experiments.v3_s3_collection import (
    S3_AGENT_COUNTS,
    S3_LAYOUTS,
    S3_SPLIT_MINIMUM_COUNTS,
    S3_TARGET_STATE_CAP,
    audit_v3_s3_parallelism,
    collect_v3_s3_data,
    source_decisions,
)
from experiments.v3_s3_training import (
    EXTRA_TREES_PARAMETERS,
    HGB_PARAMETERS,
    V3_S3_TRAINING_SCHEMA,
    finalize_v3_s3_native_audit,
    train_v3_s3_controller,
    v3_s3_native_audit_identity,
    v3_s3_training_producer_identity,
)
from experiments.v3_s3 import (
    V3_S3_FEATURE_SCHEMA_SHA256,
    V3_S3_OBJECTIVE_ID,
    V3_S3_SELECTION_OBJECTIVE_ID,
    load_v3_s3_bundle,
)
from generators.config import load_json
from generators.dataset import generate_dataset


V3_S3_PIPELINE_SCHEMA = "lns2.v3_s3_pipeline.v2"
S3_SOURCE_POLICIES = ("fixed_random", "official_adaptive", "realized_dynamic")
V3_S3_PIPELINE_PRODUCER_FILES = (
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/compact_controller_model.py",
    "experiments/context_audit.py",
    "experiments/feature_schema_v2.py",
    "experiments/feature_schema_v3.py",
    "experiments/online_feature_engine.py",
    "experiments/parallel_runtime.py",
    "experiments/repair_aware.py",
    "experiments/repair_aware_training.py",
    "experiments/repair_collection.py",
    "experiments/run_output_guard.py",
    "experiments/stall_guard.py",
    "experiments/trace_replay.py",
    "experiments/v3_s3.py",
    "experiments/v3_s3_collection.py",
    "experiments/v3_s3_pipeline.py",
    "generators/config.py",
    "generators/dataset.py",
    "generators/io.py",
    "generators/models.py",
    "generators/task_flows.py",
    "generators/validation.py",
    "generators/visualization.py",
    "generators/warehouse.py",
    "scripts/run_v3_training_pipeline.py",
    "scripts/audit_v3_s3_source_replay.py",
    "src/jsonl_observer.cpp",
    "src/python_bindings.cpp",
    "src/repair_driver.cpp",
    "third_party/mapf_lns2/inc/BasicLNS.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/SIPP.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)
V3_S3_SOURCE_REPLAY_AUDIT_SCHEMA = "lns2.v3_s3_source_replay_audit.v3"


def _write_status(root: Path, *, started_at: str, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {
            "schema": V3_S3_PIPELINE_SCHEMA,
            "started_at": started_at,
            "updated_at": _utc_now(),
            **values,
        },
    )


def _write_stage_error(
    root: Path,
    *,
    started_at: str,
    phase: str,
    error: BaseException,
    **values: Any,
) -> None:
    _write_status(
        root,
        started_at=started_at,
        status="error",
        phase=f"{phase}-failed",
        error_states=1,
        error=f"{type(error).__name__}: {error}",
        **values,
    )


def _source_config(
    *,
    project_root: Path,
    dataset: Path,
    split: str,
    policy: str,
    output: Path,
    workers: int,
) -> Path:
    rows = _read_jsonl(dataset / split / "manifest.jsonl")
    base = _read_json(project_root / "configs" / "closed_loop_multiseed_collection.json")
    layouts = {
        str(row["map_id"]): str(row["layout_mode"])
        for row in rows
    }
    counts = collections.Counter(str(row["map_id"]) for row in rows)
    tasks_per_map = set(counts.values())
    if len(tasks_per_map) != 1:
        raise ValueError("v3-S3 source dataset has non-uniform tasks per map")
    base.update(
        {
            "formal": False,
            "split": split,
            "solver_seeds": [0 if split == "policy_train" else 17],
            # The closed-loop runner treats the policy list as the registered
            # comparison cohort even when ``phase`` executes only one member.
            # Keep the full cohort here and select the source policy through
            # the phase argument passed to ``run_closed_loop_collection``.
            "policies": list(S3_SOURCE_POLICIES),
            "dataset_design": {
                "map_count": len(layouts),
                "tasks_per_map": next(iter(tasks_per_map)),
                "task_variants": sorted(
                    {str(row["task_variant"]) for row in rows}
                ),
                "layout_counts": dict(collections.Counter(layouts.values())),
            },
            "environment": {
                "time_limit": 45.0,
                "max_repair_iterations": 12,
                "neighborhood_size": 8,
                "replan_algorithm": "PP",
                "use_sipp": True,
            },
            "qualification": {
                "minimum_nonzero_states": 1,
                "minimum_nonzero_states_per_layout": 0,
                "minimum_active_maps": 1,
            },
            "max_decisions": 12,
            "metric_iteration_budget": 12,
            "wall_time_budget_seconds": 45.0,
            "episode_process_timeout_seconds": 75.0,
            "workers": int(workers),
            "reference_datasets": [],
            # Source traces are replayed in fresh Adaptive environments.  PP
            # therefore needs a seed independent of the source destroy
            # heuristic's RNG consumption.
            "deterministic_pp_replay": True,
        }
    )
    path = output / "protocol" / f"source__{split}__{policy}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, base)
    return path


def _source_task_partition(dataset: Path) -> tuple[dict[tuple[str, str], list[str]], dict[str, Any]]:
    partition: dict[tuple[str, str], list[str]] = {
        (split, policy): []
        for split in ("policy_train", "policy_validation")
        for policy in S3_SOURCE_POLICIES
    }
    cells: dict[str, Any] = {}
    for split in ("policy_train", "policy_validation"):
        rows = _read_jsonl(dataset / split / "manifest.jsonl")
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            grouped[
                (
                    str(row["layout_mode"]),
                    int(row["agent_count"]),
                    str(row["map_id"]),
                )
            ].append(row)
        cell_index = 0
        for layout in S3_LAYOUTS:
            for agents in S3_AGENT_COUNTS:
                maps = sorted(
                    {
                        key[2]
                        for key in grouped
                        if key[0] == layout and key[1] == agents
                    }
                )
                if len(maps) != (4 if split == "policy_train" else 2):
                    raise ValueError(
                        f"v3-S3 source cell has unexpected map coverage: {split}/{layout}/{agents}"
                    )
                selected: list[dict[str, Any]] = []
                for map_index, map_id in enumerate(maps):
                    options = sorted(
                        grouped[(layout, agents, map_id)],
                        key=lambda row: str(row["task_variant"]),
                    )
                    if len(options) != 4:
                        raise ValueError("v3-S3 source cell requires four scenarios per map/agent")
                    schedules = ((0, 1), (2, 3), (0, 2), (1, 3))
                    selected.extend(options[index] for index in schedules[map_index])
                for index, row in enumerate(selected):
                    policy = S3_SOURCE_POLICIES[
                        (index + cell_index) % len(S3_SOURCE_POLICIES)
                    ]
                    partition[(split, policy)].append(str(row["task_id"]))
                cells[f"{split}|{layout}|{agents}"] = {
                    "episode_count": len(selected),
                    "policy_counts": dict(
                        collections.Counter(
                            S3_SOURCE_POLICIES[
                                (index + cell_index) % len(S3_SOURCE_POLICIES)
                            ]
                            for index in range(len(selected))
                        )
                    ),
                    "scenario_count": len(
                        {str(row["task_variant"]).split("_", 1)[0] for row in selected}
                    ),
                }
                cell_index += 1
    return partition, {
        "schema": V3_S3_PIPELINE_SCHEMA,
        "episode_count": sum(len(values) for values in partition.values()),
        "by_source": {
            f"{split}|{policy}": len(values)
            for (split, policy), values in sorted(partition.items())
        },
        "cells": cells,
    }


def source_roots(output: Path) -> dict[str, list[Path]]:
    return {
        split: [output / "sources" / split / policy for policy in S3_SOURCE_POLICIES]
        for split in ("policy_train", "policy_validation")
    }


def source_replay_input_identity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind a replay audit to the exact cross-platform source decision stream."""

    decisions = [
        {
            "split": str(row.get("split", "")),
            "source_run_fingerprint": str(
                row.get("source_run_fingerprint", "")
            ),
            "source_policy": str(row["source_policy"]),
            "episode_id": str(row["episode_id"]),
            "decision_index": int(row["decision_index"]),
            "before_fingerprint": str(row["before_fingerprint"]),
            "after_fingerprint": str(row["after_fingerprint"]),
            "prefix_actions": list(row["prefix_actions"]),
            "replay_action": dict(row["replay_action"]),
        }
        for row in rows
    ]
    decisions.sort(
        key=lambda row: (
            row["split"],
            row["source_run_fingerprint"],
            row["source_policy"],
            row["episode_id"],
            row["decision_index"],
            _fingerprint((row["prefix_actions"], row["replay_action"])),
        )
    )
    return {
        "source_state_count": len(rows),
        "source_decision_fingerprint": _fingerprint(decisions),
    }


def _validate_reused_source_report(
    report: Any,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("reused v3-S3 source report is not an object")
    value = dict(report)
    if (
        value.get("schema") != V3_S3_PIPELINE_SCHEMA
        or value.get("complete") is not True
    ):
        raise ValueError("reused v3-S3 source report is incomplete or invalid")
    design = value.get("design")
    reports = value.get("reports")
    qualifications = value.get("qualifications")
    if not all(
        isinstance(item, dict)
        for item in (design, reports, qualifications)
    ):
        raise ValueError("reused v3-S3 source report lacks structured evidence")
    expected_keys = {
        f"{split}|{policy}"
        for split in ("policy_train", "policy_validation")
        for policy in S3_SOURCE_POLICIES
    }
    by_source = design.get("by_source")
    if (
        design.get("schema") != V3_S3_PIPELINE_SCHEMA
        or not isinstance(by_source, dict)
        or set(by_source) != expected_keys
        or set(reports) != expected_keys
        or set(qualifications) != {"policy_train", "policy_validation"}
    ):
        raise ValueError("reused v3-S3 source coverage is invalid")
    expected_total = 0
    for source_key in sorted(expected_keys):
        expected_count = by_source.get(source_key)
        if not _strict_nonnegative_int(expected_count) or expected_count <= 0:
            raise ValueError("reused v3-S3 source count is invalid")
        expected_total += expected_count
        policy = source_key.split("|", 1)[1]
        source_report = reports.get(source_key)
        if not isinstance(source_report, dict):
            raise ValueError("reused v3-S3 policy report is not an object")
        policy_report = source_report.get(policy)
        if (
            source_report.get("schema") != "lns2.closed_loop_confirmation.v1"
            or source_report.get("schema_version") != 1
            or source_report.get("controller") != "v2-full"
            or not isinstance(source_report.get("run_fingerprint"), str)
            or not source_report["run_fingerprint"]
            or not isinstance(policy_report, dict)
            or policy_report.get("episode_count") != expected_count
            or policy_report.get("error_count") != 0
        ):
            raise ValueError("reused v3-S3 policy report failed validation")
    if (
        design.get("episode_count") != expected_total
        or not _strict_nonnegative_int(design.get("episode_count"))
    ):
        raise ValueError("reused v3-S3 source design count mismatch")
    for qualification in qualifications.values():
        if not isinstance(qualification, dict):
            raise ValueError("reused v3-S3 qualification is not an object")
        evidence = qualification.get("qualification")
        if (
            qualification.get("schema") != "lns2.closed_loop_confirmation.v1"
            or qualification.get("schema_version") != 1
            or qualification.get("controller") != "v2-full"
            or not isinstance(qualification.get("run_fingerprint"), str)
            or not qualification["run_fingerprint"]
            or not isinstance(evidence, dict)
            or evidence.get("passed") is not True
            or evidence.get("errors") != []
        ):
            raise ValueError("reused v3-S3 qualification failed validation")
    return value


def _validate_reused_source_replay_audit(
    audit: Any,
    *,
    source_root: Path,
) -> dict[str, Any]:
    if not isinstance(audit, dict):
        raise ValueError("reused v3-S3 replay audit is not an object")
    value = dict(audit)
    if (
        value.get("schema") != V3_S3_SOURCE_REPLAY_AUDIT_SCHEMA
        or value.get("passed") is not True
    ):
        raise ValueError("reused v3-S3 replay audit is invalid")
    rows = source_decisions(source_roots(source_root))
    input_identity = source_replay_input_identity(rows)
    if (
        value.get("input_identity") != input_identity
        or value.get("input_sha256") != _fingerprint(input_identity)
        or value.get("source_state_count") != len(rows)
        or value.get("matched_decision_state_count") != len(rows)
    ):
        raise ValueError("reused v3-S3 replay audit input identity mismatch")
    episode_count = len(
        {
            (str(row["source_run_fingerprint"]), str(row["episode_id"]))
            for row in rows
        }
    )
    policy_counts = dict(
        sorted(
            collections.Counter(
                str(row["source_policy"]) for row in rows
            ).items()
        )
    )
    if (
        value.get("episode_count") != episode_count
        or value.get("matched_by_source_policy") != policy_counts
        or value.get("rejected_episode_count") != 0
        or value.get("rejections") != []
        or value.get("prefix_mismatch_count") != 0
        or value.get("prefix_mismatches") != []
        or value.get("terminal_after_mismatch_count") != 0
        or value.get("terminal_after_mismatches") != []
    ):
        raise ValueError("reused v3-S3 replay audit evidence is inconsistent")
    return value


def collect_v3_s3_sources(
    *,
    project_root: Path,
    dataset: Path,
    output: Path,
    controller_bundle: Path,
    workers: int,
    resume: bool,
    started_at: str,
) -> dict[str, Any]:
    partition, design = _source_task_partition(dataset)
    _write_json(output / "source_design.json", design)
    reports: dict[str, Any] = {}
    qualification_reports: dict[str, Any] = {}
    for split in ("policy_train", "policy_validation"):
        split_tasks = sorted(
            {
                task_id
                for policy in S3_SOURCE_POLICIES
                for task_id in partition[(split, policy)]
            }
        )
        qualification_root = output / "sources" / split / "_qualification"
        qualification_config = _source_config(
            project_root=project_root,
            dataset=dataset,
            split=split,
            policy="qualification",
            output=output,
            workers=workers,
        )
        _write_status(
            output,
            started_at=started_at,
            status="running",
            phase=f"source-{split}-qualification",
            completed_sources=len(reports),
            total_sources=6,
            error_states=0,
        )
        qualification_reports[split] = run_closed_loop_collection(
            dataset,
            qualification_config,
            qualification_root,
            phase="qualify",
            resume=resume or (qualification_root / "run_config.json").is_file(),
            workers=int(workers),
            task_ids=split_tasks,
            controller="v2-full",
            feature_backend="native",
            controller_bundle=controller_bundle,
            controller_runtime="optimized",
            verification_profile="deployment",
            stopping_rule="historical",
        )
        for policy in S3_SOURCE_POLICIES:
            tasks = partition[(split, policy)]
            root = output / "sources" / split / policy
            config = _source_config(
                project_root=project_root,
                dataset=dataset,
                split=split,
                policy=policy,
                output=output,
                workers=workers,
            )
            _write_status(
                output,
                started_at=started_at,
                status="running",
                phase=f"source-{split}-{policy}",
                completed_sources=len(reports),
                total_sources=6,
                error_states=0,
            )
            common = {
                "workers": int(workers),
                "task_ids": tasks,
                "controller": "v2-full",
                "feature_backend": "native",
                "controller_bundle": controller_bundle,
                "controller_runtime": "optimized",
                "verification_profile": "deployment",
                "stopping_rule": "historical",
                "qualification_source": qualification_root,
            }
            run_closed_loop_collection(
                dataset,
                config,
                root,
                phase="qualify",
                resume=resume or (root / "run_config.json").is_file(),
                **common,
            )
            reports[f"{split}|{policy}"] = run_closed_loop_collection(
                dataset,
                config,
                root,
                phase=policy,
                resume=True,
                **common,
            )
    report = {
        "schema": V3_S3_PIPELINE_SCHEMA,
        "complete": all(
            int(dict(value.get(source_key.split("|", 1)[1], {})).get("error_count", 0))
            == 0
            and int(
                dict(value.get(source_key.split("|", 1)[1], {})).get(
                    "episode_count", -1
                )
            )
            == int(design["by_source"][source_key])
            for source_key, value in reports.items()
        )
        and set(reports) == set(design["by_source"])
        and all(
            dict(value.get("qualification", {})).get("passed") is True
            for value in qualification_reports.values()
        ),
        "design": design,
        "qualifications": qualification_reports,
        "reports": reports,
    }
    _write_json(output / "source_report.json", report)
    return report


def _pipeline_identity(
    *,
    project_root: Path,
    dataset_config: Path,
    controller_bundle: Path,
    workers: str,
    parallelism_audit: bool,
    reuse_source_output: Path | None = None,
) -> dict[str, Any]:
    identity = {
        "runner": "run_v3_training_pipeline.sequence-pilot",
        "schema_version": 2,
        "mode": "sequence-pilot",
        "dataset_config": str(dataset_config),
        "dataset_config_sha256": sha256_file(dataset_config),
        "controller_bundle": str(controller_bundle),
        "controller_manifest_sha256": sha256_file(
            controller_bundle / "controller_manifest.json"
        ),
        "workers": str(workers),
        "parallelism_audit": bool(parallelism_audit),
        "producer_identity": producer_identity(
            project_root=project_root,
            source_files=V3_S3_PIPELINE_PRODUCER_FILES,
            native_required=True,
            optional_package_names=("joblib", "numpy", "scikit-learn"),
        ),
        "automatic_full": False,
        "automatic_quick": False,
        "automatic_formal": False,
    }
    if reuse_source_output is not None:
        source_root = Path(reuse_source_output).resolve()
        source_report = source_root / "source_report.json"
        if not source_report.is_file():
            raise FileNotFoundError(
                f"reused v3-S3 source report does not exist: {source_report}"
            )
        _validate_reused_source_report(_read_json(source_report))
        replay_audit = source_root / "source_replay_audit.json"
        if not replay_audit.is_file():
            raise ValueError(
                "reused v3-S3 sources require a passed source_replay_audit.json"
            )
        _validate_reused_source_replay_audit(
            _read_json(replay_audit),
            source_root=source_root,
        )
        identity.update(
            {
                "reuse_source_output": str(source_root),
                "reuse_source_report_sha256": sha256_file(source_report),
                "reuse_source_replay_audit_sha256": sha256_file(replay_audit),
            }
        )
    else:
        identity["reuse_source_output"] = None
    return identity


def _parse_workers(workers: str | int, *, fallback: int = 4) -> tuple[bool, int]:
    text = str(workers).strip().lower()
    if text == "auto":
        values = candidate_lane_counts()
        return True, min(fallback, max(values))
    value = int(text)
    if value <= 0:
        raise ValueError("v3-S3 workers must be positive or 'auto'")
    return False, value


def run_v3_s3_collection_stage(
    *,
    project_root: Path,
    output: Path,
    controller_bundle: Path,
    dataset_config: Path,
    workers: str | int,
    resume: bool,
    parallelism_audit: bool,
    stop_after_sources: bool = False,
    reuse_source_output: Path | None = None,
) -> dict[str, Any]:
    automatic, source_workers = _parse_workers(workers)
    identity = _pipeline_identity(
        project_root=project_root,
        dataset_config=dataset_config,
        controller_bundle=controller_bundle,
        workers=str(workers),
        parallelism_audit=parallelism_audit,
        reuse_source_output=reuse_source_output,
    )
    prepare_run_output(output, resume=resume, identity=identity)
    started_at = str(
        (_read_json(output / "status.json").get("started_at") if (output / "status.json").is_file() else None)
        or _utc_now()
    )
    reused_source_root = (
        Path(reuse_source_output).resolve()
        if reuse_source_output is not None
        else None
    )
    dataset = (
        reused_source_root / "dataset"
        if reused_source_root is not None
        else output / "dataset"
    )
    if reused_source_root is None and not (dataset / "dataset_summary.json").is_file():
        _write_status(
            output,
            started_at=started_at,
            status="running",
            phase="dataset",
            completed_states=0,
            total_states=None,
            target_state_cap=S3_TARGET_STATE_CAP,
            error_states=0,
        )
        try:
            generate_dataset(load_json(dataset_config), dataset)
        except BaseException as error:
            _write_stage_error(
                output,
                started_at=started_at,
                phase="dataset",
                error=error,
                completed_states=0,
                total_states=None,
                target_state_cap=S3_TARGET_STATE_CAP,
            )
            raise
    if reused_source_root is not None:
        source_report_path = reused_source_root / "source_report.json"
        sources = _read_json(source_report_path)
        _write_json(
            output / "source_reuse_report.json",
            {
                "schema": V3_S3_PIPELINE_SCHEMA,
                "source_output": str(reused_source_root),
                "source_report": str(source_report_path),
                "source_report_sha256": sha256_file(source_report_path),
                "source_replay_audit": str(
                    reused_source_root / "source_replay_audit.json"
                ),
                "source_replay_audit_sha256": sha256_file(
                    reused_source_root / "source_replay_audit.json"
                ),
                "complete": sources.get("complete") is True,
            },
        )
    else:
        source_report_path = output / "source_report.json"
    if reused_source_root is None and (
        not source_report_path.is_file()
        or _read_json(source_report_path).get("complete") is not True
    ):
        try:
            sources = collect_v3_s3_sources(
                project_root=project_root,
                dataset=dataset,
                output=output,
                controller_bundle=controller_bundle,
                workers=source_workers,
                resume=resume,
                started_at=started_at,
            )
        except BaseException as error:
            previous_status = (
                _read_json(output / "status.json")
                if (output / "status.json").is_file()
                else {}
            )
            _write_stage_error(
                output,
                started_at=started_at,
                phase=str(previous_status.get("phase") or "source"),
                error=error,
                completed_sources=int(previous_status.get("completed_sources", 0)),
                total_sources=6,
            )
            raise
        if sources.get("complete") is not True:
            error = RuntimeError("v3-S3 source collection completed with errors")
            _write_stage_error(
                output,
                started_at=started_at,
                phase="source-collection",
                error=error,
                completed_sources=0,
                total_sources=6,
            )
            raise error
    elif reused_source_root is None:
        sources = _read_json(source_report_path)
    if stop_after_sources:
        report = {
            "schema": V3_S3_PIPELINE_SCHEMA,
            "complete": True,
            "stage": "sources",
            "sources": sources,
            "next_stage": "collection",
        }
        _write_status(
            output,
            started_at=started_at,
            status="waiting",
            phase="awaiting-sequence-collection",
            completed_sources=6,
            total_sources=6,
            error_states=0,
        )
        return report

    source_base = reused_source_root if reused_source_root is not None else output
    roots = source_roots(source_base)
    if automatic and parallelism_audit:
        _write_status(
            output,
            started_at=started_at,
            status="running",
            phase="parallelism-audit",
            completed_states=0,
            total_states=None,
            target_state_cap=S3_TARGET_STATE_CAP,
            error_states=0,
        )
        try:
            # The audit validates its own source/controller/implementation
            # fingerprint before reusing a prior report.
            audit = audit_v3_s3_parallelism(
                source_roots=roots,
                output=output / "parallelism_audit",
                controller_bundle=controller_bundle,
            )
        except BaseException as error:
            _write_stage_error(
                output,
                started_at=started_at,
                phase="parallelism-audit",
                error=error,
                completed_states=0,
                total_states=None,
                target_state_cap=S3_TARGET_STATE_CAP,
            )
            raise
        collection_workers = int(audit["selected_lanes"])
    else:
        audit = {
            "schema": "lns2.training_parallelism_audit.v1",
            "selected_lanes": int(source_workers),
            "skipped": not parallelism_audit,
        }
        collection_workers = int(source_workers)
    _write_status(
        output,
        started_at=started_at,
        status="running",
        phase="sequence-collection",
        completed_states=0,
        total_states=None,
        target_state_cap=S3_TARGET_STATE_CAP,
        error_states=0,
        progress_file=str(output / "collection" / "status.json"),
        selected_lanes=collection_workers,
    )
    try:
        collection = collect_v3_s3_data(
            source_roots=roots,
            output=output / "collection",
            controller_bundle=controller_bundle,
            workers=collection_workers,
            resume=resume or (output / "collection" / "run_config.json").is_file(),
        )
    except BaseException as error:
        _write_stage_error(
            output,
            started_at=started_at,
            phase="sequence-collection",
            error=error,
            completed_states=0,
            total_states=None,
            target_state_cap=S3_TARGET_STATE_CAP,
            selected_lanes=collection_workers,
        )
        raise
    if collection.get("complete") is not True:
        error = RuntimeError("v3-S3 sequence collection did not complete")
        _write_stage_error(
            output,
            started_at=started_at,
            phase="sequence-collection",
            error=error,
            completed_states=int(collection.get("completed_state_count", 0)),
            total_states=int(collection.get("requested_state_count", 0)),
            selected_lanes=collection_workers,
        )
        raise error
    report = {
        "schema": V3_S3_PIPELINE_SCHEMA,
        "complete": True,
        "stage": "collection",
        "dataset": str(dataset),
        "sources": sources,
        "parallelism_audit": audit,
        "collection": collection,
        "next_stage": "windows-training",
        "full_started": False,
        "quick_started": False,
        "formal_started": False,
    }
    _write_json(output / "collection_stage_report.json", report)
    _write_status(
        output,
        started_at=started_at,
        status="waiting",
        phase="awaiting-windows-training",
        completed_states=int(collection["completed_state_count"]),
        total_states=int(collection["requested_state_count"]),
        error_states=int(collection["error_state_count"]),
        selected_lanes=collection_workers,
    )
    return report


def _training_jobs(value: str | int) -> int:
    text = str(value).strip().lower()
    if text == "auto":
        return max(1, min(10, (os.cpu_count() or 1) // 2))
    jobs = int(text)
    if jobs <= 0:
        raise ValueError("training jobs must be positive or 'auto'")
    return jobs


def _collection_root(
    output: Path, collection_source: Path | None = None
) -> Path:
    if collection_source is not None:
        return Path(collection_source).resolve()
    local = output / "collection"
    if local.is_dir():
        return local.resolve()
    reference_path = output / "collection_reference.json"
    if not reference_path.is_file():
        return local.resolve()
    reference = _read_json(reference_path)
    relative = reference.get("collection_root_relative")
    root = (
        Path(__file__).resolve().parents[1] / str(relative)
        if relative
        else Path(str(reference["collection_root"]))
    ).resolve()
    report_path = root / "collection_report.json"
    if sha256_file(report_path) != str(
        reference["collection_report_sha256"]
    ):
        raise ValueError("v3-S3 referenced collection SHA256 mismatch")
    return root


def _verified_training_inputs(
    output: Path, collection_source: Path | None = None
) -> dict[str, Any]:
    collection_root = _collection_root(output, collection_source)
    report_path = collection_root / "collection_report.json"
    report = _read_json(report_path)
    completed = _completed_collection_state_count(report)
    specifications = {
        "sequence_features": (
            collection_root / "sequence_features.jsonl",
            str(report.get("sequence_features_sha256") or ""),
        ),
        "sequence_trials": (
            collection_root / "sequence_trials.jsonl",
            str(report.get("sequence_trials_sha256") or ""),
        ),
        "external_baselines": (
            collection_root / "external_baselines.jsonl",
            str(report.get("external_baselines_sha256") or ""),
        ),
    }
    artifacts = {}
    for name, (path, expected_sha256) in specifications.items():
        if not path.is_file():
            raise FileNotFoundError(f"v3-S3 training input is missing: {path}")
        actual_sha256 = sha256_file(path)
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise ValueError(
                f"v3-S3 training input SHA256 mismatch: {name}; "
                f"expected={expected_sha256!r}, actual={actual_sha256!r}"
            )
        artifacts[name] = {
            "file": str(path.resolve()),
            "sha256": actual_sha256,
            "size_bytes": path.stat().st_size,
        }
    return {
        "collection_report": {
            "file": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
        },
        "collection_state_count": completed,
        "artifacts": artifacts,
    }


def _training_identity(
    *,
    project_root: Path,
    output: Path,
    jobs: int,
    collection_source: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _verified_training_inputs(output, collection_source)
    training_producer_identity = v3_s3_training_producer_identity(project_root)
    identity = {
        "runner": "run_v3_training_pipeline.sequence-pilot.train",
        "schema_version": 2,
        "feature_schema_sha256": V3_S3_FEATURE_SCHEMA_SHA256,
        "training_objective_id": V3_S3_OBJECTIVE_ID,
        "selection_objective_id": V3_S3_SELECTION_OBJECTIVE_ID,
        "training_jobs": int(jobs),
        "model_parameters": {
            "hist_gradient_boosting": HGB_PARAMETERS,
            "extra_trees": EXTRA_TREES_PARAMETERS,
        },
        "inputs": inputs,
        "producer_identity": training_producer_identity,
        "automatic_native_audit": False,
        "automatic_full": False,
        "automatic_quick": False,
        "automatic_formal": False,
    }
    return identity, inputs


def _completed_collection_state_count(collection: dict[str, Any]) -> int:
    requested = int(collection.get("requested_state_count", 0))
    completed = int(collection.get("completed_state_count", 0))
    if collection.get("complete") is not True or requested <= 0 or completed != requested:
        raise ValueError("v3-S3 requires a complete adaptive state collection")
    selected_by_split = dict(
        dict(collection.get("selection") or {}).get("selected_by_split") or {}
    )
    shortages = {
        split: {
            "minimum": int(minimum),
            "selected": int(selected_by_split.get(split, 0)),
        }
        for split, minimum in S3_SPLIT_MINIMUM_COUNTS.items()
        if int(selected_by_split.get(split, 0)) < int(minimum)
    }
    if shortages:
        raise ValueError(
            f"v3-S3 adaptive collection is below split minimums: {shortages}"
        )
    return completed


def _json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pre_native_training_stage_report(
    *, report: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    if report.get("native_audit_completed") is not True:
        return {**report, "manifest": manifest}
    provisional = str(report["provisional_model_family"])
    diagnostic = dict(dict(report["model_family_diagnostics"])[provisional])
    pre_report = json.loads(json.dumps(report))
    for key in (
        "selected_model_family",
        "native_latency_seconds_per_sequence",
        "native_benchmark_sequence_count",
        "native_parity_probe_count",
        "sequence_row_construction_seconds",
        "planner_seconds_per_new_state",
        "native_parity",
        "native_audit_identity",
        "native_audit_identity_fingerprint",
        "pilot_passed",
    ):
        pre_report.pop(key, None)
    pre_report.update(
        {
            "thresholds": dict(diagnostic["thresholds"]),
            "prediction_intervals": dict(diagnostic["prediction_intervals"]),
            "continuation_calibration": dict(
                diagnostic["continuation_calibration"]
            ),
            "diagnostic": {
                "v3_s3": dict(diagnostic["v3_s3"]),
                "v2_full": dict(report["diagnostic"])["v2_full"],
                "official_adaptive": dict(report["diagnostic"])[
                    "official_adaptive"
                ],
                "continuation_possible_count": int(
                    diagnostic["continuation_possible_count"]
                ),
                "continuation_reused_count": int(
                    diagnostic["continuation_reused_count"]
                ),
                "continuation_reuse_fraction": float(
                    diagnostic["continuation_reuse_fraction"]
                ),
            },
            "pilot_checks": dict(diagnostic["pilot_checks"]),
            "native_audit_completed": False,
            "decision": "awaiting_v3_s3_native_audit",
        }
    )
    pre_manifest = json.loads(json.dumps(manifest))
    for key in (
        "native_audit_identity",
        "native_audit_identity_fingerprint",
    ):
        pre_manifest.pop(key, None)
    pre_manifest.update(
        {
            "models": {
                name: dict(raw)
                for name, raw in dict(
                    dict(report["model_exports"])[provisional]["models"]
                ).items()
            },
            "model_family": provisional,
            "thresholds": dict(diagnostic["thresholds"]),
            "prediction_intervals": dict(diagnostic["prediction_intervals"]),
            "continuation_calibration": dict(
                diagnostic["continuation_calibration"]
            ),
            "native_audit_completed": False,
        }
    )
    pre_manifest["training_report"] = {
        "file": str(dict(manifest["training_report"])["file"]),
        "sha256": _json_sha256(pre_report),
    }
    return {**pre_report, "manifest": pre_manifest}


def run_v3_s3_training_stage(
    *,
    project_root: Path,
    output: Path,
    training_jobs: str | int,
    resume: bool,
    collection_source: Path | None = None,
) -> dict[str, Any]:
    jobs = _training_jobs(training_jobs)
    identity, inputs = _training_identity(
        project_root=project_root,
        output=output,
        jobs=jobs,
        collection_source=collection_source,
    )
    prepare_run_output(
        output / "training-control",
        resume=resume,
        identity=identity,
    )
    if collection_source is not None:
        collection_report = dict(inputs["collection_report"])
        resolved_collection = Path(collection_source).resolve()
        try:
            collection_relative = resolved_collection.relative_to(
                project_root.resolve()
            ).as_posix()
        except ValueError:
            collection_relative = None
        _write_json(
            output / "collection_reference.json",
            {
                "schema": "lns2.v3_s3_collection_reference.v1",
                "collection_root": str(resolved_collection),
                "collection_root_relative": collection_relative,
                "collection_report_sha256": str(
                    collection_report["sha256"]
                ),
            },
        )
    collection_state_count = int(inputs["collection_state_count"])
    controller = output / "controller"
    partial_controller = output / "controller.partial"
    status = (
        _read_json(output / "status.json")
        if (output / "status.json").is_file()
        else {}
    )
    started_at = str(status.get("started_at") or _utc_now())
    if controller.exists():
        if not resume:
            raise ValueError(
                "v3-S3 controller output already exists; pass --resume only "
                "when its training identity is unchanged"
            )
        bundle = load_v3_s3_bundle(controller)
        report_path = output / "training_stage_report.json"
        if not report_path.is_file():
            raise ValueError(
                "v3-S3 controller exists without training_stage_report.json"
            )
        training = _read_json(report_path)
        expected_training = {**bundle.report, "manifest": bundle.manifest}
        if training != expected_training:
            recoverable_pre_native = _pre_native_training_stage_report(
                report=bundle.report,
                manifest=bundle.manifest,
            )
            if training != recoverable_pre_native:
                raise ValueError(
                    "completed v3-S3 training_stage_report.json failed "
                    "semantic validation"
                )
            _write_json(report_path, expected_training)
            training = expected_training
        if (
            str(training.get("schema"))
            != V3_S3_TRAINING_SCHEMA
            or training.get("producer_identity")
            != identity["producer_identity"]
        ):
            raise ValueError(
                "completed v3-S3 training artifact belongs to a legacy or "
                "different producer"
            )
        if training.get("native_audit_completed") is not True:
            _write_status(
                output,
                started_at=started_at,
                status="waiting",
                phase="awaiting-native-audit",
                completed_states=collection_state_count,
                total_states=collection_state_count,
                error_states=0,
                training_jobs=jobs,
                provisional_model_family=training[
                    "provisional_model_family"
                ],
            )
        return training
    if partial_controller.exists():
        suffix = _utc_now().replace(":", "").replace("+", "_")
        partial_controller.replace(
            output / f"controller.interrupted-{suffix}"
        )
    _write_status(
        output,
        started_at=started_at,
        status="running",
        phase="windows-training",
        completed_states=collection_state_count,
        total_states=collection_state_count,
        error_states=0,
        training_jobs=jobs,
    )
    try:
        training = train_v3_s3_controller(
            sequence_features=Path(
                inputs["artifacts"]["sequence_features"]["file"]
            ),
            sequence_trials=Path(
                inputs["artifacts"]["sequence_trials"]["file"]
            ),
            external_baselines=Path(
                inputs["artifacts"]["external_baselines"]["file"]
            ),
            output=partial_controller,
            training_jobs=jobs,
            producer_identity_payload=identity["producer_identity"],
        )
        load_v3_s3_bundle(partial_controller)
        partial_controller.replace(controller)
    except BaseException as error:
        _write_stage_error(
            output,
            started_at=started_at,
            phase="windows-training",
            error=error,
            completed_states=collection_state_count,
            total_states=collection_state_count,
            training_jobs=jobs,
        )
        raise
    _write_json(output / "training_stage_report.json", training)
    _write_status(
        output,
        started_at=started_at,
        status="waiting",
        phase="awaiting-native-audit",
        completed_states=collection_state_count,
        total_states=collection_state_count,
        error_states=0,
        training_jobs=jobs,
        provisional_model_family=training["provisional_model_family"],
    )
    return training


def _markdown(report: dict[str, Any]) -> str:
    training = dict(report["training"])
    diagnostic = dict(training["diagnostic"])
    s3 = dict(diagnostic["v3_s3"])
    v2 = dict(diagnostic["v2_full"])
    adaptive = dict(diagnostic["official_adaptive"])
    return "\n".join(
        [
            "# Independent v3-S3 mixed-load pilot",
            "",
            f"Decision: `{training['decision']}`",
            "",
            "This pilot does not promote a deployment model and did not start full, quick, or formal evaluation.",
            "",
            "| Controller | Effective rate | No-progress rate | Mean reduction | Reduction / full second |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| v3-S3 | {float(s3['effective_rate']):.3%} | {float(s3['no_progress_rate']):.3%} | {float(s3['mean_conflict_reduction']):.4f} | {float(s3['conflict_reduction_per_total_second']):.4f} |",
            f"| v2-full external baseline | {float(v2['effective_rate']):.3%} | {float(v2['no_progress_rate']):.3%} | {float(v2['mean_conflict_reduction']):.4f} | {float(v2['conflict_reduction_per_total_second']):.4f} |",
            f"| Adaptive external baseline | {float(adaptive['effective_rate']):.3%} | {float(adaptive['no_progress_rate']):.3%} | {float(adaptive['mean_conflict_reduction']):.4f} | {float(adaptive['conflict_reduction_per_total_second']):.4f} |",
            "",
            f"- Selected model family: `{training['selected_model_family']}`",
            f"- Declared features: {training['declared_feature_count']}",
            f"- Direct continuation fraction: {float(diagnostic['continuation_reuse_fraction']):.3%}",
            f"- Runtime v2 calls: {training['manifest']['v2_runtime_call_count']}",
            f"- Runtime Adaptive calls: {training['manifest']['adaptive_runtime_call_count']}",
            "",
            "## Gate",
            "",
            "```json",
            json.dumps(training["pilot_checks"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _completed_native_stage_is_valid(
    *, output: Path, expected: dict[str, Any]
) -> bool:
    report_path = output / "v3_s3_pilot_report.json"
    if not report_path.is_file():
        return False
    existing = _read_json(report_path)
    if existing.get("complete") is not True:
        return False
    if existing != expected:
        raise ValueError(
            "completed v3-S3 native-audit report failed semantic validation"
        )
    markdown_path = output / "v3_s3_pilot_report.md"
    if not markdown_path.is_file():
        return False
    if markdown_path.read_text(encoding="utf-8") != _markdown(expected):
        raise ValueError(
            "completed v3-S3 native-audit Markdown failed validation"
        )
    status_path = output / "status.json"
    if not status_path.is_file():
        return False
    status = _read_json(status_path)
    if (
        str(status.get("schema")) != V3_S3_PIPELINE_SCHEMA
        or str(status.get("status")) != "complete"
        or str(status.get("phase")) != "complete"
        or str(status.get("decision")) != str(expected["decision"])
    ):
        raise ValueError(
            "completed v3-S3 native-audit status failed validation"
        )
    return True


def run_v3_s3_native_audit_stage(
    *,
    output: Path,
    resume: bool = False,
    benchmark_rows: int = 36,
) -> dict[str, Any]:
    collection = _read_json(
        _collection_root(output) / "collection_report.json"
    )
    collection_state_count = _completed_collection_state_count(collection)
    # Validate the complete Windows artifact before mutating the shared status.
    bundle = load_v3_s3_bundle(output / "controller")
    training_stage_path = output / "training_stage_report.json"
    if not training_stage_path.is_file():
        raise FileNotFoundError(
            "v3-S3 native audit requires training_stage_report.json"
        )
    stage_training = _read_json(training_stage_path)
    expected_stage_training = {**bundle.report, "manifest": bundle.manifest}
    if stage_training != expected_stage_training and stage_training != (
        _pre_native_training_stage_report(
            report=bundle.report,
            manifest=bundle.manifest,
        )
    ):
        raise ValueError(
            "v3-S3 native audit found an invalid training_stage_report.json"
        )
    audit_identity = v3_s3_native_audit_identity(
        controller_output=output / "controller",
        benchmark_rows=benchmark_rows,
    )
    prepare_run_output(
        output / "native-audit-control",
        resume=resume,
        identity={
            "runner": "run_v3_training_pipeline.sequence-pilot.native-audit",
            "schema_version": 1,
            "native_audit_identity": audit_identity,
        },
    )
    controller_report = _read_json(output / "controller" / "training_report.json")
    native_audit_already_completed = (
        controller_report.get("native_audit_completed") is True
    )
    status = _read_json(output / "status.json")
    started_at = str(status.get("started_at") or _utc_now())
    if not native_audit_already_completed:
        _write_status(
            output,
            started_at=started_at,
            status="running",
            phase="native-audit",
            completed_states=collection_state_count,
            total_states=collection_state_count,
            error_states=0,
        )
    try:
        training = finalize_v3_s3_native_audit(
            controller_output=output / "controller",
            benchmark_rows=benchmark_rows,
            audit_identity=audit_identity,
        )
    except BaseException as error:
        if not native_audit_already_completed:
            _write_stage_error(
                output,
                started_at=started_at,
                phase="native-audit",
                error=error,
                completed_states=collection_state_count,
                total_states=collection_state_count,
            )
        raise
    if _read_json(training_stage_path) != training:
        _write_json(training_stage_path, training)
    report = {
        "schema": V3_S3_PIPELINE_SCHEMA,
        "complete": True,
        "mode": "sequence-pilot",
        "collection": collection,
        "training": training,
        "decision": training["decision"],
        "deployment_promoted": False,
        "full_started": False,
        "quick_started": False,
        "formal_started": False,
    }
    if native_audit_already_completed and _completed_native_stage_is_valid(
        output=output, expected=report
    ):
        return report
    _write_json(output / "v3_s3_pilot_report.json", report)
    (output / "v3_s3_pilot_report.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    _write_status(
        output,
        started_at=started_at,
        status="complete",
        phase="complete",
        completed_states=collection_state_count,
        total_states=collection_state_count,
        error_states=0,
        decision=training["decision"],
        report=str(output / "v3_s3_pilot_report.md"),
        automatic_followup=False,
    )
    return report


__all__ = [
    "V3_S3_PIPELINE_SCHEMA",
    "V3_S3_SOURCE_REPLAY_AUDIT_SCHEMA",
    "collect_v3_s3_sources",
    "run_v3_s3_collection_stage",
    "run_v3_s3_native_audit_stage",
    "run_v3_s3_training_stage",
    "source_replay_input_identity",
    "source_roots",
]
