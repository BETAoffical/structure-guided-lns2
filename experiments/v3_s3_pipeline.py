from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.parallel_runtime import candidate_lane_counts
from experiments.repair_collection import _read_json, _read_jsonl, _utc_now, _write_json
from experiments.run_output_guard import prepare_run_output
from experiments.v3_s3_collection import (
    S3_AGENT_COUNTS,
    S3_LAYOUTS,
    S3_SPLIT_MINIMUM_COUNTS,
    S3_TARGET_STATE_CAP,
    audit_v3_s3_parallelism,
    collect_v3_s3_data,
)
from experiments.v3_s3_training import (
    EXTRA_TREES_PARAMETERS,
    HGB_PARAMETERS,
    finalize_v3_s3_native_audit,
    train_v3_s3_controller,
)
from experiments.v3_s3 import (
    V3_S3_FEATURE_SCHEMA_SHA256,
    V3_S3_OBJECTIVE_ID,
    load_v3_s3_bundle,
)
from generators.config import load_json
from generators.dataset import generate_dataset


V3_S3_PIPELINE_SCHEMA = "lns2.v3_s3_pipeline.v1"
S3_SOURCE_POLICIES = ("fixed_random", "official_adaptive", "realized_dynamic")


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
            bool(dict(value.get("qualification", {})).get("passed", False))
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
    implementation_files = (
        "experiments/trace_replay.py",
        "experiments/v3_s3.py",
        "experiments/v3_s3_collection.py",
        "experiments/v3_s3_pipeline.py",
        "experiments/parallel_runtime.py",
        "experiments/closed_loop_confirmation.py",
        "experiments/online_feature_engine.py",
        "generators/dataset.py",
        "generators/task_flows.py",
        "scripts/run_v3_training_pipeline.py",
        "src/python_bindings.cpp",
        "third_party/mapf_lns2/inc/RepairPolicy.h",
        "third_party/mapf_lns2/src/InitLNS.cpp",
    )
    identity = {
        "runner": "run_v3_training_pipeline.sequence-pilot",
        "schema_version": 1,
        "mode": "sequence-pilot",
        "dataset_config": str(dataset_config),
        "dataset_config_sha256": sha256_file(dataset_config),
        "controller_bundle": str(controller_bundle),
        "controller_manifest_sha256": sha256_file(
            controller_bundle / "controller_manifest.json"
        ),
        "workers": str(workers),
        "parallelism_audit": bool(parallelism_audit),
        "implementation": {
            name: sha256_file(project_root / name) for name in implementation_files
        },
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
        if not bool(_read_json(source_report).get("complete")):
            raise ValueError("reused v3-S3 source report is incomplete")
        replay_audit = source_root / "source_replay_audit.json"
        if not replay_audit.is_file() or not bool(_read_json(replay_audit).get("passed")):
            raise ValueError(
                "reused v3-S3 sources require a passed source_replay_audit.json"
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
                "complete": bool(sources.get("complete")),
            },
        )
    else:
        source_report_path = output / "source_report.json"
    if reused_source_root is None and (
        not source_report_path.is_file()
        or not bool(_read_json(source_report_path).get("complete"))
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
        if not bool(sources["complete"]):
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
    if not bool(collection["complete"]):
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


def _verified_training_inputs(output: Path) -> dict[str, Any]:
    collection_root = output / "collection"
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _verified_training_inputs(output)
    implementation_files = (
        "experiments/repair_aware.py",
        "experiments/v3_s3.py",
        "experiments/v3_s3_training.py",
    )
    identity = {
        "runner": "run_v3_training_pipeline.sequence-pilot.train",
        "schema_version": 1,
        "feature_schema_sha256": V3_S3_FEATURE_SCHEMA_SHA256,
        "training_objective_id": V3_S3_OBJECTIVE_ID,
        "training_jobs": int(jobs),
        "model_parameters": {
            "hist_gradient_boosting": HGB_PARAMETERS,
            "extra_trees": EXTRA_TREES_PARAMETERS,
        },
        "inputs": inputs,
        "implementation": {
            name: sha256_file(project_root / name)
            for name in implementation_files
        },
        "automatic_native_audit": False,
        "automatic_full": False,
        "automatic_quick": False,
        "automatic_formal": False,
    }
    return identity, inputs


def _completed_collection_state_count(collection: dict[str, Any]) -> int:
    requested = int(collection.get("requested_state_count", 0))
    completed = int(collection.get("completed_state_count", 0))
    if not bool(collection.get("complete")) or requested <= 0 or completed != requested:
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


def run_v3_s3_training_stage(
    *,
    project_root: Path,
    output: Path,
    training_jobs: str | int,
    resume: bool,
) -> dict[str, Any]:
    jobs = _training_jobs(training_jobs)
    identity, inputs = _training_identity(
        project_root=project_root,
        output=output,
        jobs=jobs,
    )
    prepare_run_output(
        output / "training-control",
        resume=resume,
        identity=identity,
    )
    collection_state_count = int(inputs["collection_state_count"])
    controller = output / "controller"
    partial_controller = output / "controller.partial"
    status = _read_json(output / "status.json")
    started_at = str(status.get("started_at") or _utc_now())
    if controller.exists():
        if not resume:
            raise ValueError(
                "v3-S3 controller output already exists; pass --resume only "
                "when its training identity is unchanged"
            )
        load_v3_s3_bundle(controller)
        report_path = output / "training_stage_report.json"
        if not report_path.is_file():
            raise ValueError(
                "v3-S3 controller exists without training_stage_report.json"
            )
        training = _read_json(report_path)
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
            sequence_features=output / "collection" / "sequence_features.jsonl",
            sequence_trials=output / "collection" / "sequence_trials.jsonl",
            external_baselines=output / "collection" / "external_baselines.jsonl",
            output=partial_controller,
            training_jobs=jobs,
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


def run_v3_s3_native_audit_stage(*, output: Path) -> dict[str, Any]:
    collection = _read_json(output / "collection" / "collection_report.json")
    collection_state_count = _completed_collection_state_count(collection)
    # Validate the complete Windows artifact before mutating the shared status.
    load_v3_s3_bundle(output / "controller")
    if not (output / "training_stage_report.json").is_file():
        raise FileNotFoundError(
            "v3-S3 native audit requires training_stage_report.json"
        )
    status = _read_json(output / "status.json")
    started_at = str(status.get("started_at") or _utc_now())
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
            controller_output=output / "controller"
        )
    except BaseException as error:
        _write_stage_error(
            output,
            started_at=started_at,
            phase="native-audit",
            error=error,
            completed_states=collection_state_count,
            total_states=collection_state_count,
        )
        raise
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
    "collect_v3_s3_sources",
    "run_v3_s3_collection_stage",
    "run_v3_s3_native_audit_stage",
    "run_v3_s3_training_stage",
    "source_roots",
]
