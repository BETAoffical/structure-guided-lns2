from __future__ import annotations

import collections
import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.high_load_rescue import select_failure_decisions
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.rescue_lite_confirmation import (
    AGENT_COUNTS,
    LAYOUTS,
    _run_source_wave,
    build_confirmation_source_config,
    dataset_map_hashes,
)
from experiments.trace_replay import decision_rows
from generators.config import load_json
from generators.dataset import generate_dataset


RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA = (
    "lns2.rescue_confirmation_qualification.v1"
)
QUALIFICATION_SPLIT = "policy_qualification"
DEFAULT_MAX_DECISIONS = 40
DEFAULT_WALL_TIME_SECONDS = 180.0
DEFAULT_MAX_STATES_PER_TASK = 2
DEFAULT_MINIMUM_STATES_PER_CELL = 5
DEFAULT_MINIMUM_TASKS_PER_CELL = 3
DEFAULT_MINIMUM_MAPS_PER_CELL = 3


def _cell(layout: str, agent_count: int) -> str:
    if layout not in LAYOUTS or int(agent_count) not in AGENT_COUNTS:
        raise ValueError(f"unexpected qualification cell: {layout}/{agent_count}")
    return f"{layout}__agents_{int(agent_count)}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("qualification CSV cannot be empty")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_status(root: Path, *, phase: str, status: str, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {
            "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
            "phase": phase,
            "status": status,
            **values,
        },
    )


def validate_qualification_isolation(
    dataset: str | Path,
    references: Iterable[str | Path],
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    current = dataset_map_hashes(dataset_root, QUALIFICATION_SPLIT)
    current_hashes = set(current.values())
    overlaps = []
    reference_rows = []
    for value in references:
        root = Path(value).resolve()
        summary_path = root / "dataset_summary.json"
        if not summary_path.is_file():
            continue
        summary = dict(read_json(summary_path))
        hashes = set()
        for split in dict(summary["splits"]):
            hashes.update(dataset_map_hashes(root, str(split)).values())
        shared = sorted(current_hashes & hashes)
        overlaps.extend(shared)
        reference_rows.append(
            {
                "dataset": str(root),
                "master_seed": int(summary["master_seed"]),
                "map_count": len(hashes),
                "shared_map_hash_count": len(shared),
            }
        )
    report = {
        "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
        "passed": not overlaps,
        "qualification_map_count": len(current),
        "qualification_map_hashes": current,
        "reference_datasets": reference_rows,
        "shared_map_hashes": sorted(set(overlaps)),
    }
    if overlaps:
        raise ValueError("qualification maps overlap previous rescue datasets")
    return report


def _source_decision_counts(
    source: Path,
    tasks: dict[str, dict[str, Any]],
) -> collections.Counter[tuple[str, str]]:
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for manifest in _read_jsonl(source / "realized_dynamic_manifest.jsonl"):
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        task = tasks[str(manifest["task_id"])]
        cell = _cell(str(task["layout_mode"]), int(task["agent_count"]))
        decisions, _events = decision_rows(source, manifest)
        counts[(cell, str(task["task_variant"]))] += len(decisions)
    return counts


def summarize_recipe_yield(
    *,
    task_rows: list[dict[str, Any]],
    failure_decisions: list[dict[str, Any]],
    decision_counts: collections.Counter[tuple[str, str]],
    max_states_per_task: int = DEFAULT_MAX_STATES_PER_TASK,
    minimum_states: int = DEFAULT_MINIMUM_STATES_PER_CELL,
    minimum_tasks: int = DEFAULT_MINIMUM_TASKS_PER_CELL,
    minimum_maps: int = DEFAULT_MINIMUM_MAPS_PER_CELL,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    if max_states_per_task <= 0:
        raise ValueError("max_states_per_task must be positive")
    tasks = {str(row["task_id"]): row for row in task_rows}
    recipes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in task_rows:
        cell = _cell(str(row["layout_mode"]), int(row["agent_count"]))
        recipes[(cell, str(row["task_variant"]))].add(str(row["task_id"]))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for decision in failure_decisions:
        task = tasks[str(decision["task_id"])]
        cell = _cell(str(task["layout_mode"]), int(task["agent_count"]))
        grouped[(cell, str(task["task_variant"]))].append(decision)

    summaries = []
    for cell, recipe in sorted(recipes):
        failures = sorted(
            grouped.get((cell, recipe), []),
            key=lambda row: (
                str(row["map_id"]),
                str(row["task_id"]),
                int(row["decision_index"]),
            ),
        )
        per_task: collections.Counter[str] = collections.Counter()
        capped = []
        for row in failures:
            task_id = str(row["task_id"])
            if per_task[task_id] >= max_states_per_task:
                continue
            per_task[task_id] += 1
            capped.append(row)
        task_ids = {str(row["task_id"]) for row in capped}
        map_ids = {str(row["map_id"]) for row in capped}
        total_decisions = int(decision_counts.get((cell, recipe), 0))
        passed = bool(
            len(capped) >= minimum_states
            and len(task_ids) >= minimum_tasks
            and len(map_ids) >= minimum_maps
        )
        summaries.append(
            {
                "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
                "cell": cell,
                "recipe": recipe,
                "source_task_count": len(recipes[(cell, recipe)]),
                "source_decision_count": total_decisions,
                "raw_no_progress_state_count": len(failures),
                "capped_no_progress_state_count": len(capped),
                "no_progress_task_count": len(task_ids),
                "no_progress_map_count": len(map_ids),
                "raw_no_progress_fraction": (
                    len(failures) / total_decisions if total_decisions else 0.0
                ),
                "qualification_passed": passed,
            }
        )

    selected: dict[str, str] = {}
    cell_reports = {}
    for layout in LAYOUTS:
        for agent_count in AGENT_COUNTS:
            cell = _cell(layout, agent_count)
            rows = [row for row in summaries if str(row["cell"]) == cell]
            eligible = [row for row in rows if bool(row["qualification_passed"])]
            ordered = sorted(
                eligible,
                key=lambda row: (
                    -int(row["capped_no_progress_state_count"]),
                    -int(row["no_progress_task_count"]),
                    -float(row["raw_no_progress_fraction"]),
                    str(row["recipe"]),
                ),
            )
            if ordered:
                selected[cell] = str(ordered[0]["recipe"])
            cell_reports[cell] = {
                "passed": bool(ordered),
                "selected_recipe": selected.get(cell),
                "eligible_recipe_count": len(eligible),
                "best_observed_state_count": max(
                    (int(row["capped_no_progress_state_count"]) for row in rows),
                    default=0,
                ),
            }
    gate = {
        "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
        "passed": len(selected) == len(LAYOUTS) * len(AGENT_COUNTS),
        "required_cell_count": len(LAYOUTS) * len(AGENT_COUNTS),
        "passed_cell_count": len(selected),
        "minimum_states_per_cell": minimum_states,
        "minimum_tasks_per_cell": minimum_tasks,
        "minimum_maps_per_cell": minimum_maps,
        "maximum_states_per_task": max_states_per_task,
        "cells": cell_reports,
    }
    return summaries, selected, gate


def run_rescue_confirmation_qualification(
    *,
    project_root: str | Path,
    output: str | Path,
    dataset_config: str | Path,
    controller_bundle: str | Path,
    reference_datasets: Iterable[str | Path],
    workers: int = 4,
    resume: bool = False,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    wall_time_seconds: float = DEFAULT_WALL_TIME_SECONDS,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    config_path = Path(dataset_config).resolve()
    controller_path = Path(controller_bundle).resolve()
    references = [Path(value).resolve() for value in reference_datasets]
    if workers <= 0 or max_decisions <= 0 or wall_time_seconds <= 0:
        raise ValueError("workers and source budgets must be positive")
    identity = {
        "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
        "dataset_config_sha256": sha256_file(config_path),
        "controller_manifest_sha256": sha256_file(
            controller_path / "controller_manifest.json"
        ),
        "max_decisions": int(max_decisions),
        "wall_time_seconds": float(wall_time_seconds),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if output_root.is_dir() and any(output_root.iterdir()):
        if not resume:
            raise ValueError("qualification output already exists; pass --resume")
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("qualification resume fingerprint mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    dataset = output_root / "dataset"
    _write_status(output_root, phase="dataset", status="running")
    if not (dataset / "dataset_summary.json").is_file():
        generate_dataset(load_json(config_path), dataset)
    isolation = validate_qualification_isolation(dataset, references)
    _write_json(output_root / "dataset_isolation.json", isolation)

    task_rows = _load_dataset_rows(dataset, [QUALIFICATION_SPLIT])
    task_ids = sorted(str(row["task_id"]) for row in task_rows)
    source_config = build_confirmation_source_config(
        dataset=dataset,
        output=output_root,
        project_root=root,
        split=QUALIFICATION_SPLIT,
        max_decisions=max_decisions,
        wall_time_seconds=wall_time_seconds,
    )
    source = output_root / "source"
    _write_status(
        output_root,
        phase="source",
        status="running",
        task_count=len(task_ids),
    )
    _run_source_wave(
        dataset=dataset,
        config=source_config,
        output=source,
        task_ids=task_ids,
        project_root=root,
        workers=workers,
        resume=resume,
    )

    failures = select_failure_decisions(source, maximum=100000)
    tasks = {str(row["task_id"]): row for row in task_rows}
    enriched = []
    for decision in failures:
        task = tasks[str(decision["task_id"])]
        enriched.append(
            {
                **decision,
                "task_variant": str(task["task_variant"]),
                "agent_count": int(task["agent_count"]),
            }
        )
    counts = _source_decision_counts(source, tasks)
    summaries, selected, gate = summarize_recipe_yield(
        task_rows=task_rows,
        failure_decisions=enriched,
        decision_counts=counts,
    )
    _write_csv(output_root / "recipe_yield.csv", summaries)
    decision = (
        "qualification_passed_freeze_recipes"
        if bool(gate["passed"])
        else "qualification_failed_redesign_or_v3"
    )
    report = {
        "schema": RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA,
        "decision": decision,
        "run_fingerprint": run_fingerprint,
        "qualification_gate": gate,
        "frozen_recipe_by_cell": selected,
        "source_task_count": len(task_ids),
        "source_no_progress_state_count": len(enriched),
        "dataset_isolation": isolation,
        "locked_confirmation_started": False,
        "branch_trials_started": False,
        "quick_formal_v3_started": False,
    }
    _write_json(output_root / "qualification_report.json", report)
    markdown = [
        "# Rescue confirmation data qualification",
        "",
        f"- Decision: `{decision}`",
        f"- Passed cells: {gate['passed_cell_count']}/{gate['required_cell_count']}",
        f"- Source tasks: {len(task_ids)}",
        f"- Raw no-progress states: {len(enriched)}",
        "- Locked confirmation started: false",
        "- Branch trials started: false",
        "",
        "## Frozen recipes",
        "",
    ]
    for layout in LAYOUTS:
        for agent_count in AGENT_COUNTS:
            cell = _cell(layout, agent_count)
            markdown.append(f"- `{cell}`: `{selected.get(cell, 'none')}`")
    markdown.extend(
        [
            "",
            "A recipe is frozen only when at least five capped no-progress states",
            "cover all three qualification tasks and maps in its cell.",
            "",
        ]
    )
    (output_root / "qualification_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    _write_status(
        output_root,
        phase="complete",
        status="complete" if bool(gate["passed"]) else "insufficient",
        decision=decision,
    )
    return report


__all__ = [
    "QUALIFICATION_SPLIT",
    "RESCUE_CONFIRMATION_QUALIFICATION_SCHEMA",
    "run_rescue_confirmation_qualification",
    "summarize_recipe_yield",
    "validate_qualification_isolation",
]
