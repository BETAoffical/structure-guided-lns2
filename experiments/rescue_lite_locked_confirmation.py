from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _read_json,
    _write_json,
)
from experiments.rescue_lite_confirmation import (
    AGENT_COUNTS,
    CONFIRMATION_SPLIT,
    DEFAULT_STATE_QUOTA_PER_CELL,
    DEFAULT_TRIALS,
    FROZEN_POLICY_ID,
    LAYOUTS,
    SIZES,
    _candidate_decisions,
    _prepare_decisions,
    _run_source_wave,
    _run_trials,
    analyze_confirmation,
    build_confirmation_source_config,
    select_confirmation_states,
    validate_dataset_isolation,
)
from generators.config import load_json
from generators.dataset import generate_dataset


LOCKED_RESCUE_CONFIRMATION_SCHEMA = "lns2.locked_rescue_confirmation.v1"
DEFAULT_MAX_DECISIONS = 40
DEFAULT_WALL_TIME_SECONDS = 180.0


def _cell(layout: str, agent_count: int) -> str:
    if layout not in LAYOUTS or int(agent_count) not in AGENT_COUNTS:
        raise ValueError(f"unexpected locked confirmation cell: {layout}/{agent_count}")
    return f"{layout}__agents_{int(agent_count)}"


def _write_status(root: Path, *, phase: str, status: str, **values: Any) -> None:
    _write_json(
        root / "status.json",
        {
            "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
            "phase": phase,
            "status": status,
            **values,
        },
    )


def validate_frozen_qualification(report: dict[str, Any]) -> dict[str, str]:
    if str(report.get("decision")) != "qualification_passed_freeze_recipes":
        raise ValueError("qualification report did not freeze recipes")
    gate = dict(report.get("qualification_gate", {}))
    if not bool(gate.get("passed")):
        raise ValueError("qualification gate did not pass")
    recipes = {
        str(cell): str(recipe)
        for cell, recipe in dict(report.get("frozen_recipe_by_cell", {})).items()
    }
    expected = {
        _cell(layout, agent_count)
        for layout in LAYOUTS
        for agent_count in AGENT_COUNTS
    }
    if set(recipes) != expected or any(not recipe for recipe in recipes.values()):
        raise ValueError("qualification recipe coverage is incomplete")
    return recipes


def select_locked_task_ids(
    task_rows: list[dict[str, Any]],
    frozen_recipe_by_cell: dict[str, str],
    *,
    expected_tasks_per_cell: int = 4,
) -> tuple[list[str], dict[str, int]]:
    selected = []
    counts: collections.Counter[str] = collections.Counter()
    for row in task_rows:
        cell = _cell(str(row["layout_mode"]), int(row["agent_count"]))
        if str(row["task_variant"]) != frozen_recipe_by_cell[cell]:
            continue
        selected.append(str(row["task_id"]))
        counts[cell] += 1
    expected = set(frozen_recipe_by_cell)
    if set(counts) != expected or any(
        counts[cell] != expected_tasks_per_cell for cell in expected
    ):
        raise ValueError("locked confirmation task coverage is incomplete")
    return sorted(selected), {cell: counts[cell] for cell in sorted(counts)}


def source_state_capacity(
    decisions: list[dict[str, Any]], *, maximum_states_per_task: int = 2
) -> dict[str, dict[str, int]]:
    if maximum_states_per_task <= 0:
        raise ValueError("maximum_states_per_task must be positive")
    grouped: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    maps: dict[str, set[str]] = collections.defaultdict(set)
    for row in decisions:
        cell = str(row["cell"])
        grouped[cell][str(row["task_id"])] += 1
        maps[cell].add(str(row["map_id"]))
    result = {}
    for layout in LAYOUTS:
        for agent_count in AGENT_COUNTS:
            cell = _cell(layout, agent_count)
            per_task = grouped.get(cell, collections.Counter())
            result[cell] = {
                "raw_state_count": sum(per_task.values()),
                "capped_state_capacity": sum(
                    min(maximum_states_per_task, count)
                    for count in per_task.values()
                ),
                "task_count": len(per_task),
                "map_count": len(maps.get(cell, set())),
            }
    return result


def run_locked_rescue_confirmation(
    *,
    project_root: str | Path,
    output: str | Path,
    dataset_config: str | Path,
    qualification_report: str | Path,
    controller_bundle: str | Path,
    repair_aware_bundle: str | Path,
    reference_datasets: Iterable[str | Path],
    workers: int = 4,
    resume: bool = False,
    quota_per_cell: int = DEFAULT_STATE_QUOTA_PER_CELL,
    trial_count: int = DEFAULT_TRIALS,
    expected_tasks_per_cell: int = 4,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    wall_time_seconds: float = DEFAULT_WALL_TIME_SECONDS,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    config_path = Path(dataset_config).resolve()
    qualification_path = Path(qualification_report).resolve()
    controller_path = Path(controller_bundle).resolve()
    repair_aware_path = Path(repair_aware_bundle).resolve()
    references = [Path(value).resolve() for value in reference_datasets]
    if (
        workers <= 0
        or quota_per_cell <= 0
        or trial_count <= 0
        or expected_tasks_per_cell <= 0
        or max_decisions <= 0
        or wall_time_seconds <= 0
    ):
        raise ValueError(
            "workers, quotas, trials, task coverage and source budgets must be positive"
        )
    qualification = _read_json(qualification_path)
    frozen = validate_frozen_qualification(qualification)
    helper_path = root / "experiments" / "rescue_lite_confirmation.py"
    identity = {
        "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
        "dataset_config_sha256": sha256_file(config_path),
        "qualification_report_sha256": sha256_file(qualification_path),
        "controller_manifest_sha256": sha256_file(
            controller_path / "controller_manifest.json"
        ),
        "repair_aware_manifest_sha256": sha256_file(
            repair_aware_path / "repair_aware_manifest.json"
        ),
        "quota_per_cell": int(quota_per_cell),
        "trial_count": int(trial_count),
        "expected_tasks_per_cell": int(expected_tasks_per_cell),
        "max_decisions": int(max_decisions),
        "wall_time_seconds": float(wall_time_seconds),
        "sizes": list(SIZES),
        "frozen_policy_id": FROZEN_POLICY_ID,
        "frozen_recipe_by_cell": frozen,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "confirmation_helper_sha256": sha256_file(helper_path),
        "reference_datasets": sorted(map(str, references)),
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if output_root.is_dir() and any(output_root.iterdir()):
        if not resume:
            raise ValueError("locked confirmation output exists; pass --resume")
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("locked confirmation resume fingerprint mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    dataset = output_root / "dataset"
    _write_status(output_root, phase="dataset", status="running")
    if not (dataset / "dataset_summary.json").is_file():
        generate_dataset(load_json(config_path), dataset)
    isolation = validate_dataset_isolation(dataset, references)
    _write_json(output_root / "dataset_isolation.json", isolation)
    task_rows = _load_dataset_rows(dataset, [CONFIRMATION_SPLIT])
    task_ids, task_counts = select_locked_task_ids(
        task_rows,
        frozen,
        expected_tasks_per_cell=expected_tasks_per_cell,
    )
    _write_json(
        output_root / "frozen_protocol.json",
        {
            "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "qualification_report": str(qualification_path),
            "qualification_report_sha256": sha256_file(qualification_path),
            "frozen_recipe_by_cell": frozen,
            "expected_tasks_per_cell": int(expected_tasks_per_cell),
            "selected_task_count_by_cell": task_counts,
            "selected_task_ids": task_ids,
        },
    )

    source_config = build_confirmation_source_config(
        dataset=dataset,
        output=output_root,
        project_root=root,
        split=CONFIRMATION_SPLIT,
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

    decisions = _candidate_decisions([source], dataset, CONFIRMATION_SPLIT)
    source_capacity = source_state_capacity(decisions)
    source_shortfall = {
        cell: values
        for cell, values in source_capacity.items()
        if int(values["capped_state_capacity"]) < quota_per_cell
    }
    if source_shortfall:
        report = {
            "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
            "decision": "locked_confirmation_insufficient_states",
            "insufficient_phase": "source-coverage",
            "run_fingerprint": run_fingerprint,
            "required_state_count_per_cell": quota_per_cell,
            "expected_tasks_per_cell": int(expected_tasks_per_cell),
            "selected_task_count_by_cell": task_counts,
            "source_state_capacity_by_cell": source_capacity,
            "source_shortfall": source_shortfall,
            "source_no_progress_state_count": len(decisions),
            "prepared_state_count": 0,
            "frozen_recipe_by_cell": frozen,
            "dataset_isolation": isolation,
            "branch_trials_started": False,
            "quick_formal_v3_started": False,
        }
        _write_json(output_root / "locked_confirmation_report.json", report)
        _write_status(
            output_root,
            phase="complete",
            status="insufficient",
            decision=report["decision"],
        )
        return report
    _write_status(
        output_root,
        phase="prepare-states",
        status="running",
        source_no_progress_state_count=len(decisions),
    )
    prepared = _prepare_decisions(
        decisions=decisions,
        output=output_root,
        run_fingerprint=run_fingerprint,
        controller_bundle=controller_path,
        repair_aware_bundle=repair_aware_path,
        workers=workers,
        maximum_per_cell=max(quota_per_cell * 4, quota_per_cell + 8),
    )
    selected, counts = select_confirmation_states(
        prepared, quota_per_cell=quota_per_cell
    )
    required = {
        _cell(layout, agent_count): quota_per_cell
        for layout in LAYOUTS
        for agent_count in AGENT_COUNTS
    }
    invalid_reasons = collections.Counter(
        str(row.get("invalid_reason", "unknown"))
        for row in prepared
        if not bool(row.get("valid"))
    )
    if counts != required:
        report = {
            "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
            "decision": "locked_confirmation_insufficient_states",
            "run_fingerprint": run_fingerprint,
            "required_state_count_by_cell": required,
            "expected_tasks_per_cell": int(expected_tasks_per_cell),
            "selected_task_count_by_cell": task_counts,
            "observed_state_count_by_cell": counts,
            "source_no_progress_state_count": len(decisions),
            "source_state_capacity_by_cell": source_capacity,
            "prepared_state_count": len(prepared),
            "invalid_preparation_reasons": dict(invalid_reasons),
            "frozen_recipe_by_cell": frozen,
            "dataset_isolation": isolation,
            "branch_trials_started": False,
            "quick_formal_v3_started": False,
        }
        _write_json(output_root / "locked_confirmation_report.json", report)
        _write_status(
            output_root,
            phase="complete",
            status="insufficient",
            decision=report["decision"],
        )
        return report

    _write_json(
        output_root / "selection_manifest.json",
        {
            "schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "state_count_by_cell": counts,
            "states": [
                {
                    "state": row["state"],
                    "top_candidate_by_size": row["top_candidate_by_size"],
                    "learned_candidate_id": row["learned_candidate_id"],
                }
                for row in selected
            ],
        },
    )
    _write_status(
        output_root,
        phase="paired-trials",
        status="running",
        state_count=len(selected),
    )
    results = _run_trials(
        selected=selected,
        output=output_root,
        run_fingerprint=run_fingerprint,
        trial_count=trial_count,
        workers=workers,
    )
    _write_status(output_root, phase="analysis", status="running")
    analysis = analyze_confirmation(results=results, output=output_root)
    report = {
        **analysis,
        "protocol_schema": LOCKED_RESCUE_CONFIRMATION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "state_count_by_cell": counts,
        "expected_tasks_per_cell": int(expected_tasks_per_cell),
        "selected_task_count_by_cell": task_counts,
        "source_no_progress_state_count": len(decisions),
        "source_state_capacity_by_cell": source_capacity,
        "frozen_recipe_by_cell": frozen,
        "qualification_report_sha256": sha256_file(qualification_path),
        "dataset_isolation": isolation,
        "branch_trials_started": True,
        "quick_formal_v3_started": False,
    }
    _write_json(output_root / "locked_confirmation_report.json", report)
    _write_status(
        output_root,
        phase="complete",
        status="complete",
        decision=str(report["decision"]),
    )
    return report


__all__ = [
    "LOCKED_RESCUE_CONFIRMATION_SCHEMA",
    "run_locked_rescue_confirmation",
    "select_locked_task_ids",
    "source_state_capacity",
    "validate_frozen_qualification",
]
