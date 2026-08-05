from __future__ import annotations

import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_robustaction_expansion import (
    STRUCTPOOL_DESIGN_SCHEMA,
    validate_robustaction_expansion_design,
)


CONFIG_SCHEMA = "lns2.stride.robustaction_structpool_qualification_design.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_structpool_qualification_report.v1"
SPLIT = "balanced_wall_clock"
FORBIDDEN_FIELDS = {
    "candidate_repair_outcome",
    "candidate_conflicts_after",
    "candidate_runtime",
    "selected_candidate",
    "controller_action",
    "controller_ttf",
    "future_trajectory",
    "repair_runtime",
    "conflicts_after",
    "repair_outcome",
}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"robust-action qualification input changed: {spec['path']}")
    return path


def _task_variant(task_id: str) -> str:
    if "__derived_opposite_exchange__" in task_id:
        return "opposite_exchange"
    if "__derived_uniform_random__" in task_id:
        return "uniform_random"
    raise ValueError(f"unregistered robust-action task variant: {task_id}")


def validate_robustaction_qualification_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_structpool_initial_pp_qualification"
        or config.get("data_line_id") != "stride-robustaction-structpool-data-v1"
        or config.get("pre_registration_git_commit")
        != "ed35f347acff94d38d829b016df7736a8b8b8895"
        or int(config.get("expected_map_count", -1)) != 20
        or int(config.get("expected_task_count", -1)) != 124
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("expected_qualification_job_count", -1)) != 248
        or int(config.get("tasks_selected_per_map", -1)) != 2
        or int(config.get("expected_selected_task_count", -1)) != 40
        or list(config.get("target_mean_initial_conflicts") or ()) != [25, 100]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0)) != 0.5
        or float(config.get("minimum_mean_initial_conflicts", -1.0)) != 1.0
        or not bool(config.get("prefer_distinct_task_variants"))
        or bool(config.get("candidate_outcomes_read"))
        or bool(config.get("controller_outcomes_read"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("robust-action qualification identity changed")
    if set(map(str, config.get("forbidden_selection_inputs") or ())) != {
        "candidate_repair_outcome",
        "candidate_conflicts_after",
        "candidate_runtime",
        "selected_candidate",
        "controller_action",
        "controller_ttf",
        "future_trajectory",
        "repair_runtime",
    }:
        raise ValueError("robust-action qualification outcome boundary changed")
    if (
        not bool(config.get("qualification_must_pass"))
        or not bool(config.get("all_resets_must_be_complete"))
        or not bool(config.get("all_maps_must_have_nonzero_state"))
        or config.get("underloaded_rule")
        != "registered_highest_agent_count_is_already_in_pool_then_fail_if_two_repairable_tasks_are_unavailable"
    ):
        raise ValueError("robust-action qualification failure policy changed")

    inputs = dict(config.get("inputs") or {})
    if set(inputs) != {"data_design", "runtime", "dataset_manifest", "dataset_summary"}:
        raise ValueError("robust-action qualification input registry changed")
    paths = {name: _registered(project_root, dict(spec)) for name, spec in inputs.items()}
    design = _read_json(paths["data_design"])
    validate_robustaction_expansion_design(design)
    if design.get("schema") != STRUCTPOOL_DESIGN_SCHEMA:
        raise ValueError("qualification does not reference the StructPool data design")
    runtime = _read_json(paths["runtime"])
    environment = dict(runtime.get("environment") or {})
    qualification = dict(runtime.get("qualification") or {})
    dataset_design = dict(runtime.get("dataset_design") or {})
    if (
        bool(runtime.get("formal"))
        or list(runtime.get("solver_seeds") or ()) != [1, 2]
        or float(environment.get("time_limit", -1.0)) != 600.0
        or int(environment.get("max_repair_iterations", -1)) != 0
        or float(runtime.get("episode_process_timeout_seconds", -1.0)) != 660.0
        or int(runtime.get("max_decisions", -1)) != 0
        or int(dataset_design.get("map_count", -1)) != 20
        or int(dataset_design.get("instance_count", -1)) != 124
        or not bool(qualification.get("enforce_registered_thresholds"))
        or int(qualification.get("minimum_active_maps", -1)) != 20
    ):
        raise ValueError("robust-action qualification runtime changed")


def select_qualified_tasks(
    summaries: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    by_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_map[str(row["map_id"])].append(row)
    selected: list[dict[str, Any]] = []
    underloaded: list[str] = []
    minimum_fraction = float(config["minimum_nonzero_solver_seed_fraction"])
    minimum_mean = float(config["minimum_mean_initial_conflicts"])
    for map_id in sorted(by_map):
        eligible = [
            row
            for row in by_map[map_id]
            if float(row["nonzero_solver_seed_fraction"]) >= minimum_fraction
            and float(row["mean_initial_conflicts"]) >= minimum_mean
        ]
        chosen: list[dict[str, Any]] = []
        for target in map(float, config["target_mean_initial_conflicts"]):
            remaining = [row for row in eligible if row not in chosen]
            if not remaining:
                break
            distinct_available = bool(chosen) and any(
                str(row["task_variant"]) != str(chosen[0]["task_variant"])
                for row in remaining
            )

            def score(row: dict[str, Any]) -> tuple[Any, ...]:
                mean_conflicts = float(row["mean_initial_conflicts"])
                return (
                    int(
                        distinct_available
                        and str(row["task_variant"])
                        == str(chosen[0]["task_variant"])
                    ),
                    abs(math.log1p(mean_conflicts) - math.log1p(target)),
                    -float(row["nonzero_solver_seed_fraction"]),
                    int(row["agent_count"]),
                    int(_fingerprint([map_id, row["task_id"], target])[:16], 16),
                )

            chosen.append(min(remaining, key=score))
        if len(chosen) != int(config["tasks_selected_per_map"]):
            underloaded.append(map_id)
            continue
        selected.extend(chosen)
    selected.sort(key=lambda row: (str(row["map_id"]), str(row["task_id"])))
    return selected, underloaded


def _copy_dataset_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"robust-action selected dataset collision: {target}")
        return
    shutil.copy2(source, target)


def analyze_robustaction_qualification(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robustaction_qualification_design(config, project_root=project_root)
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    manifest_path = dataset / SPLIT / "manifest.jsonl"
    qualification_path = qualification / "qualification_manifest.jsonl"
    qualification_report_path = qualification / "qualification_report.json"
    run_config_path = qualification / "run_config.json"
    rows = _read_jsonl(manifest_path)
    results = _read_jsonl(qualification_path)
    qualification_report = _read_json(qualification_report_path)
    run_config = _read_json(run_config_path)
    registered_runtime = _read_json(
        project_root / str(config["inputs"]["runtime"]["path"])
    )

    dataset_index = {str(row["task_id"]): row for row in rows}
    expected = {
        (task_id, seed)
        for task_id in dataset_index
        for seed in map(int, config["solver_seeds"])
    }
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    forbidden_hits: set[str] = set()
    for row in results:
        forbidden_hits.update(FORBIDDEN_FIELDS & set(row))
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source = dataset_index.get(key[0])
        if source is None or key not in expected:
            errors.append(f"unexpected:{key[0]}:{key[1]}")
            continue
        if (
            str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
            or str(row.get("map_id")) != str(source["map_id"])
            or int(row.get("agent_count", -1)) != int(source["agent_count"])
        ):
            errors.append(f"invalid:{key[0]}:{key[1]}")

    summaries = []
    for task_id, source in sorted(dataset_index.items()):
        values = [indexed.get((task_id, seed)) for seed in map(int, config["solver_seeds"])]
        if any(value is None for value in values):
            continue
        conflicts = [int(value["initial_conflicts"]) for value in values if value is not None]
        summaries.append(
            {
                "map_id": str(source["map_id"]),
                "task_id": task_id,
                "task_variant": _task_variant(task_id),
                "layout_mode": str(source["layout_mode"]),
                "agent_count": int(source["agent_count"]),
                "solver_seed_count": len(conflicts),
                "nonzero_solver_seed_count": sum(value > 0 for value in conflicts),
                "nonzero_solver_seed_fraction": sum(value > 0 for value in conflicts)
                / len(conflicts),
                "mean_initial_conflicts": statistics.fmean(conflicts),
                "minimum_initial_conflicts": min(conflicts),
                "maximum_initial_conflicts": max(conflicts),
            }
        )
    selected, underloaded = select_qualified_tasks(summaries, config)
    active_maps = {str(row["map_id"]) for row in summaries if int(row["maximum_initial_conflicts"]) > 0}
    gates = {
        "dataset_dimensions": len(rows) == 124 and len(dataset_index) == 124,
        "complete_qualification_product": set(indexed) == expected,
        "all_rows_valid": not errors,
        "forbidden_outcomes_absent": not forbidden_hits,
        "registered_runtime_exact": dict(run_config.get("configuration") or {})
        == registered_runtime,
        "qualification_passed": bool(qualification_report.get("passed")),
        "all_maps_have_nonzero_state": len(active_maps) == 20,
        "two_tasks_selected_per_map": len(selected) == 40 and not underloaded,
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_blind_initial_pp_qualification",
        "data_line_id": "stride-robustaction-structpool-data-v1",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "qualification_manifest_sha256": sha256_file(qualification_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "map_count": len({str(row["map_id"]) for row in rows}),
            "task_count": len(rows),
            "qualification_job_count": len(results),
            "selected_task_count": len(selected),
            "active_map_count": len(active_maps),
        },
        "underloaded_maps": underloaded,
        "task_summaries": summaries,
        "selected_tasks": selected,
        "forbidden_fields_found": sorted(forbidden_hits),
        "row_errors": errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "qualification_selection_report.json", report)
    if not passed:
        return report

    selected_ids = {str(row["task_id"]) for row in selected}
    selected_manifest = []
    for raw in rows:
        if str(raw["task_id"]) not in selected_ids:
            continue
        row = dict(raw)
        for field, directory in (
            ("map_file", "maps"),
            ("map_metadata_file", "maps"),
            ("scenario_file", "scenarios"),
            ("task_file", "tasks"),
        ):
            source = dataset / SPLIT / str(row[field])
            target = output / SPLIT / directory / source.name
            _copy_dataset_file(source, target)
            row[field] = f"{directory}/{target.name}"
        selected_manifest.append(row)
    selected_manifest.sort(key=lambda row: str(row["task_id"]))
    manifest_output = output / SPLIT / "manifest.jsonl"
    _write_jsonl(manifest_output, selected_manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": "stride-robustaction-structpool-source-v1",
        "configuration_fingerprint": _fingerprint(
            {
                "design": sha256_file(config_path),
                "qualification": sha256_file(qualification_path),
                "selected_task_ids": sorted(selected_ids),
            }
        ),
        "source": "checksum-pinned MovingAI tasks selected by initial PP only",
        "splits": {
            SPLIT: {
                "map_count": 20,
                "instance_count": 40,
                "source_counts": {"movingai": 40},
            }
        },
    }
    _write_json(output / "dataset_summary.json", summary)
    report["selected_manifest_sha256"] = sha256_file(manifest_output)
    report["dataset_summary_sha256"] = sha256_file(output / "dataset_summary.json")
    _write_json(output / "qualification_selection_report.json", report)
    return report


__all__ = [
    "analyze_robustaction_qualification",
    "select_qualified_tasks",
    "validate_robustaction_qualification_design",
]
