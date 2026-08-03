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


CONFIG_SCHEMA = "lns2.stride.repairability_source_cohort_config.v1"
REPORT_SCHEMA = "lns2.stride.repairability_source_cohort.v1"
SPLIT = "balanced_wall_clock"
FORBIDDEN_FIELDS = {
    "candidate_repair_outcome",
    "selected_candidate",
    "controller_action",
    "future_trajectory",
    "controller_ttf",
    "repair_runtime",
    "repair_outcome",
    "conflicts_after",
}


def _registered_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _task_variant(task_id: str) -> str:
    if "__derived_opposite_exchange__" in task_id:
        return "opposite_exchange"
    if "__derived_uniform_random__" in task_id:
        return "uniform_random"
    return "official_scenario"


def validate_repairability_source_cohort_config(config: dict[str, Any]) -> None:
    split = dict(config.get("effective_map_split") or {})
    train = list(map(str, split.get("train") or ()))
    validation = list(map(str, split.get("validation") or ()))
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_blind_source_task_selection"
        or config.get("controller_id") != "stride-augcontrol-v1"
        or int(config.get("tasks_per_map", -1)) != 2
        or int(config.get("expected_map_count", -1)) != 22
        or int(config.get("expected_task_count", -1)) != 44
        or list(config.get("target_median_initial_conflicts") or ()) != [20, 150]
        or float(config.get("minimum_nonzero_solver_seed_fraction", -1.0))
        != 2.0 / 3.0
        or int(config.get("minimum_median_initial_conflicts", -1)) != 1
        or not bool(config.get("prefer_distinct_task_variants"))
        or bool(config.get("candidate_outcomes_read"))
        or bool(config.get("formal_ood_data_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("repairability source-cohort contract changed")
    if (
        len(train) != 16
        or len(validation) != 6
        or len(set(train)) != 16
        or len(set(validation)) != 6
        or set(train) & set(validation)
    ):
        raise ValueError("repairability effective map split changed")
    if dict(config.get("replacements") or {}) != {
        "arena2": "den206d",
        "den001d": "den011d",
        "den005d": "ht_mansion_n",
    }:
        raise ValueError("repairability source replacements changed")
    if set(map(str, config.get("selection_inputs_forbidden") or ())) != {
        "candidate_repair_outcome",
        "selected_candidate",
        "controller_action",
        "future_trajectory",
        "controller_ttf",
        "repair_runtime",
    }:
        raise ValueError("repairability source outcome boundary changed")
    sources = list(config.get("sources") or ())
    if len(sources) != 4 or len({str(row.get("id")) for row in sources}) != 4:
        raise ValueError("repairability source registry changed")


def _copy_registered_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"repairability dataset file collision: {target}")
        return
    shutil.copy2(source, target)


def prepare_repairability_source_cohort(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_repairability_source_cohort_config(config)
    data_design_path = _registered_path(project_root, str(config["data_design"]))
    if sha256_file(data_design_path) != str(config["data_design_sha256"]):
        raise ValueError("repairability source data-design SHA differs")
    design = _read_json(data_design_path)
    ratios = {
        str(name): float(value)
        for name, value in dict(design["static_low_degree_cell_ratio_by_map"]).items()
    }
    threshold = float(design["topology_boundary_threshold"])
    map_split = {
        str(map_id): research_split
        for research_split, map_ids in dict(config["effective_map_split"]).items()
        for map_id in map_ids
    }

    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    source_evidence = []
    forbidden_hits: set[str] = set()
    for source_order, raw_source in enumerate(config["sources"]):
        source = dict(raw_source)
        dataset = _registered_path(project_root, str(source["dataset"]))
        qualification = _registered_path(project_root, str(source["qualification"]))
        dataset_manifest = dataset / SPLIT / "manifest.jsonl"
        qualification_manifest = qualification / "qualification_manifest.jsonl"
        if sha256_file(dataset_manifest) != str(source["dataset_manifest_sha256"]):
            raise ValueError(f"repairability dataset SHA differs: {source['id']}")
        if sha256_file(qualification_manifest) != str(
            source["qualification_manifest_sha256"]
        ):
            raise ValueError(f"repairability qualification SHA differs: {source['id']}")
        dataset_rows = {
            str(row["task_id"]): row for row in _read_jsonl(dataset_manifest)
        }
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(qualification_manifest):
            forbidden_hits.update(FORBIDDEN_FIELDS & set(row))
            task_id = str(row.get("task_id"))
            if task_id in dataset_rows:
                grouped[task_id].append(row)
        expected_seeds = set(map(int, source["solver_seeds"]))
        for task_id, values in grouped.items():
            dataset_row = dataset_rows[task_id]
            map_id = str(dataset_row["map_id"])
            if map_id not in map_split:
                continue
            indexed = {int(row.get("solver_seed", -1)): row for row in values}
            if set(indexed) != expected_seeds or any(
                str(row.get("status")) != "ok"
                or not bool(row.get("initial_complete"))
                or not str(row.get("state_fingerprint", ""))
                for row in indexed.values()
            ):
                continue
            conflicts = [int(indexed[seed]["initial_conflicts"]) for seed in sorted(indexed)]
            candidates[map_id].append(
                {
                    "source_id": str(source["id"]),
                    "source_order": source_order,
                    "dataset_root": str(dataset),
                    "dataset_row": dataset_row,
                    "task_id": task_id,
                    "map_id": map_id,
                    "task_variant": _task_variant(task_id),
                    "agent_count": int(dataset_row["agent_count"]),
                    "solver_seed_count": len(conflicts),
                    "nonzero_solver_seed_count": sum(value > 0 for value in conflicts),
                    "nonzero_solver_seed_fraction": sum(value > 0 for value in conflicts)
                    / len(conflicts),
                    "median_initial_conflicts": statistics.median(conflicts),
                    "mean_initial_conflicts": statistics.fmean(conflicts),
                    "minimum_initial_conflicts": min(conflicts),
                    "maximum_initial_conflicts": max(conflicts),
                }
            )
        source_evidence.append(
            {
                "id": str(source["id"]),
                "dataset_manifest_sha256": sha256_file(dataset_manifest),
                "qualification_manifest_sha256": sha256_file(
                    qualification_manifest
                ),
            }
        )
    if forbidden_hits:
        raise ValueError(
            f"repairability source qualification contains outcomes: {sorted(forbidden_hits)}"
        )

    selected = []
    selection_reports = []
    minimum_fraction = float(config["minimum_nonzero_solver_seed_fraction"])
    minimum_median = float(config["minimum_median_initial_conflicts"])
    for map_id in sorted(map_split):
        eligible = [
            row
            for row in candidates.get(map_id, ())
            if float(row["nonzero_solver_seed_fraction"]) >= minimum_fraction
            and float(row["median_initial_conflicts"]) >= minimum_median
        ]
        chosen = []
        for target in map(float, config["target_median_initial_conflicts"]):
            remaining = [row for row in eligible if row not in chosen]
            if not remaining:
                break
            variants = {str(row["task_variant"]) for row in remaining}
            require_other = bool(chosen) and any(
                variant != str(chosen[0]["task_variant"]) for variant in variants
            )

            def rank(row: dict[str, Any]) -> tuple[Any, ...]:
                median = float(row["median_initial_conflicts"])
                return (
                    int(
                        require_other
                        and str(row["task_variant"])
                        == str(chosen[0]["task_variant"])
                    ),
                    abs(math.log1p(median) - math.log1p(target)),
                    -float(row["nonzero_solver_seed_fraction"]),
                    int(row["agent_count"]),
                    int(
                        _fingerprint(
                            [
                                "stride-repairability-source-task-v1",
                                map_id,
                                row["task_id"],
                            ]
                        )[:16],
                        16,
                    ),
                )

            chosen.append(min(remaining, key=rank))
        if len(chosen) != int(config["tasks_per_map"]):
            raise ValueError(f"repairability map lacks two qualified tasks: {map_id}")
        selected.extend(chosen)
        selection_reports.append(
            {
                "map_id": map_id,
                "research_split": map_split[map_id],
                "candidate_task_count": len(candidates.get(map_id, ())),
                "eligible_task_count": len(eligible),
                "selected": [
                    {key: value for key, value in row.items() if key != "dataset_row"}
                    for row in chosen
                ],
            }
        )

    output = Path(output).resolve()
    summary_path = output / "dataset_summary.json"
    identity = {
        "config_sha256": sha256_file(config_path),
        "data_design_sha256": sha256_file(data_design_path),
        "selected_task_ids": sorted(str(row["task_id"]) for row in selected),
        "source_evidence": source_evidence,
    }
    fingerprint = _fingerprint(identity)
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != fingerprint:
            raise ValueError("repairability source dataset belongs to another cohort")
        return existing
    if output.is_dir() and any(output.iterdir()):
        raise ValueError("repairability source output is non-empty without summary")

    manifest = []
    for selected_row in selected:
        dataset_root = Path(str(selected_row["dataset_root"]))
        row = dict(selected_row["dataset_row"])
        for field, directory in (
            ("map_file", "maps"),
            ("map_metadata_file", "maps"),
            ("scenario_file", "scenarios"),
            ("task_file", "tasks"),
        ):
            source_file = dataset_root / SPLIT / str(row[field])
            target_file = output / SPLIT / directory / source_file.name
            _copy_registered_file(source_file, target_file)
            row[field] = f"{directory}/{target_file.name}"
        row.update(
            {
                "split": SPLIT,
                "source_group": "movingai",
                "source_cohort_id": str(selected_row["source_id"]),
                "research_split": map_split[str(row["map_id"])],
                "layout_mode": (
                    "boundary_relevant"
                    if ratios[str(row["map_id"])] >= threshold
                    else "control"
                ),
                "static_low_degree_cell_ratio": ratios[str(row["map_id"])],
            }
        )
        manifest.append(row)
    manifest.sort(key=lambda row: str(row["task_id"]))
    if (
        len(manifest) != int(config["expected_task_count"])
        or len({str(row["map_id"]) for row in manifest})
        != int(config["expected_map_count"])
    ):
        raise ValueError("repairability source cohort dimensions differ")
    _write_jsonl(output / SPLIT / "manifest.jsonl", manifest)
    report = {
        "schema": REPORT_SCHEMA,
        "configuration_fingerprint": fingerprint,
        "config_sha256": sha256_file(config_path),
        "data_design_sha256": sha256_file(data_design_path),
        "result_blind": True,
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "source_evidence": source_evidence,
        "map_count": len({str(row["map_id"]) for row in manifest}),
        "task_count": len(manifest),
        "research_split_task_counts": {
            research_split: sum(
                str(row["research_split"]) == research_split for row in manifest
            )
            for research_split in ("train", "validation")
        },
        "topology_group_task_counts": {
            group: sum(str(row["layout_mode"]) == group for row in manifest)
            for group in ("control", "boundary_relevant")
        },
        "selection_reports": selection_reports,
        "selected_task_ids": [str(row["task_id"]) for row in manifest],
        "passed": True,
    }
    _write_json(output / "source_cohort_report.json", report)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "dataset_revision": "stride-repairability-source-cohort-v1",
            "configuration_fingerprint": fingerprint,
            "source": "checksum-pinned MovingAI tasks selected by initial PP only",
            "splits": {
                SPLIT: {
                    "map_count": report["map_count"],
                    "instance_count": report["task_count"],
                    "source_counts": {"movingai": report["task_count"]},
                }
            },
        },
    )
    return report


__all__ = [
    "prepare_repairability_source_cohort",
    "validate_repairability_source_cohort_config",
]
