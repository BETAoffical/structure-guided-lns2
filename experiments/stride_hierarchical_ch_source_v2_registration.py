"""Static, outcome-blind registration for hierarchical C/H source-v2.

Only the registered train/development map and scenario files are parsed.  The
sealed-final sources and all source-v1 qualification/repair outcomes are outside
this module's input boundary.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import median_low
from typing import Any, Iterable

from experiments._common import registered_input, sha256_file
from experiments.stride_hierarchical_ch_bucket_mix_v2 import (
    ALGORITHM_SCHEMA,
    TASK_SCHEMA,
    VARIANT_IDS as MATERIALIZER_VARIANT_IDS,
    build_registered_bucket_mix_variants,
)
from experiments.stride_hierarchical_ch_h1_v2 import SEALED_FINAL_MAPS


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_source_v2_registration.v1"
REPORT_SCHEMA = "lns2.stride.hierarchical_ch_source_v2_registration_report.v1"
EXPERIMENT_ID = "stride_hierarchical_ch_source_v2"
DEFAULT_OUTPUT_NAME = "stride-hierarchical-ch-source-v2-registration"
TRAIN_MAPS = (
    "brc000d", "den000d", "hrt000d", "lak100d", "lgt605d", "orz999d",
    "ost000t", "rmtst03",
)
DEVELOPMENT_MAPS = (
    "brc203d", "den400d", "lak201d", "lgt602d", "orz302d", "ost004d",
    "brc997d", "den504d",
)
VARIANT_IDS = ("bucket_mix_00", "bucket_mix_01")
SOLVER_SEEDS = (41, 42)


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("source-v2 registration must be a JSON object")
    return value


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("source-v2 registration identity changed")
    if config.get("status") != "REGISTERED" or config.get("training_authorized") is not False:
        raise ValueError("source-v2 must remain registered and training-disabled")
    if config.get("allowed_splits") != ["train", "development"]:
        raise ValueError("only train/development source splits are allowed")
    expected_splits = {"train": list(TRAIN_MAPS), "development": list(DEVELOPMENT_MAPS)}
    if config.get("map_splits") != expected_splits:
        raise ValueError("source-v2 map membership/order changed")
    if set(TRAIN_MAPS) & set(DEVELOPMENT_MAPS):
        raise ValueError("train/development maps overlap")
    forbidden_ids = set(SEALED_FINAL_MAPS)
    leaked = forbidden_ids & set(_strings(config))
    if leaked:
        raise ValueError(f"sealed-final map ID leaked into source-v2 registration: {sorted(leaked)}")
    if config.get("variant_ids") != list(VARIANT_IDS):
        raise ValueError("exact two registered bucket-mix variants are required")
    if config.get("solver_seeds") != list(SOLVER_SEEDS):
        raise ValueError("solver seed registration changed")
    maps = (*TRAIN_MAPS, *DEVELOPMENT_MAPS)
    loads = config.get("per_map_load_grid")
    if not isinstance(loads, dict) or set(loads) != set(maps):
        raise ValueError("per-map load grid must cover the 16 registered maps exactly")
    for map_id in maps:
        pair = loads[map_id]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int or value <= 0 for value in pair)
            or pair[0] >= pair[1]
        ):
            raise ValueError(f"{map_id} requires two increasing positive integer loads")
    if config.get("expected_map_count") != 16:
        raise ValueError("expected_map_count must be 16")
    if config.get("expected_task_count") != 64:
        raise ValueError("expected_task_count must be 16 maps x 2 variants x 2 loads")
    if config.get("expected_episode_count") != 128:
        raise ValueError("expected_episode_count must include both registered solver seeds")
    if config.get("maximum_source_episodes_per_map") != 8 or config.get("workers") != 20:
        raise ValueError("registered episode-per-map/worker contract changed")

    task = config.get("task_materialization")
    if not isinstance(task, dict) or task.get("algorithm_schema") != ALGORITHM_SCHEMA:
        raise ValueError("bucket-mix v2 algorithm schema is required")
    if task.get("task_schema") != TASK_SCHEMA or tuple(MATERIALIZER_VARIANT_IDS) != VARIANT_IDS:
        raise ValueError("bucket-mix task schema or implementation variants changed")
    if task.get("variant_ids") != list(VARIANT_IDS):
        raise ValueError("task materializer variants differ from registration")
    for forbidden_flag in (
        "repair_outcome_used_for_sampling",
        "qualification_outcome_used_for_sampling",
        "runtime_or_pp_seconds_used_for_sampling",
    ):
        if task.get(forbidden_flag) is not False:
            raise ValueError(f"{forbidden_flag} must remain false")

    qualification = config.get("reset_only_qualification")
    if not isinstance(qualification, dict):
        raise ValueError("reset-only qualification contract is required")
    required = {
        "expected_reset_count": {"train": 64, "development": 64, "total": 128},
        "minimum_nonzero_states": {"train": 33, "development": 33},
        "minimum_active_maps": {"train": 8, "development": 8},
        "minimum_nonzero_resets_per_map": 2,
        "minimum_nonzero_resets_per_map_variant": 1,
        "minimum_nonzero_high_load_resets_per_map": 1,
        "minimum_resets_with_at_least_4_conflicts": {"train": 12, "development": 12},
        "minimum_resets_with_at_least_16_conflicts": {"train": 4, "development": 4},
        "minimum_maps_with_a_reset_at_least_4_conflicts": {"train": 4, "development": 4},
        "minimum_maps_with_a_reset_at_least_16_conflicts": {"train": 2, "development": 2},
        "failure_action": "STATE_SUPPLY_FAIL_NO_BACKFILL",
    }
    for key, expected in required.items():
        if qualification.get(key) != expected:
            raise ValueError(f"reset-only qualification gate changed: {key}")
    for flag in (
        "atomic_whole_cohort_gate",
        "all_resets_complete",
        "all_resets_valid",
        "all_initial_states_consistent",
    ):
        if qualification.get(flag) is not True:
            raise ValueError(f"reset-only qualification flag must remain true: {flag}")
    for flag in (
        "outcome_based_subset_selection_allowed",
        "per_task_or_per_map_load_adaptation_allowed",
        "reserve_or_replacement_backfill_allowed",
    ):
        if qualification.get(flag) is not False:
            raise ValueError(f"qualification adaptation/backfill must remain false: {flag}")

    boundary = config.get("claim_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(key) is not False
        for key in (
            "sealed_final_semantic_access_authorized",
            "sealed_final_source_registered_here",
            "source_v1_outcomes_used_for_map_or_load_selection",
            "model_fit_executed",
            "training_authorized",
            "runtime_or_ttf_claim_authorized",
        )
    ):
        raise ValueError("source-v2 claim boundary changed")


def _parse_map(path: Path) -> tuple[int, int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or lines[0] != "type octile" or lines[3] != "map":
        raise ValueError(f"invalid MovingAI map header: {path}")
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError(f"MovingAI map dimensions do not match header: {path}")
    walkable = sum(character in ".GS" for row in grid for character in row)
    return width, height, walkable


def _parse_scenario(path: Path, *, map_id: str, width: int, height: int) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "version 1":
        raise ValueError(f"invalid MovingAI scenario header: {path}")
    buckets: set[int] = set()
    starts: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    distances: list[float] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(f"invalid scenario row {path}:{line_number}")
        bucket = int(fields[0])
        row_map, row_width, row_height = fields[1], int(fields[2]), int(fields[3])
        sx, sy, gx, gy = map(int, fields[4:8])
        distance = float(fields[8])
        if (
            bucket < 0
            or row_map != f"{map_id}.map"
            or (row_width, row_height) != (width, height)
            or not (0 <= sx < width and 0 <= gx < width and 0 <= sy < height and 0 <= gy < height)
            or not math.isfinite(distance)
            or distance < 0.0
        ):
            raise ValueError(f"invalid scenario metadata {path}:{line_number}")
        buckets.add(bucket)
        starts.add((sx, sy))
        goals.add((gx, gy))
        distances.append(distance)
    return {
        "scenario_rows": len(distances),
        "bucket_count": len(buckets),
        "unique_start_count": len(starts),
        "unique_goal_count": len(goals),
        "median_shortest_distance": median_low(sorted(distances)),
    }


def validate_static_sources(project_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    static = config["static_load_design"]["per_map_static_metadata"]
    for split in config["allowed_splits"]:
        for map_id in config["map_splits"][split]:
            source = config["registered_sources"][map_id]
            map_path = registered_input(project_root, source["map"], label=f"{map_id} map")
            scenario_path = registered_input(
                project_root, source["scenario"], label=f"{map_id} scenario"
            )
            width, height, walkable = _parse_map(map_path)
            scenario = _parse_scenario(
                scenario_path, map_id=map_id, width=width, height=height
            )
            expected = static[map_id]
            for key in ("walkable_cells", "scenario_rows", "bucket_count"):
                actual = walkable if key == "walkable_cells" else scenario[key]
                if actual != expected[key]:
                    raise ValueError(f"{map_id} static metadata changed: {key}")
            if not math.isclose(
                scenario["median_shortest_distance"],
                expected["median_shortest_distance"],
                rel_tol=0.0,
                abs_tol=5e-9,
            ):
                raise ValueError(f"{map_id} median shortest distance changed")
            max_load = config["per_map_load_grid"][map_id][1]
            if min(scenario["unique_start_count"], scenario["unique_goal_count"]) < max_load:
                raise ValueError(f"{map_id} cannot supply registered high load")
            build = build_registered_bucket_mix_variants(
                scenario_path,
                expected_source_sha256=source["scenario"]["sha256"],
                map_id=map_id,
                registered_loads=config["per_map_load_grid"][map_id],
                allowed_map_ids=(*TRAIN_MAPS, *DEVELOPMENT_MAPS),
                sealed_map_ids=SEALED_FINAL_MAPS,
            )
            variant_capacity = {
                variant.variant_id: variant.unique_capacity for variant in build.variants
            }
            if set(variant_capacity) != set(VARIANT_IDS) or any(
                capacity < max_load for capacity in variant_capacity.values()
            ):
                raise ValueError(f"{map_id} bucket-mix variant capacity is below high load")
            observed[map_id] = {
                "split": split,
                "width": width,
                "height": height,
                "walkable_cells": walkable,
                **scenario,
                "variant_unique_capacity": variant_capacity,
                "registered_loads": config["per_map_load_grid"][map_id],
            }
    return observed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def run_preflight(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    project_root = path.parents[1]
    config = _read_json(path)
    validate_config(config)
    parent = config["parent_h1_registration"]
    registered_input(project_root, parent["config"], label="parent H1 config")
    registered_input(project_root, parent["report"], label="parent H1 report")
    observed = validate_static_sources(project_root, config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "REGISTERED",
        "passed": True,
        "config_path": path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(path),
        "allowed_splits": config["allowed_splits"],
        "map_splits": config["map_splits"],
        "registered_map_count": 16,
        "registered_source_file_count": 32,
        "variant_ids": list(VARIANT_IDS),
        "per_map_load_grid": config["per_map_load_grid"],
        "expected_task_count": 64,
        "expected_episode_count": 128,
        "workers": 20,
        "reset_only_qualification": config["reset_only_qualification"],
        "static_source_validation_passed": True,
        "static_source_observations": observed,
        "source_v1_outcomes_read": False,
        "sealed_final_semantic_access": False,
        "sealed_final_source_registered_here": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }
    _atomic_json(Path(output).resolve() / "registration_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_OUTPUT_NAME",
    "DEVELOPMENT_MAPS",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "SOLVER_SEEDS",
    "TRAIN_MAPS",
    "VARIANT_IDS",
    "run_preflight",
    "validate_config",
    "validate_static_sources",
]
