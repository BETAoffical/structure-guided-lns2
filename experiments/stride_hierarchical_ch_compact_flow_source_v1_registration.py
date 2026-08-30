"""Registration-only boundary for compact-flow hierarchical C/H source data.

The design reuses checksum-pinned, project-derived MovingAI tasks and their
historical reset-only qualification as sequential design evidence.  It does
not import any source-v2 reset row, access sealed-final map/scenario content,
collect H1 outcomes, fit a model, or authorize training.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import registered_input, sha256_file


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_source_registration.v1"
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_source_registration_report.v1"
)
TASK_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_registered_task.v1"
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_source_v1"
DEFAULT_OUTPUT_NAME = "stride-hierarchical-ch-compact-flow-source-v1-registration"

TRAIN_MAPS = (
    "den404d",
    "lak101d",
    "den201d",
    "hrt002d",
    "den009d",
    "den101d",
    "den203d",
    "den308d",
)
DEVELOPMENT_MAPS = (
    "lak108d",
    "lak110d",
    "ost102d",
    "den408d",
    "den202d",
    "den207d",
    "den998d",
    "den020d",
)
OD_VARIANTS = ("opposite_exchange", "uniform_random")
SOLVER_SEEDS = (41, 42)
HISTORICAL_SOLVER_SEEDS = (1, 2, 3)

LEGACY_H1_MAPS = (
    "den520d",
    "maze-128-128-10",
    "warehouse-10-20-10-2-1",
    "maze-128-128-1",
    "lak303d",
    "random-32-32-10",
    "room-32-32-4",
    "random-32-32-20",
    "maze-32-32-2",
    "random-64-64-10",
    "warehouse-20-40-10-2-1",
    "maze-32-32-4",
    "random-64-64-20",
    "room-64-64-8",
    "warehouse-20-40-10-2-2",
    "room-64-64-16",
)
SEALED_FINAL_MAP_IDS = (
    "brc503d",
    "combat2",
    "den501d",
    "hrt201n",
    "lak200d",
    "lgt601d",
    "orz500d",
    "ost002d",
)

EXPECTED_MAP_REGISTRATION: dict[str, dict[str, Any]] = {
    "den404d": {"split": "train", "source_revision": "targeted_load_v5", "capacity_free_cells": 358, "capacity_stratum": "ultra", "loads": [80, 180, 280], "task_seeds": [23, 29]},
    "lak101d": {"split": "train", "source_revision": "targeted_load_v5", "capacity_free_cells": 318, "capacity_stratum": "ultra", "loads": [70, 160, 240], "task_seeds": [23, 29]},
    "den201d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 538, "capacity_stratum": "small", "loads": [40, 100, 180], "task_seeds": [11, 17]},
    "hrt002d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 754, "capacity_stratum": "small", "loads": [50, 100, 180], "task_seeds": [11, 17]},
    "den009d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 1003, "capacity_stratum": "medium", "loads": [60, 120, 200], "task_seeds": [11, 17]},
    "den101d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 1360, "capacity_stratum": "medium", "loads": [80, 160, 240], "task_seeds": [11, 17]},
    "den203d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 2629, "capacity_stratum": "large", "loads": [100, 200, 320], "task_seeds": [11, 17]},
    "den308d": {"split": "train", "source_revision": "map_derived_v4", "capacity_free_cells": 3155, "capacity_stratum": "large", "loads": [100, 200, 320], "task_seeds": [11, 17]},
    "lak108d": {"split": "development", "source_revision": "targeted_load_v5", "capacity_free_cells": 286, "capacity_stratum": "ultra", "loads": [60, 140, 220], "task_seeds": [23, 29]},
    "lak110d": {"split": "development", "source_revision": "targeted_load_v5", "capacity_free_cells": 168, "capacity_stratum": "ultra", "loads": [40, 90, 140], "task_seeds": [23, 29]},
    "ost102d": {"split": "development", "source_revision": "targeted_load_v5", "capacity_free_cells": 249, "capacity_stratum": "ultra", "loads": [50, 120, 200], "task_seeds": [23, 29]},
    "den408d": {"split": "development", "source_revision": "targeted_load_v5", "capacity_free_cells": 548, "capacity_stratum": "small", "loads": [100, 240, 400], "task_seeds": [23, 29]},
    "den202d": {"split": "development", "source_revision": "map_derived_v4", "capacity_free_cells": 593, "capacity_stratum": "small", "loads": [40, 100, 180], "task_seeds": [11, 17]},
    "den207d": {"split": "development", "source_revision": "map_derived_v4", "capacity_free_cells": 874, "capacity_stratum": "medium", "loads": [60, 120, 200], "task_seeds": [11, 17]},
    "den998d": {"split": "development", "source_revision": "map_derived_v4", "capacity_free_cells": 1853, "capacity_stratum": "medium", "loads": [80, 160, 240], "task_seeds": [11, 17]},
    "den020d": {"split": "development", "source_revision": "map_derived_v4", "capacity_free_cells": 3102, "capacity_stratum": "large", "loads": [100, 200, 320], "task_seeds": [11, 17]},
}

EXPECTED_CAPACITY_STRATA = {
    "metric": "topology_metrics.free_cell_count",
    "ultra": {"minimum_inclusive": 0, "maximum_inclusive": 400},
    "small": {"minimum_inclusive": 401, "maximum_inclusive": 800},
    "medium": {"minimum_inclusive": 801, "maximum_inclusive": 2000},
    "large": {"minimum_inclusive": 2001, "maximum_inclusive": None},
}
EXPECTED_CAPACITY_COUNTS = {
    "train": {"ultra": 2, "small": 2, "medium": 2, "large": 2},
    "development": {"ultra": 3, "small": 2, "medium": 2, "large": 1},
}


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL row: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _safe_child(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or not relative or ".." in value.parts:
        raise ValueError(f"{label} must be a safe relative path")
    resolved_root = root.resolve()
    result = (resolved_root / value).resolve()
    if result != resolved_root and resolved_root not in result.parents:
        raise ValueError(f"{label} escapes its registered dataset root")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_config(config: dict[str, Any], *, allow_unpinned_manifest: bool = False) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("status") != "REGISTERED"
        or config.get("scientific_status") != "sequential_design_only"
    ):
        raise ValueError("compact-flow source registration identity changed")
    expected_splits = {
        "train": list(TRAIN_MAPS),
        "development": list(DEVELOPMENT_MAPS),
    }
    if config.get("map_splits") != expected_splits:
        raise ValueError("compact-flow map split membership/order changed")
    selected = set(TRAIN_MAPS) | set(DEVELOPMENT_MAPS)
    if len(selected) != 16 or set(TRAIN_MAPS) & set(DEVELOPMENT_MAPS):
        raise ValueError("train/development must be exactly map-disjoint")
    if selected & set(LEGACY_H1_MAPS):
        raise ValueError("legacy H1 map leaked into compact-flow registration")
    if selected & set(SEALED_FINAL_MAP_IDS):
        raise ValueError("sealed-final map ID leaked into compact-flow registration")
    if config.get("map_registration") != EXPECTED_MAP_REGISTRATION:
        raise ValueError("per-map source/load/seed/capacity registration changed")
    if config.get("capacity_strata") != EXPECTED_CAPACITY_STRATA:
        raise ValueError("capacity stratum definition changed")
    if config.get("required_capacity_counts") != EXPECTED_CAPACITY_COUNTS:
        raise ValueError("capacity stratum quotas changed")
    observed_counts: dict[str, collections.Counter[str]] = {
        "train": collections.Counter(),
        "development": collections.Counter(),
    }
    for registration in config["map_registration"].values():
        observed_counts[registration["split"]][registration["capacity_stratum"]] += 1
    if {key: dict(value) for key, value in observed_counts.items()} != EXPECTED_CAPACITY_COUNTS:
        raise ValueError("per-map registrations do not realize capacity quotas")

    product = config.get("task_product")
    expected_product = {
        "od_variants": list(OD_VARIANTS),
        "solver_seeds": list(SOLVER_SEEDS),
        "loads_per_map": 3,
        "task_seeds_per_map": 2,
        "tasks_per_map": 12,
        "tasks_per_split": 96,
        "registered_task_count": 192,
        "episodes_per_split": 192,
        "expected_episode_count": 384,
        "workers": 16,
    }
    if product != expected_product:
        raise ValueError("complete compact-flow task product changed")
    aliases = {
        "expected_task_count": product["registered_task_count"],
        "expected_episode_count": product["expected_episode_count"],
        "solver_seeds": product["solver_seeds"],
        "workers": product["workers"],
    }
    if any(config.get(key) != value for key, value in aliases.items()):
        raise ValueError("top-level task-product alias changed")

    legacy = config.get("legacy_h1_exclusion")
    if legacy != {"map_ids": list(LEGACY_H1_MAPS), "overlap_allowed": False}:
        raise ValueError("legacy H1 exclusion contract changed")
    registered_manifest = config.get("registered_task_manifest")
    if not isinstance(registered_manifest, dict):
        raise ValueError("registered task manifest contract is required")
    expected_counts = {
        "schema": TASK_SCHEMA,
        "row_count": 192,
        "unique_map_file_count": 16,
        "unique_map_metadata_file_count": 16,
        "unique_scenario_file_count": 192,
        "unique_task_file_count": 192,
    }
    for key, expected in expected_counts.items():
        if registered_manifest.get(key) != expected:
            raise ValueError(f"registered task manifest changed: {key}")
    if registered_manifest.get("path") != (
        "build/stride-hierarchical-ch-compact-flow-source-v1-registration/"
        "registered_tasks.jsonl"
    ):
        raise ValueError("registered task manifest path changed")
    manifest_hash = registered_manifest.get("sha256")
    if not _valid_sha256(manifest_hash) and not (
        allow_unpinned_manifest and manifest_hash == "__TO_BE_COMPUTED__"
    ):
        raise ValueError("registered task manifest SHA256 is not frozen")

    parent = config.get("parent_source_v2_terminal_failure")
    if not isinstance(parent, dict):
        raise ValueError("source-v2 terminal failure must be pinned")
    if (
        parent.get("required_status") != "STATE_SUPPLY_FAIL_NO_BACKFILL"
        or parent.get("required_qualification_job_count") != 128
        or parent.get("required_realized_episode_count") != 0
        or parent.get("completed_v2_reset_rows_imported") != 0
        or parent.get("full_schedule_restart") is not True
        or parent.get("preserve_parent_artifacts") is not True
    ):
        raise ValueError("source-v2 terminal failure/import boundary changed")
    if parent.get("run_fingerprints") != {
        "train": "612148ec368292aa4489c7f84d03d5d7b82698e4ae84b693114ebf7eac052971",
        "development": "a39cfaee4c1784c71b42e8675635e6867a26466f1f02b1c0fc91515f66521f58",
    }:
        raise ValueError("source-v2 run fingerprints changed")
    artifacts = parent.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 10:
        raise ValueError("complete source-v2 terminal artifact pins are required")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not _valid_sha256(item.get("sha256"))
        for item in artifacts.values()
    ):
        raise ValueError("invalid source-v2 terminal artifact pin")

    sources = config.get("source_revisions")
    if not isinstance(sources, dict) or set(sources) != {
        "map_derived_v4",
        "targeted_load_v5",
    }:
        raise ValueError("exact v4/v5 source revisions are required")
    for source_id, source in sources.items():
        if source.get("historical_solver_seeds") != list(HISTORICAL_SOLVER_SEEDS):
            raise ValueError(f"historical solver seeds changed: {source_id}")
        if source.get("usage") != "sequential_design_evidence_only":
            raise ValueError(f"historical evidence boundary changed: {source_id}")
        for key in (
            "candidate_config",
            "candidate_manifest",
            "qualification_config",
            "qualification_manifest",
            "qualification_report",
        ):
            item = source.get(key)
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not _valid_sha256(item.get("sha256"))
            ):
                raise ValueError(f"invalid historical evidence pin: {source_id}/{key}")

    boundary = config.get("claim_boundary")
    if boundary != {
        "sequential_design_only": True,
        "historical_qualification_used_for_design": True,
        "historical_rows_imported_as_new_source_rows": 0,
        "completed_v2_reset_rows_imported": 0,
        "sealed_final_semantic_access": False,
        "sealed_final_source_registered": False,
        "h1_executed": False,
        "model_fit_executed": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }:
        raise ValueError("compact-flow claim boundary changed")


def _validate_parent_failure(project_root: Path, config: dict[str, Any]) -> None:
    parent = config["parent_source_v2_terminal_failure"]
    paths: dict[str, Path] = {}
    for name, registration in parent["artifacts"].items():
        paths[name] = registered_input(
            project_root, registration, label=f"source-v2 terminal artifact {name}"
        )
    terminal = _read_json(paths["terminal_trust"])
    if (
        terminal.get("status") != parent["required_status"]
        or terminal.get("qualification_job_count")
        != parent["required_qualification_job_count"]
        or terminal.get("realized_episode_count")
        != parent["required_realized_episode_count"]
        or terminal.get("sealed_final_semantic_access") is not False
        or terminal.get("training_authorized") is not False
    ):
        raise ValueError("pinned source-v2 artifact is not the registered terminal failure")


def _load_source_evidence(
    project_root: Path, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source_id, source in config["source_revisions"].items():
        paths = {
            key: registered_input(
                project_root, source[key], label=f"{source_id} {key}"
            )
            for key in (
                "candidate_config",
                "candidate_manifest",
                "qualification_config",
                "qualification_manifest",
                "qualification_report",
            )
        }
        candidate_config = _read_json(paths["candidate_config"])
        benchmarks = candidate_config.get("benchmarks")
        if not isinstance(benchmarks, list):
            raise ValueError(f"candidate benchmark list missing: {source_id}")
        benchmark_by_id = {
            str(item["id"]): item for item in benchmarks if isinstance(item, dict)
        }
        result[source_id] = {
            "dataset_root": (project_root / source["dataset_root"]).resolve(),
            "candidate_rows": _read_jsonl(paths["candidate_manifest"]),
            "qualification_rows": _read_jsonl(paths["qualification_manifest"]),
            "benchmark_by_id": benchmark_by_id,
        }
    return result


def collect_registered_tasks(
    project_root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    evidence = _load_source_evidence(project_root, config)
    records: list[dict[str, Any]] = []
    task_ids_by_map: dict[str, set[str]] = {}

    for split, map_ids in config["map_splits"].items():
        for map_id in map_ids:
            registration = config["map_registration"][map_id]
            source_id = registration["source_revision"]
            source = evidence[source_id]
            rows = [row for row in source["candidate_rows"] if row.get("map_id") == map_id]
            if len(rows) != 12:
                raise ValueError(f"{map_id} must have exactly 12 registered source tasks")
            expected_product = {
                (agent_count, od_variant, task_seed)
                for agent_count in registration["loads"]
                for od_variant in OD_VARIANTS
                for task_seed in registration["task_seeds"]
            }
            observed_product: set[tuple[int, str, int]] = set()
            task_ids: set[str] = set()
            benchmark = source["benchmark_by_id"].get(map_id)
            if not isinstance(benchmark, dict):
                raise ValueError(f"registered benchmark missing: {map_id}")
            if benchmark.get("agent_counts") != registration["loads"]:
                raise ValueError(f"candidate-config loads changed: {map_id}")

            for row in rows:
                dataset_root = source["dataset_root"]
                map_path = _safe_child(dataset_root, str(row["map_file"]), f"{map_id} map")
                metadata_path = _safe_child(
                    dataset_root, str(row["map_metadata_file"]), f"{map_id} map metadata"
                )
                scenario_path = _safe_child(
                    dataset_root, str(row["scenario_file"]), f"{map_id} scenario"
                )
                task_path = _safe_child(dataset_root, str(row["task_file"]), f"{map_id} task")
                task = _read_json(task_path)
                metadata = _read_json(metadata_path)
                agent_count = int(task.get("agent_count", -1))
                od_variant = str(task.get("od_variant", ""))
                task_seed = int(task.get("task_seed", -1))
                product_key = (agent_count, od_variant, task_seed)
                if product_key not in expected_product or product_key in observed_product:
                    raise ValueError(f"unexpected/duplicate task product row: {map_id}/{product_key}")
                observed_product.add(product_key)
                task_id = str(row.get("task_id", ""))
                if not task_id or task_id in task_ids:
                    raise ValueError(f"duplicate/empty task ID: {map_id}")
                task_ids.add(task_id)

                map_sha = sha256_file(map_path)
                metadata_sha = sha256_file(metadata_path)
                scenario_sha = sha256_file(scenario_path)
                task_sha = sha256_file(task_path)
                if map_sha != benchmark.get("member_sha256"):
                    raise ValueError(f"registered map SHA changed: {map_id}")
                if (
                    task.get("benchmark_id") != map_id
                    or task.get("source_map_sha256") != map_sha
                    or task.get("scenario_sha256") != scenario_sha
                    or int(row.get("agent_count", -1)) != agent_count
                ):
                    raise ValueError(f"task/map/scenario registration disagrees: {task_id}")
                free_cells = int(metadata["topology_metrics"]["free_cell_count"])
                if (
                    free_cells != registration["capacity_free_cells"]
                    or int(row["topology_metrics"]["free_cell_count"]) != free_cells
                ):
                    raise ValueError(f"capacity metadata changed: {map_id}")
                load_tier = ("low", "middle", "high")[
                    registration["loads"].index(agent_count)
                ]
                rel = lambda path: path.relative_to(project_root).as_posix()
                records.append(
                    {
                        "schema": TASK_SCHEMA,
                        "experiment_id": EXPERIMENT_ID,
                        "split": split,
                        "source_revision": source_id,
                        "map_id": map_id,
                        "capacity_free_cells": free_cells,
                        "capacity_stratum": registration["capacity_stratum"],
                        "load_tier": load_tier,
                        "agent_count": agent_count,
                        "od_variant": od_variant,
                        "task_seed": task_seed,
                        "task_id": task_id,
                        "map": {"path": rel(map_path), "sha256": map_sha},
                        "map_metadata": {
                            "path": rel(metadata_path),
                            "sha256": metadata_sha,
                        },
                        "scenario": {"path": rel(scenario_path), "sha256": scenario_sha},
                        "task": {"path": rel(task_path), "sha256": task_sha},
                    }
                )
            if observed_product != expected_product:
                raise ValueError(f"incomplete task product: {map_id}")
            task_ids_by_map[map_id] = task_ids

    split_order = {"train": 0, "development": 1}
    map_order = {
        map_id: index
        for index, map_id in enumerate((*TRAIN_MAPS, *DEVELOPMENT_MAPS))
    }
    records.sort(
        key=lambda row: (
            split_order[row["split"]],
            map_order[row["map_id"]],
            row["agent_count"],
            row["od_variant"],
            row["task_seed"],
        )
    )

    historical_summary: dict[str, dict[str, Any]] = {}
    for map_id, task_ids in task_ids_by_map.items():
        registration = config["map_registration"][map_id]
        rows = [
            row
            for row in evidence[registration["source_revision"]]["qualification_rows"]
            if row.get("map_id") == map_id
        ]
        expected_keys = {
            (task_id, seed) for task_id in task_ids for seed in HISTORICAL_SOLVER_SEEDS
        }
        observed_keys = {(str(row.get("task_id")), int(row.get("solver_seed", -1))) for row in rows}
        if len(rows) != 36 or observed_keys != expected_keys:
            raise ValueError(f"historical qualification product changed: {map_id}")
        if any(
            row.get("status") != "ok"
            or row.get("initial_complete") is not True
            or row.get("error") is not None
            for row in rows
        ):
            raise ValueError(f"historical qualification is not 36/36 valid: {map_id}")
        by_load: dict[str, Any] = {}
        for load_tier, agent_count in zip(("low", "middle", "high"), registration["loads"]):
            values = [int(row["initial_conflicts"]) for row in rows if int(row["agent_count"]) == agent_count]
            if len(values) != 12:
                raise ValueError(f"historical load support changed: {map_id}/{agent_count}")
            by_load[load_tier] = {
                "agent_count": agent_count,
                "count": 12,
                "minimum_initial_conflicts": min(values),
                "maximum_initial_conflicts": max(values),
                "mean_initial_conflicts": sum(values) / len(values),
                "nonzero_count": sum(value > 0 for value in values),
            }
        historical_summary[map_id] = {
            "source_revision": registration["source_revision"],
            "valid_reset_count": 36,
            "error_count": 0,
            "timeout_count": 0,
            "by_load": by_load,
        }
    return records, historical_summary


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def _report_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def run_preflight(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    project_root = path.parents[1]
    config = _read_json(path)
    validate_config(config)
    _validate_parent_failure(project_root, config)
    records, historical = collect_registered_tasks(project_root, config)
    payload = _jsonl_bytes(records)
    observed_manifest_sha = _sha256_bytes(payload)
    expected_manifest_sha = config["registered_task_manifest"]["sha256"]
    if observed_manifest_sha != expected_manifest_sha:
        raise ValueError("registered task manifest SHA256 differs from frozen config")
    unique_maps = {(row["map"]["path"], row["map"]["sha256"]) for row in records}
    unique_metadata = {
        (row["map_metadata"]["path"], row["map_metadata"]["sha256"])
        for row in records
    }
    unique_scenarios = {
        (row["scenario"]["path"], row["scenario"]["sha256"]) for row in records
    }
    unique_tasks = {(row["task"]["path"], row["task"]["sha256"]) for row in records}
    if (len(records), len(unique_maps), len(unique_metadata), len(unique_scenarios), len(unique_tasks)) != (
        192,
        16,
        16,
        192,
        192,
    ):
        raise ValueError("registered per-file task product counts changed")

    output_root = Path(output).resolve()
    manifest_path = output_root / "registered_tasks.jsonl"
    _atomic_write(manifest_path, payload)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "REGISTERED",
        "passed": True,
        "scientific_status": "sequential_design_only",
        "config_path": path.relative_to(project_root).as_posix(),
        "config_sha256": sha256_file(path),
        "map_splits": config["map_splits"],
        "capacity_strata": config["capacity_strata"],
        "required_capacity_counts": config["required_capacity_counts"],
        "map_registration": config["map_registration"],
        "task_product": config["task_product"],
        "expected_task_count": config["expected_task_count"],
        "expected_episode_count": config["expected_episode_count"],
        "solver_seeds": config["solver_seeds"],
        "workers": config["workers"],
        "registered_task_manifest": {
            "path": _report_path(manifest_path, project_root),
            "schema": TASK_SCHEMA,
            "sha256": observed_manifest_sha,
            "row_count": len(records),
            "unique_map_file_count": len(unique_maps),
            "unique_map_metadata_file_count": len(unique_metadata),
            "unique_scenario_file_count": len(unique_scenarios),
            "unique_task_file_count": len(unique_tasks),
        },
        "historical_qualification": historical,
        "historical_qualification_used_for_design": True,
        "historical_rows_imported_as_new_source_rows": 0,
        "parent_source_v2_status": "STATE_SUPPLY_FAIL_NO_BACKFILL",
        "completed_v2_reset_rows_imported": 0,
        "legacy_h1_overlap": [],
        "sealed_final_semantic_access": False,
        "sealed_final_source_registered": False,
        "h1_executed": False,
        "model_fit_executed": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }
    _atomic_write(
        output_root / "registration_report.json",
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_OUTPUT_NAME",
    "DEVELOPMENT_MAPS",
    "EXPERIMENT_ID",
    "OD_VARIANTS",
    "REPORT_SCHEMA",
    "SOLVER_SEEDS",
    "TASK_SCHEMA",
    "TRAIN_MAPS",
    "collect_registered_tasks",
    "run_preflight",
    "validate_config",
]
