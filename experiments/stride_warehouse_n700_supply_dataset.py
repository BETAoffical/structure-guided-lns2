from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import registered_input, sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _derived_endpoint_seed,
    _derived_endpoints,
    _four_neighbor_distances,
    _largest_four_connected_component,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


CONFIG_SCHEMA = "lns2.stride.warehouse_n700_supply_dataset_config.v1"
DATASET_SCHEMA = "lns2.stride.warehouse_n700_supply_dataset.v1"
Q0_SCHEMA = "lns2.stride.warehouse_n700_supply_dataset_q0.v1"
STATUS_SCHEMA = "lns2.stride.warehouse_n700_supply_dataset_status.v1"
EXPERIMENT_ID = "stride-warehouse-n700-supply-dataset-v1"
SCIENTIFIC_STATUS = (
    "preregistered_geometry_only_n700_supply_before_any_reset_screen_or_controller"
)

MAP_ID = "warehouse-10-20-10-2-1"
MAP_CODE = "w1020a"
MAP_SHA256 = "c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0"
MAP_ARCHIVE_SHA256 = (
    "96db9e07ca14986a2786e1771a7eab7686b89d628e67700bb4b3c02b84c4c462"
)
MASTER_SEED = 20260808
TASK_SEEDS = (233, 277)
TASK_VARIANTS = ("opposite_exchange", "uniform_random")
AGENT_COUNT = 700
TASK_IDS = (
    "w1020a__oe__t0233__n0700",
    "w1020a__oe__t0277__n0700",
    "w1020a__ur__t0233__n0700",
    "w1020a__ur__t0277__n0700",
)
ENDPOINT_SEEDS = {
    "w1020a__oe__t0233__n0700": 898049359781102072,
    "w1020a__oe__t0277__n0700": 9814746312160663332,
    "w1020a__ur__t0233__n0700": 3565506677784553453,
    "w1020a__ur__t0277__n0700": 17835522936825645677,
}
FORBIDDEN_OUTPUT_ROOTS = (
    "build/wh-f16-v1-r2",
    "build/wh-f16-v2-r2",
    "build/movingai-dev",
    "build/stride-v2first-consensus16-warehouse-n800-supply-v1",
    "build/stride-v2first-consensus16-warehouse-n800-supply-v2",
    "build/stride-v2first-consensus16-warehouse-n800-supply-v3",
)
OUTPUT_CONTRACT = {
    "dataset_subdirectory": "dataset",
    "same_identity_is_idempotent": True,
    "foreign_or_partial_dataset_rejected": True,
    "old_r2_or_n800_output_reuse_forbidden": True,
    "reset_or_controller_invocation": False,
}
CLAIM_BOUNDARY = (
    "geometry_only_four_task_n700_state_supply_not_reset_repair_ttf_or_"
    "performance_evidence"
)


def _task_id(variant: str, task_seed: int) -> str:
    variant_code = {"opposite_exchange": "oe", "uniform_random": "ur"}[variant]
    return f"{MAP_CODE}__{variant_code}__t{task_seed:04d}__n{AGENT_COUNT:04d}"


def task_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    del config
    rows = [
        {
            "map_id": MAP_ID,
            "variant": variant,
            "task_seed": task_seed,
            "agent_count": AGENT_COUNT,
            "task_id": _task_id(variant, task_seed),
            "endpoint_seed": ENDPOINT_SEEDS[_task_id(variant, task_seed)],
        }
        for task_seed in TASK_SEEDS
        for variant in TASK_VARIANTS
    ]
    return sorted(rows, key=lambda row: str(row["task_id"]))


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("Warehouse N700 supply dataset identity changed")
    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("split") != SPLIT
        or cohort.get("map_id") != MAP_ID
        or cohort.get("map_code") != MAP_CODE
        or int(cohort.get("master_seed", -1)) != MASTER_SEED
        or tuple(map(int, cohort.get("task_seeds") or ())) != TASK_SEEDS
        or tuple(map(str, cohort.get("task_variants") or ())) != TASK_VARIANTS
        or int(cohort.get("agent_count", -1)) != AGENT_COUNT
        or tuple(sorted(map(str, cohort.get("task_ids") or ()))) != TASK_IDS
        or cohort.get("endpoint_seed_contract")
        != "sha256_compact_json_first_8_bytes_big_endian"
        or {str(key): int(value) for key, value in dict(cohort.get("endpoint_seeds") or {}).items()}
        != ENDPOINT_SEEDS
    ):
        raise ValueError("Warehouse N700 supply dataset cohort changed")
    inputs = dict(config.get("inputs") or {})
    map_spec = dict(inputs.get("map") or {})
    archive_spec = dict(inputs.get("map_archive") or {})
    if (
        str(map_spec.get("sha256") or "") != MAP_SHA256
        or str(archive_spec.get("sha256") or "") != MAP_ARCHIVE_SHA256
        or archive_spec.get("member") != f"{MAP_ID}.map"
        or inputs.get("official_scenario_archive_used") is not False
        or dict(config.get("output_contract") or {}) != OUTPUT_CONTRACT
    ):
        raise ValueError("Warehouse N700 supply dataset input/output contract changed")
    registered_input(root, map_spec, label="Warehouse N700 map")
    registered_input(root, archive_spec, label="Warehouse N700 map archive")
    for item in task_specs(config):
        observed = _derived_endpoint_seed(
            MASTER_SEED,
            MAP_ID,
            int(item["task_seed"]),
            str(item["variant"]),
            AGENT_COUNT,
        )
        if observed != int(item["endpoint_seed"]):
            raise ValueError("Warehouse N700 registered endpoint seed changed")
    return path, root, config


def _guard_output(root: Path, output: str | Path) -> Path:
    resolved = Path(output).resolve()
    for relative in FORBIDDEN_OUTPUT_ROOTS:
        forbidden = (root / relative).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(
                "Warehouse N700 dataset output must not reuse a protected old/source root"
            )
    return resolved


def _identity_fingerprint(path: Path, config: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "experiment_id": EXPERIMENT_ID,
            "implementation_sha256": sha256_file(Path(__file__)),
            "cohort": dict(config["cohort"]),
            "inputs": dict(config["inputs"]),
        }
    )


def _artifact_hashes(dataset: Path, manifest: list[Mapping[str, Any]]) -> dict[str, Any]:
    split = dataset / SPLIT
    tasks: dict[str, Any] = {}
    for row in manifest:
        task_id = str(row["task_id"])
        tasks[task_id] = {
            "scenario_sha256": sha256_file(split / str(row["scenario_file"])),
            "task_sha256": sha256_file(split / str(row["task_file"])),
        }
    return {
        "map_sha256": sha256_file(split / "maps" / f"{MAP_ID}.map"),
        "map_metadata_sha256": sha256_file(split / "maps" / f"{MAP_ID}.json"),
        "manifest_sha256": sha256_file(split / "manifest.jsonl"),
        "q0_geometry_audit_sha256": sha256_file(dataset / "q0_geometry_audit.json"),
        "tasks": tasks,
    }


def _audit_dataset(
    path: Path,
    config: Mapping[str, Any],
    dataset: Path,
    summary: Mapping[str, Any],
) -> None:
    split = dataset / SPLIT
    expected_fingerprint = _identity_fingerprint(path, config)
    if (
        summary.get("schema") != DATASET_SCHEMA
        or summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("configuration_fingerprint") != expected_fingerprint
        or summary.get("config_sha256") != sha256_file(path)
        or summary.get("implementation_sha256") != sha256_file(Path(__file__))
        or summary.get("solver_or_controller_invoked") is not False
        or summary.get("old_r2_or_n800_artifacts_modified") is not False
    ):
        raise ValueError("Warehouse N700 dataset summary identity changed")
    manifest = _read_jsonl(split / "manifest.jsonl")
    if len(manifest) != 4 or tuple(sorted(str(row.get("task_id")) for row in manifest)) != TASK_IDS:
        raise ValueError("Warehouse N700 dataset manifest identity changed")
    if len({str(row.get("task_id")) for row in manifest}) != 4:
        raise ValueError("Warehouse N700 dataset manifest has duplicate task IDs")
    q0 = _read_json(dataset / "q0_geometry_audit.json")
    if (
        q0.get("schema") != Q0_SCHEMA
        or q0.get("experiment_id") != EXPERIMENT_ID
        or q0.get("passed") is not True
        or q0.get("solver_or_controller_invoked") is not False
        or int(q0.get("task_count", -1)) != 4
    ):
        raise ValueError("Warehouse N700 Q0 identity changed")
    q0_by_id = {str(row.get("task_id")): dict(row) for row in q0.get("checks") or ()}
    if tuple(sorted(q0_by_id)) != TASK_IDS:
        raise ValueError("Warehouse N700 Q0 task coverage changed")
    for row in manifest:
        task_id = str(row["task_id"])
        expected = next(item for item in task_specs(config) if item["task_id"] == task_id)
        if (
            row.get("map_id") != MAP_ID
            or int(row.get("agent_count", -1)) != AGENT_COUNT
            or row.get("scenario_file") != f"scenarios/{task_id}.scen"
            or row.get("task_file") != f"tasks/{task_id}.json"
            or row.get("map_file") != f"maps/{MAP_ID}.map"
        ):
            raise ValueError(f"Warehouse N700 manifest row changed: {task_id}")
        task = _read_json(split / str(row["task_file"]))
        if (
            task.get("benchmark_id") != MAP_ID
            or task.get("od_variant") != expected["variant"]
            or int(task.get("task_seed", -1)) != int(expected["task_seed"])
            or int(task.get("endpoint_seed", -1)) != int(expected["endpoint_seed"])
            or int(task.get("agent_count", -1)) != AGENT_COUNT
            or task.get("source_map_sha256") != MAP_SHA256
            or task.get("unique_starts") is not True
            or task.get("unique_goals") is not True
            or int(task.get("fixed_point_count", -1)) != 0
            or task.get("scenario_sha256")
            != sha256_file(split / str(row["scenario_file"]))
        ):
            raise ValueError(f"Warehouse N700 task metadata changed: {task_id}")
        check = q0_by_id[task_id]
        if (
            check.get("passed") is not True
            or check.get("all_endpoints_reachable") is not True
            or check.get("unique_starts") is not True
            or check.get("unique_goals") is not True
            or int(check.get("fixed_point_count", -1)) != 0
            or int(check.get("endpoint_seed", -1)) != int(expected["endpoint_seed"])
        ):
            raise ValueError(f"Warehouse N700 Q0 row changed: {task_id}")
    artifacts = _artifact_hashes(dataset, manifest)
    if dict(summary.get("artifacts") or {}) != artifacts:
        raise ValueError("Warehouse N700 dataset artifact hashes changed")


def plan(config_path: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "command": "plan",
        "config_sha256": sha256_file(path),
        "map_id": MAP_ID,
        "map_sha256": MAP_SHA256,
        "map_archive_sha256": MAP_ARCHIVE_SHA256,
        "agent_count": AGENT_COUNT,
        "task_count": 4,
        "tasks": task_specs(config),
        "solver_or_controller_invoked": False,
        "old_r2_or_n800_artifacts_modified": False,
    }


def prepare_dataset(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = _guard_output(root, output)
    dataset = output / str(dict(config["output_contract"])["dataset_subdirectory"])
    summary_path = dataset / "dataset_summary.json"
    fingerprint = _identity_fingerprint(path, config)
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("configuration_fingerprint") != fingerprint:
            raise ValueError("Warehouse N700 dataset belongs to a different identity")
        _audit_dataset(path, config, dataset, summary)
        return summary
    if dataset.is_dir() and any(dataset.iterdir()):
        raise ValueError("Warehouse N700 dataset is non-empty without its identity")

    split = dataset / SPLIT
    map_spec = dict(dict(config["inputs"])["map"])
    source_map = registered_input(root, map_spec, label="Warehouse N700 map")
    map_path = split / "maps" / f"{MAP_ID}.map"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_map, map_path)
    rows, cols, _grid, passable = _movingai_passable_cells(map_path)
    component = _largest_four_connected_component(passable)
    metrics = _map_metrics(map_path)
    metadata_path = split / "maps" / f"{MAP_ID}.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "benchmark_id": MAP_ID,
            "source": "checksum-pinned existing MovingAI Warehouse map",
            "source_path": str(map_spec["path"]),
            "source_archive_sha256": MAP_ARCHIVE_SHA256,
            "map_sha256": sha256_file(map_path),
            "largest_four_connected_component": len(component),
            "topology_metrics": metrics,
        },
    )

    manifest: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    for item in task_specs(config):
        task_seed = int(item["task_seed"])
        variant = str(item["variant"])
        task_id = str(item["task_id"])
        endpoint_seed = _derived_endpoint_seed(
            MASTER_SEED, MAP_ID, task_seed, variant, AGENT_COUNT
        )
        if endpoint_seed != int(item["endpoint_seed"]):
            raise RuntimeError(f"Warehouse N700 endpoint seed changed: {task_id}")
        starts, goals = _derived_endpoints(
            component, AGENT_COUNT, variant, endpoint_seed
        )
        distances = _four_neighbor_distances(passable, starts, goals)
        scenario_path = split / "scenarios" / f"{task_id}.scen"
        _write_derived_scenario(
            scenario_path,
            map_path.name,
            rows,
            cols,
            starts,
            goals,
            distances,
        )
        task_payload = {
            "schema_version": 1,
            "task_semantics_version": 1,
            "task_semantics": "geometry-only derived Warehouse MAPF OD",
            "benchmark_id": MAP_ID,
            "source_map_sha256": sha256_file(map_path),
            "od_variant": variant,
            "task_seed": task_seed,
            "endpoint_seed": endpoint_seed,
            "agent_count": AGENT_COUNT,
            "unique_starts": len(set(starts)) == AGENT_COUNT,
            "unique_goals": len(set(goals)) == AGENT_COUNT,
            "fixed_point_count": sum(
                start == goal for start, goal in zip(starts, goals)
            ),
            "distance_metric": "four_neighbor_unit",
            "minimum_shortest_distance": min(distances),
            "maximum_shortest_distance": max(distances),
            "mean_shortest_distance": statistics.fmean(distances),
            "scenario_sha256": sha256_file(scenario_path),
        }
        task_path = split / "tasks" / f"{task_id}.json"
        _write_json(task_path, task_payload)
        q0_passed = bool(
            task_payload["unique_starts"]
            and task_payload["unique_goals"]
            and task_payload["fixed_point_count"] == 0
            and len(distances) == AGENT_COUNT
            and min(distances) > 0
        )
        q0_rows.append(
            {
                "task_id": task_id,
                "variant": variant,
                "task_seed": task_seed,
                "agent_count": AGENT_COUNT,
                "endpoint_seed": endpoint_seed,
                "endpoint_fingerprint": _fingerprint(
                    {"starts": starts, "goals": goals}
                ),
                "scenario_sha256": task_payload["scenario_sha256"],
                "unique_starts": task_payload["unique_starts"],
                "unique_goals": task_payload["unique_goals"],
                "fixed_point_count": task_payload["fixed_point_count"],
                "all_endpoints_reachable": len(distances) == AGENT_COUNT,
                "passed": q0_passed,
            }
        )
        manifest.append(
            {
                "split": SPLIT,
                "source_group": "movingai",
                "instance_origin": "warehouse_n700_supply_derived_od",
                "map_id": MAP_ID,
                "task_id": task_id,
                "map_file": f"maps/{map_path.name}",
                "scenario_file": f"scenarios/{scenario_path.name}",
                "map_metadata_file": f"maps/{metadata_path.name}",
                "task_file": f"tasks/{task_path.name}",
                "layout_mode": "warehouse",
                "layout_variant": MAP_ID,
                "scenario_type": f"warehouse_derived_{variant}",
                "task_variant": f"{variant}_seed_{task_seed}_agents_{AGENT_COUNT}",
                "agent_count": AGENT_COUNT,
                "topology_metrics": metrics,
                "dominant_flow_ratio": 0.0,
                "hotspot_skew": 0.0,
                "required_bottleneck_crossing_ratio": 0.0,
                "mean_shortest_distance": task_payload["mean_shortest_distance"],
            }
        )

    manifest.sort(key=lambda row: str(row["task_id"]))
    q0_rows.sort(key=lambda row: str(row["task_id"]))
    if len(manifest) != 4 or tuple(row["task_id"] for row in manifest) != TASK_IDS:
        raise RuntimeError("Warehouse N700 dataset dimensions changed")
    q0 = {
        "schema": Q0_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_count": 4,
        "checks": q0_rows,
        "passed": all(row["passed"] for row in q0_rows),
        "solver_or_controller_invoked": False,
        "old_r2_or_n800_artifacts_modified": False,
    }
    if q0["passed"] is not True:
        raise RuntimeError("Warehouse N700 Q0 geometry audit failed")
    _write_jsonl(split / "manifest.jsonl", manifest)
    _write_json(dataset / "q0_geometry_audit.json", q0)
    artifacts = _artifact_hashes(dataset, manifest)
    summary = {
        "schema": DATASET_SCHEMA,
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "configuration_fingerprint": fingerprint,
        "config_sha256": sha256_file(path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "dataset_revision": EXPERIMENT_ID,
        "source": "one checksum-pinned MovingAI Warehouse map",
        "task_semantics": "geometry_only_derived_not_official_mapf_scenarios",
        "claim_boundary": CLAIM_BOUNDARY,
        "splits": {
            SPLIT: {
                "map_count": 1,
                "instance_count": 4,
                "source_counts": {"movingai": 4},
                "layout_counts": {"warehouse": 4},
            }
        },
        "artifacts": artifacts,
        "solver_or_controller_invoked": False,
        "old_r2_or_n800_artifacts_modified": False,
    }
    _write_json(summary_path, summary)
    _audit_dataset(path, config, dataset, summary)
    return summary


__all__ = [
    "AGENT_COUNT",
    "CONFIG_SCHEMA",
    "DATASET_SCHEMA",
    "ENDPOINT_SEEDS",
    "EXPERIMENT_ID",
    "MAP_ARCHIVE_SHA256",
    "MAP_ID",
    "MAP_SHA256",
    "Q0_SCHEMA",
    "STATUS_SCHEMA",
    "TASK_IDS",
    "TASK_SEEDS",
    "TASK_VARIANTS",
    "load_config",
    "plan",
    "prepare_dataset",
    "task_specs",
]
