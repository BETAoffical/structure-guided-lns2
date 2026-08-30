from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_hierarchical_ch_h1_v2 import SOURCE_LOAD_GRID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_source_config.v1"
MATERIALIZATION_SCHEMA = "lns2.stride.hierarchical_ch_source_materialization.v1"
PLAN_SCHEMA = "lns2.stride.hierarchical_ch_source_plan.v1"
COLLECTION_TRUST_SCHEMA = "lns2.stride.hierarchical_ch_source_collection_trust.v1"
ALLOWED_SPLITS = ("train", "development")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{number}")
        rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _write_exact(path: Path, payload: bytes, *, resume: bool) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing output differs from deterministic source plan: {path}")
        if not resume:
            raise FileExistsError(f"output already exists; use --resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_project_path(project_root: Path, registered: str) -> Path:
    path = Path(registered)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"registered path must be project-relative: {registered}")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"registered path escapes the project: {registered}") from error
    return resolved


def _pinned_path(project_root: Path, pin: dict[str, Any], label: str) -> Path:
    path = _resolve_project_path(project_root, str(pin.get("path", "")))
    expected = str(pin.get("sha256", "")).lower()
    if len(expected) != 64 or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def _reject_sealed(value: str, sealed_ids: set[str], label: str) -> None:
    tokens = {value, value.split("__", 1)[0], Path(value).stem}
    if tokens & sealed_ids or value == "sealed_final":
        raise ValueError(f"sealed-final {label} is not authorized: {value}")


def load_registered_source_context(
    config_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_config_path = Path(config_path)
    if not source_config_path.is_absolute():
        source_config_path = (root / source_config_path).resolve()
    source_config = _read_json(source_config_path)
    if source_config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported hierarchical C/H source config")
    if source_config.get("claim_boundary", {}).get("training_authorized") is not False:
        raise ValueError("source collection must remain training_authorized=false")
    registration = dict(source_config.get("registered_design") or {})
    design_path = _pinned_path(root, dict(registration.get("config") or {}), "v2 config")
    report_path = _pinned_path(
        root, dict(registration.get("registration_report") or {}), "registration report"
    )
    design = _read_json(design_path)
    report = _read_json(report_path)
    design_sha = _sha256(design_path)
    if (
        report.get("schema") != registration.get("required_report_schema")
        or report.get("status") != registration.get("required_status")
        or report.get("passed") is not True
        or report.get("config_sha256") != design_sha
        or report.get("training_authorized") is not False
        or design.get("claim_boundary", {}).get("training_authorized") is not False
    ):
        raise ValueError("v2 registration is not a frozen REGISTERED no-train design")
    expected_grid = {key: list(value) for key, value in SOURCE_LOAD_GRID.items()}
    observed_grid = dict(design.get("new_source_design", {}).get("per_map_load_grid") or {})
    if observed_grid != expected_grid:
        raise ValueError("v2 per-map load grid differs from exact SOURCE_LOAD_GRID")
    if list(design.get("new_source_design", {}).get("source_seeds") or []) != [41, 42]:
        raise ValueError("v2 solver-seed registration changed")
    map_splits = {name: list(values) for name, values in design["map_splits"].items()}
    allowed_ids = {map_id for split in ALLOWED_SPLITS for map_id in map_splits[split]}
    sealed_ids = set(map_splits["sealed_final"])
    if (
        len(allowed_ids) != 16
        or allowed_ids & sealed_ids
        or set(expected_grid) != allowed_ids
        or any(report["map_splits"][split] != map_splits[split] for split in (*ALLOWED_SPLITS, "sealed_final"))
    ):
        raise ValueError("registered map split or SOURCE_LOAD_GRID integrity failed")
    encoded_source_config = json.dumps(source_config, sort_keys=True)
    if any(map_id in encoded_source_config for map_id in sealed_ids):
        raise ValueError("source execution config must not name sealed-final map IDs")
    source = dict(source_config.get("source") or {})
    if tuple(source.get("allowed_splits") or ()) != ALLOWED_SPLITS:
        raise ValueError("only train and development source splits are executable")
    collection = dict(source_config.get("collection") or {})
    base_path = _pinned_path(root, dict(collection.get("base_config") or {}), "collection base")
    controller_pin = dict(collection.get("controller_bundle") or {})
    controller_root = _resolve_project_path(root, str(controller_pin.get("path", "")))
    controller_manifest = controller_root / "controller_manifest.json"
    if (
        not controller_manifest.is_file()
        or _sha256(controller_manifest) != str(controller_pin.get("manifest_sha256", "")).lower()
    ):
        raise ValueError("v2-full controller bundle manifest SHA256 mismatch")
    if (
        collection.get("controller") != "v2-full"
        or list(collection.get("solver_seeds") or []) != [41, 42]
        or collection.get("executed_policy") != "realized_dynamic"
        or collection.get("deterministic_pp_replay") is not True
        or collection.get("feature_backend") != "native"
        or int(collection.get("expected_episode_count", -1)) != 64
    ):
        raise ValueError("registered source collection contract changed")
    return {
        "project_root": root,
        "source_config_path": source_config_path,
        "source_config": source_config,
        "design_path": design_path,
        "design": design,
        "design_sha256": design_sha,
        "report_path": report_path,
        "report": report,
        "report_sha256": _sha256(report_path),
        "base_config_path": base_path,
        "base_config": _read_json(base_path),
        "controller_root": controller_root,
        "map_splits": map_splits,
        "allowed_ids": allowed_ids,
        "sealed_ids": sealed_ids,
        "load_grid": expected_grid,
    }


def _map_details(payload: bytes, path: Path) -> dict[str, Any]:
    lines = payload.decode("utf-8").splitlines()
    headers: dict[str, str] = {}
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "map":
            marker = index
            break
        pieces = line.split(maxsplit=1)
        if len(pieces) == 2:
            headers[pieces[0].lower()] = pieces[1]
    if marker is None or "height" not in headers or "width" not in headers:
        raise ValueError(f"invalid MovingAI map: {path}")
    rows, cols = int(headers["height"]), int(headers["width"])
    grid = lines[marker + 1 : marker + 1 + rows]
    if len(grid) != rows or any(len(line) != cols for line in grid):
        raise ValueError(f"MovingAI map dimensions differ from header: {path}")
    free = {
        (x, y)
        for y, line in enumerate(grid)
        for x, value in enumerate(line)
        if value in {".", "G", "S"}
    }
    if not free:
        raise ValueError(f"MovingAI map has no passable cells: {path}")
    degrees = []
    for x, y in free:
        degrees.append(sum((x + dx, y + dy) in free for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))))
    return {
        "rows": rows,
        "cols": cols,
        "free": free,
        "metrics": {
            "rows": rows,
            "cols": cols,
            "free_cell_count": len(free),
            "obstacle_count": rows * cols - len(free),
            "obstacle_ratio": (rows * cols - len(free)) / (rows * cols),
            "average_free_degree": sum(degrees) / len(degrees),
            "minimum_free_degree": min(degrees),
            "maximum_free_degree": max(degrees),
            "dead_end_cell_count": sum(value <= 1 for value in degrees),
            "low_degree_cell_ratio": sum(value <= 2 for value in degrees) / len(degrees),
        },
    }


def _scenario_prefix(
    payload: bytes,
    *,
    path: Path,
    map_id: str,
    map_details: dict[str, Any],
    agent_count: int,
) -> dict[str, Any]:
    lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not lines or not lines[0].lower().startswith("version"):
        raise ValueError(f"MovingAI scenario version header is missing: {path}")
    records = [line.split() for line in lines[1 : 1 + agent_count]]
    if len(records) != agent_count or any(len(row) < 9 for row in records):
        raise ValueError(f"scenario has fewer than {agent_count} valid rows: {path}")
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    distances: list[float] = []
    for row in records:
        if (
            row[1] != f"{map_id}.map"
            or int(row[2]) != int(map_details["cols"])
            or int(row[3]) != int(map_details["rows"])
        ):
            raise ValueError(f"scenario row is registered for a different map: {path}")
        start = (int(row[4]), int(row[5]))
        goal = (int(row[6]), int(row[7]))
        distance = float(row[8])
        if start not in map_details["free"] or goal not in map_details["free"]:
            raise ValueError(f"scenario prefix uses a blocked/out-of-bounds cell: {path}")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"scenario prefix has an invalid distance: {path}")
        starts.append(start)
        goals.append(goal)
        distances.append(distance)
    if len(starts) != len(set(starts)) or len(goals) != len(set(goals)):
        raise ValueError(f"scenario prefix repeats a start or goal: {path}")
    return {
        "agent_count": agent_count,
        "unique_start_count": len(set(starts)),
        "unique_goal_count": len(set(goals)),
        "minimum_shortest_distance": min(distances),
        "maximum_shortest_distance": max(distances),
        "mean_shortest_distance": sum(distances) / len(distances),
    }


def _source_file(ctx: dict[str, Any], map_id: str, kind: str) -> tuple[Path, str]:
    _reject_sealed(map_id, ctx["sealed_ids"], "map ID")
    if map_id not in ctx["allowed_ids"]:
        raise ValueError(f"unregistered train/development map ID: {map_id}")
    row = dict(ctx["design"]["registered_sources"][map_id][kind])
    path = _resolve_project_path(ctx["project_root"], str(row["path"]))
    source_root = _resolve_project_path(ctx["project_root"], str(ctx["design"]["source_root"]))
    try:
        path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"registered {kind} escapes source root: {map_id}") from error
    return path, str(row["sha256"]).lower()


def materialize_source_dataset(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    dataset_root = output_root / str(ctx["source_config"]["source"]["dataset_directory"])
    source_root = _resolve_project_path(ctx["project_root"], str(ctx["design"]["source_root"]))
    if dataset_root == source_root or source_root in dataset_root.parents:
        raise ValueError("new dataset output must not be inside the registered raw source")
    prospective = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "DRY_RUN" if dry_run else "MATERIALIZED",
        "training_authorized": False,
        "sealed_final_semantic_access": False,
        "registered_config_sha256": ctx["design_sha256"],
        "registration_report_sha256": ctx["report_sha256"],
        "split_map_counts": {split: len(ctx["map_splits"][split]) for split in ALLOWED_SPLITS},
        "expected_task_count": 32,
        "expected_raw_file_count": 32,
    }
    if dry_run:
        return prospective
    expected_files: set[Path] = set()
    source_manifest: list[dict[str, Any]] = []
    all_manifest: dict[str, list[dict[str, Any]]] = {}
    for split in ALLOWED_SPLITS:
        manifest: list[dict[str, Any]] = []
        for map_id in ctx["map_splits"][split]:
            map_path, map_hash = _source_file(ctx, map_id, "map")
            scenario_path, scenario_hash = _source_file(ctx, map_id, "scenario")
            map_payload = map_path.read_bytes()
            scenario_payload = scenario_path.read_bytes()
            if hashlib.sha256(map_payload).hexdigest() != map_hash:
                raise ValueError(f"registered map SHA256 mismatch: {map_id}")
            if hashlib.sha256(scenario_payload).hexdigest() != scenario_hash:
                raise ValueError(f"registered scenario SHA256 mismatch: {map_id}")
            details = _map_details(map_payload, map_path)
            split_root = dataset_root / split
            map_output = split_root / "maps" / f"{map_id}.map"
            scenario_output = split_root / "scenarios" / f"{map_id}.map.scen"
            metadata_output = split_root / "maps" / f"{map_id}.json"
            for destination, payload in ((map_output, map_payload), (scenario_output, scenario_payload)):
                _write_exact(destination, payload, resume=resume)
                expected_files.add(destination)
            metadata = {
                "schema_version": 1,
                "benchmark_id": map_id,
                "split": split,
                "source": "registered MovingAI MAPF benchmark",
                "map_sha256": map_hash,
                "scenario_sha256": scenario_hash,
                "topology_metrics": details["metrics"],
                "training_authorized": False,
            }
            _write_exact(metadata_output, _json_bytes(metadata), resume=resume)
            expected_files.add(metadata_output)
            source_manifest.append(
                {
                    "split": split,
                    "map_id": map_id,
                    "map_file": str(map_output.relative_to(dataset_root)).replace("\\", "/"),
                    "map_sha256": map_hash,
                    "scenario_file": str(scenario_output.relative_to(dataset_root)).replace("\\", "/"),
                    "scenario_sha256": scenario_hash,
                }
            )
            for agent_count in ctx["load_grid"][map_id]:
                prefix = _scenario_prefix(
                    scenario_payload,
                    path=scenario_path,
                    map_id=map_id,
                    map_details=details,
                    agent_count=int(agent_count),
                )
                task_id = f"{map_id}__random_00__agents_{int(agent_count):04d}"
                _reject_sealed(task_id, ctx["sealed_ids"], "task ID")
                task_output = split_root / "tasks" / f"{task_id}.json"
                task_payload = {
                    "schema_version": 1,
                    "task_semantics": "static MovingAI official scenario-0 prefix",
                    "benchmark_id": map_id,
                    "split": split,
                    "scenario_index": 0,
                    "map_sha256": map_hash,
                    "scenario_sha256": scenario_hash,
                    **prefix,
                }
                _write_exact(task_output, _json_bytes(task_payload), resume=resume)
                expected_files.add(task_output)
                group = str(ctx["design"]["map_groups"][map_id])
                manifest.append(
                    {
                        "split": split,
                        "source_group": "movingai_registered_official_scenario",
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_file": f"maps/{map_id}.map",
                        "scenario_file": f"scenarios/{map_id}.map.scen",
                        "map_metadata_file": f"maps/{map_id}.json",
                        "task_file": f"tasks/{task_id}.json",
                        "map_sha256": map_hash,
                        "scenario_sha256": scenario_hash,
                        "layout_mode": group,
                        "layout_variant": map_id,
                        "scenario_type": "movingai_random_0",
                        "task_variant": f"random_0_agents_{int(agent_count)}",
                        "agent_count": int(agent_count),
                        "topology_metrics": details["metrics"],
                        "dominant_flow_ratio": 0.0,
                        "hotspot_skew": 0.0,
                        "required_bottleneck_crossing_ratio": 0.0,
                        "mean_shortest_distance": prefix["mean_shortest_distance"],
                        "training_authorized": False,
                    }
                )
        manifest.sort(key=lambda row: str(row["task_id"]))
        manifest_path = dataset_root / split / "manifest.jsonl"
        _write_exact(manifest_path, _jsonl_bytes(manifest), resume=resume)
        expected_files.add(manifest_path)
        all_manifest[split] = manifest
    source_manifest.sort(key=lambda row: (str(row["split"]), str(row["map_id"])))
    source_manifest_path = dataset_root / "source_manifest.jsonl"
    _write_exact(source_manifest_path, _jsonl_bytes(source_manifest), resume=resume)
    expected_files.add(source_manifest_path)
    summary = {
        "schema_version": 1,
        "schema": MATERIALIZATION_SCHEMA,
        "source": "hash-pinned registered MovingAI official scenarios",
        "registered_config_sha256": ctx["design_sha256"],
        "registration_report_sha256": ctx["report_sha256"],
        "training_authorized": False,
        "splits": {
            split: {"map_count": len(ctx["map_splits"][split]), "instance_count": len(all_manifest[split])}
            for split in ALLOWED_SPLITS
        },
    }
    summary_path = dataset_root / "dataset_summary.json"
    _write_exact(summary_path, _json_bytes(summary), resume=resume)
    expected_files.add(summary_path)
    observed_files = {path for path in dataset_root.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        extras = sorted(str(path.relative_to(dataset_root)) for path in observed_files - expected_files)
        missing = sorted(str(path.relative_to(dataset_root)) for path in expected_files - observed_files)
        raise ValueError(f"materialized dataset file set differs: extras={extras}, missing={missing}")
    prospective.update(
        {
            "dataset_root": str(dataset_root),
            "task_count": sum(map(len, all_manifest.values())),
            "raw_file_count": len(source_manifest) * 2,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "split_manifests": {
                split: {
                    "path": str(dataset_root / split / "manifest.jsonl"),
                    "sha256": _sha256(dataset_root / split / "manifest.jsonl"),
                    "task_count": len(all_manifest[split]),
                }
                for split in ALLOWED_SPLITS
            },
        }
    )
    _write_exact(output_root / "materialization_trust_report.json", _json_bytes(prospective), resume=resume)
    return prospective


def _validate_materialized(ctx: dict[str, Any], output_root: Path) -> dict[str, list[dict[str, Any]]]:
    dataset_root = output_root / str(ctx["source_config"]["source"]["dataset_directory"])
    trust = _read_json(output_root / "materialization_trust_report.json")
    if (
        trust.get("schema") != MATERIALIZATION_SCHEMA
        or trust.get("status") != "MATERIALIZED"
        or trust.get("training_authorized") is not False
        or trust.get("registered_config_sha256") != ctx["design_sha256"]
        or trust.get("registration_report_sha256") != ctx["report_sha256"]
    ):
        raise ValueError("materialization trust report is not pinned to registration")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ALLOWED_SPLITS:
        manifest_path = dataset_root / split / "manifest.jsonl"
        expected_hash = trust["split_manifests"][split]["sha256"]
        if _sha256(manifest_path) != expected_hash:
            raise ValueError(f"materialized {split} manifest SHA256 mismatch")
        rows = _read_jsonl(manifest_path)
        expected = {
            f"{map_id}__random_00__agents_{agent_count:04d}"
            for map_id in ctx["map_splits"][split]
            for agent_count in ctx["load_grid"][map_id]
        }
        if {str(row["task_id"]) for row in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"materialized {split} task product differs from registration")
        for row in rows:
            map_id = str(row["map_id"])
            _reject_sealed(map_id, ctx["sealed_ids"], "manifest map ID")
            if map_id not in ctx["map_splits"][split] or row.get("training_authorized") is not False:
                raise ValueError(f"materialized {split} manifest violates split/no-train contract")
            for field, hash_field in (("map_file", "map_sha256"), ("scenario_file", "scenario_sha256")):
                path = (dataset_root / split / str(row[field])).resolve()
                try:
                    path.relative_to((dataset_root / split).resolve())
                except ValueError as error:
                    raise ValueError(f"materialized path escapes split: {path}") from error
                if _sha256(path) != str(row[hash_field]):
                    raise ValueError(f"materialized task source hash mismatch: {row['task_id']}")
            task = _read_json(dataset_root / split / str(row["task_file"]))
            if (
                int(task.get("agent_count", -1)) != int(row["agent_count"])
                or task.get("map_sha256") != row.get("map_sha256")
                or task.get("scenario_sha256") != row.get("scenario_sha256")
                or int(task.get("unique_start_count", -1)) != int(row["agent_count"])
                or int(task.get("unique_goal_count", -1)) != int(row["agent_count"])
            ):
                raise ValueError(f"materialized task metadata mismatch: {row['task_id']}")
        result[split] = rows
    return result


def _collection_config(ctx: dict[str, Any], split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _reject_sealed(split, ctx["sealed_ids"], "split")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported source split: {split}")
    collection = ctx["source_config"]["collection"]
    config = json.loads(json.dumps(ctx["base_config"]))
    # Dataset-design family replication is defined over maps, not over the two
    # load-specific task rows emitted for each map.
    family_counts = collections.Counter(
        str(ctx["design"]["map_groups"][map_id])
        for map_id in ctx["map_splits"][split]
    )
    config.update(
        {
            "formal": True,
            "split": split,
            "solver_seeds": list(collection["solver_seeds"]),
            "policies": list(collection["policies_in_config"]),
            "dataset_design": {
                "mode": "movingai_ood",
                "map_count": len(ctx["map_splits"][split]),
                "task_count": len(rows),
                "scenario_indices": [0],
                "layout_family_counts": dict(sorted(family_counts.items())),
                "maps": [
                    {
                        "map_id": map_id,
                        "layout_family": str(ctx["design"]["map_groups"][map_id]),
                        "agent_counts": list(ctx["load_grid"][map_id]),
                    }
                    for map_id in ctx["map_splits"][split]
                ],
                "historical_map_ids": list(ctx["design"]["legacy_current_h1"]["map_ids"]),
            },
            "environment": {
                **dict(config["environment"]),
                "time_limit": float(collection["environment_time_limit_seconds"]),
            },
            "qualification": {
                "mode": "movingai_ood",
                "minimum_nonzero_states": len(rows),
                "minimum_active_maps": len(ctx["map_splits"][split]),
                "required_layout_families": sorted(family_counts),
            },
            "max_decisions": int(collection["max_decisions"]),
            "metric_iteration_budget": int(collection["max_decisions"]),
            "wall_time_budget_seconds": float(collection["wall_time_budget_seconds"]),
            "episode_process_timeout_seconds": float(collection["episode_process_timeout_seconds"]),
            "workers": int(collection["workers"]),
            "deterministic_pp_replay": True,
            "reference_datasets": [],
            "scientific_status": "registered_source_collection_training_unauthorized",
            "training_authorized": False,
        }
    )
    return config


def plan_source_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    ctx = load_registered_source_context(config_path, project_root=project_root)
    output_root = Path(output).resolve()
    manifests = _validate_materialized(ctx, output_root)
    seeds = list(ctx["source_config"]["collection"]["solver_seeds"])
    schedule = [
        {
            "split": split,
            "task_id": row["task_id"],
            "map_id": row["map_id"],
            "agent_count": row["agent_count"],
            "solver_seed": seed,
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "qualification_required": True,
            "training_authorized": False,
        }
        for split in ALLOWED_SPLITS
        for row in manifests[split]
        for seed in seeds
    ]
    schedule.sort(key=lambda row: (str(row["split"]), str(row["task_id"]), int(row["solver_seed"])))
    if len(schedule) != 64 or len({(row["split"], row["task_id"], row["solver_seed"]) for row in schedule}) != 64:
        raise ValueError("source collection schedule is not the exact 64-job product")
    report = {
        "schema": PLAN_SCHEMA,
        "status": "DRY_RUN" if dry_run else "PLANNED",
        "registered_config_sha256": ctx["design_sha256"],
        "registration_report_sha256": ctx["report_sha256"],
        "controller": "v2-full",
        "executed_policy": "realized_dynamic",
        "solver_seeds": seeds,
        "task_count": 32,
        "episode_count": 64,
        "split_episode_counts": {
            split: len(manifests[split]) * len(seeds) for split in ALLOWED_SPLITS
        },
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
    }
    if dry_run:
        return {**report, "schedule": schedule}
    for split in ALLOWED_SPLITS:
        path = output_root / "collection_configs" / f"{split}.json"
        _write_exact(path, _json_bytes(_collection_config(ctx, split, manifests[split])), resume=resume)
    schedule_path = output_root / "source_schedule.jsonl"
    _write_exact(schedule_path, _jsonl_bytes(schedule), resume=resume)
    report.update({"schedule_path": str(schedule_path), "schedule_sha256": _sha256(schedule_path)})
    _write_exact(output_root / "source_plan_report.json", _json_bytes(report), resume=resume)
    return report


def _manifest_job_product(path: Path, expected: set[tuple[str, int]], label: str) -> None:
    rows = _read_jsonl(path)
    observed = {(str(row["task_id"]), int(row["solver_seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{label} manifest differs from the exact task/seed product")


def collect_source_episodes(
    config_path: str | Path,
    output: str | Path,
    *,
    split: str = "all",
    dry_run: bool = False,
    resume: bool = False,
    project_root: str | Path = PROJECT_ROOT,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runner = runner or run_closed_loop_collection
    ctx = load_registered_source_context(config_path, project_root=project_root)
    _reject_sealed(split, ctx["sealed_ids"], "split")
    if split not in {"all", *ALLOWED_SPLITS}:
        raise ValueError(f"unsupported source split: {split}")
    output_root = Path(output).resolve()
    manifests = _validate_materialized(ctx, output_root)
    plan = _read_json(output_root / "source_plan_report.json")
    schedule_path = output_root / "source_schedule.jsonl"
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "PLANNED"
        or plan.get("training_authorized") is not False
        or _sha256(schedule_path) != plan.get("schedule_sha256")
    ):
        raise ValueError("source collection plan is missing, changed, or training-authorized")
    selected_splits = ALLOWED_SPLITS if split == "all" else (split,)
    collection = ctx["source_config"]["collection"]
    calls: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for selected in selected_splits:
        task_ids = [str(row["task_id"]) for row in manifests[selected]]
        jobs = {(task_id, int(seed)) for task_id in task_ids for seed in collection["solver_seeds"]}
        if len(jobs) != 32:
            raise ValueError(f"{selected} collection is not the exact 16-task x 2-seed product")
        collection_output = output_root / "collection" / selected
        common = {
            "dataset": output_root / str(ctx["source_config"]["source"]["dataset_directory"]),
            "config_path": output_root / "collection_configs" / f"{selected}.json",
            "output": collection_output,
            "workers": int(collection["workers"]),
            "dry_run": dry_run,
            "trace_format": str(collection["trace_format"]),
            "controller": "v2-full",
            "feature_backend": "native",
            "controller_bundle": ctx["controller_root"],
            "job_keys": jobs,
            "cohort_job_keys": jobs,
            "wall_time_budget_seconds": float(collection["wall_time_budget_seconds"]),
            "episode_process_timeout_seconds": float(collection["episode_process_timeout_seconds"]),
            "environment_time_limit_seconds": float(collection["environment_time_limit_seconds"]),
            "qualification_process_timeout_seconds": float(collection["qualification_process_timeout_seconds"]),
            "deterministic_pp_replay": True,
        }
        qualification_source = None
        if (
            resume
            and (collection_output / "qualification_manifest.jsonl").is_file()
            and (collection_output / "run_config.json").is_file()
        ):
            qualification_source = collection_output
        executions.append(
            {
                "split": selected,
                "jobs": jobs,
                "output": collection_output,
                "common": common,
                "qualification_source": qualification_source,
            }
        )

    planned_episode_count = 32 * len(selected_splits)
    if dry_run:
        # Dry-run remains a complete dispatch preview, but it cannot create a
        # production trust artifact or make a supply decision.
        for execution in executions:
            runner(
                phase="qualify",
                resume=resume,
                qualification_source=execution["qualification_source"],
                **execution["common"],
            )
            calls.append({"split": execution["split"], "phase": "qualify"})
        for execution in executions:
            runner(phase="realized_dynamic", resume=True, **execution["common"])
            calls.append({"split": execution["split"], "phase": "realized_dynamic"})
        report = {
            "schema": COLLECTION_TRUST_SCHEMA,
            "status": "DRY_RUN_DISPATCHED",
            "registered_config_sha256": ctx["design_sha256"],
            "registration_report_sha256": ctx["report_sha256"],
            "splits": list(selected_splits),
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "episode_count": planned_episode_count,
            "planned_episode_count": planned_episode_count,
            "realized_episode_count": 0,
            "qualification_job_count": planned_episode_count,
            "outcome_filtering": False,
            "sealed_final_semantic_access": False,
            "training_authorized": False,
            "calls": calls,
        }
        return report

    # Fail closed across the whole selected cohort: qualify every split before
    # any policy episode is allowed to start.
    qualification_passed: dict[str, bool] = {}
    for execution in executions:
        runner(
            phase="qualify",
            resume=resume,
            qualification_source=execution["qualification_source"],
            **execution["common"],
        )
        calls.append({"split": execution["split"], "phase": "qualify"})
        _manifest_job_product(
            execution["output"] / "qualification_manifest.jsonl",
            execution["jobs"],
            "qualification",
        )
        qualification_report = _read_json(execution["output"] / "qualification_report.json")
        if type(qualification_report.get("passed")) is not bool:
            raise ValueError(f"{execution['split']} qualification report lacks passed=true/false")
        qualification_passed[execution["split"]] = bool(qualification_report["passed"])

    if not all(qualification_passed.values()):
        report = {
            "schema": COLLECTION_TRUST_SCHEMA,
            "status": "STATE_SUPPLY_FAIL_NO_BACKFILL",
            "registered_config_sha256": ctx["design_sha256"],
            "registration_report_sha256": ctx["report_sha256"],
            "splits": list(selected_splits),
            "controller": "v2-full",
            "policy": "realized_dynamic",
            "episode_count": planned_episode_count,
            "planned_episode_count": planned_episode_count,
            "realized_episode_count": 0,
            "qualification_job_count": planned_episode_count,
            "qualification_passed": qualification_passed,
            "outcome_filtering": False,
            "sealed_final_semantic_access": False,
            "training_authorized": False,
            "calls": calls,
        }
        _write_exact(
            output_root / f"collection_trust_report__{split}.json",
            _json_bytes(report),
            resume=resume,
        )
        return report

    for execution in executions:
        # Qualification established the shared run_config/cohort, so policy
        # execution is always a continuation on both fresh and resumed runs.
        runner(phase="realized_dynamic", resume=True, **execution["common"])
        calls.append({"split": execution["split"], "phase": "realized_dynamic"})
        _manifest_job_product(
            execution["output"] / "realized_dynamic_manifest.jsonl",
            execution["jobs"],
            "realized_dynamic",
        )
    report = {
        "schema": COLLECTION_TRUST_SCHEMA,
        "status": "COLLECTED_EXACT_PRODUCT",
        "registered_config_sha256": ctx["design_sha256"],
        "registration_report_sha256": ctx["report_sha256"],
        "splits": list(selected_splits),
        "controller": "v2-full",
        "policy": "realized_dynamic",
        "episode_count": planned_episode_count,
        "planned_episode_count": planned_episode_count,
        "realized_episode_count": planned_episode_count,
        "qualification_job_count": planned_episode_count,
        "qualification_passed": qualification_passed,
        "outcome_filtering": False,
        "sealed_final_semantic_access": False,
        "training_authorized": False,
        "calls": calls,
    }
    _write_exact(
        output_root / f"collection_trust_report__{split}.json",
        _json_bytes(report),
        resume=resume,
    )
    return report


__all__ = [
    "ALLOWED_SPLITS",
    "COLLECTION_TRUST_SCHEMA",
    "CONFIG_SCHEMA",
    "MATERIALIZATION_SCHEMA",
    "PLAN_SCHEMA",
    "collect_source_episodes",
    "load_registered_source_context",
    "materialize_source_dataset",
    "plan_source_collection",
]
