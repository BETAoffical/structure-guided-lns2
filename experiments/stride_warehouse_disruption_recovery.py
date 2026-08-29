from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Mapping

from experiments._common import sha256_file
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.warehouse_disruption_checkpoints import (
    checkpoint_manifest_fields,
    compute_checkpoint_identity_sha256,
    inject_global_hotspot_delays,
    warehouse_checkpoint_gate,
    warehouse_hotspot_definition,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.warehouse_disruption_recovery_screen_config.v1"
EXPERIMENT_ID = "stride-warehouse-disruption-recovery-screen-v1"
PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_plan.v1"
REPORT_SCHEMA = "lns2.stride.warehouse_disruption_checkpoint_report.v1"
CHECKPOINT_NATIVE_SCHEMA = "lns2.warehouse_disruption_checkpoint.native.v1"
MANIFEST_FILENAME = "checkpoint_manifest.jsonl"
REPORT_FILENAME = "checkpoint_qualification_report.json"
PLAN_FILENAME = "checkpoint_plan.json"
INCUMBENT_SUPPLY_ERROR = (
    "RuntimeError: Official incumbent did not produce a feasible path set"
)


def _project_path(root: Path, value: Any, *, label: str) -> Path:
    relative = Path(str(value or ""))
    if not str(value or "") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained project-relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root") from error
    return resolved


def _registered_file(root: Path, value: Mapping[str, Any], *, label: str) -> Path:
    path = _project_path(root, value.get("path"), label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    expected = str(value.get("sha256") or "").lower()
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 changed")
    return path


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("Warehouse disruption screen identity changed")

    dataset = dict(config.get("dataset") or {})
    dataset_config = _registered_file(
        root,
        {
            "path": dataset.get("config_path"),
            "sha256": dataset.get("config_sha256"),
        },
        label="Warehouse disruption dataset config",
    )
    runtime = dict(config.get("runtime") or {})
    runtime_template = _registered_file(
        root,
        {
            "path": runtime.get("template_path"),
            "sha256": runtime.get("template_sha256"),
        },
        label="Warehouse disruption runtime template",
    )
    controller = dict(config.get("controller_contract") or {})
    for prefix in ("v2", "mixed"):
        manifest = (
            _project_path(root, controller[f"{prefix}_bundle"], label=f"{prefix} bundle")
            / "controller_manifest.json"
        )
        if not manifest.is_file() or sha256_file(manifest) != str(
            controller[f"{prefix}_manifest_sha256"]
        ):
            raise ValueError(f"frozen {prefix} controller manifest changed")

    generation = dict(config.get("checkpoint_generation") or {})
    gate = dict(generation.get("gate") or {})
    expected_gate = {
        "minimum_conflict_pair_count": 16,
        "minimum_active_conflict_agent_count": 32,
        "minimum_largest_conflict_component_size": 16,
    }
    if (
        dataset.get("split") != "development"
        or int(dataset.get("expected_map_count", -1)) != 4
        or int(dataset.get("expected_task_count", -1)) != 16
        or tuple(dataset.get("screen_task_variants") or ())
        != ("station_rush_d10", "balanced_od_d125")
        or int(generation.get("expected_candidate_count", -1)) != 8
        or gate != expected_gate
        or generation.get("global_time_rule")
        != "earliest_maximum_hotspot_occupancy_after_t0"
        or generation.get("selection_blind_to_controller_outcomes") is not True
        or generation.get("no_failed_candidate_replacement") is not True
        or tuple(config.get("controllers") or ())
        != ("official_adaptive", "v2_only", "mixed_full_v2", "dual16")
        or int(runtime.get("timed_workers", -1)) != 1
    ):
        raise ValueError("Warehouse disruption screen contract changed")

    config["_dataset_config"] = str(dataset_config)
    config["_runtime_template"] = str(runtime_template)
    config["_dataset_root"] = str(
        _project_path(root, dataset["output_path"], label="dataset output")
    )
    return path, root, config


def _ensure_dataset(root: Path, config: Mapping[str, Any]) -> Path:
    dataset_root = Path(str(config["_dataset_root"]))
    manifests = [
        dataset_root / split / "manifest.jsonl"
        for split in ("development", "controller_held_out")
    ]
    if not all(path.is_file() for path in manifests):
        missing = [str(path) for path in manifests if not path.is_file()]
        raise FileNotFoundError(
            "Generate the registered warehouse dataset before planning: "
            + ", ".join(missing)
        )

    rows = [
        row
        for split in ("development", "controller_held_out")
        for row in _load_dataset_rows(dataset_root, [split])
    ]
    if len(rows) != 24 or len({str(row["task_id"]) for row in rows}) != 24:
        raise ValueError("Warehouse disruption dataset task product changed")
    map_files = {
        (
            str(row["split"]),
            str(row["map_id"]),
            str(row["map_file"]),
        )
        for row in rows
    }
    hashes = {
        sha256_file(dataset_root / split / map_file)
        for split, _map_id, map_file in map_files
    }
    if len(map_files) != 6 or len(hashes) != 6:
        raise ValueError("Warehouse disruption maps are not six byte-distinct layouts")
    return dataset_root


def _screen_schedule(config: Mapping[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    dataset = dict(config["dataset"])
    rows = _load_dataset_rows(dataset_root, [str(dataset["split"])])
    variant_order = {
        name: index for index, name in enumerate(dataset["screen_task_variants"])
    }
    selected = [
        dict(row)
        for row in rows
        if str(row.get("task_variant")) in variant_order
    ]
    selected.sort(
        key=lambda row: (str(row["map_id"]), variant_order[str(row["task_variant"])])
    )
    generation = dict(config["checkpoint_generation"])
    runtime = dict(config["runtime"])
    if (
        len(selected) != int(generation["expected_candidate_count"])
        or len({str(row["map_id"]) for row in selected}) != 4
        or collections.Counter(str(row["task_variant"]) for row in selected)
        != {"station_rush_d10": 4, "balanced_od_d125": 4}
    ):
        raise ValueError("Warehouse disruption screen task schedule changed")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        result.append(
            {
                "key_index": index,
                "key_id": f"{row['task_id']}#delay-{index:02d}",
                "checkpoint_id": f"warehouse-disruption-dev-{index:02d}",
                "map_id": str(row["map_id"]),
                "task_id": str(row["task_id"]),
                "task_variant": str(row["task_variant"]),
                "split": str(row["split"]),
                "agent_count": int(row["agent_count"]),
                "incumbent_solver_seed": int(
                    generation["incumbent_solver_seed_base"]
                )
                + index,
                "selection_seed": int(generation["selection_seed_base"]) + index,
                "screen_solver_seed": int(runtime["screen_solver_seed_base"]) + index,
                "row": row,
            }
        )
    return result


def plan(config_path: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    dataset_root = _ensure_dataset(root, config)
    schedule = _screen_schedule(config, dataset_root)
    generation = dict(config["checkpoint_generation"])
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "dataset_root": str(dataset_root),
        "candidate_count": len(schedule),
        "map_count": len({row["map_id"] for row in schedule}),
        "task_variant_counts": dict(
            sorted(collections.Counter(row["task_variant"] for row in schedule).items())
        ),
        "workers": int(generation["workers"]),
        "controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "schedule": [{key: value for key, value in row.items() if key != "row"} for row in schedule],
    }


def _target_delay_count(agent_count: int, fraction: float) -> int:
    minimum = math.ceil(agent_count * 0.10)
    maximum = math.floor(agent_count * 0.15)
    nearest = math.floor(agent_count * fraction + 0.5)
    return min(max(nearest, minimum), maximum)


def _select_global_time(
    paths: list[list[int]], hotspot: Mapping[str, Any], fraction: float
) -> dict[str, int]:
    cols = int(hotspot["cols"])
    hotspot_ids = {
        int(row) * cols + int(col) for row, col in hotspot["hotspot_cells"]
    }
    target = _target_delay_count(len(paths), fraction)
    horizon = max(map(len, paths)) - 1
    occupancies = []
    for time in range(1, horizon):
        count = sum(
            time < len(path) - 1 and int(path[time]) in hotspot_ids
            for path in paths
        )
        occupancies.append((count, time))
    if not occupancies:
        raise ValueError("incumbent has no positive nonterminal disturbance time")
    maximum, selected = min(
        ((-count, time) for count, time in occupancies), key=lambda value: value
    )
    maximum = -maximum
    if maximum < target:
        raise ValueError(
            f"maximum hotspot occupancy {maximum} is below required delayed agents {target}"
        )
    return {
        "global_time": selected,
        "eligible_agent_count": maximum,
        "required_delayed_agent_count": target,
    }


def _checkpoint_worker(job: dict[str, Any]) -> dict[str, Any]:
    item = dict(job["item"])
    row = dict(item.pop("row"))
    dataset_root = Path(str(job["dataset_root"]))
    checkpoint_root = Path(str(job["checkpoint_root"]))
    generation = dict(job["generation"])
    environment_config = {
        "time_limit": float(generation["incumbent_time_limit_seconds"]),
        "max_repair_iterations": int(generation["incumbent_max_repair_iterations"]),
        "neighborhood_size": 8,
        "replan_algorithm": "PP",
        "use_sipp": True,
    }
    environment = _make_environment(
        str(dataset_root), row, environment_config, "Adaptive"
    )
    state = _plain(environment.reset(seed=int(item["incumbent_solver_seed"])))
    incumbent_repairs = 0
    while not bool(state["done"]):
        state = _plain(environment.step({"mode": "official"}))["observation"]
        incumbent_repairs += 1
    if not bool(state["feasible"]) or int(state["num_of_colliding_pairs"]) != 0:
        raise RuntimeError("Official incumbent did not produce a feasible path set")
    agents = sorted(state["agents"], key=lambda value: int(value["id"]))
    paths = [list(map(int, agent["path"])) for agent in agents]

    split_root = dataset_root / str(row["split"])
    map_document = _read_json(split_root / str(row["map_metadata_file"]))
    metadata = dict(map_document["metadata"])
    hotspot = warehouse_hotspot_definition(
        dict(metadata["station_zones"]),
        list(metadata["semantic_cell_types"]),
        neighborhood_radius=int(generation["hotspot_neighborhood_radius"]),
    )
    timing = _select_global_time(
        paths, hotspot, float(generation["delayed_agent_fraction"])
    )
    disturbance = inject_global_hotspot_delays(
        paths,
        semantic_cell_types=list(metadata["semantic_cell_types"]),
        station_zones=dict(metadata["station_zones"]),
        global_time=int(timing["global_time"]),
        delay_ticks=int(generation["delay_ticks"]),
        delayed_agent_fraction=float(generation["delayed_agent_fraction"]),
        selection_seed=int(item["selection_seed"]),
        neighborhood_radius=int(generation["hotspot_neighborhood_radius"]),
    )
    pre_native = checkpoint_manifest_fields(
        map_id=str(row["map_id"]),
        task_id=str(row["task_id"]),
        map_sha256=sha256_file(split_root / str(row["map_file"])),
        task_sha256=sha256_file(split_root / str(row["task_file"])),
        source_paths=paths,
        disturbance=disturbance,
        checkpoint_id=str(item["checkpoint_id"]),
    )

    restore = _make_environment(str(dataset_root), row, environment_config, "Adaptive")
    disturbed_state = _plain(
        restore.reset_paths(
            list(disturbance["paths"]), seed=int(item["selection_seed"])
        )
    )
    complexity = summarize_initial_state_complexity(disturbed_state)
    native_gate = warehouse_checkpoint_gate(complexity, **dict(generation["gate"]))
    for key in (
        "conflict_pair_count",
        "active_conflict_agent_count",
        "largest_conflict_component_size",
    ):
        if int(complexity[key]) != int(disturbance["conflicts"][key]):
            raise RuntimeError(f"native disturbed-state metric differs: {key}")
    relative_blob, blob_path = write_state_blob(checkpoint_root, disturbed_state)

    final = {
        **pre_native,
        "schema": CHECKPOINT_NATIVE_SCHEMA,
        "source_kind": "checkpoint_blob_v1",
        "key_index": int(item["key_index"]),
        "key_id": str(item["key_id"]),
        "split": str(row["split"]),
        "task_variant": str(row["task_variant"]),
        "screen_solver_seed": int(item["screen_solver_seed"]),
        "incumbent": {
            "planner": "official_adaptive",
            "solver_seed": int(item["incumbent_solver_seed"]),
            "repair_decision_count": incumbent_repairs,
            "state_fingerprint": state_fingerprint(state),
            "paths_sha256": str(disturbance["source_paths_sha256"]),
            "feasible": True,
            "conflict_pair_count": 0,
        },
        "global_time_selection": {
            "rule": str(generation["global_time_rule"]),
            **timing,
        },
        "qualification": native_gate,
        "native_complexity": complexity,
        "state_blob": relative_blob,
        "state_blob_sha256": sha256_file(blob_path),
        "expected_fingerprint": state_fingerprint(disturbed_state),
        "repair_structure_fingerprint": repair_structure_fingerprint(
            disturbed_state
        ),
        "expected_conflicts": int(disturbed_state["num_of_colliding_pairs"]),
        "restore_seed": int(item["selection_seed"]),
        "native_solver_or_controller_invoked": True,
        "checkpoint_selection_controller_outcomes_consulted": False,
    }
    final["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(final)
    return {
        "status": "ok",
        "state_count": 1,
        "outcome_count": 0,
        "key_index": int(item["key_index"]),
        "checkpoint": final,
    }


def _checkpoint_failure(job: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    item = dict(job["item"])
    normalized_status = (
        "state_supply_unavailable"
        if status == "error" and error == INCUMBENT_SUPPLY_ERROR
        else status
    )
    return {
        "status": normalized_status,
        "error": error,
        "state_count": 0,
        "outcome_count": 0,
        "key_index": int(item["key_index"]),
        "key_id": str(item["key_id"]),
        "map_id": str(item["map_id"]),
        "task_id": str(item["task_id"]),
    }


def analyze_checkpoints(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output_path = Path(output).resolve()
    manifest_path = output_path / "checkpoints" / MANIFEST_FILENAME
    rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    generation = dict(config["checkpoint_generation"])
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    maps = {str(row["map_id"]) for row in qualified}
    raw_failures = _read_json(output_path / "checkpoints" / "worker_results.json").get(
        "failures", []
    ) if (output_path / "checkpoints" / "worker_results.json").is_file() else []
    supply_unavailable = [
        dict(row)
        for row in raw_failures
        if row.get("status") == "state_supply_unavailable"
        or (
            row.get("status") == "error"
            and str(row.get("error")) == INCUMBENT_SUPPLY_ERROR
        )
    ]
    execution_failures = [
        dict(row) for row in raw_failures if row not in supply_unavailable
    ]
    attempted_count = len(rows) + len(raw_failures)
    passed = (
        attempted_count == int(generation["expected_candidate_count"])
        and len(qualified) >= int(generation["minimum_qualified_count"])
        and len(maps) >= int(generation["minimum_qualified_map_count"])
        and not execution_failures
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "candidate_count": int(generation["expected_candidate_count"]),
        "attempted_candidate_count": attempted_count,
        "completed_checkpoint_count": len(rows),
        "qualified_checkpoint_count": len(qualified),
        "qualified_map_count": len(maps),
        "qualified_checkpoint_ids": [str(row["checkpoint_id"]) for row in qualified],
        "state_supply_unavailable_count": len(supply_unavailable),
        "state_supply_unavailable": supply_unavailable,
        "execution_failure_count": len(execution_failures),
        "execution_failures": execution_failures,
        "gate": dict(generation["gate"]),
        "minimum_qualified_count": int(generation["minimum_qualified_count"]),
        "minimum_qualified_map_count": int(generation["minimum_qualified_map_count"]),
        "passed": passed,
        "controller_step_or_ttf_invoked": False,
        "checkpoint_selection_controller_outcomes_consulted": False,
        "failed_candidate_replacement": False,
        "claim_boundary": str(config["claim_boundary"]),
        "per_checkpoint": [
            {
                "checkpoint_id": str(row["checkpoint_id"]),
                "map_id": str(row["map_id"]),
                "task_id": str(row["task_id"]),
                "task_variant": str(row["task_variant"]),
                "initial_conflicts": int(row["expected_conflicts"]),
                "active_conflict_agent_count": int(
                    row["native_complexity"]["active_conflict_agent_count"]
                ),
                "largest_conflict_component_size": int(
                    row["native_complexity"]["largest_conflict_component_size"]
                ),
                "qualified": bool(row["qualification"]["passed"]),
            }
            for row in rows
        ],
    }
    _write_json(output_path / "checkpoints" / REPORT_FILENAME, report)
    return report


def prepare_checkpoints(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    dataset_root = _ensure_dataset(root, config)
    output_path = Path(output).resolve()
    checkpoint_root = output_path / "checkpoints"
    report_path = checkpoint_root / REPORT_FILENAME
    if resume and report_path.is_file():
        report = _read_json(report_path)
        if report.get("config_sha256") != sha256_file(path):
            raise ValueError("existing checkpoint report belongs to another config")
        return report

    schedule = _screen_schedule(config, dataset_root)
    plan_payload = plan(path)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / PLAN_FILENAME, plan_payload)
    generation = dict(config["checkpoint_generation"])
    jobs = [
        {
            "job_id": str(item["key_id"]),
            "item": item,
            "dataset_root": str(dataset_root),
            "checkpoint_root": str(checkpoint_root),
            "generation": generation,
        }
        for item in schedule
    ]
    actual_workers = int(workers if workers is not None else generation["workers"])
    results = _run_jobs(
        _checkpoint_worker,
        jobs,
        actual_workers,
        phase="warehouse-checkpoints",
        output_root=checkpoint_root,
        run_fingerprint=_fingerprint(
            {
                "config_sha256": sha256_file(path),
                "schedule": plan_payload["schedule"],
            }
        ),
        timeout_seconds=float(generation["process_timeout_seconds"]),
        failure_result=_checkpoint_failure,
        stop_on_failure=False,
    )
    completed = sorted(
        (dict(row["checkpoint"]) for row in results if row.get("status") == "ok"),
        key=lambda row: int(row["key_index"]),
    )
    failures = [dict(row) for row in results if row.get("status") != "ok"]
    _write_jsonl(checkpoint_root / MANIFEST_FILENAME, completed)
    _write_json(
        checkpoint_root / "worker_results.json",
        {"completed_count": len(completed), "failures": failures},
    )
    return analyze_checkpoints(path, output_path)


__all__ = [
    "EXPERIMENT_ID",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_checkpoints",
    "load_config",
    "plan",
    "prepare_checkpoints",
]
