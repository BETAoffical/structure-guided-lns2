"""Fresh-map three-band Warehouse disruption recovery extension.

The cohort is fixed before controller execution.  Dataset construction and
checkpoint selection never inspect controller outcomes; timed lanes are paired
and strictly serial.
"""
from __future__ import annotations

import collections
from functools import partial
import random
from pathlib import Path
from typing import Any, Mapping

from experiments._common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _load_dataset_rows, _run_jobs
from experiments.stride_warehouse_disruption_recovery import INCUMBENT_SUPPLY_ERROR, _checkpoint_worker
from experiments.stride_warehouse_disruption_recovery_heldout_ttf import (
    TTF_OVERRIDE_SCHEMA, _common_kwargs, _controller_kwargs, _controller_summary, _mean,
)
from experiments.warehouse_disruption_checkpoints import compute_checkpoint_identity_sha256
from experiments.warehouse_checkpoint_resume import (
    checkpoint_failure_record,
    reusable_checkpoint_report,
)
from generators.dataset import generate_dataset

SCHEMA = "lns2.stride.warehouse_disruption_recovery_load_extension_config.v1"
EXPERIMENT_ID = "stride-warehouse-disruption-recovery-load-extension-v1"
PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_load_extension_plan.v1"
CHECKPOINT_REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_load_extension_checkpoint_report.v1"
TTF_PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_load_extension_ttf_plan.v1"
TTF_REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_load_extension_ttf_report.v1"
VARIANTS = ("balanced_od_d10", "balanced_od_d125", "balanced_od_d15")
BANDS = ("medium_high", "high", "very_high")
CONTROLLERS = ("official_adaptive", "dual16")
MANIFEST_FILENAME = "load_extension_checkpoint_manifest.jsonl"
REPORT_FILENAME = "load_extension_checkpoint_report.json"
WORKER_FILENAME = "load_extension_worker_results.json"
PLAN_FILENAME = "load_extension_plan.json"
TIMED_MANIFESTS = {"official_adaptive": "official_adaptive_manifest.jsonl", "dual16": "realized_dynamic_manifest.jsonl"}


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path, root = Path(path).resolve(), PROJECT_ROOT
    config = read_json(path)
    if config.get("schema") != SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("load-extension config identity changed")
    dataset = dict(config.get("dataset") or {})
    dataset_config = root / str(dataset.get("config_path"))
    runtime = dict(config.get("runtime") or {})
    runtime_template = root / str(runtime.get("template_path"))
    controller_manifest = root / str(config["controller_contract"]["v2_bundle"]) / "controller_manifest.json"
    for candidate, expected, label in (
        (dataset_config, dataset.get("config_sha256"), "dataset config"),
        (runtime_template, runtime.get("template_sha256"), "runtime template"),
        (controller_manifest, config["controller_contract"].get("v2_manifest_sha256"), "V2 manifest"),
    ):
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"registered {label} changed")
    if (dataset.get("expected_map_count"), dataset.get("expected_task_count")) != (6, 18):
        raise ValueError("dataset cardinality changed")
    if tuple(dataset.get("load_bands", {}).keys()) != VARIANTS or tuple(dataset["load_bands"].values()) != BANDS:
        raise ValueError("load bands changed")
    generation = dict(config.get("checkpoint_generation") or {})
    if (generation.get("expected_candidate_count"), generation.get("disturbance_replicas_per_task")) != (36, 2):
        raise ValueError("checkpoint product changed")
    if dict(generation.get("gate") or {}) != {"minimum_conflict_pair_count": 16, "minimum_active_conflict_agent_count": 32, "minimum_largest_conflict_component_size": 16}:
        raise ValueError("checkpoint gate changed")
    if tuple(config.get("controllers") or ()) != CONTROLLERS or runtime.get("timed_workers") != 1 or runtime.get("execution_order") != "rotating_strict_two_controller_serial":
        raise ValueError("strict paired runtime changed")
    excluded = tuple(map(str, config.get("excluded_map_sha256") or ()))
    if len(excluded) != 12 or len(set(excluded)) != 12:
        raise ValueError("old-map exclusion registry changed")
    config["_dataset_config"] = str(dataset_config)
    config["_dataset_root"] = str(root / dataset["output_path"])
    config["_runtime_template"] = str(runtime_template)
    return path, root, config


def dry_run(config_path: str | Path) -> dict[str, Any]:
    path, _root_path, config = load_config(config_path)
    data = read_json(Path(config["_dataset_config"]))
    return {"schema": PLAN_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
            "dataset_configuration_fingerprint": _fingerprint(data), "map_count": 6, "task_count": 18,
            "checkpoint_candidate_count": 36, "maximum_timed_episode_count": 72,
            "load_bands": list(BANDS), "native_or_controller_invoked": False}


def generate_fresh_dataset(config_path: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    _path, _root_path, config = load_config(config_path)
    payload = read_json(Path(config["_dataset_config"]))
    destination = Path(str(config["_dataset_root"])).resolve()
    if output is not None and Path(output).resolve() != destination:
        raise ValueError("load-extension dataset output is frozen by the registered config")
    result = generate_dataset(payload, output_override=destination)
    _ensure_dataset(config, destination)
    return result


def _ensure_dataset(config: Mapping[str, Any], root: Path | None = None) -> Path:
    root = (root or Path(str(config["_dataset_root"]))).resolve()
    rows = [dict(row) for row in _load_dataset_rows(root, ["load_extension"])]
    if len(rows) != 18 or len({str(r["task_id"]) for r in rows}) != 18:
        raise ValueError("load-extension dataset must contain 18 unique tasks")
    split = root / "load_extension"
    maps = {(str(r["map_id"]), str(r["map_file"])) for r in rows}
    hashes = [sha256_file(split / filename) for _map, filename in maps]
    if len(maps) != 6 or len(set(hashes)) != 6 or set(hashes) & set(config["excluded_map_sha256"]):
        raise ValueError("new maps are not six byte-distinct maps disjoint from all prior maps")
    counts = collections.Counter(str(r.get("task_variant")) for r in rows)
    if counts != {name: 6 for name in VARIANTS}:
        raise ValueError("three-band task product changed")
    return root


def _checkpoint_schedule(config: Mapping[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    order = {name: i for i, name in enumerate(VARIANTS)}
    rows = sorted((dict(r) for r in _load_dataset_rows(dataset_root, ["load_extension"])), key=lambda r: (str(r["map_id"]), order[str(r["task_variant"])]))
    generation, runtime, result = config["checkpoint_generation"], config["runtime"], []
    for task_index, row in enumerate(rows):
        for replica in range(2):
            key = len(result)
            result.append({"key_index": key, "task_index": task_index, "disturbance_replica": replica,
                "key_id": f"{row['task_id']}#load-extension-disturbance-{replica}", "checkpoint_id": f"load-extension-{task_index:02d}-disturbance-{replica}",
                "map_id": str(row["map_id"]), "task_id": str(row["task_id"]), "task_variant": str(row["task_variant"]),
                "load_band": str(config["dataset"]["load_bands"][str(row["task_variant"])]), "split": "load_extension",
                "agent_count": int(row["agent_count"]), "incumbent_solver_seed": int(generation["incumbent_solver_seed_base"]) + task_index,
                "selection_seed": int(generation["selection_seed_base"]) + key, "screen_solver_seed": int(runtime["screen_solver_seed_base"]) + key, "row": row})
    if len(result) != 36 or len({r["key_id"] for r in result}) != 36:
        raise ValueError("checkpoint job keys changed")
    return result


def plan(config_path: str | Path) -> dict[str, Any]:
    path, _root_path, config = load_config(config_path)
    dataset_root = _ensure_dataset(config)
    schedule = _checkpoint_schedule(config, dataset_root)
    return {"schema": PLAN_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
            "candidate_count": 36, "map_count": 6, "task_count": 18,
            "load_band_counts": dict(collections.Counter(r["load_band"] for r in schedule)),
            "controller_outcomes_consulted": False, "failed_candidate_replacement": False,
            "schedule": [{k: v for k, v in r.items() if k != "row"} for r in schedule]}


_failure = partial(
    checkpoint_failure_record,
    incumbent_supply_error=INCUMBENT_SUPPLY_ERROR,
    include_count_fields=False,
)


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    result = _checkpoint_worker(job)
    if result.get("status") == "ok":
        checkpoint = dict(result["checkpoint"]); item = job["item"]
        checkpoint.update({"disturbance_replica": item["disturbance_replica"], "load_band": item["load_band"]})
        checkpoint["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(checkpoint)
        result["checkpoint"] = checkpoint
    return result


def _checkpoint_report(
    path: Path,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    qualified = [r for r in rows if dict(r.get("qualification") or {}).get("passed")]
    bands = collections.Counter(str(r.get("load_band")) for r in qualified)
    maps = {str(r.get("map_id")) for r in qualified}; cells = collections.Counter((str(r.get("map_id")), str(r.get("load_band"))) for r in qualified)
    execution = [r for r in failures if r.get("status") != "state_supply_unavailable"]
    passed = len(rows) + len(failures) == 36 and 30 <= len(qualified) <= 36 and len(maps) == 6 and all(bands[b] >= 10 for b in BANDS) and len(cells) == 18 and all(v >= 1 for v in cells.values()) and not execution
    return {"schema": CHECKPOINT_REPORT_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
        "attempted_candidate_count": len(rows) + len(failures), "completed_checkpoint_count": len(rows), "qualified_checkpoint_count": len(qualified),
        "qualified_map_count": len(maps), "qualified_per_load_band": dict(bands), "qualified_map_band_cell_count": len(cells),
        "qualified_checkpoint_ids": [r["checkpoint_id"] for r in qualified], "failures": failures, "execution_failure_count": len(execution),
        "checkpoint_selection_controller_outcomes_consulted": False, "failed_candidate_replacement": False, "passed": passed}


def analyze_checkpoints(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root_path, config = load_config(config_path); root = Path(output).resolve() / "checkpoints"
    rows = [dict(r) for r in read_jsonl(root / MANIFEST_FILENAME)] if (root / MANIFEST_FILENAME).is_file() else []
    worker = read_json(root / WORKER_FILENAME) if (root / WORKER_FILENAME).is_file() else {"failures": []}
    failures = [dict(row) for row in worker.get("failures") or ()]
    report = _checkpoint_report(path, config, rows, failures)
    write_json(root / REPORT_FILENAME, report); return report


def prepare_checkpoints(config_path: str | Path, output: str | Path, *, workers: int | None = None, resume: bool = False) -> dict[str, Any]:
    path, _root_path, config = load_config(config_path); dataset_root = _ensure_dataset(config); out = Path(output).resolve(); cp = out / "checkpoints"
    payload = plan(path)
    if resume:
        report = reusable_checkpoint_report(
            report_path=cp / REPORT_FILENAME,
            plan_path=out / PLAN_FILENAME,
            manifest_path=cp / MANIFEST_FILENAME,
            worker_path=cp / WORKER_FILENAME,
            expected_schema=CHECKPOINT_REPORT_SCHEMA,
            experiment_id=EXPERIMENT_ID,
            config_sha256=sha256_file(path),
            expected_plan=payload,
            schedule=payload["schedule"],
            build_report=lambda rows, failures: _checkpoint_report(path, config, rows, failures),
        )
        if report is not None:
            return report
    cp.mkdir(parents=True, exist_ok=True); write_json(out / PLAN_FILENAME, payload)
    jobs = [{"job_id": r["key_id"], "item": r, "dataset_root": str(dataset_root), "checkpoint_root": str(cp), "generation": config["checkpoint_generation"]} for r in _checkpoint_schedule(config, dataset_root)]
    results = _run_jobs(_worker, jobs, int(workers or config["checkpoint_generation"]["workers"]), phase="warehouse-load-extension-checkpoints", output_root=cp,
        run_fingerprint=_fingerprint({"config_sha256": sha256_file(path), "schedule": payload["schedule"]}), timeout_seconds=float(config["checkpoint_generation"]["process_timeout_seconds"]), failure_result=_failure, stop_on_failure=False)
    completed = sorted((dict(r["checkpoint"]) for r in results if r.get("status") == "ok"), key=lambda r: int(r["key_index"])); failures = [dict(r) for r in results if r.get("status") != "ok"]
    write_jsonl(cp / MANIFEST_FILENAME, completed); write_json(cp / WORKER_FILENAME, {"completed_count": len(completed), "failures": failures})
    return analyze_checkpoints(path, out)


def _checkpoint_inputs(config_path: str | Path, output: str | Path) -> tuple[Path, Path, dict[str, Any], Path, list[dict[str, Any]]]:
    path, root, config = load_config(config_path); dataset_root = _ensure_dataset(config); cp = Path(output).resolve() / "checkpoints"
    report = read_json(cp / REPORT_FILENAME); rows = [dict(r) for r in read_jsonl(cp / MANIFEST_FILENAME)]
    worker = read_json(cp / WORKER_FILENAME)
    failures = [dict(r) for r in worker.get("failures") or ()]
    execution_failures = [r for r in failures if r.get("status") != "state_supply_unavailable"]
    qualified = [r for r in rows if dict(r.get("qualification") or {}).get("passed")]
    bands = collections.Counter(str(r.get("load_band")) for r in qualified)
    cells = collections.Counter((str(r.get("map_id")), str(r.get("load_band"))) for r in qualified)
    if (
        report.get("schema") != CHECKPOINT_REPORT_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("passed") is not True
        or report.get("config_sha256") != sha256_file(path)
        or int(report.get("attempted_candidate_count", -1)) != 36
        or len(rows) + len(failures) != 36
        or int(worker.get("completed_count", -1)) != len(rows)
        or int(report.get("completed_checkpoint_count", -1)) != len(rows)
        or not 30 <= len(qualified) <= 36
        or int(report.get("qualified_checkpoint_count", -1)) != len(qualified)
        or int(report.get("qualified_map_count", -1)) != 6
        or int(report.get("qualified_map_band_cell_count", -1)) != 18
        or {str(k): int(v) for k, v in dict(report.get("qualified_per_load_band") or {}).items()} != dict(bands)
        or set(map(str, report.get("qualified_checkpoint_ids") or ())) != {str(r.get("checkpoint_id")) for r in qualified}
        or int(report.get("execution_failure_count", -1)) != 0
        or execution_failures
        or list(report.get("failures") or ()) != failures
        or report.get("checkpoint_selection_controller_outcomes_consulted") is not False
        or report.get("failed_candidate_replacement") is not False
        or any(bands[band] < 10 for band in BANDS)
        or len(cells) != 18
        or any(value < 1 for value in cells.values())
    ):
        raise ValueError("checkpoint report did not pass its frozen contract")
    registered = {str(r["task_id"]): r for r in _load_dataset_rows(dataset_root, ["load_extension"])}
    planned = {int(r["key_index"]): r for r in _checkpoint_schedule(config, dataset_root)}
    split = dataset_root / "load_extension"; seen = set()
    for row in qualified:
        task = registered.get(str(row.get("task_id"))); expected = planned.get(int(row.get("key_index", -1)))
        blob = cp / str(row.get("state_blob") or "")
        key = (str(row.get("task_id")), int(row.get("screen_solver_seed", -1)))
        checks = (
            task is not None and expected is not None,
            key not in seen,
            str(row.get("split")) == "load_extension",
            task is not None and expected is not None and str(row.get("map_id")) == str(task.get("map_id")) == str(expected.get("map_id")),
            expected is not None and str(row.get("task_id")) == str(expected.get("task_id")),
            task is not None and expected is not None and str(row.get("task_variant")) == str(task.get("task_variant")) == str(expected.get("task_variant")),
            expected is not None and str(row.get("load_band")) == str(expected.get("load_band")),
            task is not None and expected is not None and int(row.get("agent_count", -1)) == int(task.get("agent_count", -2)) == int(expected.get("agent_count", -3)),
            expected is not None and int(row.get("disturbance_replica", -1)) == int(expected.get("disturbance_replica", -2)),
            expected is not None and str(row.get("checkpoint_id")) == str(expected.get("checkpoint_id")),
            expected is not None and int(row.get("screen_solver_seed", -1)) == int(expected.get("screen_solver_seed", -2)),
            expected is not None and int(row.get("restore_seed", -1)) == int(expected.get("selection_seed", -2)),
            expected is not None and int(dict(row.get("incumbent") or {}).get("solver_seed", -1)) == int(expected.get("incumbent_solver_seed", -2)),
            task is not None and str(row.get("map_sha256")) == sha256_file(split / str(task.get("map_file"))),
            task is not None and str(row.get("task_sha256")) == sha256_file(split / str(task.get("task_file"))),
            row.get("source_kind") == "checkpoint_blob_v1",
            row.get("checkpoint_identity_sha256") == compute_checkpoint_identity_sha256(row),
            blob.is_file() and sha256_file(blob) == row.get("state_blob_sha256"),
        )
        if not all(checks):
            raise ValueError("checkpoint identity, blob, or registered input changed")
        seen.add(key)
    return path, root, config, cp, sorted(qualified, key=lambda r: int(r["key_index"]))


def _schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key = int(row["key_index"])
        for pos in range(2):
            controller = CONTROLLERS[(key % 2 + pos) % 2]
            result.append({"schedule_index": len(result), "key_index": key, "within_key_position": pos, "controller": controller,
                **{k: row[k] for k in ("checkpoint_id", "checkpoint_identity_sha256", "map_id", "task_id", "task_variant", "load_band", "disturbance_replica")}, "solver_seed": int(row["screen_solver_seed"])})
    return result


def plan_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root_path, _config, cp, rows = _checkpoint_inputs(config_path, output); schedule = _schedule(rows)
    return {"schema": TTF_PLAN_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path), "checkpoint_root": str(cp),
            "checkpoint_count": len(rows), "episode_count": len(schedule), "controllers": list(CONTROLLERS), "timed_workers": 1,
            "execution_order": "rotating_strict_two_controller_serial", "schedule": schedule}


def _runtime_config(config: Mapping[str, Any], rows: list[dict[str, Any]], destination: Path) -> Path:
    payload = dict(read_json(Path(str(config["_runtime_template"])))); runtime = config["runtime"]
    payload.update({"formal": False, "split": "load_extension", "solver_seeds": sorted(int(r["screen_solver_seed"]) for r in rows), "policies": ["official_adaptive", "realized_dynamic"],
        "wall_time_budget_seconds": 60.0, "episode_process_timeout_seconds": 90.0, "workers": 1, "deterministic_pp_replay": False, "repair_seed_policy": "episode_stream",
        "dataset_design": {"mode": "structured", "map_count": 6, "tasks_per_map": 3, "task_variants": list(VARIANTS), "layout_counts": {"station_centric": 6}}})
    env = dict(payload["environment"]); env["time_limit"] = float(runtime["environment_time_limit_seconds"]); payload["environment"] = env
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and read_json(destination) != payload: raise ValueError("existing runtime config changed")
    if not destination.is_file(): write_json(destination, payload)
    return destination


def _lane(root: Path, item: Mapping[str, Any]) -> Path:
    return root / "lanes" / f"key_{int(item['key_index']):02d}" / f"pos_{int(item['within_key_position'])}_{item['controller']}"


def _install_checkpoint_qualification_anchor(anchor: Path, rows: list[dict[str, Any]]) -> None:
    manifest = anchor / "qualification_manifest.jsonl"
    natural = {(str(r["task_id"]), int(r["solver_seed"])): dict(r) for r in read_jsonl(manifest)}
    expected = {(str(r["task_id"]), int(r["screen_solver_seed"])) for r in rows}
    if set(natural) != expected or len(rows) != len(expected):
        raise ValueError("checkpoint qualification anchor keys changed")
    qualified = []
    for checkpoint in rows:
        key = (str(checkpoint["task_id"]), int(checkpoint["screen_solver_seed"]))
        complexity = dict(checkpoint.get("native_complexity") or {})
        if int(checkpoint.get("expected_conflicts", 0)) <= 0 or not dict(checkpoint.get("qualification") or {}).get("passed"):
            raise ValueError("checkpoint qualification anchor requires a qualified disturbed state")
        qualified.append({**natural[key], "status": "ok", "error": None, "initial_complete": True,
            "initial_feasible": False, "repairable": True, "initial_conflicts": int(checkpoint["expected_conflicts"]),
            "initial_complexity": complexity, "state_fingerprint": str(checkpoint["expected_fingerprint"]),
            "qualification_source_kind": "authenticated_checkpoint_blob_v1",
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "checkpoint_identity_sha256": str(checkpoint["checkpoint_identity_sha256"])})
    write_jsonl(manifest, sorted(qualified, key=lambda r: (str(r["task_id"]), int(r["solver_seed"]))))


def collect_ttf(config_path: str | Path, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    path, root, config, cp, rows = _checkpoint_inputs(config_path, output); dataset = _ensure_dataset(config); ttf = Path(output).resolve() / "ttf"; schedule = _schedule(rows)
    ttf.mkdir(parents=True, exist_ok=True); schedule_path = ttf / "execution_schedule.jsonl"
    if schedule_path.is_file() and read_jsonl(schedule_path) != schedule: raise ValueError("execution schedule changed")
    write_jsonl(schedule_path, schedule); runtime = _runtime_config(config, rows, ttf / "runtime_config.json")
    keys = {(str(r["task_id"]), int(r["screen_solver_seed"])) for r in rows}; anchor = ttf / "qualification_anchor"; exists = (anchor / "run_config.json").is_file()
    if exists and not resume: raise ValueError("qualification anchor exists; pass resume")
    run_closed_loop_collection(dataset, runtime, anchor, phase="qualify", workers=8, resume=exists, task_ids=sorted(t for t, _ in keys), job_keys=keys, cohort_job_keys=keys,
        qualification_process_timeout_seconds=90.0, **{k: v for k, v in _common_kwargs(config).items() if k != "workers"})
    _install_checkpoint_qualification_anchor(anchor, rows)
    run_closed_loop_collection(dataset, runtime, anchor, phase="qualify", workers=1, resume=True,
        task_ids=sorted(t for t, _ in keys), job_keys=keys, cohort_job_keys=keys, qualification_source=anchor,
        qualification_process_timeout_seconds=90.0, **{k: v for k, v in _common_kwargs(config).items() if k != "workers"})
    by_id = {str(r["checkpoint_id"]): r for r in rows}; progress = []
    for item in schedule:
        row = by_id[str(item["checkpoint_id"])]; key = (str(item["task_id"]), int(item["solver_seed"])); lane = _lane(ttf, item); exists = (lane / "run_config.json").is_file()
        if exists and not resume: raise ValueError(f"lane exists; pass resume: {lane}")
        overrides = {key: {"schema": TTF_OVERRIDE_SCHEMA, "state_id": row["checkpoint_id"], "initial_restore": {**row, "collection_root": str(cp)}}}
        phase, kwargs = _controller_kwargs(root, config, str(item["controller"])); common = {"task_ids": [item["task_id"]], "job_keys": {key}, "cohort_job_keys": {key}, "qualification_source": anchor, "episode_overrides": overrides}
        run_closed_loop_collection(dataset, runtime, lane, phase="qualify", resume=exists, **common, **kwargs)
        result = run_closed_loop_collection(dataset, runtime, lane, phase=phase, resume=True, **common, **kwargs); progress.append({**item, "lane": str(lane), "collection": result})
        write_json(ttf / "collection_progress.json", {"completed": len(progress), "expected": len(schedule), "rows": progress})
    return analyze_ttf(path, output)


def _comparison(rows: list[dict[str, Any]], summaries: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {c: {r["checkpoint_id"]: r for r in rows if r["controller"] == c} for c in CONTROLLERS}; ids = sorted(set(indexed[CONTROLLERS[0]]) & set(indexed[CONTROLLERS[1]]))
    left = [indexed[CONTROLLERS[0]][i]["summary"] for i in ids]; right = [indexed[CONTROLLERS[1]][i]["summary"] for i in ids]; lc = [float(r["capped_wall_time_to_feasible"]) for r in left]; rc = [float(r["capped_wall_time_to_feasible"]) for r in right]
    base, target = float(summaries[CONTROLLERS[0]]["successes_per_observed_hour"]), float(summaries[CONTROLLERS[1]]["successes_per_observed_hour"])
    throughput = (target-base)/base if base else (None if target == 0 else float("inf"))
    result = {"paired_key_count": len(ids), "paired_win_count": sum(r < l for l, r in zip(lc, rc)), "paired_win_rate": sum(r < l for l, r in zip(lc, rc))/len(ids) if ids else 0.0,
        "mean_capped_ttf_improvement": (_mean(lc)-_mean(rc))/_mean(lc) if _mean(lc) else 0.0, "throughput_improvement": throughput,
        "success_rate_loss": float(summaries[CONTROLLERS[0]]["success_rate"])-float(summaries[CONTROLLERS[1]]["success_rate"]),
        "additional_timeout_count": sum(bool(r["external_timeout"]) and not bool(l["external_timeout"]) for l,r in zip(left,right)), "additional_censor_count": sum(bool(l["success"]) and not bool(r["success"]) for l,r in zip(left,right))}
    result["screen_gate_passed"] = bool(ids and result["paired_win_rate"] >= gate["minimum_paired_win_rate"] and result["mean_capped_ttf_improvement"] >= gate["minimum_mean_capped_ttf_improvement"] and throughput is not None and throughput >= gate["minimum_successes_per_hour_improvement"] and result["success_rate_loss"] <= gate["maximum_success_rate_loss"] and result["additional_timeout_count"] == result["additional_censor_count"] == 0)
    maps = collections.defaultdict(list)
    for checkpoint_id in ids: maps[indexed[CONTROLLERS[0]][checkpoint_id]["map_id"]].append((float(indexed[CONTROLLERS[0]][checkpoint_id]["summary"]["capped_wall_time_to_feasible"]), float(indexed[CONTROLLERS[1]][checkpoint_id]["summary"]["capped_wall_time_to_feasible"])))
    result["map_cluster_count"] = len(maps); result["map_cluster_mean_improvements"] = {m: (_mean([x for x,_ in v])-_mean([y for _,y in v]))/_mean([x for x,_ in v]) for m,v in sorted(maps.items())}
    rng = random.Random(2026091501); clusters = list(maps)
    boots = []
    if clusters:
        for _ in range(2000):
            sample = [rng.choice(clusters) for _ in clusters]; pairs = [p for m in sample for p in maps[m]]; a,b = _mean([x for x,_ in pairs]),_mean([y for _,y in pairs]); boots.append((a-b)/a if a else 0.0)
    result["map_cluster_bootstrap_95ci"] = [sorted(boots)[49], sorted(boots)[1949]] if boots else None
    return result


def analyze_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root_path, config, _cp, checkpoints = _checkpoint_inputs(config_path, output); ttf = Path(output).resolve()/"ttf"; schedule = _schedule(checkpoints)
    if not (ttf/"execution_schedule.jsonl").is_file() or read_jsonl(ttf/"execution_schedule.jsonl") != schedule: raise ValueError("execution schedule missing or changed")
    by_id = {r["checkpoint_id"]: r for r in checkpoints}; collected=[]; missing=[]
    for item in schedule:
        manifest = _lane(ttf,item)/TIMED_MANIFESTS[item["controller"]]; matches=[r for r in (read_jsonl(manifest) if manifest.is_file() else []) if r.get("task_id")==item["task_id"] and int(r.get("solver_seed",-1))==item["solver_seed"]]
        if len(matches)!=1 or matches[0].get("status") not in {"ok","resumed"}: missing.append(item); continue
        summary=dict(matches[0].get("summary") or {}); cp=by_id[item["checkpoint_id"]]
        collected.append({**item,"summary":summary,"initial_fingerprint_matches":summary.get("initial_fingerprint")==cp.get("expected_fingerprint"),"initial_conflicts_match":int(summary.get("initial_conflicts",-1))==int(cp["expected_conflicts"]),"bounded_summary_valid":summary.get("ttf_clock_schema")=="lns2.ttf.reset_inclusive_wall.v1" and summary.get("capped_wall_time_to_feasible") is not None and float(summary.get("wall_time_budget_seconds",-1))==60.0 and int(summary.get("invalid_action_count",-1))==0 and int(summary.get("fingerprint_mismatch_count",-1))==0 and str(summary.get("stop_reason")) in {"success","wall_timeout","controller_stalled","native_terminal"}})
    integrity={"complete_strict_serial_schedule":len(collected)==len(schedule) and not missing,"paired_initial_fingerprints":all(r["initial_fingerprint_matches"] for r in collected),"paired_initial_conflicts":all(r["initial_conflicts_match"] for r in collected),"registered_clock":all(r["bounded_summary_valid"] for r in collected),"map_disjoint":True}
    analysis={}
    for band in BANDS:
        group=[r for r in collected if r["load_band"]==band]; summaries={c:_controller_summary([r for r in group if r["controller"]==c]) for c in CONTROLLERS}; analysis[band]={"controllers":summaries,"comparison_vs_official":_comparison(group,summaries,config["screen_gate"])}
    passed={b:bool(all(integrity.values()) and analysis[b]["comparison_vs_official"]["screen_gate_passed"]) for b in BANDS}; passed_bands=[b for b in BANDS if passed[b]]
    route="all_three" if len(passed_bands)==3 else "_and_".join(passed_bands) if passed_bands else "none"
    report={"schema":TTF_REPORT_SCHEMA,"experiment_id":EXPERIMENT_ID,"config_sha256":sha256_file(path),"checkpoint_count":len(checkpoints),"expected_episode_count":len(schedule),"completed_episode_count":len(collected),"integrity_gates":integrity,"missing_or_failed_schedule_rows":missing,"analysis":analysis,"band_gate_passed":passed,"route_recommendation":route,"timing_boundary":"checkpoint_restore_inclusive_ttf","global_or_default_promotion_allowed":False}
    write_json(ttf/"load_extension_ttf_report.json",report); return report


__all__ = ["BANDS", "CONTROLLERS", "analyze_checkpoints", "analyze_ttf", "collect_ttf", "dry_run", "generate_fresh_dataset", "load_config", "plan", "plan_ttf", "prepare_checkpoints", "_comparison", "_schedule"]
