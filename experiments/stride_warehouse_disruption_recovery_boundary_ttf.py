from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments._common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_warehouse_disruption_recovery_boundary import (
    EXPERIMENT_ID, MANIFEST_FILENAME, REPORT_FILENAME, VARIANTS,
    WORKER_RESULTS_FILENAME,
    _boundary_schedule, _ensure_datasets, load_config,
)
from experiments.stride_warehouse_disruption_recovery_heldout_ttf import (
    TTF_OVERRIDE_SCHEMA, _common_kwargs, _controller_kwargs, _controller_summary, _mean,
)
from experiments.warehouse_disruption_checkpoints import compute_checkpoint_identity_sha256


TTF_PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_boundary_ttf_plan.v1"
TTF_REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_boundary_ttf_report.v1"
CONTROLLERS = ("official_adaptive", "dual16")
TIMED_MANIFESTS = {"official_adaptive": "official_adaptive_manifest.jsonl", "dual16": "realized_dynamic_manifest.jsonl"}


def _checkpoint_inputs(config_path: str | Path, output: str | Path) -> tuple[Path, Path, dict[str, Any], Path, list[dict[str, Any]]]:
    path, root, config = load_config(config_path)
    dataset_root = _ensure_datasets(config)
    checkpoint_root = Path(output).resolve() / "checkpoints"
    report_path, manifest_path = checkpoint_root / REPORT_FILENAME, checkpoint_root / MANIFEST_FILENAME
    if not report_path.is_file() or not manifest_path.is_file():
        raise ValueError("Boundary checkpoint evidence is missing")
    report, rows = read_json(report_path), [dict(row) for row in read_jsonl(manifest_path)]
    worker_path = checkpoint_root / WORKER_RESULTS_FILENAME
    if not worker_path.is_file():
        raise ValueError("Boundary checkpoint worker evidence is missing")
    worker = read_json(worker_path)
    failures = [dict(row) for row in worker.get("failures", [])]
    execution_failures = [
        row for row in failures if row.get("status") != "state_supply_unavailable"
    ]
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    generation = dict(config["checkpoint_generation"])
    bands = {str(value) for value in config["dataset"]["load_bands"].values()}
    if (
        report.get("experiment_id") != EXPERIMENT_ID or report.get("passed") is not True
        or report.get("config_sha256") != sha256_file(path)
        or int(report.get("attempted_candidate_count", -1)) != 24
        or len(rows) + len(failures) != 24
        or int(worker.get("completed_count", -1)) != len(rows)
        or int(report.get("completed_checkpoint_count", -1)) != len(rows)
        or not 18 <= len(qualified) <= 24
        or int(report.get("qualified_checkpoint_count", -1)) != len(qualified)
        or int(report.get("qualified_map_count", -1)) != 6
        or int(report.get("qualified_map_band_cell_count", -1)) != 12
        or int(report.get("execution_failure_count", -1)) != 0
        or execution_failures
        or int(report.get("state_supply_unavailable_count", -1))
        != len(failures)
        or set(map(str, report.get("qualified_checkpoint_ids") or ()))
        != {str(row.get("checkpoint_id")) for row in qualified}
        or report.get("checkpoint_selection_controller_outcomes_consulted") is not False
        or report.get("failed_candidate_replacement") is not False
    ):
        raise ValueError("Boundary checkpoint report did not pass its frozen contract")
    schedule = {int(row["key_index"]): row for row in _boundary_schedule(config, dataset_root)}
    from experiments.repair_collection import _load_dataset_rows
    registered_rows = _load_dataset_rows(dataset_root, ["boundary_confirmation"])
    registered = {str(row["task_id"]): row for row in registered_rows}
    prior_root = Path(str(config["_prior_dataset_root"]))
    prior_rows = _load_dataset_rows(prior_root, ["development", "controller_held_out"])
    prior_ids = {str(row["map_id"]) for row in prior_rows}
    prior_hashes = {sha256_file(prior_root / str(row["split"]) / str(row["map_file"])) for row in prior_rows}
    observed_keys: set[tuple[str, int]] = set()
    cell_counts: dict[tuple[str, str], int] = {}
    band_counts = {band: 0 for band in bands}
    for row in qualified:
        planned, task = schedule.get(int(row.get("key_index", -1))), registered.get(str(row.get("task_id")))
        if planned is None or task is None:
            raise ValueError("Boundary checkpoint is not registered")
        split_root = dataset_root / "boundary_confirmation"
        band = str(row.get("load_band"))
        key = (str(row["task_id"]), int(row["screen_solver_seed"]))
        checks = (
            str(row.get("split")) == "boundary_confirmation",
            str(row.get("map_id")) == str(task["map_id"]) == str(planned["map_id"]),
            str(row.get("task_variant")) == str(task["task_variant"]),
            int(row.get("agent_count", -1)) == int(task["agent_count"]),
            band == str(planned["load_band"]) and band in bands,
            int(row.get("disturbance_replica", -1)) == int(planned["disturbance_replica"]),
            str(row.get("checkpoint_id")) == str(planned["checkpoint_id"]),
            int(row.get("screen_solver_seed", -1)) == int(planned["screen_solver_seed"]),
            int(row.get("restore_seed", -1)) == int(planned["selection_seed"]),
            int(dict(row.get("incumbent") or {}).get("solver_seed", -1)) == int(planned["incumbent_solver_seed"]),
            str(row.get("map_sha256")) == sha256_file(split_root / str(task["map_file"])),
            str(row.get("task_sha256")) == sha256_file(split_root / str(task["task_file"])),
            str(row.get("checkpoint_identity_sha256")) == compute_checkpoint_identity_sha256(row),
            row.get("source_kind") == "checkpoint_blob_v1",
        )
        blob = checkpoint_root / str(row.get("state_blob") or "")
        if not all(checks) or key in observed_keys or not blob.is_file() or sha256_file(blob) != str(row.get("state_blob_sha256")):
            raise ValueError("Boundary checkpoint identity, seed, replica, blob, or registered hash changed")
        map_hash = sha256_file(split_root / str(task["map_file"]))
        if str(row["map_id"]) in prior_ids or map_hash in prior_hashes:
            raise ValueError("Boundary checkpoint map is not disjoint")
        observed_keys.add(key); band_counts[band] += 1
        cell = (str(row["map_id"]), band); cell_counts[cell] = cell_counts.get(cell, 0) + 1
    if len(observed_keys) != len(qualified) or len(cell_counts) != 12 or any(value < 8 for value in band_counts.values()):
        raise ValueError("Boundary qualified cohort coverage changed")
    if {
        str(name): int(value)
        for name, value in dict(report.get("qualified_per_load_band") or {}).items()
    } != band_counts:
        raise ValueError("Boundary checkpoint report load-band counts changed")
    qualified.sort(key=lambda row: int(row["key_index"]))
    return path, root, config, checkpoint_root, qualified


def _schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        key_index = int(row["key_index"])
        for position in range(2):
            controller = CONTROLLERS[(key_index % 2 + position) % 2]
            result.append({"schedule_index": len(result), "key_index": key_index,
                "checkpoint_id": str(row["checkpoint_id"]), "checkpoint_identity_sha256": str(row["checkpoint_identity_sha256"]),
                "map_id": str(row["map_id"]), "task_id": str(row["task_id"]), "task_variant": str(row["task_variant"]),
                "load_band": str(row["load_band"]), "disturbance_replica": int(row["disturbance_replica"]),
                "solver_seed": int(row["screen_solver_seed"]), "controller": controller, "within_key_position": position})
    return result


def plan_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, _config, checkpoint_root, rows = _checkpoint_inputs(config_path, output)
    schedule = _schedule(rows)
    return {"schema": TTF_PLAN_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
        "checkpoint_root": str(checkpoint_root), "checkpoint_count": len(rows), "episode_count": len(schedule),
        "controllers": list(CONTROLLERS), "qualification_workers": 8, "timed_workers": 1,
        "execution_order": "rotating_strict_two_controller_serial", "map_disjoint": True,
        "global_claim_allowed": False, "default_replacement_allowed": False, "schedule": schedule}


def _runtime_config(config: Mapping[str, Any], rows: list[dict[str, Any]], destination: Path) -> Path:
    payload = dict(read_json(Path(str(config["_runtime_template"]))))
    payload.update({"formal": False, "split": "boundary_confirmation", "solver_seeds": sorted(int(row["screen_solver_seed"]) for row in rows),
        "policies": ["official_adaptive", "realized_dynamic"], "wall_time_budget_seconds": 120.0,
        "episode_process_timeout_seconds": 150.0, "workers": 1, "deterministic_pp_replay": False,
        "repair_seed_policy": "episode_stream", "dataset_design": {"mode": "structured", "map_count": 6,
        "tasks_per_map": 2, "task_variants": list(VARIANTS), "layout_counts": {"station_centric": 6}}})
    environment = dict(payload["environment"]); environment["time_limit"] = 120.0; payload["environment"] = environment
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and read_json(destination) != payload: raise ValueError("existing boundary runtime changed")
    if not destination.is_file(): write_json(destination, payload)
    return destination


def _lane_root(root: Path, item: Mapping[str, Any]) -> Path:
    return root / "lanes" / f"key_{int(item['key_index']):02d}" / f"pos_{int(item['within_key_position'])}_{item['controller']}"


def collect_ttf(config_path: str | Path, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    path, root, config, checkpoint_root, rows = _checkpoint_inputs(config_path, output)
    dataset_root, ttf_root = _ensure_datasets(config), Path(output).resolve() / "ttf"
    schedule = _schedule(rows); ttf_root.mkdir(parents=True, exist_ok=True)
    schedule_path = ttf_root / "execution_schedule.jsonl"
    if schedule_path.is_file() and read_jsonl(schedule_path) != schedule: raise ValueError("boundary execution schedule changed")
    write_jsonl(schedule_path, schedule)
    runtime_path = _runtime_config(config, rows, ttf_root / "runtime_config.json")
    by_id = {str(row["checkpoint_id"]): row for row in rows}
    keys = {(str(row["task_id"]), int(row["screen_solver_seed"])) for row in rows}
    if len(keys) != len(rows): raise ValueError("Boundary qualified job keys are not unique")
    anchor = ttf_root / "qualification_anchor"; exists = (anchor / "run_config.json").is_file()
    if exists and not resume: raise ValueError("boundary anchor exists; pass resume")
    run_closed_loop_collection(dataset_root, runtime_path, anchor, phase="qualify", workers=8, resume=exists,
        task_ids=sorted({task for task, _seed in keys}), job_keys=keys, cohort_job_keys=keys,
        qualification_process_timeout_seconds=150.0,
        **{k: v for k, v in _common_kwargs(config).items() if k != "workers"})
    progress: list[dict[str, Any]] = []
    for item in schedule:
        row, key = by_id[str(item["checkpoint_id"])], (str(item["task_id"]), int(item["solver_seed"]))
        overrides = {key: {"schema": TTF_OVERRIDE_SCHEMA, "state_id": str(row["checkpoint_id"]),
            "initial_restore": {**row, "collection_root": str(checkpoint_root.resolve())}}}
        lane = _lane_root(ttf_root, item); lane_exists = (lane / "run_config.json").is_file()
        if lane_exists and not resume: raise ValueError(f"boundary lane exists; pass resume: {lane}")
        phase, kwargs = _controller_kwargs(root, config, str(item["controller"]))
        common = {"task_ids": [str(item["task_id"])], "job_keys": {key}, "cohort_job_keys": {key}, "qualification_source": anchor, "episode_overrides": overrides}
        run_closed_loop_collection(dataset_root, runtime_path, lane, phase="qualify", resume=lane_exists, **common, **kwargs)
        result = run_closed_loop_collection(dataset_root, runtime_path, lane, phase=phase, resume=True, **common, **kwargs)
        progress.append({**item, "lane": str(lane), "collection": result})
        write_json(
            ttf_root / "collection_progress.json",
            {
                "schema": TTF_PLAN_SCHEMA,
                "completed": len(progress),
                "expected": len(schedule),
                "rows": progress,
            },
        )
    return analyze_ttf(path, output)


def _comparison(rows: list[dict[str, Any]], summaries: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {name: {str(row["checkpoint_id"]): row for row in rows if row["controller"] == name} for name in CONTROLLERS}
    ids = sorted(set(indexed["official_adaptive"]) & set(indexed["dual16"]))
    left = [indexed["official_adaptive"][key]["summary"] for key in ids]; right = [indexed["dual16"][key]["summary"] for key in ids]
    lc = [float(row["capped_wall_time_to_feasible"]) for row in left]; rc = [float(row["capped_wall_time_to_feasible"]) for row in right]
    base, target = float(summaries["official_adaptive"]["successes_per_observed_hour"]), float(summaries["dual16"]["successes_per_observed_hour"])
    throughput = (target - base) / base if base else (None if target == 0 else float("inf"))
    result = {"paired_key_count": len(ids), "paired_win_count": sum(r < l for l, r in zip(lc, rc)),
        "paired_win_rate": sum(r < l for l, r in zip(lc, rc)) / len(ids) if ids else 0.0,
        "mean_capped_ttf_improvement": (_mean(lc) - _mean(rc)) / _mean(lc) if _mean(lc) else 0.0,
        "throughput_improvement": throughput,
        "success_rate_loss": float(summaries["official_adaptive"]["success_rate"]) - float(summaries["dual16"]["success_rate"]),
        "additional_timeout_count": sum(bool(r["external_timeout"]) and not bool(l["external_timeout"]) for l, r in zip(left, right)),
        "additional_censor_count": sum(bool(l["success"]) and not bool(r["success"]) for l, r in zip(left, right))}
    no_extra = result["additional_timeout_count"] == result["additional_censor_count"] == 0
    result["screen_gate_passed"] = bool(ids
        and result["paired_win_rate"] >= float(gate["minimum_paired_win_rate"])
        and result["mean_capped_ttf_improvement"] >= float(gate["minimum_mean_capped_ttf_improvement"])
        and throughput is not None and throughput >= float(gate["minimum_successes_per_hour_improvement"])
        and result["success_rate_loss"] <= float(gate["maximum_success_rate_loss"])
        and (not bool(gate["no_additional_timeout_or_censor"]) or no_extra))
    return result


def analyze_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config, _checkpoint_root, checkpoints = _checkpoint_inputs(config_path, output)
    ttf_root, schedule = Path(output).resolve() / "ttf", _schedule(checkpoints)
    schedule_path = ttf_root / "execution_schedule.jsonl"
    if not schedule_path.is_file() or read_jsonl(schedule_path) != schedule: raise ValueError("boundary execution schedule is missing or changed")
    by_id = {str(row["checkpoint_id"]): row for row in checkpoints}; collected, missing = [], []
    for item in schedule:
        manifest = _lane_root(ttf_root, item) / TIMED_MANIFESTS[str(item["controller"])]
        matches = [row for row in (read_jsonl(manifest) if manifest.is_file() else []) if str(row.get("task_id")) == str(item["task_id"]) and int(row.get("solver_seed", -1)) == int(item["solver_seed"])]
        if len(matches) != 1 or matches[0].get("status") not in {"ok", "resumed"}: missing.append(item); continue
        summary, checkpoint = dict(matches[0].get("summary") or {}), by_id[str(item["checkpoint_id"])]
        valid = bool(summary.get("ttf_clock_schema") == "lns2.ttf.reset_inclusive_wall.v1" and summary.get("capped_wall_time_to_feasible") is not None
            and float(summary.get("wall_time_budget_seconds", -1)) == 120.0 and int(summary.get("invalid_action_count", -1)) == 0
            and int(summary.get("fingerprint_mismatch_count", -1)) == 0 and str(summary.get("stop_reason")) in {"success", "wall_timeout", "controller_stalled", "native_terminal"})
        collected.append({**item, "summary": summary, "initial_fingerprint_matches": summary.get("initial_fingerprint") == checkpoint.get("expected_fingerprint"),
            "initial_conflicts_match": int(summary.get("initial_conflicts", -1)) == int(checkpoint["expected_conflicts"]), "bounded_summary_valid": valid})
    groups = {"overall": collected, "medium_high": [r for r in collected if r["load_band"] == "medium_high"], "high": [r for r in collected if r["load_band"] == "high"]}
    gate, analysis = dict(config["screen_gate"]), {}
    for name, group in groups.items():
        summaries = {controller: _controller_summary([row for row in group if row["controller"] == controller]) for controller in CONTROLLERS}
        analysis[name] = {"controllers": summaries, "comparison_vs_official": _comparison(group, summaries, gate)}
    integrity = {"complete_strict_serial_schedule": len(collected) == len(schedule) and not missing,
        "execution_schedule_exact": True, "paired_initial_fingerprints": all(r["initial_fingerprint_matches"] for r in collected),
        "paired_initial_conflicts": all(r["initial_conflicts_match"] for r in collected),
        "registered_clock_and_stop_reason": all(r["bounded_summary_valid"] for r in collected), "map_disjoint": True}
    passed = {band: bool(all(integrity.values()) and analysis[band]["comparison_vs_official"]["screen_gate_passed"]) for band in ("medium_high", "high")}
    route = "both" if all(passed.values()) else "high_only" if passed["high"] else "none"
    report = {"schema": TTF_REPORT_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
        "checkpoint_count": len(checkpoints), "expected_episode_count": len(schedule), "completed_episode_count": len(collected),
        "integrity_gates": integrity, "missing_or_failed_schedule_rows": missing, "analysis": analysis, "screen_gate": gate,
        "band_gate_passed": passed, "route_recommendation": route, "global_claim_allowed": False,
        "default_replacement_allowed": False, "formal_promotion_allowed": False, "timing_boundary": "checkpoint_restore_inclusive_ttf"}
    write_json(ttf_root / "boundary_ttf_report.json", report); return report


__all__ = ["CONTROLLERS", "TTF_PLAN_SCHEMA", "TTF_REPORT_SCHEMA", "_checkpoint_inputs", "_comparison", "_schedule", "analyze_ttf", "collect_ttf", "plan_ttf"]
