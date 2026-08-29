from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_warehouse_disruption_recovery_heldout_ttf import _controller_kwargs
from experiments.warehouse_disruption_checkpoints import warehouse_checkpoint_gate


CONFIG_SCHEMA = "lns2.stride.dual16_room_maze_highload_extension_config.v1"
EXPERIMENT_ID = "stride-dual16-room-maze-highload-extension-v1"
PLAN_SCHEMA = "lns2.stride.dual16_room_maze_highload_extension_plan.v1"
REPORT_SCHEMA = "lns2.stride.dual16_room_maze_highload_extension_report.v1"
CONTROLLERS = ("official_adaptive", "dual16")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path, root = Path(path).resolve(), Path(__file__).resolve().parents[1]
    config = read_json(config_path)
    runtime, qualification = dict(config.get("runtime") or {}), dict(config.get("qualification") or {})
    template, manifest = root / str(config.get("runtime_template")), root / str(config.get("v2_bundle")) / "controller_manifest.json"
    if (config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status") != "sequential_family_diagnostic_not_map_disjoint"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
        or config.get("controller_blind_preregistration") is not True
        or config.get("global_or_default_promotion_allowed") is not False
        or dict(config.get("maps") or {}) != {"maze": ["maze-32-32-4", "maze-128-128-1"], "room": ["room-64-64-8", "room-64-64-16"]}
        or dict(config.get("map_agent_counts") or {}) != {"maze-32-32-4": 200, "maze-128-128-1": 400, "room-64-64-8": 600, "room-64-64-16": 600}
        or dict(config.get("diagnostic_roles") or {}) != {"maze-128-128-1": "high_conflict_pressure_negative_control"}
        or list(config.get("scenario_indices") or ()) != [4, 5]
        or list(config.get("solver_seeds") or ()) != [51, 52]
        or qualification != {"workers": 8, "minimum_conflict_pair_count": 16, "minimum_active_conflict_agent_count": 32,
            "minimum_largest_conflict_component_size": 16, "minimum_initial_conflicts": 101, "minimum_qualified_per_family": 4}
        or runtime != {"wall_time_seconds": 60.0, "environment_time_limit_seconds": 60.0, "process_fuse_seconds": 90.0, "timed_workers": 1}
        or not template.is_file() or sha256_file(template) != str(config.get("runtime_template_sha256"))
        or not manifest.is_file() or sha256_file(manifest) != str(config.get("v2_manifest_sha256"))):
        raise ValueError("room/maze extension contract changed")
    config["_dataset_root"] = str((root / str(config["dataset_root"])).resolve())
    config["_runtime_template"] = str(template.resolve())
    config["controller_contract"] = {"v2_bundle": str(config["v2_bundle"])}
    config["runtime"] = {
        **runtime,
        "wall_time_budget_seconds": float(runtime["wall_time_seconds"]),
        "environment_time_limit_seconds": float(runtime["environment_time_limit_seconds"]),
        "episode_process_timeout_seconds": float(runtime["process_fuse_seconds"]),
    }
    return config_path, root, config


def _dataset_rows(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from experiments.repair_collection import _load_dataset_rows
    root = Path(str(config["_dataset_root"])); rows = _load_dataset_rows(root, [str(config["split"])])
    wanted = {map_id for values in config["maps"].values() for map_id in values}
    counts = {str(key): int(value) for key, value in config["map_agent_counts"].items()}
    selected = {str(row["task_id"]): dict(row) for row in rows if str(row["map_id"]) in wanted
        and int(row["agent_count"]) == counts[str(row["map_id"])]
        and str(row.get("scenario_type")) in {"movingai_random_4", "movingai_random_5"}}
    if len(selected) != 8 or {str(r["map_id"]) for r in selected.values()} != wanted:
        raise ValueError("registered room/maze task product changed")
    return selected


def _family(config: Mapping[str, Any], map_id: str) -> str:
    return next(name for name, maps in config["maps"].items() if map_id in maps)


def _schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _dataset_rows(config); result = []
    for task_id, row in sorted(rows.items(), key=lambda item: (str(item[1]["map_id"]), str(item[0]))):
        for seed in config["solver_seeds"]:
            key_index = len(result)
            result.append({"key_index": key_index, "task_id": task_id, "solver_seed": int(seed),
                "map_id": str(row["map_id"]), "family": _family(config, str(row["map_id"])),
                "agent_count": int(row["agent_count"]), "diagnostic_role": str(dict(config.get("diagnostic_roles") or {}).get(str(row["map_id"]), "ordinary_high_load_case"))})
    if len(result) != 16 or len({(r["task_id"], r["solver_seed"]) for r in result}) != 16:
        raise ValueError("room/maze preregistered keys changed")
    return result


def _execution_schedule(keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in keys:
        for position in range(2):
            controller = CONTROLLERS[(int(row["key_index"]) % 2 + position) % 2]
            out.append({**row, "schedule_index": len(out), "within_key_position": position, "controller": controller})
    return out


def plan(config_path: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path); keys = _schedule(config)
    return {"schema": PLAN_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
        "scientific_status": config["scientific_status"], "candidate_key_count": 16, "maximum_timed_episode_count": 32,
        "controller_outcomes_consulted": False, "map_disjoint_confirmation": False,
        "family_key_counts": dict(sorted(collections.Counter(r["family"] for r in keys).items())), "keys": keys}


def _runtime_config(config: Mapping[str, Any], output: Path) -> Path:
    payload = dict(read_json(Path(str(config["_runtime_template"]))))
    payload.update({"formal": False, "split": "movingai_ood", "solver_seeds": list(config["solver_seeds"]),
        "policies": ["official_adaptive", "realized_dynamic"], "wall_time_budget_seconds": 60.0,
        "episode_process_timeout_seconds": 90.0, "workers": 1, "deterministic_pp_replay": False,
        "repair_seed_policy": "episode_stream", "dataset_design": {"mode": "movingai_ood", "map_count": 4,
        "task_count": 8, "scenario_indices": [4, 5], "layout_family_counts": {"maze": 2, "room": 2}}})
    env = dict(payload["environment"]); env["time_limit"] = 60.0; payload["environment"] = env
    if output.is_file() and read_json(output) != payload: raise ValueError("existing runtime config changed")
    if not output.is_file(): write_json(output, payload)
    return output


def _qualified(config: Mapping[str, Any], anchor: Path, keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(anchor / "qualification_manifest.jsonl"); indexed = {(str(r["task_id"]), int(r["solver_seed"])): r for r in rows}
    result = []
    for key in keys:
        row = indexed.get((key["task_id"], key["solver_seed"]))
        if not row or row.get("status") not in {"ok", "resumed"}:
            continue
        complexity = dict(row.get("initial_complexity") or {})
        gate = warehouse_checkpoint_gate(complexity, minimum_conflict_pair_count=16,
            minimum_active_conflict_agent_count=32, minimum_largest_conflict_component_size=16)
        if int(row.get("initial_conflicts", -1)) >= 101 and gate["passed"]:
            result.append({**key, "initial_fingerprint": str(row["state_fingerprint"]), "initial_conflicts": int(row["initial_conflicts"])})
    counts = collections.Counter(r["family"] for r in result)
    if any(counts[name] < 4 for name in ("room", "maze")): raise ValueError("insufficient reset-only qualified room/maze keys")
    return result


def collect(config_path: str | Path, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    path, root, config = load_config(config_path); output_root = Path(output).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    keys = _schedule(config); write_json(output_root / "plan.json", plan(path)); runtime = _runtime_config(config, output_root / "runtime_config.json")
    job_keys = {(r["task_id"], r["solver_seed"]) for r in keys}; anchor = output_root / "qualification_anchor"
    exists = (anchor / "run_config.json").is_file()
    if exists and not resume: raise ValueError("qualification exists; pass resume")
    run_closed_loop_collection(Path(config["_dataset_root"]), runtime, anchor, phase="qualify", workers=8, resume=exists,
        task_ids=sorted({r["task_id"] for r in keys}), job_keys=job_keys, cohort_job_keys=job_keys,
        stopping_rule="wall-clock", repair_seed_policy="episode_stream", deterministic_pp_replay=False,
        wall_time_budget_seconds=60.0, environment_time_limit_seconds=60.0, episode_process_timeout_seconds=90.0,
        qualification_process_timeout_seconds=90.0, use_global_collection_lock=False)
    qualified = _qualified(config, anchor, keys); schedule = _execution_schedule(qualified)
    schedule_path = output_root / "execution_schedule.jsonl"
    if schedule_path.is_file() and read_jsonl(schedule_path) != schedule: raise ValueError("execution schedule changed")
    write_jsonl(schedule_path, schedule)
    for item in schedule:
        key = (item["task_id"], item["solver_seed"]); lane = output_root / "lanes" / f"key_{item['key_index']:02d}" / f"pos_{item['within_key_position']}_{item['controller']}"
        lane_exists = (lane / "run_config.json").is_file()
        if lane_exists and not resume: raise ValueError(f"lane exists; pass resume: {lane}")
        phase, kwargs = _controller_kwargs(root, config, item["controller"])
        common = {"task_ids": [item["task_id"]], "job_keys": {key}, "cohort_job_keys": {key}, "qualification_source": anchor}
        run_closed_loop_collection(Path(config["_dataset_root"]), runtime, lane, phase="qualify", resume=lane_exists, **common, **kwargs)
        run_closed_loop_collection(Path(config["_dataset_root"]), runtime, lane, phase=phase, resume=True, **common, **kwargs)
    return analyze(path, output_root)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = [r["summary"] for r in rows]; successes = sum(bool(x["success"]) for x in s); capped = [float(x["capped_wall_time_to_feasible"]) for x in s]; observed = sum(float(x["episode_observed_wall_seconds"]) for x in s)
    return {"episode_count": len(s), "success_count": successes, "success_rate": successes / len(s) if s else 0.0,
        "mean_capped_ttf_seconds": statistics.fmean(capped) if capped else 0.0, "median_capped_ttf_seconds": statistics.median(capped) if capped else 0.0,
        "successes_per_observed_hour": successes * 3600 / observed if observed else 0.0,
        "mean_controller_wall_seconds": statistics.fmean(float(dict(x.get("controller_totals") or {}).get("controller_seconds_before_repair", 0)) for x in s) if s else 0.0,
        "mean_repair_wall_seconds": statistics.fmean(float(x.get("repair_wall_seconds", 0)) for x in s) if s else 0.0}


def analyze(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path); root = Path(output).resolve()
    schedule_path = root / "execution_schedule.jsonl"
    expected_schedule = _execution_schedule(_qualified(config, root / "qualification_anchor", _schedule(config)))
    schedule = read_jsonl(schedule_path) if schedule_path.is_file() else []
    execution_schedule_exact = bool(schedule_path.is_file() and schedule == expected_schedule)
    if not execution_schedule_exact:
        raise ValueError("room/maze execution schedule is missing or changed")
    collected, missing = [], []
    for item in schedule:
        name = "official_adaptive_manifest.jsonl" if item["controller"] == "official_adaptive" else "realized_dynamic_manifest.jsonl"
        lane = root / "lanes" / f"key_{item['key_index']:02d}" / f"pos_{item['within_key_position']}_{item['controller']}"
        matches = [r for r in (read_jsonl(lane / name) if (lane / name).is_file() else []) if r.get("task_id") == item["task_id"] and int(r.get("solver_seed", -1)) == item["solver_seed"]]
        if len(matches) != 1 or matches[0].get("status") not in {"ok", "resumed"}:
            missing.append(item); continue
        summary = dict(matches[0].get("summary") or {})
        bounded_summary_valid = bool(
            summary.get("ttf_clock_schema") == "lns2.ttf.reset_inclusive_wall.v1"
            and summary.get("capped_wall_time_to_feasible") is not None
            and float(summary.get("wall_time_budget_seconds", -1)) == 60.0
            and int(summary.get("invalid_action_count", -1)) == 0
            and int(summary.get("fingerprint_mismatch_count", -1)) == 0
            and str(summary.get("stop_reason")) in {"success", "wall_timeout", "controller_stalled", "native_terminal"}
        )
        if not bounded_summary_valid:
            missing.append(item); continue
        collected.append({**item, "summary": summary,
            "initial_identity_matches": summary.get("initial_fingerprint") == item["initial_fingerprint"] and int(summary.get("initial_conflicts", -1)) == item["initial_conflicts"],
            "bounded_summary_valid": True})
    def group_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = {c: _summary([r for r in rows if r["controller"] == c]) for c in CONTROLLERS}; indexed = {c: {r["key_index"]: r for r in rows if r["controller"] == c} for c in CONTROLLERS}; ids = sorted(set(indexed[CONTROLLERS[0]]) & set(indexed[CONTROLLERS[1]]))
        pairs = [(indexed[CONTROLLERS[0]][i]["summary"], indexed[CONTROLLERS[1]][i]["summary"]) for i in ids]; wins = sum(float(r["capped_wall_time_to_feasible"]) < float(l["capped_wall_time_to_feasible"]) for l, r in pairs)
        return {"controllers": summaries, "paired": {"pair_count": len(pairs), "dual16_win_count": wins, "dual16_win_rate": wins / len(pairs) if pairs else 0.0,
            "additional_censor_count": sum(bool(l["success"]) and not bool(r["success"]) for l, r in pairs),
            "additional_timeout_count": sum(bool(r.get("external_timeout")) and not bool(l.get("external_timeout")) for l, r in pairs)}}
    groups = {"overall": collected, "room": [r for r in collected if r["family"] == "room"], "maze": [r for r in collected if r["family"] == "maze"]}
    analysis = {name: group_analysis(rows) for name, rows in groups.items()}
    per_map = {map_id: group_analysis([r for r in collected if r["map_id"] == map_id])
        for map_id in sorted({map_id for maps in config["maps"].values() for map_id in maps})}
    roles = {str(row["diagnostic_role"]) for row in expected_schedule}
    per_role = {role: group_analysis([r for r in collected if r["diagnostic_role"] == role]) for role in sorted(roles)}
    qualification_rows = read_jsonl(root / "qualification_anchor" / "qualification_manifest.jsonl")
    qualified_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in expected_schedule}
    qualification = {"candidate_key_count": len(_schedule(config)), "qualified_key_count": len(expected_schedule) // 2,
        "excluded_key_count": len(_schedule(config)) - len(expected_schedule) // 2,
        "excluded": [{"task_id": str(row.get("task_id")), "solver_seed": int(row.get("solver_seed", -1)),
            "map_id": str(row.get("map_id")), "status": str(row.get("status")), "initial_conflicts": row.get("initial_conflicts"),
            "error": row.get("error")} for row in qualification_rows
            if (str(row.get("task_id")), int(row.get("solver_seed", -1))) not in qualified_keys]}
    complete = len(collected) == len(schedule) and not missing
    identity_consistent = bool(complete and all(r["initial_identity_matches"] for r in collected))
    integrity = {"execution_schedule_exact": execution_schedule_exact, "complete_strict_serial_schedule": complete,
        "paired_initial_identity": identity_consistent, "registered_clock_and_stop_reason": bool(complete and all(r["bounded_summary_valid"] for r in collected))}
    report = {"schema": REPORT_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path), "scientific_status": config["scientific_status"],
        "expected_episode_count": len(schedule), "completed_episode_count": len(collected), "missing": missing,
        "integrity_gates": integrity, "initial_identity_consistent": identity_consistent,
        "qualification": qualification, "analysis": analysis, "per_map": per_map, "per_diagnostic_role": per_role,
        "map_disjoint_confirmation": False, "global_or_default_promotion_allowed": False}
    write_json(root / "report.json", report); return report


__all__ = ["CONTROLLERS", "analyze", "collect", "load_config", "plan", "_execution_schedule", "_schedule"]
