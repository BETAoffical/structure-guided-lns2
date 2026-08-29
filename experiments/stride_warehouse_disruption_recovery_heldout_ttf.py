from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_warehouse_disruption_recovery_heldout import (
    EXPERIMENT_ID,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    _ensure_dataset,
    _heldout_schedule,
    load_config,
)
from experiments.stride_warehouse_disruption_recovery_ttf import (
    TTF_OVERRIDE_SCHEMA,
    _common_kwargs,
    _controller_summary,
    _mean,
)
from experiments.warehouse_disruption_checkpoints import compute_checkpoint_identity_sha256
from lns2_selector.runtime.structshell_dual16 import (
    structshell_dual16_augmentation,
    validate_structshell_dual16_augmentation,
)


TTF_PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_heldout_ttf_plan.v1"
TTF_REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_heldout_ttf_report.v1"
CONTROLLERS = ("official_adaptive", "dual16")
TIMED_MANIFESTS = {
    "official_adaptive": "official_adaptive_manifest.jsonl",
    "dual16": "realized_dynamic_manifest.jsonl",
}


def _checkpoint_inputs(
    config_path: str | Path, output: str | Path
) -> tuple[Path, Path, dict[str, Any], Path, list[dict[str, Any]]]:
    path, root, config = load_config(config_path)
    checkpoint_root = Path(output).resolve() / "checkpoints"
    report_path = checkpoint_root / REPORT_FILENAME
    manifest_path = checkpoint_root / MANIFEST_FILENAME
    if not report_path.is_file() or not manifest_path.is_file():
        raise ValueError("Held-out checkpoint evidence is missing")
    report = read_json(report_path)
    if (
        report.get("experiment_id") != EXPERIMENT_ID
        or report.get("passed") is not True
        or report.get("config_sha256") != sha256_file(path)
        or report.get("checkpoint_selection_controller_outcomes_consulted") is not False
        or report.get("failed_candidate_replacement") is not False
    ):
        raise ValueError("Held-out checkpoint report did not pass")
    rows = [dict(row) for row in read_jsonl(manifest_path)]
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    if (
        len(rows) < int(config["checkpoint_generation"]["minimum_qualified_count"])
        or len(rows) > int(config["checkpoint_generation"]["expected_candidate_count"])
        or len(qualified) < int(config["checkpoint_generation"]["minimum_qualified_count"])
        or len({str(row["map_id"]) for row in qualified})
        < int(config["checkpoint_generation"]["minimum_qualified_map_count"])
        or int(report.get("attempted_candidate_count", -1)) != 4
        or int(report.get("qualified_checkpoint_count", -1)) != len(qualified)
        or int(report.get("execution_failure_count", -1)) != 0
        or set(map(str, report.get("qualified_checkpoint_ids") or ()))
        != {str(row.get("checkpoint_id")) for row in qualified}
    ):
        raise ValueError("Held-out TTF requires the qualified frozen checkpoint cohort")
    from experiments.repair_collection import _load_dataset_rows

    dataset_root = Path(str(config["_dataset_root"]))
    development_rows = _load_dataset_rows(dataset_root, ["development"])
    heldout_dataset_rows = _load_dataset_rows(dataset_root, ["controller_held_out"])
    development_maps = {str(row["map_id"]) for row in development_rows}
    heldout_maps = {str(row["map_id"]) for row in qualified}
    if not development_maps or development_maps & heldout_maps:
        raise ValueError("Held-out checkpoint maps overlap development maps")
    development_hashes = {
        sha256_file(dataset_root / "development" / str(row["map_file"]))
        for row in development_rows
    }
    heldout_hashes = {
        sha256_file(dataset_root / "controller_held_out" / str(row["map_file"]))
        for row in heldout_dataset_rows
    }
    if development_hashes & heldout_hashes:
        raise ValueError("Held-out checkpoint map bytes overlap development maps")
    registered_by_task = {
        str(row["task_id"]): row
        for row in heldout_dataset_rows
        if str(row.get("task_variant"))
        in set(config["dataset"]["screen_task_variants"])
    }
    scheduled_by_task = {
        str(item["task_id"]): item
        for item in _heldout_schedule(config, dataset_root)
    }
    if len({str(row.get("checkpoint_id")) for row in qualified}) != len(qualified):
        raise ValueError("checkpoint ids are empty or duplicated")
    if len({int(row.get("key_index", -1)) for row in qualified}) != len(qualified):
        raise ValueError("checkpoint key indices are duplicated")
    for row in qualified:
        if str(row.get("split")) != "controller_held_out":
            raise ValueError("held-out checkpoint split changed")
        registered = registered_by_task.get(str(row.get("task_id")))
        scheduled = scheduled_by_task.get(str(row.get("task_id")))
        if registered is None or scheduled is None:
            raise ValueError("held-out checkpoint task is not in the registered cohort")
        split_root = dataset_root / "controller_held_out"
        if (
            str(row.get("map_id")) != str(registered["map_id"])
            or str(row.get("task_variant")) != str(registered["task_variant"])
            or int(row.get("agent_count", -1)) != int(registered["agent_count"])
            or str(row.get("checkpoint_id")) != str(scheduled["checkpoint_id"])
            or int(row.get("key_index", -1)) != int(scheduled["key_index"])
            or int(row.get("screen_solver_seed", -1))
            != int(scheduled["screen_solver_seed"])
            or int(row.get("restore_seed", -1)) != int(scheduled["selection_seed"])
            or int(dict(row.get("incumbent") or {}).get("solver_seed", -1))
            != int(scheduled["incumbent_solver_seed"])
            or str(row.get("map_sha256"))
            != sha256_file(split_root / str(registered["map_file"]))
            or str(row.get("task_sha256"))
            != sha256_file(split_root / str(registered["task_file"]))
        ):
            raise ValueError("held-out checkpoint differs from its registered dataset task")
        if str(row.get("checkpoint_identity_sha256") or "") != compute_checkpoint_identity_sha256(row):
            raise ValueError(f"checkpoint identity changed: {row.get('checkpoint_id')}")
        if row.get("source_kind") != "checkpoint_blob_v1":
            raise ValueError("Held-out TTF requires checkpoint_blob_v1 inputs")
        blob = checkpoint_root / str(row.get("state_blob") or "")
        if not blob.is_file() or sha256_file(blob) != str(row.get("state_blob_sha256")):
            raise ValueError(f"checkpoint blob changed: {row.get('checkpoint_id')}")
    qualified.sort(key=lambda row: int(row["key_index"]))
    return path, root, config, checkpoint_root, qualified


def _schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for checkpoint in rows:
        key_index = int(checkpoint["key_index"])
        for position in range(2):
            controller = CONTROLLERS[(key_index % 2 + position) % 2]
            result.append({
                "schedule_index": len(result),
                "key_index": key_index,
                "checkpoint_id": str(checkpoint["checkpoint_id"]),
                "checkpoint_identity_sha256": str(checkpoint["checkpoint_identity_sha256"]),
                "map_id": str(checkpoint["map_id"]),
                "task_id": str(checkpoint["task_id"]),
                "task_variant": str(checkpoint["task_variant"]),
                "agent_count": int(checkpoint["agent_count"]),
                "solver_seed": int(checkpoint["screen_solver_seed"]),
                "controller": controller,
                "within_key_position": position,
            })
    return result


def plan_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config, checkpoint_root, checkpoints = _checkpoint_inputs(config_path, output)
    schedule = _schedule(checkpoints)
    return {
        "schema": TTF_PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_count": len(checkpoints),
        "controller_count": 2,
        "episode_count": len(schedule),
        "controllers": list(CONTROLLERS),
        "execution_order": str(config["runtime"]["execution_order"]),
        "timed_workers": 1,
        "qualification_workers": 4,
        "timing_boundary": str(config["runtime"]["timing_boundary"]),
        "map_disjoint_from_development": True,
        "targeted_expert_confirmation_only": True,
        "global_or_default_promotion_allowed": False,
        "schedule": schedule,
    }


def _runtime_config(config: Mapping[str, Any], checkpoints: list[dict[str, Any]], destination: Path) -> Path:
    payload = dict(read_json(Path(str(config["_runtime_template"]))))
    runtime = dict(config["runtime"])
    payload.update({
        "formal": False,
        "split": "controller_held_out",
        "solver_seeds": sorted(int(row["screen_solver_seed"]) for row in checkpoints),
        "policies": ["official_adaptive", "realized_dynamic"],
        "wall_time_budget_seconds": 120.0,
        "episode_process_timeout_seconds": 150.0,
        "workers": 1,
        "deterministic_pp_replay": False,
        "repair_seed_policy": "episode_stream",
        "dataset_design": {
            "mode": "structured", "map_count": 2, "tasks_per_map": 4,
            "task_variants": ["station_rush_d10", "balanced_od_d125"],
            "layout_counts": {"station_centric": 2},
        },
    })
    environment = dict(payload["environment"])
    environment["time_limit"] = float(runtime["environment_time_limit_seconds"])
    payload["environment"] = environment
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and read_json(destination) != payload:
        raise ValueError("existing held-out TTF runtime config changed")
    if not destination.is_file():
        write_json(destination, payload)
    return destination


def _controller_kwargs(root: Path, config: Mapping[str, Any], controller: str) -> tuple[str, dict[str, Any]]:
    common = _common_kwargs(config)
    if controller == "official_adaptive":
        return "official_adaptive", {
            **common, "controller": "official_adaptive", "feature_backend": "auto",
            "controller_runtime": "reference", "verification_profile": "audit",
        }
    if controller != "dual16":
        raise ValueError(f"unknown held-out controller: {controller}")
    bundle_path = str(config["controller_contract"].get("v2_bundle") or "artifacts/initlns-closed-loop-controller-v2")
    augmentation = validate_structshell_dual16_augmentation(structshell_dual16_augmentation())
    assert augmentation is not None
    return "realized_dynamic", {
        **common, "controller": "v2-full", "controller_bundle": str((root / bundle_path).resolve()),
        "feature_backend": "native", "controller_runtime": "optimized",
        "verification_profile": "deployment", "hybridstructpool_augmentation": augmentation,
    }


def _lane_root(ttf_root: Path, item: Mapping[str, Any]) -> Path:
    return ttf_root / "lanes" / f"key_{int(item['key_index']):02d}" / f"pos_{int(item['within_key_position'])}_{item['controller']}"


def collect_ttf(config_path: str | Path, output: str | Path, *, resume: bool = False) -> dict[str, Any]:
    path, root, config, checkpoint_root, checkpoints = _checkpoint_inputs(config_path, output)
    dataset_root = _ensure_dataset(root, config)
    ttf_root = Path(output).resolve() / "ttf"
    schedule = _schedule(checkpoints)
    ttf_root.mkdir(parents=True, exist_ok=True)
    schedule_path = ttf_root / "execution_schedule.jsonl"
    if schedule_path.is_file() and read_jsonl(schedule_path) != schedule:
        raise ValueError("existing held-out TTF schedule changed")
    write_jsonl(schedule_path, schedule)
    runtime_path = _runtime_config(config, checkpoints, ttf_root / "runtime_config.json")
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    all_keys = {(str(row["task_id"]), int(row["screen_solver_seed"])) for row in checkpoints}
    anchor = ttf_root / "qualification_anchor"
    anchor_exists = (anchor / "run_config.json").is_file()
    if anchor_exists and not resume:
        raise ValueError("held-out qualification anchor exists; pass resume")
    run_closed_loop_collection(
        dataset_root, runtime_path, anchor, phase="qualify", workers=4, resume=anchor_exists,
        task_ids=sorted(task for task, _seed in all_keys), job_keys=all_keys,
        cohort_job_keys=all_keys, qualification_process_timeout_seconds=150.0,
        **{key: value for key, value in _common_kwargs(config).items() if key != "workers"},
    )
    progress: list[dict[str, Any]] = []
    for item in schedule:
        checkpoint = checkpoint_by_id[str(item["checkpoint_id"])]
        key = (str(item["task_id"]), int(item["solver_seed"]))
        overrides = {key: {
            "schema": TTF_OVERRIDE_SCHEMA, "state_id": str(checkpoint["checkpoint_id"]),
            "initial_restore": {**checkpoint, "collection_root": str(checkpoint_root.resolve())},
        }}
        lane = _lane_root(ttf_root, item)
        lane_exists = (lane / "run_config.json").is_file()
        if lane_exists and not resume:
            raise ValueError(f"held-out TTF lane exists; pass resume: {lane}")
        phase, kwargs = _controller_kwargs(root, config, str(item["controller"]))
        common = dict(task_ids=[str(item["task_id"])], job_keys={key}, cohort_job_keys={key},
                      qualification_source=anchor, episode_overrides=overrides)
        run_closed_loop_collection(dataset_root, runtime_path, lane, phase="qualify", resume=lane_exists, **common, **kwargs)
        result = run_closed_loop_collection(dataset_root, runtime_path, lane, phase=phase, resume=True, **common, **kwargs)
        progress.append({**item, "lane": str(lane), "collection": result})
        write_json(ttf_root / "collection_progress.json", {"schema": TTF_PLAN_SCHEMA, "completed": len(progress), "rows": progress})
    return analyze_ttf(path, output)


def analyze_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config, _checkpoint_root, checkpoints = _checkpoint_inputs(config_path, output)
    ttf_root = Path(output).resolve() / "ttf"
    schedule = _schedule(checkpoints)
    schedule_path = ttf_root / "execution_schedule.jsonl"
    schedule_integrity = bool(
        schedule_path.is_file() and read_jsonl(schedule_path) == schedule
    )
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    collected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in schedule:
        manifest = _lane_root(ttf_root, item) / TIMED_MANIFESTS[str(item["controller"])]
        matches = [row for row in (read_jsonl(manifest) if manifest.is_file() else [])
                   if str(row.get("task_id")) == str(item["task_id"])
                   and int(row.get("solver_seed", -1)) == int(item["solver_seed"])]
        if len(matches) != 1 or matches[0].get("status") not in {"ok", "resumed"}:
            missing.append(dict(item)); continue
        checkpoint = checkpoint_by_id[str(item["checkpoint_id"])]
        summary = dict(matches[0].get("summary") or {})
        collected.append({
            **item, "manifest": matches[0], "summary": summary,
            "initial_fingerprint_matches": summary.get("initial_fingerprint") == checkpoint.get("expected_fingerprint"),
            "initial_conflicts_match": int(summary.get("initial_conflicts", -1)) == int(checkpoint["expected_conflicts"]),
            "bounded_summary_valid": bool(
                summary.get("ttf_clock_schema") == "lns2.ttf.reset_inclusive_wall.v1"
                and summary.get("capped_wall_time_to_feasible") is not None
                and float(summary.get("wall_time_budget_seconds", -1)) == 120.0
                and int(summary.get("invalid_action_count", -1)) == 0
                and int(summary.get("fingerprint_mismatch_count", -1)) == 0
                and str(summary.get("stop_reason"))
                in {"success", "wall_timeout", "controller_stalled", "native_terminal"}),
        })
    by_controller = {name: [row for row in collected if row["controller"] == name] for name in CONTROLLERS}
    summaries = {name: _controller_summary(rows) for name, rows in by_controller.items()}
    indexed = {name: {str(row["checkpoint_id"]): row for row in rows} for name, rows in by_controller.items()}
    common_ids = sorted(set(indexed["official_adaptive"]) & set(indexed["dual16"]))
    pairs = [(indexed["official_adaptive"][key], indexed["dual16"][key]) for key in common_ids]
    left = [dict(pair[0]["summary"]) for pair in pairs]
    right = [dict(pair[1]["summary"]) for pair in pairs]
    left_capped = [float(row["capped_wall_time_to_feasible"]) for row in left]
    right_capped = [float(row["capped_wall_time_to_feasible"]) for row in right]
    left_mean, right_mean = _mean(left_capped), _mean(right_capped)
    base_rate = float(summaries["official_adaptive"]["successes_per_observed_hour"])
    target_rate = float(summaries["dual16"]["successes_per_observed_hour"])
    throughput = (target_rate - base_rate) / base_rate if base_rate else (None if target_rate == 0 else float("inf"))
    additional_timeout = sum(bool(r["external_timeout"]) and not bool(l["external_timeout"]) for l, r in zip(left, right))
    additional_censor = sum(bool(l["success"]) and not bool(r["success"]) for l, r in zip(left, right))
    gate = dict(config["screen_gate"])
    comparison = {
        "paired_key_count": len(pairs),
        "paired_win_count": sum(r < l for l, r in zip(left_capped, right_capped)),
        "paired_win_rate": sum(r < l for l, r in zip(left_capped, right_capped)) / len(pairs) if pairs else 0.0,
        "mean_capped_ttf_improvement": (left_mean - right_mean) / left_mean if left_mean else 0.0,
        "throughput_improvement": throughput,
        "success_rate_loss": summaries["official_adaptive"]["success_rate"] - summaries["dual16"]["success_rate"],
        "additional_timeout_count": additional_timeout,
        "additional_censor_count": additional_censor,
    }
    comparison["screen_gate_passed"] = bool(
        len(pairs) == len(checkpoints)
        and comparison["paired_win_rate"] >= float(gate["minimum_paired_win_rate"])
        and comparison["mean_capped_ttf_improvement"] >= float(gate["minimum_mean_capped_ttf_improvement"])
        and throughput is not None and throughput >= float(gate["minimum_successes_per_hour_improvement"])
        and comparison["success_rate_loss"] <= float(gate["maximum_success_rate_loss"])
        and (not gate["no_additional_timeout_or_censor"] or additional_timeout == additional_censor == 0))
    integrity = {
        "complete_strict_serial_schedule": schedule_integrity and len(collected) == len(schedule) and not missing,
        "paired_initial_fingerprints": all(row["initial_fingerprint_matches"] for row in collected),
        "paired_initial_conflicts": all(row["initial_conflicts_match"] for row in collected),
        "registered_reset_inclusive_ttf_clock": all(row["bounded_summary_valid"] for row in collected),
        "all_qualified_rows_per_controller": all(len(rows) == len(checkpoints) for rows in by_controller.values()),
        "map_disjoint_from_development": True,
    }
    complete = all(integrity.values())
    report = {
        "schema": TTF_REPORT_SCHEMA, "experiment_id": EXPERIMENT_ID, "config_sha256": sha256_file(path),
        "checkpoint_count": len(checkpoints), "expected_episode_count": len(schedule),
        "completed_episode_count": len(collected), "complete_pairing_and_initial_state_identity": complete,
        "integrity_gates": integrity, "missing_or_failed_schedule_rows": missing,
        "controllers": summaries, "comparison_vs_official": comparison, "screen_gate": gate,
        "targeted_expert_confirmation": bool(complete and comparison["screen_gate_passed"]),
        "global_claim_allowed": False, "default_replacement_allowed": False,
        "formal_promotion_allowed": False,
        "promotion_decision": "targeted_expert_confirmation_only_no_global_or_default_promotion",
        "timing_boundary": "checkpoint_restore_inclusive_ttf",
    }
    write_json(ttf_root / "heldout_ttf_report.json", report)
    return report


__all__ = ["CONTROLLERS", "TTF_PLAN_SCHEMA", "TTF_REPORT_SCHEMA", "_checkpoint_inputs", "_schedule", "analyze_ttf", "collect_ttf", "plan_ttf"]
