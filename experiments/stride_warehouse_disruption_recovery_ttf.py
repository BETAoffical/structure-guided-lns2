from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_warehouse_disruption_recovery import (
    EXPERIMENT_ID,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    _ensure_dataset,
    load_config,
)
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)
from lns2_selector.runtime.structshell_dual16 import (
    structshell_dual16_augmentation,
    validate_structshell_dual16_augmentation,
)


TTF_PLAN_SCHEMA = "lns2.stride.warehouse_disruption_recovery_ttf_plan.v1"
TTF_REPORT_SCHEMA = "lns2.stride.warehouse_disruption_recovery_ttf_report.v1"
TTF_OVERRIDE_SCHEMA = "lns2.warehouse_disruption_recovery_episode_override.v1"
CONTROLLERS = ("official_adaptive", "v2_only", "mixed_full_v2", "dual16")
TIMED_MANIFESTS = {
    "official_adaptive": "official_adaptive_manifest.jsonl",
    "v2_only": "realized_dynamic_manifest.jsonl",
    "mixed_full_v2": "realized_dynamic_manifest.jsonl",
    "dual16": "realized_dynamic_manifest.jsonl",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _checkpoint_inputs(
    config_path: str | Path, output: str | Path
) -> tuple[Path, Path, dict[str, Any], Path, list[dict[str, Any]]]:
    path, root, config = load_config(config_path)
    output_root = Path(output).resolve()
    checkpoint_root = output_root / "checkpoints"
    report_path = checkpoint_root / REPORT_FILENAME
    manifest_path = checkpoint_root / MANIFEST_FILENAME
    if not report_path.is_file() or not manifest_path.is_file():
        raise ValueError("Warehouse disruption checkpoint evidence is missing")
    report = read_json(report_path)
    if (
        report.get("experiment_id") != EXPERIMENT_ID
        or report.get("passed") is not True
        or report.get("config_sha256") != sha256_file(path)
        or report.get("checkpoint_selection_controller_outcomes_consulted") is not False
        or report.get("failed_candidate_replacement") is not False
    ):
        raise ValueError("Warehouse disruption checkpoint report did not pass")
    rows = [dict(row) for row in read_jsonl(manifest_path)]
    qualified = [row for row in rows if dict(row.get("qualification") or {}).get("passed")]
    qualified_ids = {str(row.get("checkpoint_id")) for row in qualified}
    if (
        len(rows) != 7
        or len(qualified) != 7
        or int(report.get("attempted_candidate_count", -1)) != 8
        or int(report.get("qualified_checkpoint_count", -1)) != 7
        or int(report.get("state_supply_unavailable_count", -1)) != 1
        or int(report.get("execution_failure_count", -1)) != 0
        or set(map(str, report.get("qualified_checkpoint_ids") or ()))
        != qualified_ids
    ):
        raise ValueError("TTF screen requires the frozen seven qualified checkpoints")
    if len({str(row.get("checkpoint_id")) for row in rows}) != 7:
        raise ValueError("checkpoint ids are empty or duplicated")
    if len({int(row.get("key_index", -1)) for row in rows}) != 7:
        raise ValueError("checkpoint key indices are duplicated")
    for row in rows:
        observed = str(row.get("checkpoint_identity_sha256") or "")
        if observed != compute_checkpoint_identity_sha256(row):
            raise ValueError(f"checkpoint identity changed: {row.get('checkpoint_id')}")
        if row.get("source_kind") != "checkpoint_blob_v1":
            raise ValueError("TTF requires checkpoint_blob_v1 inputs")
        blob = checkpoint_root / str(row.get("state_blob") or "")
        if not blob.is_file() or sha256_file(blob) != str(row.get("state_blob_sha256")):
            raise ValueError(f"checkpoint blob changed: {row.get('checkpoint_id')}")
    rows.sort(key=lambda row: int(row["key_index"]))
    return path, root, config, checkpoint_root, rows


def _schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for checkpoint in rows:
        key_index = int(checkpoint["key_index"])
        offset = key_index % len(CONTROLLERS)
        for position in range(len(CONTROLLERS)):
            controller = CONTROLLERS[(offset + position) % len(CONTROLLERS)]
            result.append(
                {
                    "schedule_index": len(result),
                    "key_index": key_index,
                    "checkpoint_id": str(checkpoint["checkpoint_id"]),
                    "checkpoint_identity_sha256": str(
                        checkpoint["checkpoint_identity_sha256"]
                    ),
                    "map_id": str(checkpoint["map_id"]),
                    "task_id": str(checkpoint["task_id"]),
                    "task_variant": str(checkpoint["task_variant"]),
                    "agent_count": int(checkpoint["agent_count"]),
                    "solver_seed": int(checkpoint["screen_solver_seed"]),
                    "controller": controller,
                    "within_key_position": position,
                }
            )
    return result


def plan_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config, checkpoint_root, checkpoints = _checkpoint_inputs(
        config_path, output
    )
    schedule = _schedule(checkpoints)
    return {
        "schema": TTF_PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_count": len(checkpoints),
        "controller_count": len(CONTROLLERS),
        "episode_count": len(schedule),
        "controllers": list(CONTROLLERS),
        "execution_order": str(config["runtime"]["execution_order"]),
        "timed_workers": 1,
        "qualification_workers": int(config["runtime"]["qualification_workers"]),
        "timing_boundary": str(config["runtime"]["timing_boundary"]),
        "controller_outcomes_consulted_for_checkpoint_selection": False,
        "formal_promotion_allowed": False,
        "schedule": schedule,
    }


def _runtime_config(
    root: Path,
    config: Mapping[str, Any],
    checkpoints: list[dict[str, Any]],
    destination: Path,
) -> Path:
    template = read_json(Path(str(config["_runtime_template"])))
    runtime = dict(config["runtime"])
    payload = dict(template)
    payload.update(
        {
            "formal": False,
            "split": "development",
            "solver_seeds": sorted(int(row["screen_solver_seed"]) for row in checkpoints),
            "policies": ["official_adaptive", "realized_dynamic"],
            "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
            "episode_process_timeout_seconds": float(
                runtime["episode_process_timeout_seconds"]
            ),
            "workers": 1,
            "deterministic_pp_replay": False,
            "repair_seed_policy": "episode_stream",
            "dataset_design": {
                "mode": "structured",
                "map_count": 4,
                "tasks_per_map": 4,
                "task_variants": [
                    "station_rush_d10",
                    "station_release_d10",
                    "station_dominant_d125",
                    "balanced_od_d125",
                ],
                "layout_counts": {"station_centric": 4},
            },
        }
    )
    environment = dict(payload["environment"])
    environment["time_limit"] = float(runtime["environment_time_limit_seconds"])
    payload["environment"] = environment
    model_registration = dict(payload["model_registration"])
    registered_bundles = dict(
        model_registration.get("registered_controller_bundles") or {}
    )
    registered_bundles["mixed-full-v2"] = {
        "controller_manifest_sha256": str(
            config["controller_contract"]["mixed_manifest_sha256"]
        )
    }
    model_registration["registered_controller_bundles"] = registered_bundles
    payload["model_registration"] = model_registration
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError("existing TTF runtime config changed")
    else:
        write_json(destination, payload)
    return destination


def _common_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    return {
        "workers": 1,
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
        "use_global_collection_lock": False,
    }


def _controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> tuple[str, dict[str, Any]]:
    common = _common_kwargs(config)
    contract = dict(config["controller_contract"])
    if controller == "official_adaptive":
        return "official_adaptive", {
            **common,
            "controller": "official_adaptive",
            "feature_backend": "auto",
            "controller_runtime": "reference",
            "verification_profile": "audit",
        }
    if controller == "mixed_full_v2":
        mode = "mixed-full-v2"
        bundle = (root / str(contract["mixed_bundle"])).resolve()
    else:
        mode = "v2-full"
        bundle = (root / str(contract["v2_bundle"])).resolve()
    result: dict[str, Any] = {
        **common,
        "controller": mode,
        "controller_bundle": str(bundle),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if controller == "dual16":
        augmentation = validate_structshell_dual16_augmentation(
            structshell_dual16_augmentation()
        )
        assert augmentation is not None
        result["hybridstructpool_augmentation"] = augmentation
    elif controller not in {"v2_only", "mixed_full_v2"}:
        raise ValueError(f"unknown Warehouse disruption controller: {controller}")
    return "realized_dynamic", result


def _lane_root(ttf_root: Path, item: Mapping[str, Any]) -> Path:
    return (
        ttf_root
        / "lanes"
        / f"key_{int(item['key_index']):02d}"
        / f"pos_{int(item['within_key_position'])}_{item['controller']}"
    )


def collect_ttf(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    path, root, config, checkpoint_root, checkpoints = _checkpoint_inputs(
        config_path, output
    )
    dataset_root = _ensure_dataset(root, config)
    output_root = Path(output).resolve()
    ttf_root = output_root / "ttf"
    plan = plan_ttf(path, output_root)
    schedule = list(plan["schedule"])
    ttf_root.mkdir(parents=True, exist_ok=True)
    plan_path = ttf_root / "execution_schedule.jsonl"
    if plan_path.is_file() and read_jsonl(plan_path) != schedule:
        raise ValueError("existing TTF schedule changed")
    write_jsonl(plan_path, schedule)
    runtime_path = _runtime_config(
        root, config, checkpoints, ttf_root / "runtime_config.json"
    )
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    all_keys = {
        (str(row["task_id"]), int(row["screen_solver_seed"]))
        for row in checkpoints
    }
    task_ids = sorted({task_id for task_id, _seed in all_keys})
    anchor = ttf_root / "qualification_anchor"
    anchor_exists = (anchor / "run_config.json").is_file()
    if anchor_exists and not resume:
        raise ValueError("TTF qualification anchor exists; pass resume")
    run_closed_loop_collection(
        dataset_root,
        runtime_path,
        anchor,
        phase="qualify",
        workers=int(config["runtime"]["qualification_workers"]),
        resume=anchor_exists,
        task_ids=task_ids,
        job_keys=all_keys,
        cohort_job_keys=all_keys,
        qualification_process_timeout_seconds=float(
            config["runtime"]["episode_process_timeout_seconds"]
        ),
        **{
            key: value
            for key, value in _common_kwargs(config).items()
            if key not in {"workers"}
        },
    )
    progress: list[dict[str, Any]] = []
    for item in schedule:
        checkpoint = checkpoint_by_id[str(item["checkpoint_id"])]
        key = (str(item["task_id"]), int(item["solver_seed"]))
        override = {
            "schema": TTF_OVERRIDE_SCHEMA,
            "state_id": str(checkpoint["checkpoint_id"]),
            "initial_restore": {
                **checkpoint,
                "collection_root": str(checkpoint_root.resolve()),
            },
        }
        overrides = {key: override}
        lane = _lane_root(ttf_root, item)
        lane_exists = (lane / "run_config.json").is_file()
        if lane_exists and not resume:
            raise ValueError(f"TTF lane exists; pass resume: {lane}")
        phase, kwargs = _controller_kwargs(root, config, str(item["controller"]))
        run_closed_loop_collection(
            dataset_root,
            runtime_path,
            lane,
            phase="qualify",
            resume=lane_exists,
            task_ids=[str(item["task_id"])],
            job_keys={key},
            cohort_job_keys={key},
            qualification_source=anchor,
            episode_overrides=overrides,
            **kwargs,
        )
        result = run_closed_loop_collection(
            dataset_root,
            runtime_path,
            lane,
            phase=phase,
            resume=True,
            task_ids=[str(item["task_id"])],
            job_keys={key},
            cohort_job_keys={key},
            qualification_source=anchor,
            episode_overrides=overrides,
            **kwargs,
        )
        progress.append({**item, "lane": str(lane), "collection": result})
        write_json(
            ttf_root / "collection_progress.json",
            {"schema": TTF_PLAN_SCHEMA, "completed": len(progress), "rows": progress},
        )
    return analyze_ttf(path, output_root)


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(row["summary"]) for row in rows]
    successes = sum(bool(row["success"]) for row in summaries)
    capped = [float(row["capped_wall_time_to_feasible"]) for row in summaries]
    observed = sum(float(row["episode_observed_wall_seconds"]) for row in summaries)
    return {
        "episode_count": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_capped_ttf_seconds": _mean(capped),
        "median_capped_ttf_seconds": statistics.median(capped) if capped else 0.0,
        "mean_successful_ttf_seconds": _mean(
            [
                float(row["wall_time_to_feasible"])
                for row in summaries
                if bool(row["success"])
            ]
        ),
        "successes_per_observed_hour": successes * 3600.0 / observed if observed else 0.0,
        "total_observed_wall_seconds": observed,
        "external_timeout_count": sum(bool(row["external_timeout"]) for row in summaries),
        "controller_stalled_count": sum(
            str(row.get("stop_reason")) == "controller_stalled" for row in summaries
        ),
        "mean_reset_seconds": _mean(
            [float(row.get("reset_wall_seconds", 0.0)) for row in summaries]
        ),
        "mean_repair_wall_seconds": _mean(
            [float(row.get("repair_wall_seconds", 0.0)) for row in summaries]
        ),
        "mean_repair_iterations": _mean(
            [float(row.get("repair_iterations", 0)) for row in summaries]
        ),
    }


def analyze_ttf(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    path, _root, config, _checkpoint_root, checkpoints = _checkpoint_inputs(
        config_path, output
    )
    ttf_root = Path(output).resolve() / "ttf"
    schedule = _schedule(checkpoints)
    checkpoint_by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    collected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in schedule:
        lane = _lane_root(ttf_root, item)
        manifest = lane / TIMED_MANIFESTS[str(item["controller"])]
        rows = read_jsonl(manifest) if manifest.is_file() else []
        matches = [
            row
            for row in rows
            if str(row.get("task_id")) == str(item["task_id"])
            and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
        ]
        if len(matches) != 1 or matches[0].get("status") not in {"ok", "resumed"}:
            missing.append(dict(item))
            continue
        checkpoint = checkpoint_by_id[str(item["checkpoint_id"])]
        summary = dict(matches[0].get("summary") or {})
        collected.append(
            {
                **item,
                "manifest": matches[0],
                "summary": summary,
                "initial_fingerprint_matches": summary.get("initial_fingerprint")
                == checkpoint.get("expected_fingerprint"),
                "initial_conflicts_match": int(summary.get("initial_conflicts", -1))
                == int(checkpoint["expected_conflicts"]),
                "bounded_summary_valid": bool(
                    summary.get("ttf_clock_schema")
                    == "lns2.ttf.reset_inclusive_wall.v1"
                    and summary.get("capped_wall_time_to_feasible") is not None
                    and float(summary.get("wall_time_budget_seconds", -1.0))
                    == float(config["runtime"]["wall_time_budget_seconds"])
                    and int(summary.get("invalid_action_count", -1)) == 0
                    and int(summary.get("fingerprint_mismatch_count", -1)) == 0
                    and str(summary.get("stop_reason"))
                    in {
                        "success",
                        "wall_timeout",
                        "controller_stalled",
                        "native_terminal",
                    }
                ),
            }
        )
    by_controller = {
        controller: [row for row in collected if row["controller"] == controller]
        for controller in CONTROLLERS
    }
    summaries = {
        controller: _controller_summary(rows)
        for controller, rows in by_controller.items()
    }
    indexed = {
        controller: {str(row["checkpoint_id"]): row for row in rows}
        for controller, rows in by_controller.items()
    }
    baseline = indexed["official_adaptive"]
    gate = dict(config["screen_gate"])
    comparisons: dict[str, Any] = {}
    for controller in CONTROLLERS[1:]:
        common_ids = sorted(set(baseline) & set(indexed[controller]))
        pairs = [(baseline[key], indexed[controller][key]) for key in common_ids]
        left = [dict(pair[0]["summary"]) for pair in pairs]
        right = [dict(pair[1]["summary"]) for pair in pairs]
        left_capped = [float(row["capped_wall_time_to_feasible"]) for row in left]
        right_capped = [float(row["capped_wall_time_to_feasible"]) for row in right]
        wins = sum(r < l for l, r in zip(left_capped, right_capped))
        left_mean, right_mean = _mean(left_capped), _mean(right_capped)
        base_throughput = summaries["official_adaptive"]["successes_per_observed_hour"]
        candidate_throughput = summaries[controller]["successes_per_observed_hour"]
        success_loss = summaries["official_adaptive"]["success_rate"] - summaries[controller]["success_rate"]
        additional_timeouts = sum(
            bool(r["external_timeout"]) and not bool(l["external_timeout"])
            for l, r in zip(left, right)
        )
        additional_censors = sum(
            bool(l["success"]) and not bool(r["success"])
            for l, r in zip(left, right)
        )
        throughput_improvement = (
            (candidate_throughput - base_throughput) / base_throughput
            if base_throughput
            else None
        )
        throughput_gate_passed = (
            throughput_improvement
            >= float(gate["minimum_successes_per_hour_improvement"])
            if throughput_improvement is not None
            else candidate_throughput > 0.0
        )
        values = {
            "paired_key_count": len(pairs),
            "paired_win_count": wins,
            "paired_win_rate": wins / len(pairs) if pairs else 0.0,
            "mean_capped_ttf_improvement": (
                (left_mean - right_mean) / left_mean if left_mean else 0.0
            ),
            "throughput_improvement": throughput_improvement,
            "throughput_gate_passed": throughput_gate_passed,
            "success_rate_loss": success_loss,
            "additional_timeout_count": additional_timeouts,
            "additional_censor_count": additional_censors,
        }
        values["screen_gate_passed"] = bool(
            len(pairs) == len(checkpoints)
            and values["paired_win_rate"] >= float(gate["minimum_paired_win_rate"])
            and values["mean_capped_ttf_improvement"]
            >= float(gate["minimum_mean_capped_ttf_improvement"])
            and throughput_gate_passed
            and success_loss <= float(gate["maximum_success_rate_loss"])
            and (
                not bool(gate["no_additional_timeout_or_censor"])
                or additional_timeouts == 0 and additional_censors == 0
            )
        )
        comparisons[controller] = values
    complete = bool(
        len(collected) == len(schedule)
        and not missing
        and all(row["initial_fingerprint_matches"] for row in collected)
        and all(row["initial_conflicts_match"] for row in collected)
        and all(row["bounded_summary_valid"] for row in collected)
        and all(len(rows) == len(checkpoints) for rows in by_controller.values())
    )
    integrity = {
        "complete_strict_serial_schedule": len(collected) == len(schedule) and not missing,
        "paired_initial_fingerprints": all(
            row["initial_fingerprint_matches"] for row in collected
        ),
        "paired_initial_conflicts": all(
            row["initial_conflicts_match"] for row in collected
        ),
        "registered_reset_inclusive_ttf_clock": all(
            row["bounded_summary_valid"] for row in collected
        ),
        "seven_rows_per_controller": all(
            len(rows) == len(checkpoints) for rows in by_controller.values()
        ),
    }
    report = {
        "schema": TTF_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "checkpoint_count": len(checkpoints),
        "expected_episode_count": len(schedule),
        "completed_episode_count": len(collected),
        "complete_pairing_and_initial_state_identity": complete,
        "integrity_gates": integrity,
        "missing_or_failed_schedule_rows": missing,
        "controllers": summaries,
        "comparisons_vs_official": comparisons,
        "screen_gate": gate,
        "screen_gate_passed_controllers": sorted(
            controller
            for controller, value in comparisons.items()
            if value["screen_gate_passed"]
        ),
        "formal_promotion_allowed": False,
        "promotion_decision": "development_screen_only_no_promotion",
        "timing_boundary": "checkpoint_restore_inclusive_ttf",
    }
    write_json(ttf_root / "ttf_report.json", report)
    return report


__all__ = ["analyze_ttf", "collect_ttf", "plan_ttf"]
