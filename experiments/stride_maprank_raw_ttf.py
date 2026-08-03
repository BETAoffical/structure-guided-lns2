from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.balanced_wall_clock import prepare_movingai_dataset
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_maprank import CONTROLLER_ID, validate_maprank_evaluation_config
from experiments.stride_maprank_evaluation import _bundle_paths, _training_evidence
from experiments.stride_robuststep_preflight import _mean
from experiments.stride_stage3 import _project_path


REPORT_SCHEMA = "lns2.stride.maprank_raw_ttf_report.v1"
STATUS_SCHEMA = "lns2.stride.maprank_raw_ttf_status.v1"
CONTROLLERS = ("v2-full", "v2-augmented-pool", CONTROLLER_ID)
TTF_CLOCK_SCHEMA = "lns2.ttf.reset_inclusive_wall.v1"
LAYER_NAMES = ("high_load_development", "fresh_map_raw_ttf")


def _load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_maprank_evaluation_config(config)
    return path, root, config


def _registered_tasks(cohort: dict[str, Any], fresh: bool) -> list[str]:
    if not fresh:
        return list(map(str, cohort.get("tasks") or ()))
    map_id = str(cohort["map_id"])
    agents = int(cohort["agent_count"])
    return [
        f"{map_id}__random_{index:02d}__agents_{agents:04d}" for index in (4, 5)
    ]


def _layer_context(
    root: Path, config: dict[str, Any], layer_name: str
) -> tuple[Path, list[dict[str, Any]], tuple[int, ...]]:
    if layer_name not in LAYER_NAMES:
        raise ValueError("MapRank raw-TTF layer identity changed")
    fresh = layer_name == "fresh_map_raw_ttf"
    layer = dict(config[layer_name])
    runtime = _project_path(root, str(layer["runtime_config"]))
    if fresh:
        dataset = _project_path(root, str(layer["dataset"]))
        datasets = {str(row["id"]): dataset for row in layer["cohorts"]}
    else:
        datasets = {
            str(row["id"]): _project_path(root, str(row["dataset"]))
            for row in layer["cohorts"]
        }
    cohorts = []
    for source in layer["cohorts"]:
        cohort = dict(source)
        cohort_id = str(cohort["id"])
        cohort["dataset_path"] = datasets[cohort_id]
        cohort["split"] = str(cohort.get("split", layer.get("split")))
        cohort["tasks"] = _registered_tasks(cohort, fresh)
        indexed = _dataset_tasks(cohort["dataset_path"], cohort["split"])
        if set(cohort["tasks"]) - set(indexed):
            raise ValueError(f"{layer_name}/{cohort_id} dataset tasks differ")
        cohorts.append(cohort)
    return runtime, cohorts, tuple(map(int, layer["solver_seeds"]))


def _schedule(
    cohorts: list[dict[str, Any]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    keys = sorted(
        (str(cohort["id"]), task, seed)
        for cohort in cohorts
        for task in cohort["tasks"]
        for seed in seeds
    )
    schedule = []
    for key_index, (cohort_id, task_id, seed) in enumerate(keys):
        offset = key_index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        for position, controller in enumerate(order):
            schedule.append(
                {
                    "ordinal": len(schedule),
                    "cohort_id": cohort_id,
                    "task_id": task_id,
                    "solver_seed": seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
    return schedule


def _controller_kwargs(
    controller: str, bundles: dict[str, Path], augmentation: dict[str, Any]
) -> dict[str, Any]:
    if controller == "v2-full":
        mode, bundle, topology = "v2-full", bundles["v2-full"], None
    elif controller == "v2-augmented-pool":
        mode, bundle, topology = "v2-full", bundles["v2-full"], augmentation
    elif controller == CONTROLLER_ID:
        mode, bundle, topology = CONTROLLER_ID, bundles[CONTROLLER_ID], augmentation
    else:
        raise ValueError(f"unexpected MapRank raw-TTF controller: {controller}")
    result = {
        "controller": mode,
        "controller_bundle": str(bundle),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "stopping_rule": "run-to-completion",
    }
    if topology is not None:
        result["topology_boundary_augmentation"] = topology
    return result


def _prerequisites(root: Path, config: dict[str, Any], output: Path, layer: str) -> None:
    _training_evidence(root, config)
    evaluation_root = output.parent
    shadow = _read_json(
        evaluation_root / "shadow-fresh-v1" / "maprank_shadow_report.json"
    )
    if shadow.get("passed") is not True:
        raise ValueError("MapRank Shadow gate did not pass")
    if layer == "fresh_map_raw_ttf":
        development = _read_json(
            evaluation_root / "high-load-v1" / "maprank_raw_ttf_report.json"
        )
        if development.get("performance_passed") is not True:
            raise ValueError(
                "MapRank high-load gate did not pass; fresh maps remain unread"
            )


def prepare_maprank_fresh_dataset(config_path: str | Path) -> dict[str, Any]:
    _path, root, config = _load_config(config_path)
    layer = dict(config["fresh_map_raw_ttf"])
    return prepare_movingai_dataset(
        _project_path(root, str(layer["fetched_dataset"])),
        _project_path(root, str(layer["dataset_config"])),
        _project_path(root, str(layer["dataset"])),
    )


def run_maprank_raw_ttf_layer(
    config_path: str | Path,
    output: str | Path,
    *,
    layer_name: str,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    output = Path(output).resolve()
    if layer_name == "fresh_map_raw_ttf" and not dry_run:
        _prerequisites(root, config, output, layer_name)
        prepare_maprank_fresh_dataset(path)
    runtime, cohorts, seeds = _layer_context(root, config, layer_name)
    schedule = _schedule(cohorts, seeds)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "layer": layer_name,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    _prerequisites(root, config, output, layer_name)
    bundles = _bundle_paths(root, config)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "evaluation_status.json"
    status_base = {
        "schema": STATUS_SCHEMA,
        "layer": layer_name,
        "config_sha256": sha256_file(path),
        "schedule_sha256": _fingerprint(schedule),
        "total_schedule_entries": len(schedule),
    }
    if status_path.is_file() and not resume:
        raise ValueError("MapRank raw-TTF output exists; pass resume")
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    _write_json(
        status_path,
        {**status_base, "completed_schedule_entries": 0, "complete": False},
    )
    by_cohort = {str(row["id"]): row for row in cohorts}
    augmentation = dict(config["topology_boundary_augmentation"])
    for cohort in cohorts:
        cohort_id = str(cohort["id"])
        keys = {(task, seed) for task in cohort["tasks"] for seed in seeds}
        qualification = output / "cohorts" / cohort_id / "qualification"
        run_closed_loop_collection(
            cohort["dataset_path"],
            runtime,
            qualification,
            phase="qualify",
            workers=1,
            resume=qualification.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs("v2-full", bundles, augmentation),
        )
        for controller in CONTROLLERS:
            controller_root = (
                output / "cohorts" / cohort_id / "controllers" / controller
            )
            run_closed_loop_collection(
                cohort["dataset_path"],
                runtime,
                controller_root,
                phase="qualify",
                workers=1,
                resume=controller_root.joinpath("run_config.json").is_file(),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification,
                **_controller_kwargs(controller, bundles, augmentation),
            )
    completed = 0
    for item in schedule:
        cohort = by_cohort[str(item["cohort_id"])]
        keys = {(task, seed) for task in cohort["tasks"] for seed in seeds}
        controller = str(item["controller"])
        controller_root = (
            output
            / "cohorts"
            / str(item["cohort_id"])
            / "controllers"
            / controller
        )
        manifest = controller_root / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                cohort["dataset_path"],
                runtime,
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=keys,
                job_keys={key},
                **_controller_kwargs(controller, bundles, augmentation),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "current": item,
                "complete": False,
            },
        )
    report = analyze_maprank_raw_ttf_layer(path, output, layer_name=layer_name)
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / "maprank_raw_ttf_report.json"),
        },
    )
    return report


def _metric(source: dict[str, Any], name: str) -> float:
    value = source.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [
        dict(row.get("summary") or {}) for row in rows if row.get("status") == "ok"
    ]
    successes = [row for row in summaries if bool(row.get("success"))]
    totals = [dict(row.get("controller_totals") or {}) for row in summaries]
    ttf = [_metric(row, "wall_time_to_feasible") for row in successes]
    return {
        "episode_count": len(rows),
        "execution_error_count": len(rows) - len(summaries),
        "success_count": len(successes),
        "mean_raw_wall_time_to_feasible": _mean(ttf),
        "median_raw_wall_time_to_feasible": statistics.median(ttf) if ttf else 0.0,
        "mean_repair_iterations": _mean(
            [float(row.get("repair_iterations", 0)) for row in summaries]
        ),
        "mean_normalized_wall_clock_conflict_auc": _mean(
            [_metric(row, "normalized_wall_clock_conflict_auc") for row in summaries]
        ),
        "mean_pp_replan_seconds": _mean(
            [_metric(row, "pp_replan_seconds") for row in totals]
        ),
        "mean_controller_seconds_before_repair": _mean(
            [_metric(row, "controller_seconds_before_repair") for row in totals]
        ),
        "mean_neighborhood_selection_seconds": _mean(
            [_metric(row, "neighborhood_selection_seconds") for row in totals]
        ),
        "invalid_action_count": sum(
            int(row.get("invalid_action_count", 0)) for row in summaries
        ),
        "fingerprint_mismatch_count": sum(
            int(row.get("fingerprint_mismatch_count", 0)) for row in summaries
        ),
    }


def _paired_comparison(
    baseline: dict[tuple[str, str, int], dict[str, Any]],
    challenger: dict[tuple[str, str, int], dict[str, Any]],
    keys: list[tuple[str, str, int]],
) -> dict[str, Any]:
    pairs = []
    for key in keys:
        left = baseline.get(key)
        right = challenger.get(key)
        if (
            left is None
            or right is None
            or left.get("status") != "ok"
            or right.get("status") != "ok"
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        left_summary = dict(left.get("summary") or {})
        right_summary = dict(right.get("summary") or {})
        if not bool(left_summary.get("success")) or not bool(
            right_summary.get("success")
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        pairs.append((key, left_summary, right_summary))
    left_ttf = [_metric(left, "wall_time_to_feasible") for _key, left, _ in pairs]
    right_ttf = [_metric(right, "wall_time_to_feasible") for _key, _, right in pairs]
    baseline_mean = _mean(left_ttf)
    challenger_mean = _mean(right_ttf)
    deltas = [right - left for left, right in zip(left_ttf, right_ttf)]
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_mean_raw_ttf": baseline_mean,
        "challenger_mean_raw_ttf": challenger_mean,
        "mean_raw_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_raw_ttf_relative_improvement": (
            (baseline_mean - challenger_mean) / baseline_mean
            if baseline_mean
            else 0.0
        ),
        "faster_count": sum(value < -1e-9 for value in deltas),
        "slower_count": sum(value > 1e-9 for value in deltas),
        "tied_count": sum(abs(value) <= 1e-9 for value in deltas),
        "paired_faster_fraction": (
            sum(value < -1e-9 for value in deltas) / len(deltas) if deltas else 0.0
        ),
        "mean_repair_iterations_delta": _mean(
            [
                float(right.get("repair_iterations", 0))
                - float(left.get("repair_iterations", 0))
                for _key, left, right in pairs
            ]
        ),
        "mean_normalized_wall_auc_delta": _mean(
            [
                _metric(right, "normalized_wall_clock_conflict_auc")
                - _metric(left, "normalized_wall_clock_conflict_auc")
                for _key, left, right in pairs
            ]
        ),
    }


def analyze_maprank_raw_ttf_layer(
    config_path: str | Path, output: str | Path, *, layer_name: str
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    output = Path(output).resolve()
    _runtime, cohorts, seeds = _layer_context(root, config, layer_name)
    expected = {
        (str(cohort["id"]), task, seed)
        for cohort in cohorts
        for task in cohort["tasks"]
        for seed in seeds
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    manifest_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    errors = []
    for controller in CONTROLLERS:
        rows_by_key = {}
        for cohort in cohorts:
            cohort_id = str(cohort["id"])
            manifest = (
                output
                / "cohorts"
                / cohort_id
                / "controllers"
                / controller
                / "realized_dynamic_manifest.jsonl"
            )
            rows = _read_jsonl(manifest)
            manifest_hashes[controller][cohort_id] = sha256_file(manifest)
            for row in rows:
                key = (cohort_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in rows_by_key:
                    errors.append(f"{controller}: duplicate episode {key}")
                rows_by_key[key] = row
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = 0
    conflict_mismatches = 0
    bad_clock = 0
    capped_values = 0
    for key in sorted(expected):
        present = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in present):
            continue
        summaries_for_key = [dict(row["summary"]) for row in present]
        fingerprint_mismatches += (
            len({str(row.get("initial_fingerprint")) for row in summaries_for_key})
            != 1
        )
        conflict_mismatches += (
            len({int(row.get("initial_conflicts", -1)) for row in summaries_for_key})
            != 1
        )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
            for row in summaries_for_key
        )
        capped_values += sum(
            row.get("capped_wall_time_to_feasible") is not None
            for row in summaries_for_key
        )
    summaries = {
        name: _controller_summary(list(indexed[name].values()))
        for name in CONTROLLERS
    }
    all_keys = sorted(expected)
    total = _paired_comparison(indexed["v2-full"], indexed[CONTROLLER_ID], all_keys)
    pool = _paired_comparison(
        indexed["v2-full"], indexed["v2-augmented-pool"], all_keys
    )
    ranker = _paired_comparison(
        indexed["v2-augmented-pool"], indexed[CONTROLLER_ID], all_keys
    )
    per_cohort = {}
    for cohort in cohorts:
        cohort_id = str(cohort["id"])
        keys = [key for key in all_keys if key[0] == cohort_id]
        per_cohort[cohort_id] = {
            "total_vs_v2_full": _paired_comparison(
                indexed["v2-full"], indexed[CONTROLLER_ID], keys
            ),
            "pool_effect_vs_v2_full": _paired_comparison(
                indexed["v2-full"], indexed["v2-augmented-pool"], keys
            ),
            "ranker_effect_vs_v2_augmented": _paired_comparison(
                indexed["v2-augmented-pool"], indexed[CONTROLLER_ID], keys
            ),
        }
    integrity = {
        "complete_paired_coverage": not any("coverage" in value for value in errors),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for values in indexed.values()
            for row in values.values()
        ),
        "all_controllers_succeeded": all(
            summary["success_count"] == len(expected) for summary in summaries.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0
            for summary in summaries.values()
        ),
    }
    gates = dict(config[layer_name]["gates"])
    maximum_regression = max(
        (
            -float(
                row["total_vs_v2_full"].get(
                    "mean_raw_ttf_relative_improvement", 0.0
                )
            )
            for row in per_cohort.values()
        ),
        default=0.0,
    )
    performance = {
        "total_raw_ttf_improvement": float(
            total.get("mean_raw_ttf_relative_improvement", -1.0)
        )
        >= float(gates["minimum_raw_ttf_improvement_vs_v2_full"]),
        "ranker_raw_ttf_improvement": float(
            ranker.get("mean_raw_ttf_relative_improvement", -1.0)
        )
        >= float(gates["minimum_ranker_raw_ttf_improvement_vs_v2_augmented"]),
        "maximum_group_regression": maximum_regression
        <= float(
            gates.get(
                "maximum_cohort_raw_ttf_regression",
                gates.get("maximum_map_raw_ttf_regression"),
            )
        ),
        "repair_iterations_noninferior": float(
            total.get("mean_repair_iterations_delta", 1.0)
        )
        <= 0.0,
        "success_count_noninferior": summaries[CONTROLLER_ID]["success_count"]
        >= summaries["v2-full"]["success_count"],
    }
    if layer_name == "fresh_map_raw_ttf":
        performance["paired_faster_fraction"] = float(
            total.get("paired_faster_fraction", 0.0)
        ) >= float(gates["minimum_paired_faster_fraction"])
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "layer": layer_name,
        "scientific_status": layer_name,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "formal_speed_claim_supported": (
            layer_name == "fresh_map_raw_ttf" and performance_passed
        ),
        "primary_metric": "mean_raw_wall_time_to_feasible",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparisons": {
            "total_method_vs_v2_full": total,
            "pool_effect_vs_v2_full": pool,
            "ranker_effect_vs_v2_augmented": ranker,
        },
        "per_cohort": per_cohort,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "initial_conflict_mismatch_count": conflict_mismatches,
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(manifest_hashes),
        },
    }
    _write_json(output / "maprank_raw_ttf_report.json", report)
    return report


__all__ = [
    "CONTROLLERS",
    "analyze_maprank_raw_ttf_layer",
    "prepare_maprank_fresh_dataset",
    "run_maprank_raw_ttf_layer",
]
