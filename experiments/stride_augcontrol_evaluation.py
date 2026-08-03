from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.balanced_wall_clock import prepare_movingai_dataset
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_repairability_collection import BOUNDARY_AUGMENTATION
from experiments.stride_robuststep_preflight import _mean
from experiments.stride_stage3 import _project_path


CONFIG_SCHEMA = "lns2.stride.augcontrol_evaluation_config.v1"
REPORT_SCHEMA = "lns2.stride.augcontrol_evaluation_report.v1"
SHADOW_REPORT_SCHEMA = "lns2.stride.augcontrol_shadow_report.v1"
STATUS_SCHEMA = "lns2.stride.augcontrol_evaluation_status.v1"
CONTROLLER_ID = "stride-augcontrol-v1"
CONTROLLERS = ("v2-full", "v2-augmented-pool", CONTROLLER_ID)
TTF_CLOCK_SCHEMA = "lns2.ttf.reset_inclusive_wall.v1"


def _registered_tasks(cohort: dict[str, Any], formal: bool) -> list[str]:
    if not formal:
        return list(map(str, cohort.get("tasks") or ()))
    map_id = str(cohort["map_id"])
    agents = int(cohort["agent_count"])
    return [
        f"{map_id}__random_{index:02d}__agents_{agents:04d}"
        for index in (4, 5)
    ]


def validate_augcontrol_evaluation_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE augcontrol evaluation config")
    if (
        config.get("scientific_status")
        != "preregistered_before_augcontrol_training_outcomes"
        or config.get("experiment_id") != "stride-augcontrol-evaluation-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
        or config.get("primary_metric") != "mean_raw_wall_time_to_feasible"
        or config.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
        or config.get("stopping_rule") != "run-to-completion"
        or config.get("scientific_time_limit_seconds") is not None
        or config.get("environment_time_limit_seconds") is not None
        or config.get("episode_process_timeout_seconds") is not None
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or config.get("execution_order") != "strict_rotating_triplet_order"
        or int(config.get("workers", 0)) != 1
        or not bool(config.get("deterministic_pp_replay_required"))
    ):
        raise ValueError("STRIDE augcontrol evaluation identity changed")
    if dict(config.get("topology_boundary_augmentation") or {}) != BOUNDARY_AUGMENTATION:
        raise ValueError("augcontrol runtime candidate pool differs from training")
    bundles = dict(config.get("controller_bundles") or {})
    if set(bundles) != {"v2-full", CONTROLLER_ID}:
        raise ValueError("augcontrol evaluation bundles changed")
    shadow = dict(config.get("shadow") or {})
    if (
        len(set(map(str, shadow.get("validation_maps") or ()))) != 6
        or tuple(map(int, shadow.get("solver_seeds") or ())) != (1, 2)
        or shadow.get("active_comparator") != "v2-augmented-pool"
        or int(shadow.get("minimum_decisions", 0)) < 1
    ):
        raise ValueError("augcontrol Shadow registration changed")
    expected = {
        "development": ((1, 2, 3, 4), 2, False),
        "formal_ood": ((1, 2, 3), 6, True),
    }
    for name, (seeds, count, formal) in expected.items():
        layer = dict(config.get(name) or {})
        if tuple(map(int, layer.get("solver_seeds") or ())) != seeds:
            raise ValueError(f"{name} solver seeds changed")
        cohorts = list(layer.get("cohorts") or ())
        if len(cohorts) != count or len({str(row.get("id")) for row in cohorts}) != count:
            raise ValueError(f"{name} cohorts changed")
        tasks = [task for row in cohorts for task in _registered_tasks(row, formal)]
        if len(tasks) != len(set(tasks)) or len(tasks) != count * 2:
            raise ValueError(f"{name} task registration changed")
    required = set(map(str, config.get("required_metrics") or ()))
    if required != {
        "success_count",
        "raw_wall_time_to_feasible",
        "repair_iterations",
        "normalized_wall_clock_conflict_auc",
        "pp_replan_seconds",
        "controller_seconds_before_repair",
        "neighborhood_selection_seconds",
        "invalid_action_count",
        "fingerprint_mismatch_count",
    }:
        raise ValueError("augcontrol required metrics changed")


def _load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_augcontrol_evaluation_config(config)
    return path, root, config


def _training_evidence(root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _project_path(root, str(config["training_report"]))
    report = _read_json(path)
    if (
        report.get("controller_id") != CONTROLLER_ID
        or not bool(report.get("offline_passed"))
        or not bool(report.get("shadow_eligible"))
        or bool(report.get("formal_ood_data_read"))
        or bool(report.get("test_data_read"))
    ):
        raise ValueError("augcontrol model did not pass the preregistered offline gate")
    return path, report


def _bundle_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    bundles = dict(config["controller_bundles"])
    paths = {
        "v2-full": _project_path(root, str(bundles["v2-full"])),
        CONTROLLER_ID: _project_path(root, str(bundles[CONTROLLER_ID])),
    }
    for controller, path in paths.items():
        manifest = _read_json(path / "controller_manifest.json")
        expected = controller if controller != "v2-full" else "v2-full"
        actual = str(manifest.get("controller_id", manifest.get("default_controller", "")))
        if actual != expected:
            raise ValueError(f"unexpected controller bundle identity: {controller}/{actual}")
    return paths


def _controller_kwargs(
    controller: str, bundles: dict[str, Path], augmentation: dict[str, Any]
) -> dict[str, Any]:
    if controller == "v2-full":
        mode, bundle, topology = "v2-full", bundles["v2-full"], None
    elif controller == "v2-augmented-pool":
        mode, bundle, topology = "v2-full", bundles["v2-full"], augmentation
    else:
        mode, bundle, topology = CONTROLLER_ID, bundles[CONTROLLER_ID], augmentation
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


def _dataset_tasks(dataset: Path, split: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(dataset / split / "manifest.jsonl")
    return {str(row["task_id"]): row for row in rows}


def run_augcontrol_shadow(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    training_path, _ = _training_evidence(root, config)
    bundles = _bundle_paths(root, config)
    shadow = dict(config["shadow"])
    dataset = _project_path(root, str(shadow["dataset"]))
    runtime = _project_path(root, str(shadow["runtime_config"]))
    qualification = _project_path(root, str(shadow["qualification_source"]))
    rows = _dataset_tasks(dataset, "balanced_wall_clock")
    maps = set(map(str, shadow["validation_maps"]))
    task_ids = sorted(
        task_id for task_id, row in rows.items() if str(row["map_id"]) in maps
    )
    if len(task_ids) != 12 or {str(rows[task]["map_id"]) for task in task_ids} != maps:
        raise ValueError("Shadow validation-map task coverage changed")
    keys = {
        (task_id, seed)
        for task_id in task_ids
        for seed in map(int, shadow["solver_seeds"])
    }
    output = Path(output).resolve()
    run_closed_loop_collection(
        dataset,
        runtime,
        output,
        phase="realized_dynamic",
        workers=1,
        resume=resume,
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        controller="v2-full",
        controller_bundle=str(bundles["v2-full"]),
        diagnostic_shadow_bundles={CONTROLLER_ID: bundles[CONTROLLER_ID]},
        feature_backend="native",
        controller_runtime="optimized",
        verification_profile="deployment",
        topology_boundary_augmentation=dict(config["topology_boundary_augmentation"]),
        stopping_rule="wall-clock",
    )
    return analyze_augcontrol_shadow(path, output, training_path=training_path)


def analyze_augcontrol_shadow(
    config_path: str | Path,
    output: str | Path,
    *,
    training_path: Path | None = None,
) -> dict[str, Any]:
    path, root, config = _load_config(config_path)
    if training_path is None:
        training_path, _ = _training_evidence(root, config)
    output = Path(output).resolve()
    manifest = output / "realized_dynamic_manifest.jsonl"
    rows = _read_jsonl(manifest)
    expected = 12 * len(config["shadow"]["solver_seeds"])
    totals = [dict(dict(row.get("summary") or {}).get("controller_totals") or {}) for row in rows]
    key = f"diagnostic_shadow_decision_count:{CONTROLLER_ID}"
    disagreements = f"diagnostic_shadow_disagreement_count:{CONTROLLER_ID}"
    decisions = sum(int(row.get(key, 0)) for row in totals)
    disagreement_count = sum(int(row.get(disagreements, 0)) for row in totals)
    gates = {
        "complete_episode_coverage": len(rows) == expected,
        "zero_execution_errors": all(row.get("status") == "ok" for row in rows),
        "minimum_shadow_decisions": decisions >= int(config["shadow"]["minimum_decisions"]),
        "zero_action_overrides": sum(int(row.get("diagnostic_shadow_action_override_count", 0)) for row in totals) == 0,
        "zero_semantic_mismatches": sum(int(row.get("diagnostic_shadow_semantic_mismatch_count", 0)) for row in totals) == 0,
        "zero_invalid_actions": sum(int(dict(row.get("summary") or {}).get("invalid_action_count", 0)) for row in rows) == 0,
    }
    report = {
        "schema": SHADOW_REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "active_comparator": "v2-augmented-pool",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "episode_count": len(rows),
        "shadow_decision_count": decisions,
        "shadow_disagreement_count": disagreement_count,
        "shadow_disagreement_fraction": disagreement_count / decisions if decisions else 0.0,
        "gates": gates,
        "passed": all(gates.values()),
        "inputs": {
            "config_sha256": sha256_file(path),
            "training_report_sha256": sha256_file(training_path),
            "manifest_sha256": sha256_file(manifest),
        },
    }
    _write_json(output / "augcontrol_shadow_report.json", report)
    return report


def _layer_context(
    root: Path, config: dict[str, Any], layer_name: str
) -> tuple[Path, list[dict[str, Any]], tuple[int, ...]]:
    formal = layer_name == "formal_ood"
    layer = dict(config[layer_name])
    if formal:
        dataset = _project_path(root, str(layer["dataset"]))
        runtime = _project_path(root, str(layer["runtime_config"]))
        datasets = {str(row["id"]): dataset for row in layer["cohorts"]}
    else:
        runtime = _project_path(root, str(layer["runtime_config"]))
        datasets = {
            str(row["id"]): _project_path(root, str(row["dataset"]))
            for row in layer["cohorts"]
        }
    cohorts: list[dict[str, Any]] = []
    for source in layer["cohorts"]:
        cohort = dict(source)
        cohort_id = str(cohort["id"])
        cohort["dataset_path"] = datasets[cohort_id]
        cohort["split"] = str(cohort.get("split", layer.get("split")))
        cohort["tasks"] = _registered_tasks(cohort, formal)
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


def _layer_prerequisites(root: Path, config: dict[str, Any], output: Path, layer: str) -> None:
    _training_evidence(root, config)
    shadow = _read_json(output.parent / "shadow" / "augcontrol_shadow_report.json")
    if not bool(shadow.get("passed")):
        raise ValueError("augcontrol Shadow gate did not pass")
    if layer == "formal_ood":
        development = _read_json(output.parent / "development" / "augcontrol_evaluation_report.json")
        if not bool(development.get("performance_passed")):
            raise ValueError("development high-load gate did not pass; formal OOD remains unread")


def prepare_augcontrol_formal_ood(config_path: str | Path) -> dict[str, Any]:
    _path, root, config = _load_config(config_path)
    layer = dict(config["formal_ood"])
    return prepare_movingai_dataset(
        _project_path(root, str(layer["fetched_dataset"])),
        _project_path(root, str(layer["dataset_config"])),
        _project_path(root, str(layer["dataset"])),
    )


def run_augcontrol_layer(
    config_path: str | Path,
    output: str | Path,
    *,
    layer_name: str,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if layer_name not in {"development", "formal_ood"}:
        raise ValueError("augcontrol layer must be development or formal_ood")
    path, root, config = _load_config(config_path)
    layer_output = Path(output).resolve()
    _layer_prerequisites(root, config, layer_output, layer_name)
    if layer_name == "formal_ood":
        prepare_augcontrol_formal_ood(path)
    bundles = _bundle_paths(root, config)
    runtime, cohorts, seeds = _layer_context(root, config, layer_name)
    schedule = _schedule(cohorts, seeds)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "layer": layer_name,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    layer_output.mkdir(parents=True, exist_ok=True)
    status_path = layer_output / "evaluation_status.json"
    status_base = {
        "schema": STATUS_SCHEMA,
        "layer": layer_name,
        "config_sha256": sha256_file(path),
        "schedule_sha256": _fingerprint(schedule),
        "total_schedule_entries": len(schedule),
    }
    if status_path.is_file() and not resume:
        raise ValueError("augcontrol evaluation output exists; pass resume")
    _write_jsonl(layer_output / "execution_schedule.jsonl", schedule)
    _write_json(status_path, {**status_base, "completed_schedule_entries": 0, "complete": False})
    by_cohort = {str(row["id"]): row for row in cohorts}
    augmentation = dict(config["topology_boundary_augmentation"])
    for cohort in cohorts:
        cohort_id = str(cohort["id"])
        keys = {(task, seed) for task in cohort["tasks"] for seed in seeds}
        qualification = layer_output / "cohorts" / cohort_id / "qualification"
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
            controller_root = layer_output / "cohorts" / cohort_id / "controllers" / controller
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
        controller_root = layer_output / "cohorts" / str(item["cohort_id"]) / "controllers" / controller
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
            {**status_base, "completed_schedule_entries": completed, "current": item, "complete": False},
        )
    report = analyze_augcontrol_layer(path, layer_output, layer_name=layer_name)
    _write_json(
        status_path,
        {**status_base, "completed_schedule_entries": completed, "complete": True, "report_sha256": sha256_file(layer_output / "augcontrol_evaluation_report.json")},
    )
    return report


def _metric(source: dict[str, Any], name: str) -> float:
    value = source.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(row.get("summary") or {}) for row in rows if row.get("status") == "ok"]
    successes = [row for row in summaries if bool(row.get("success"))]
    totals = [dict(row.get("controller_totals") or {}) for row in summaries]
    return {
        "episode_count": len(rows),
        "execution_error_count": len(rows) - len(summaries),
        "success_count": len(successes),
        "mean_raw_wall_time_to_feasible": _mean([_metric(row, "wall_time_to_feasible") for row in successes]),
        "median_raw_wall_time_to_feasible": statistics.median([_metric(row, "wall_time_to_feasible") for row in successes]) if successes else 0.0,
        "mean_repair_iterations": _mean([float(row.get("repair_iterations", 0)) for row in summaries]),
        "mean_normalized_wall_clock_conflict_auc": _mean([_metric(row, "normalized_wall_clock_conflict_auc") for row in summaries]),
        "mean_pp_replan_seconds": _mean([_metric(row, "pp_replan_seconds") for row in totals]),
        "mean_controller_seconds_before_repair": _mean([_metric(row, "controller_seconds_before_repair") for row in totals]),
        "mean_neighborhood_selection_seconds": _mean([_metric(row, "neighborhood_selection_seconds") for row in totals]),
        "invalid_action_count": sum(int(row.get("invalid_action_count", 0)) for row in summaries),
        "fingerprint_mismatch_count": sum(int(row.get("fingerprint_mismatch_count", 0)) for row in summaries),
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
        if left is None or right is None or left.get("status") != "ok" or right.get("status") != "ok":
            return {"valid": False, "paired_episode_count": len(pairs)}
        left_summary = dict(left.get("summary") or {})
        right_summary = dict(right.get("summary") or {})
        if not bool(left_summary.get("success")) or not bool(right_summary.get("success")):
            return {"valid": False, "paired_episode_count": len(pairs)}
        pairs.append((key, left_summary, right_summary))
    left_ttf = [_metric(left, "wall_time_to_feasible") for _key, left, _right in pairs]
    right_ttf = [_metric(right, "wall_time_to_feasible") for _key, _left, right in pairs]
    baseline_mean = _mean(left_ttf)
    challenger_mean = _mean(right_ttf)
    deltas = [right - left for left, right in zip(left_ttf, right_ttf)]
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_mean_raw_ttf": baseline_mean,
        "challenger_mean_raw_ttf": challenger_mean,
        "mean_raw_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_raw_ttf_relative_improvement": (baseline_mean - challenger_mean) / baseline_mean if baseline_mean else 0.0,
        "faster_count": sum(value < -1e-9 for value in deltas),
        "slower_count": sum(value > 1e-9 for value in deltas),
        "tied_count": sum(abs(value) <= 1e-9 for value in deltas),
        "paired_faster_fraction": sum(value < -1e-9 for value in deltas) / len(deltas) if deltas else 0.0,
        "mean_repair_iterations_delta": _mean([float(right.get("repair_iterations", 0)) - float(left.get("repair_iterations", 0)) for _key, left, right in pairs]),
        "mean_normalized_wall_auc_delta": _mean([_metric(right, "normalized_wall_clock_conflict_auc") - _metric(left, "normalized_wall_clock_conflict_auc") for _key, left, right in pairs]),
        "per_episode": [
            {
                "cohort_id": key[0],
                "task_id": key[1],
                "solver_seed": key[2],
                "baseline_raw_ttf": left,
                "challenger_raw_ttf": right,
                "raw_ttf_delta_seconds": right - left,
            }
            for (key, _left, _right), left, right in zip(pairs, left_ttf, right_ttf)
        ],
    }


def analyze_augcontrol_layer(
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
    errors: list[str] = []
    for controller in CONTROLLERS:
        rows_by_key = {}
        for cohort in cohorts:
            cohort_id = str(cohort["id"])
            manifest = output / "cohorts" / cohort_id / "controllers" / controller / "realized_dynamic_manifest.jsonl"
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
        summaries = [dict(row["summary"]) for row in present]
        fingerprint_mismatches += len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        conflict_mismatches += len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        capped_values += sum(row.get("capped_wall_time_to_feasible") is not None for row in summaries)
    summaries = {
        name: _controller_summary(list(indexed[name].values())) for name in CONTROLLERS
    }
    all_keys = sorted(expected)
    total = _paired_comparison(indexed["v2-full"], indexed[CONTROLLER_ID], all_keys)
    pool = _paired_comparison(indexed["v2-full"], indexed["v2-augmented-pool"], all_keys)
    ranker = _paired_comparison(indexed["v2-augmented-pool"], indexed[CONTROLLER_ID], all_keys)
    per_cohort = {}
    for cohort in cohorts:
        cohort_id = str(cohort["id"])
        keys = [key for key in all_keys if key[0] == cohort_id]
        per_cohort[cohort_id] = {
            "total_vs_v2_full": _paired_comparison(indexed["v2-full"], indexed[CONTROLLER_ID], keys),
            "pool_effect_vs_v2_full": _paired_comparison(indexed["v2-full"], indexed["v2-augmented-pool"], keys),
            "ranker_effect_vs_v2_augmented": _paired_comparison(indexed["v2-augmented-pool"], indexed[CONTROLLER_ID], keys),
        }
    integrity = {
        "complete_paired_coverage": not any("coverage" in value for value in errors),
        "zero_execution_errors": all(row.get("status") == "ok" for values in indexed.values() for row in values.values()),
        "all_controllers_succeeded": all(summary["success_count"] == len(expected) for summary in summaries.values()),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(summary["invalid_action_count"] == 0 for summary in summaries.values()),
        "zero_semantic_mismatches": all(summary["fingerprint_mismatch_count"] == 0 for summary in summaries.values()),
    }
    gates = dict(config[layer_name]["gates"])
    maximum_regression = max(
        (
            -float(row["total_vs_v2_full"].get("mean_raw_ttf_relative_improvement", 0.0))
            for row in per_cohort.values()
        ),
        default=0.0,
    )
    performance = {
        "total_raw_ttf_improvement": float(total.get("mean_raw_ttf_relative_improvement", -1.0)) >= float(gates["minimum_raw_ttf_improvement_vs_v2_full"]),
        "ranker_raw_ttf_improvement": float(ranker.get("mean_raw_ttf_relative_improvement", -1.0)) >= float(gates["minimum_ranker_raw_ttf_improvement_vs_v2_augmented"]),
        "maximum_group_regression": maximum_regression <= float(gates.get("maximum_cohort_raw_ttf_regression", gates.get("maximum_map_raw_ttf_regression"))),
        "repair_iterations_noninferior": float(total.get("mean_repair_iterations_delta", 1.0)) <= 0.0,
        "success_count_noninferior": summaries[CONTROLLER_ID]["success_count"] >= summaries["v2-full"]["success_count"],
    }
    if layer_name == "formal_ood":
        performance["paired_faster_fraction"] = float(total.get("paired_faster_fraction", 0.0)) >= float(gates["minimum_paired_faster_fraction"])
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "layer": layer_name,
        "scientific_status": "formal_ood_evidence" if layer_name == "formal_ood" else "development_high_load",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "formal_speed_claim_supported": layer_name == "formal_ood" and performance_passed,
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
    _write_json(output / "augcontrol_evaluation_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "analyze_augcontrol_layer",
    "analyze_augcontrol_shadow",
    "prepare_augcontrol_formal_ood",
    "run_augcontrol_layer",
    "run_augcontrol_shadow",
    "validate_augcontrol_evaluation_config",
]
