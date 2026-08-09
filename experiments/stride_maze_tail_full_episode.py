from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_guardpool_maze_regression import _controller_kwargs
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA, _quick_controller_summary
from lns2_selector.runtime.online_selection import (
    slotpool_runtime_augmentation,
    validate_structpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.maze_tail_full_episode_config.v1"
STATUS_SCHEMA = "lns2.stride.maze_tail_full_episode_status.v1"
REPORT_SCHEMA = "lns2.stride.maze_tail_full_episode_report.v1"
CONTROLLERS = ("v2-full", "v2-plus-structpool", "v2-plus-slotpool")
CHALLENGERS = CONTROLLERS[1:]
STATUS_FILENAME = "full_episode_status.json"
REPORT_FILENAME = "maze_tail_full_episode_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="Maze tail full episode")


def _expected_keys(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }


def _task_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): dict(row) for row in rows}


def load_maze_tail_full_episode_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_after_outcome_blind_reset_selection_before_any_controller_episode"
        or config.get("experiment_id") != "stride-maze-tail-full-episode-v1"
        or config.get("pre_registration_parent_commit")
        != "9dbcab7df4c6ccbb8ee77d921435ea117d2ada33"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Maze tail full-episode identity changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_over_original_candidate_pool",
        "challengers": [
            "frozen_v2_over_original_plus_full_structpool",
            "frozen_v2_over_original_plus_frozen_slotpool",
        ],
        "execution_order": "strict_three_controller_rotation",
        "paired_solver_seed_required": True,
        "workers": 1,
    }:
        raise ValueError("Maze tail full-episode comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_maze_tail_evidence_preflight_runtime_v1.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
        "workers": 1,
    }:
        raise ValueError("Maze tail full-episode runtime changed")
    full = validate_structpool_augmentation(dict(config["full_structpool_augmentation"]))
    slot = slotpool_runtime_augmentation()
    if (
        full is None
        or full.get("pool_id") != "stride-structpool-v1"
        or slot.get("pool_id") != "stride-slotpool-v1"
    ):
        raise ValueError("Maze tail full-episode treatments changed")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "outcome_blind_frozen_independent_maze_tail_cohort"
        or cohort.get("dataset")
        != "build/stride-maze-tail-evidence-candidate-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (17, 29, 43)
        or int(cohort.get("paired_key_count", -1)) != 33
        or int(cohort.get("episode_count_per_controller", -1)) != 33
        or [str(row.get("id")) for row in groups]
        != ["maze-128-128-1", "maze-128-128-2", "maze-32-32-4"]
        or [len(list(row.get("tasks") or ())) for row in groups] != [4, 3, 4]
        or list(cohort.get("result_based_exclusions") or ())
        or len(_expected_keys(config)) != 33
    ):
        raise ValueError("Maze tail full-episode cohort changed")
    if dict(config.get("tail_definition") or {}) != {
        "adverse_minimum_repair_iteration_delta": 1,
        "severe_minimum_repair_iteration_delta": 10,
        "severe_minimum_repair_iteration_ratio": 2.0,
    }:
        raise ValueError("Maze tail definition changed")
    if dict(config.get("tail_evidence_gates") or {}) != {
        "minimum_adverse_or_severe_comparisons": 12,
        "minimum_severe_tail_comparisons": 4,
        "minimum_adverse_map_count": 2,
        "minimum_adverse_task_count": 6,
        "minimum_adverse_solver_seed_count": 2,
        "minimum_adverse_comparisons_per_challenger": 3,
    }:
        raise ValueError("Maze tail evidence gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "tail_incidence_evidence_only": True,
        "formal_speed_claim": False,
        "fresh_map_generalization_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "no_result_based_exclusions": True,
        "speed_metrics_descriptive_only": True,
    }:
        raise ValueError("Maze tail claim boundary changed")
    expected_inputs = {
        "dataset_summary",
        "dataset_manifest",
        "runtime_config",
        "controller_manifest",
        "slotpool_model",
        "qualification_manifest",
        "qualification_report",
        "qualification_run_config",
        "selected_cohort",
        "preflight_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("Maze tail input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    if str(slot["slotpool_model"]["sha256"]) != str(
        config["inputs"]["slotpool_model"]["sha256"]
    ):
        raise ValueError("Maze tail SlotPool model does not match registration")
    preflight = _read_json(inputs["preflight_report"])
    if (
        preflight.get("passed") is not True
        or preflight.get("selected_solver_key_count") != 33
        or preflight.get("training_allowed") is not False
        or preflight.get("controller_outcomes_read") is not False
        or preflight.get("ttf_outcomes_read") is not False
    ):
        raise ValueError("Maze tail outcome-blind preflight changed")
    selected_rows = _read_jsonl(inputs["selected_cohort"])
    configured_tasks = {
        str(task) for group in groups for task in group.get("tasks", [])
    }
    selected_tasks = {str(row["task_id"]) for row in selected_rows}
    if configured_tasks != selected_tasks or len(selected_rows) != 11:
        raise ValueError("Maze tail selected cohort does not match registration")
    group_by_task = {
        str(task): str(group["id"])
        for group in groups
        for task in group.get("tasks", [])
    }
    if any(group_by_task[str(row["task_id"])] != str(row["map_id"]) for row in selected_rows):
        raise ValueError("Maze tail selected task map changed")
    manifest_tasks = {str(row["task_id"]) for row in _read_jsonl(inputs["dataset_manifest"])}
    if configured_tasks - manifest_tasks:
        raise ValueError("Maze tail task is absent from the dataset")
    qualification_rows = _read_jsonl(inputs["qualification_manifest"])
    qualification_keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in qualification_rows
        if str(row["task_id"]) in configured_tasks and row.get("status") == "ok"
    }
    if len(qualification_keys) != 33:
        raise ValueError("Maze tail qualification source is incomplete")
    return path, root, config


def maze_tail_full_episode_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rotations = (
        CONTROLLERS,
        CONTROLLERS[1:] + CONTROLLERS[:1],
        CONTROLLERS[2:] + CONTROLLERS[:2],
    )
    for index, (group_id, task_id, solver_seed) in enumerate(
        sorted(_expected_keys(config))
    ):
        for position, controller in enumerate(rotations[index % 3]):
            rows.append(
                {
                    "group_id": group_id,
                    "task_id": task_id,
                    "solver_seed": solver_seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
    return rows


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_maze_tail_full_episode.py",
            "experiments/stride_guardpool_maze_regression.py",
            "experiments/stride_maprank_raw_ttf.py",
            "experiments/stride_structpool_ttf_quick.py",
        ),
        native_required=native_required,
    )


def _completed_counts(output: Path) -> dict[str, int]:
    result = {}
    for controller in CONTROLLERS:
        manifest = output / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest) if manifest.is_file() else []
        result[controller] = sum(row.get("status") in {"ok", "error"} for row in rows)
    return result


def run_maze_tail_full_episode(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_maze_tail_full_episode_config(config_path)
    schedule = maze_tail_full_episode_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": len(schedule) // 3,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    output = Path(output).resolve()
    status_path = output / STATUS_FILENAME
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=_producer(root),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="Maze tail full episode",
    )
    status_base = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    dataset = (root / str(config["cohort"]["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    expected = _expected_keys(config)
    job_keys = {(task, seed) for _group, task, seed in expected}
    qualification_source = (
        root / str(config["inputs"]["qualification_manifest"]["path"])
    ).resolve().parent
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            workers=1,
            resume=(prepared.resumed and collection.joinpath("run_config.json").is_file()),
            cohort_job_keys=job_keys,
            job_keys=job_keys,
            qualification_source=qualification_source,
            **_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=job_keys,
                job_keys={key},
                **_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "controller_completed_episode_counts": _completed_counts(output),
                "current": item,
                "complete": False,
            },
        )
    report = analyze_maze_tail_full_episode(
        path, output, producer=status_base["producer_identity"]
    )
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "controller_completed_episode_counts": _completed_counts(output),
            "complete": True,
            "report_sha256": sha256_file(output / REPORT_FILENAME),
        },
    )
    return report


def _pool_counts(row: dict[str, Any]) -> dict[str, int]:
    summary = dict(row.get("summary") or {})
    totals = dict(summary.get("controller_totals") or {})
    selected_families = dict(summary.get("selected_family_counts") or {})
    return {
        "gate_passed": int(totals.get("structpool_gate_passed_count", 0)),
        "generated": int(totals.get("structpool_generated_count", 0)),
        "structural_selected": sum(
            int(count)
            for family, count in selected_families.items()
            if str(family).startswith("structpool-")
        ),
        "slotpool_reduced": int(totals.get("slotpool_reduction_applied_count", 0)),
        "slotpool_selected": int(
            totals.get("slotpool_selected_structural_candidate_count", 0)
        ),
    }


def _tail_rows(
    config: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    keys: list[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    definition = dict(config["tail_definition"])
    rows: list[dict[str, Any]] = []
    for challenger in CHALLENGERS:
        for key in keys:
            baseline = dict(indexed["v2-full"][key].get("summary") or {})
            treatment = dict(indexed[challenger][key].get("summary") or {})
            left_iterations = int(baseline.get("repair_iterations", 0))
            right_iterations = int(treatment.get("repair_iterations", 0))
            delta = right_iterations - left_iterations
            ratio = right_iterations / max(left_iterations, 1)
            if (
                delta >= int(definition["severe_minimum_repair_iteration_delta"])
                and ratio >= float(definition["severe_minimum_repair_iteration_ratio"])
            ):
                category = "severe"
            elif delta >= int(definition["adverse_minimum_repair_iteration_delta"]):
                category = "adverse"
            else:
                category = "beneficial_or_tied"
            task = key[1]
            meta = metadata[task]
            rows.append(
                {
                    "challenger": challenger,
                    "map_id": key[0],
                    "task_id": task,
                    "solver_seed": key[2],
                    "task_variant_family": str(meta["task_variant_family"]),
                    "conflict_band": str(meta["conflict_band"]),
                    "agent_count": int(meta["agent_count"]),
                    "initial_conflicts": int(baseline.get("initial_conflicts", -1)),
                    "v2_repair_iterations": left_iterations,
                    "challenger_repair_iterations": right_iterations,
                    "repair_iteration_delta": delta,
                    "repair_iteration_ratio": ratio,
                    "v2_raw_ttf": float(baseline.get("wall_time_to_feasible", 0.0)),
                    "challenger_raw_ttf": float(
                        treatment.get("wall_time_to_feasible", 0.0)
                    ),
                    "tail_category": category,
                }
            )
    return rows


def analyze_maze_tail_full_episode(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_maze_tail_full_episode_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    producer = producer or _producer(root, native_required=False)
    expected = _expected_keys(config)
    selected_path = _registered(root, dict(config["inputs"]["selected_cohort"]))
    metadata = _task_metadata(_read_jsonl(selected_path))
    qualification_path = _registered(
        root, dict(config["inputs"]["qualification_manifest"])
    )
    qualification = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(qualification_path)
        if str(row["task_id"]) in metadata
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    errors: list[str] = []
    counts = {controller: defaultdict(int) for controller in CONTROLLERS}
    runtime_configs: dict[str, dict[str, Any]] = {}
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest)
        hashes[controller] = sha256_file(manifest)
        runtime_configs[controller] = _read_json(collection / "run_config.json")
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            task = str(row["task_id"])
            seed = int(row["solver_seed"])
            meta = metadata.get(task)
            if meta is None:
                errors.append(f"{controller}: unexpected episode {(task, seed)}")
                continue
            key = (str(meta["map_id"]), task, seed)
            if key not in expected:
                errors.append(f"{controller}: unexpected episode {key}")
                continue
            if key in by_key:
                errors.append(f"{controller}: duplicate episode {key}")
            by_key[key] = row
            for name, value in _pool_counts(row).items():
                counts[controller][name] += value
        if set(by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = by_key

    fingerprint_mismatches = conflict_mismatches = qualification_mismatches = 0
    bad_clock = capped = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row.get("summary") or {}) for row in rows]
        fingerprint_mismatches += len(
            {str(row.get("initial_fingerprint")) for row in summaries}
        ) != 1
        conflict_mismatches += len(
            {int(row.get("initial_conflicts", -1)) for row in summaries}
        ) != 1
        qualified = qualification.get((key[1], key[2]), {})
        qualification_mismatches += any(
            str(row.get("initial_fingerprint")) != str(qualified.get("state_fingerprint"))
            or int(row.get("initial_conflicts", -1))
            != int(qualified.get("initial_conflicts", -2))
            for row in summaries
        )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries
        )
        capped += sum(
            row.get("capped_wall_time_to_feasible") is not None for row in summaries
        )
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    keys = sorted(expected)
    comparisons = {
        challenger: _paired_comparison(indexed["v2-full"], indexed[challenger], keys)
        for challenger in CHALLENGERS
    }
    per_map = {
        map_id: {
            challenger: _paired_comparison(
                indexed["v2-full"],
                indexed[challenger],
                [key for key in keys if key[0] == map_id],
            )
            for challenger in CHALLENGERS
        }
        for map_id in sorted({key[0] for key in keys})
    }
    per_band = {
        band: {
            challenger: _paired_comparison(
                indexed["v2-full"],
                indexed[challenger],
                [key for key in keys if metadata[key[1]]["conflict_band"] == band],
            )
            for challenger in CHALLENGERS
        }
        for band in ("moderate", "high")
    }
    per_variant = {
        variant: {
            challenger: _paired_comparison(
                indexed["v2-full"],
                indexed[challenger],
                [
                    key
                    for key in keys
                    if metadata[key[1]]["task_variant_family"] == variant
                ],
            )
            for challenger in CHALLENGERS
        }
        for variant in ("opposite_exchange", "uniform_random")
    }
    tail_rows = _tail_rows(config, metadata, indexed, keys) if not errors else []
    adverse_rows = [
        row for row in tail_rows if row["tail_category"] in {"adverse", "severe"}
    ]
    severe_rows = [row for row in tail_rows if row["tail_category"] == "severe"]
    adverse_per_challenger = {
        challenger: sum(row["challenger"] == challenger for row in adverse_rows)
        for challenger in CHALLENGERS
    }
    evidence_thresholds = dict(config["tail_evidence_gates"])
    evidence_gates = {
        "minimum_adverse_or_severe_comparisons": len(adverse_rows)
        >= int(evidence_thresholds["minimum_adverse_or_severe_comparisons"]),
        "minimum_severe_tail_comparisons": len(severe_rows)
        >= int(evidence_thresholds["minimum_severe_tail_comparisons"]),
        "minimum_adverse_map_count": len({row["map_id"] for row in adverse_rows})
        >= int(evidence_thresholds["minimum_adverse_map_count"]),
        "minimum_adverse_task_count": len({row["task_id"] for row in adverse_rows})
        >= int(evidence_thresholds["minimum_adverse_task_count"]),
        "minimum_adverse_solver_seed_count": len(
            {row["solver_seed"] for row in adverse_rows}
        )
        >= int(evidence_thresholds["minimum_adverse_solver_seed_count"]),
        "minimum_adverse_comparisons_per_challenger": all(
            count
            >= int(evidence_thresholds["minimum_adverse_comparisons_per_challenger"])
            for count in adverse_per_challenger.values()
        ),
    }
    deterministic_runtime = all(
        dict(value.get("configuration") or {}).get("deterministic_pp_replay") is True
        and dict(value.get("configuration") or {}).get("stopping_rule")
        == "run-to-completion"
        and dict(dict(value.get("configuration") or {}).get("environment") or {}).get(
            "time_limit"
        )
        == 0.0
        and dict(value.get("configuration") or {}).get("wall_time_budget_seconds")
        is None
        for value in runtime_configs.values()
    )
    integrity = {
        "complete_three_controller_coverage": not any(
            "coverage" in error for error in errors
        ),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for controller_rows in indexed.values()
            for row in controller_rows.values()
        ),
        "all_controllers_succeeded": all(
            summary["success_count"] == len(expected) for summary in summaries.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "registered_qualification_states_reused": qualification_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped == 0,
        "run_to_completion_deterministic_runtime": deterministic_runtime,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0
            for summary in summaries.values()
        ),
        "full_structpool_activated": counts["v2-plus-structpool"]["gate_passed"] > 0,
        "full_structpool_selected": counts["v2-plus-structpool"]["structural_selected"]
        > 0,
        "slotpool_activated": counts["v2-plus-slotpool"]["gate_passed"] > 0,
        "slotpool_selected": counts["v2-plus-slotpool"]["structural_selected"] > 0,
        "slotpool_reduction_exercised": counts["v2-plus-slotpool"]["slotpool_reduced"]
        > 0,
    }
    integrity_passed = not errors and all(integrity.values())
    evidence_passed = integrity_passed and all(evidence_gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "independent_maze_tail_incidence_evidence_complete",
        "producer_identity": producer,
        "primary_purpose": "harmful_tail_incidence_and_first_divergence_coverage",
        "primary_speed_claim": False,
        "formal_speed_claim": False,
        "fresh_map_generalization_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "speed_metrics_descriptive_only": True,
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "paired_key_count": len(expected),
        "episode_count_per_controller": len(expected),
        "comparison_count": len(tail_rows),
        "controller_summaries": summaries,
        "v2_paired_comparisons": comparisons,
        "per_map": per_map,
        "per_conflict_band": per_band,
        "per_task_variant": per_variant,
        "tail_counts": {
            "adverse_or_severe": len(adverse_rows),
            "severe": len(severe_rows),
            "adverse_map_count": len({row["map_id"] for row in adverse_rows}),
            "adverse_task_count": len({row["task_id"] for row in adverse_rows}),
            "adverse_solver_seed_count": len(
                {row["solver_seed"] for row in adverse_rows}
            ),
            "adverse_per_challenger": adverse_per_challenger,
        },
        "tail_comparisons": tail_rows,
        "runtime_counts": {controller: dict(value) for controller, value in counts.items()},
        "integrity_gates": integrity,
        "tail_evidence_gates": evidence_gates,
        "integrity_passed": integrity_passed,
        "tail_evidence_passed": evidence_passed,
        "next_step": (
            config["next_step_on_tail_evidence_pass"]
            if evidence_passed
            else config["next_step_on_tail_evidence_failure"]
        ),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": hashes,
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "CONTROLLERS",
    "REPORT_FILENAME",
    "STATUS_FILENAME",
    "analyze_maze_tail_full_episode",
    "load_maze_tail_full_episode_config",
    "maze_tail_full_episode_schedule",
    "run_maze_tail_full_episode",
]
