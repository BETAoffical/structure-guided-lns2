from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    registered_input,
    sha256_file,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_guardpool import validate_guardpool_registration
from lns2_selector.runtime.online_selection import (
    guardpool_runtime_augmentation,
    slotpool_runtime_augmentation,
    validate_structpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.guardpool_maze_regression_config.v1"
STATUS_SCHEMA = "lns2.stride.guardpool_maze_regression_status.v1"
REPORT_SCHEMA = "lns2.stride.guardpool_maze_regression_report.v1"
CONTROLLERS = (
    "v2-full",
    "v2-plus-structpool",
    "v2-plus-slotpool",
    "stride-guardpool-v1",
)


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="GuardPool Maze")


def load_guardpool_maze_regression_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_known_regression_after_runtime_semantics_tests"
        or config.get("experiment_id") != "stride-guardpool-maze-regression-v1"
        or config.get("pre_registration_parent_commit")
        != "e371fbb06e06939695be3bf6b84323666bb3984f"
        or config.get("pre_registration_revision_reason")
        != "replace_incompatible_historical_qualification_with_fresh_single_key_reset_before_any_solver_outcome"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("GuardPool Maze regression identity changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_guardpool_maze_regression_runtime_v1.json",
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
        raise ValueError("GuardPool Maze runtime changed")
    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "dataset": "build/stride-structpool-revised-six-map-dataset-v1",
        "split": "balanced_wall_clock",
        "task_id": "maze-128-128-1__derived_opposite_exchange__task_seed_0233__agents_0100",
        "map_id": "maze-128-128-1",
        "solver_seed": 3,
        "agent_count": 100,
        "expected_initial_conflicts": 66,
        "expected_v2_repair_iterations": 15,
    }:
        raise ValueError("GuardPool known Maze cohort changed")
    if dict(config.get("gates") or {}) != {
        "guardpool_success_required": True,
        "guardpool_maximum_repair_iterations": 30,
        "v2_success_required": True,
        "v2_repair_iterations_must_equal": 15,
        "all_initial_conflicts_must_equal": 66,
        "all_initial_fingerprints_must_match": True,
        "maximum_error_count": 0,
        "maximum_invalid_action_count": 0,
        "maximum_fingerprint_mismatch_count": 0,
    }:
        raise ValueError("GuardPool known Maze gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "known_regression_only": True,
        "independent_generalization_evidence": False,
        "formal_ttf_claim": False,
        "parameter_selection_from_this_result": False,
        "default_replacement_allowed": False,
        "slotpool_without_guard_is_ablation_only": True,
    }:
        raise ValueError("GuardPool known Maze claim boundary changed")
    full = validate_structpool_augmentation(
        dict(config["full_structpool_augmentation"])
    )
    if full is None or full.get("pool_id") != "stride-structpool-v1":
        raise ValueError("GuardPool Maze full StructPool treatment changed")
    expected_inputs = {
        "guardpool_registration",
        "runtime_config",
        "controller_manifest",
        "dataset_summary",
        "dataset_manifest",
        "slotpool_model",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("GuardPool Maze input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    registration = _read_json(inputs["guardpool_registration"])
    validate_guardpool_registration(registration, project_root=root)
    if tuple(registration["known_maze_regression"]["treatments"]) != CONTROLLERS:
        raise ValueError("GuardPool registration treatment order changed")
    runtime = _read_json(inputs["runtime_config"])
    if (
        runtime.get("formal") is not False
        or runtime.get("solver_seeds") != [3]
        or dict(runtime.get("qualification") or {})
        != {
            "mode": "known_maze_regression_reset_only_v1",
            "minimum_nonzero_states": 1,
            "minimum_nonzero_states_per_layout": 1,
            "minimum_active_maps": 1,
            "minimum_nonzero_states_per_solver_seed": 1,
        }
    ):
        raise ValueError("GuardPool fresh reset-only qualification changed")
    rows = _read_jsonl(inputs["dataset_manifest"])
    matching = [row for row in rows if str(row["task_id"]) == cohort["task_id"]]
    if len(matching) != 1 or int(matching[0]["agent_count"]) != 100:
        raise ValueError("GuardPool known Maze task is missing from the dataset")
    return path, root, config


def guardpool_maze_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    cohort = dict(config["cohort"])
    return [
        {
            "controller": controller,
            "task_id": str(cohort["task_id"]),
            "solver_seed": int(cohort["solver_seed"]),
            "execution_position": index,
        }
        for index, controller in enumerate(CONTROLLERS)
    ]


def _controller_kwargs(root: Path, config: dict[str, Any], controller: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": str(config["runtime"]["feature_backend"]),
        "controller_runtime": str(config["runtime"]["controller_runtime"]),
        "verification_profile": str(config["runtime"]["verification_profile"]),
        "stopping_rule": "run-to-completion",
    }
    if controller == "v2-plus-structpool":
        result["structpool_augmentation"] = dict(
            config["full_structpool_augmentation"]
        )
    elif controller == "v2-plus-slotpool":
        result["structpool_augmentation"] = slotpool_runtime_augmentation()
    elif controller == "stride-guardpool-v1":
        result["structpool_augmentation"] = guardpool_runtime_augmentation()
    elif controller != "v2-full":
        raise ValueError(f"unknown GuardPool Maze controller: {controller}")
    return result


def run_guardpool_maze_regression(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_guardpool_maze_regression_config(config_path)
    schedule = guardpool_maze_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    output = Path(output).resolve()
    status_path = output / "regression_status.json"
    prepared = prepare_resumable_output(
        output,
        status_filename=status_path.name,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_guardpool_maze_regression.py",
                "experiments/stride_guardpool.py",
            ),
        ),
        resume=resume,
        report_filename="guardpool_maze_regression_report.json",
        report_schema=REPORT_SCHEMA,
        label="GuardPool Maze",
    )
    status_base = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    cohort = dict(config["cohort"])
    key = (str(cohort["task_id"]), int(cohort["solver_seed"]))
    keys = {key}
    dataset = (root / str(cohort["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    if config.get("qualification_source") is not None:
        raise ValueError("GuardPool Maze must collect a fresh reset qualification")
    qualification_source = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification_source,
        phase="qualify",
        workers=1,
        resume=(
            prepared.resumed
            and qualification_source.joinpath("run_config.json").is_file()
        ),
        cohort_job_keys=keys,
        job_keys=keys,
        **_controller_kwargs(root, config, "v2-full"),
    )
    for completed, item in enumerate(schedule, start=1):
        controller = str(item["controller"])
        collection = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            workers=1,
            resume=(
                prepared.resumed
                and collection.joinpath("run_config.json").is_file()
            ),
            cohort_job_keys=keys,
            job_keys=keys,
            qualification_source=qualification_source,
            **_controller_kwargs(root, config, controller),
        )
        manifest = collection / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=keys,
                job_keys=keys,
                **_controller_kwargs(root, config, controller),
            )
        _write_json(
            status_path,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "current": item,
                "complete": False,
            },
        )
    report = analyze_guardpool_maze_regression(
        path,
        output,
        producer=status_base["producer_identity"],
    )
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": len(schedule),
            "complete": True,
            "report_sha256": sha256_file(
                output / "guardpool_maze_regression_report.json"
            ),
        },
    )
    return report


def _trace_summary(collection: Path, row: dict[str, Any]) -> dict[str, Any]:
    events = read_trace_events(collection / str(row["trace_file"]))
    transitions = [event for event in events if event.get("event") == "transition"]
    summary = dict(row["summary"])
    trajectory = list(map(int, summary["conflict_trajectory"]))
    decisions = []
    for index, event in enumerate(transitions):
        controller = dict(event.get("controller") or {})
        proposal = dict(controller.get("proposal") or {})
        selected_id = str(controller.get("selected_candidate_id") or "")
        selected = next(
            (
                candidate
                for candidate in controller.get("candidate_pool", [])
                if str(candidate.get("candidate_id")) == selected_id
            ),
            {},
        )
        decisions.append(
            {
                "decision_index": index,
                "before_conflicts": trajectory[index],
                "after_conflicts": trajectory[index + 1],
                "selected_candidate_id": selected_id,
                "selected_size": selected.get("actual_size"),
                "selected_families": list(selected.get("selection_families") or ()),
                "selected_structural": bool(selected.get("structpool_family_groups")),
                "pp_random_seed": int(dict(event.get("action") or {}).get("pp_random_seed", -1)),
                "before_fingerprint": str(event.get("before_fingerprint") or ""),
                "guard_active": bool(proposal.get("guardpool_active", False)),
                "guard_triggered": bool(proposal.get("guardpool_triggered", False)),
                "guard_released": bool(proposal.get("guardpool_released", False)),
                "guard_no_progress_streak": int(
                    proposal.get("guardpool_no_progress_streak", 0)
                ),
                "slotpool_raw_structural_count": int(
                    proposal.get("slotpool_raw_structural_candidate_count", 0)
                ),
                "slotpool_selected_structural_count": int(
                    proposal.get("slotpool_selected_structural_candidate_count", 0)
                ),
            }
        )
    streak = 0
    maximum_streak = 0
    for before, after in zip(trajectory, trajectory[1:]):
        streak = 0 if after < before else streak + 1
        maximum_streak = max(maximum_streak, streak)
    trigger_decisions = [
        row["decision_index"] for row in decisions if row["guard_triggered"]
    ]
    guard_recovery_rounds = []
    for trigger in trigger_decisions:
        recovery = next(
            (
                index - trigger + 1
                for index in range(trigger, len(decisions))
                if decisions[index]["after_conflicts"]
                < decisions[index]["before_conflicts"]
            ),
            None,
        )
        guard_recovery_rounds.append(recovery)
    totals = dict(summary.get("controller_totals") or {})
    return {
        "status": str(row["status"]),
        "success": bool(summary.get("success")),
        "initial_conflicts": int(summary["initial_conflicts"]),
        "initial_fingerprint": str(summary["initial_fingerprint"]),
        "repair_iterations": int(summary["repair_iterations"]),
        "raw_wall_ttf_seconds": summary.get("wall_time_to_feasible"),
        "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
        "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
        "controller_seconds_before_repair": float(
            totals.get("controller_seconds_before_repair", 0.0)
        ),
        "slotpool_total_inference_seconds": float(
            totals.get("slotpool_total_inference_seconds", 0.0)
        ),
        "invalid_action_count": int(summary.get("invalid_action_count", 0)),
        "fingerprint_mismatch_count": int(
            summary.get("fingerprint_mismatch_count", 0)
        ),
        "maximum_no_progress_streak": maximum_streak,
        "guard_trigger_decisions": trigger_decisions,
        "guard_release_decisions": [
            row["decision_index"] for row in decisions if row["guard_released"]
        ],
        "guard_recovery_rounds": guard_recovery_rounds,
        "conflict_trajectory": trajectory,
        "decisions": decisions,
        "trace_sha256": str(row["trace_sha256"]),
    }


def _paired_pp_replay_audit(
    rows: list[dict[str, Any]],
) -> tuple[bool, int]:
    replay_groups: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    for row in rows:
        for decision in row.get("decisions") or ():
            replay_groups[
                (
                    int(decision["decision_index"]),
                    str(decision["before_fingerprint"]),
                )
            ].append(int(decision["pp_random_seed"]))
    paired = [seeds for seeds in replay_groups.values() if len(seeds) >= 2]
    return bool(paired) and all(len(set(seeds)) == 1 for seeds in paired), len(paired)


def analyze_guardpool_maze_regression(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_guardpool_maze_regression_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename="regression_status.json",
        report_filename="guardpool_maze_regression_report.json",
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_guardpool_maze_regression.py",
                "experiments/stride_guardpool.py",
            ),
            native_required=False,
        )
    by_controller: dict[str, dict[str, Any]] = {}
    error_count = 0
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        rows = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
        if len(rows) != 1:
            raise ValueError(f"GuardPool Maze {controller} manifest coverage changed")
        row = rows[0]
        if row.get("status") != "ok":
            error_count += 1
            by_controller[controller] = {
                "status": str(row.get("status")),
                "error": row.get("error"),
            }
        else:
            by_controller[controller] = _trace_summary(collection, row)
    valid = [row for row in by_controller.values() if row.get("status") == "ok"]
    fingerprints = {str(row["initial_fingerprint"]) for row in valid}
    initial_conflicts = {int(row["initial_conflicts"]) for row in valid}
    v2 = by_controller["v2-full"]
    guard = by_controller["stride-guardpool-v1"]
    pp_seed_pairing, paired_replay_group_count = _paired_pp_replay_audit(valid)
    first_divergence = {}
    v2_ids = [row["selected_candidate_id"] for row in v2.get("decisions", [])]
    for controller, row in by_controller.items():
        if controller == "v2-full" or row.get("status") != "ok":
            continue
        ids = [decision["selected_candidate_id"] for decision in row["decisions"]]
        first_divergence[controller] = next(
            (
                index
                for index, (left, right) in enumerate(zip(v2_ids, ids))
                if left != right
            ),
            None,
        )
    gates = {
        "zero_errors": error_count == int(config["gates"]["maximum_error_count"]),
        "complete_controller_coverage": set(by_controller) == set(CONTROLLERS)
        and len(valid) == len(CONTROLLERS),
        "initial_conflicts": initial_conflicts == {66},
        "initial_fingerprint": len(fingerprints) == 1,
        "v2_success": v2.get("success") is True,
        "v2_repair_iterations": v2.get("repair_iterations") == 15,
        "guardpool_success": guard.get("success") is True,
        "guardpool_repair_iterations": int(
            guard.get("repair_iterations", 10**9)
        )
        <= int(config["gates"]["guardpool_maximum_repair_iterations"]),
        "no_invalid_actions": all(
            int(row.get("invalid_action_count", 0)) == 0 for row in valid
        ),
        "no_fingerprint_mismatch": all(
            int(row.get("fingerprint_mismatch_count", 0)) == 0 for row in valid
        ),
        "paired_pp_seeds": pp_seed_pairing,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "known_regression_only_not_generalization_or_ttf_claim",
        "producer_identity": producer,
        "integrity_passed": all(
            value
            for name, value in gates.items()
            if name not in {"guardpool_success", "guardpool_repair_iterations"}
        ),
        "regression_passed": all(gates.values()),
        "gates": gates,
        "controller_results": by_controller,
        "first_divergence_vs_v2": first_divergence,
        "identical_state_paired_pp_group_count": paired_replay_group_count,
        "pp_pairing_semantics": "same_decision_index_and_before_fingerprint",
        "formal_ttf_claim": False,
        "independent_generalization_evidence": False,
        "next_step": (
            "run_guardpool_development_quick"
            if all(gates.values())
            else "stop_guardpool_runtime_before_development_ttf"
        ),
        "inputs": {
            "config_sha256": sha256_file(path),
            "controller_trace_sha256": {
                controller: row.get("trace_sha256")
                for controller, row in by_controller.items()
            },
        },
    }
    _write_json(output / "guardpool_maze_regression_report.json", report)
    return report


__all__ = [
    "analyze_guardpool_maze_regression",
    "_paired_pp_replay_audit",
    "guardpool_maze_schedule",
    "load_guardpool_maze_regression_config",
    "run_guardpool_maze_regression",
]
