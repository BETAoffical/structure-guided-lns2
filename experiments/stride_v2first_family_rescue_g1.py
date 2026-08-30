from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import (
    closed_loop_producer_identity,
    json_fingerprint,
    read_json,
    read_jsonl,
    registered_input,
    sha256_file,
    write_json,
)
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_failure_informed_rescue_continuation import _decision_rows
from experiments.stride_structshell_maze32_n300_fourmap_quick import (
    load_config as load_four_map_source_config,
)
from lns2_selector.runtime.v2_first_single_family_rescue import (
    validate_v2_first_single_family_rescue_augmentation,
    v2_first_single_family_rescue_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.v2first_family_rescue_g1_config.v1"
STATUS_SCHEMA = "lns2.stride.v2first_family_rescue_g1_status.v1"
BATCH_A_REPORT_SCHEMA = "lns2.stride.v2first_family_rescue_g1_batch_a_report.v1"
FINAL_REPORT_SCHEMA = "lns2.stride.v2first_family_rescue_g1_final_report.v1"
EXPERIMENT_ID = "stride-v2first-family-rescue-g1-v1"
PRE_REGISTRATION_PARENT_COMMIT = "4b0ffa7dabd341044230d5ad988b9d60a89d0a1c"
CONTROLLERS = (
    "v2_only",
    "component16_rescue",
    "hotspot16_rescue",
)
RESCUE_ARMS = CONTROLLERS[1:]
SOLVER_SEED = 26
STATUS_FILENAME = "collection_status.json"
BATCH_A_REPORT_FILENAME = "batch_a_report.json"
FINAL_REPORT_FILENAME = "final_report.json"

_STAGE_GROUPS = {
    "batch_a": (
        "room-64-64-16",
        "random-32-32-20-high-load",
        "warehouse-w1020a-opposite-exchange",
    ),
    "batch_b": ("maze-32-32-4-n300",),
}

_EXPECTED_RUNTIME = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 200.0,
    "environment_time_limit_seconds": 200.0,
    "episode_process_timeout_seconds": 260.0,
    "outer_job_timeout_seconds": 300.0,
    "timing_boundary": "reset_inclusive_ttf",
    "execution_order": "rotating_strict_three_controller_serial",
    "qualification": {
        "phase": "reset_only",
        "workers": 16,
        "included_in_ttf": False,
    },
    "timed_episodes": {
        "workers": 1,
        "strict_serial": True,
        "included_in_ttf": True,
    },
}

_EXPECTED_PER_MAP_GATES = {
    "success_must_not_be_below_v2": True,
    "maximum_restricted_ttf_regression": 0.20,
}
_EXPECTED_BATCH_A_GATES = {
    "minimum_rescue_triggered_map_count": 2,
    "minimum_post_trigger_repeat_rollback_rate_reduction": 0.15,
    "success_count_must_not_be_below_v2": True,
    "maximum_mean_restricted_ttf_regression": 0.10,
    "maximum_any_map_restricted_ttf_regression": 0.20,
    "each_arm_evaluated_independently_against_v2": True,
    "minimum_independently_passing_arm_count": 1,
    "cross_arm_best_selection": False,
}
_EXPECTED_BATCH_B_GATES = {
    "retain_a_batch_a_passing_arm": True,
    "maze_success_must_not_be_below_v2": True,
    "maximum_maze_restricted_ttf_regression": 0.20,
    "maximum_all_map_mean_restricted_ttf_regression": 0.10,
}


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_v2first_family_rescue_g1.py",
            "scripts/run_stride_v2first_family_rescue_g1.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/runtime/v2_first_single_family_rescue.py",
            "lns2_selector/runtime/structshell_single_family.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
        native_required=native_required,
    )


def _validate_controller_contract(config: Mapping[str, Any]) -> None:
    contract = dict(config.get("controller_contract") or {})
    trigger = dict(contract.get("trigger") or {})
    arms = dict(contract.get("arms") or {})
    if (
        contract.get("base")
        != "frozen_v2_full_native_features_optimized_copeland"
        or trigger
        != {
            "trigger_id": "stride-v2first-exact-rollback-rescue-v1",
            "minimum_consecutive_v2_exact_rollbacks": 3,
            "fingerprint_scope": "same_repair_fingerprint",
            "maximum_rescue_offers_per_episode": 1,
            "maximum_executed_rescues_per_episode": 1,
            "offer_consumed_when_challenger_unavailable": True,
            "post_rescue_policy": "permanent_v2_only",
            "time_limit_triggers_rescue": False,
        }
        or arms
        != {
            "component16_rescue": {
                "family": "conflict_component",
                "nominal_size": 16,
            },
            "hotspot16_rescue": {
                "family": "spatiotemporal_hotspot",
                "nominal_size": 16,
            },
        }
        or contract.get("rescue_selection")
        != "execute_the_only_requested_family_candidate_directly"
        or contract.get("unavailable_fallback")
        != "fresh_v2_only_and_record_unavailable"
        or contract.get("full_v2_pool_unchanged_before_trigger") is not True
        or contract.get("family_candidates_never_enter_v2_copeland_pool") is not True
        or contract.get("component_and_hotspot_never_share_a_pool") is not True
        or int(contract.get("maximum_pp_calls_per_decision", -1)) != 1
        or contract.get("gcbs_retry_plateau_router_or_new_ranker") is not False
    ):
        raise ValueError("G1 V2-first rescue controller contract changed")
    expected = {
        "component16_rescue": v2_first_single_family_rescue_augmentation(
            "conflict_component", 16
        ),
        "hotspot16_rescue": v2_first_single_family_rescue_augmentation(
            "hotspot", 16
        ),
    }
    for arm, value in expected.items():
        if validate_v2_first_single_family_rescue_augmentation(value) != value:
            raise ValueError(f"G1 runtime factory rejected {arm}")


def load_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("G1 config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_fresh_solver_seed_family_isolated_bounded_ttf_screen"
        or config.get("pre_registration_parent_commit")
        != PRE_REGISTRATION_PARENT_COMMIT
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or dict(config.get("runtime") or {}) != _EXPECTED_RUNTIME
    ):
        raise ValueError("G1 experiment identity changed")
    _validate_controller_contract(config)

    stages = dict(config.get("stages") or {})
    if (
        tuple(map(str, dict(stages.get("batch_a") or {}).get("ordered_group_ids") or ()))
        != _STAGE_GROUPS["batch_a"]
        or int(dict(stages.get("batch_a") or {}).get("paired_key_count", -1)) != 3
        or int(dict(stages.get("batch_a") or {}).get("episode_count", -1)) != 9
        or dict(stages.get("batch_a") or {}).get(
            "stop_suffix_when_no_rescue_arm_survives_completed_map_gates"
        )
        is not True
        or tuple(map(str, dict(stages.get("batch_b") or {}).get("ordered_group_ids") or ()))
        != _STAGE_GROUPS["batch_b"]
        or int(dict(stages.get("batch_b") or {}).get("paired_key_count", -1)) != 1
        or int(dict(stages.get("batch_b") or {}).get("episode_count", -1)) != 3
        or dict(stages.get("batch_b") or {}).get("requires_batch_a_pass") is not True
    ):
        raise ValueError("G1 staged schedule changed")

    cohort = dict(config.get("cohort") or {})
    seed_audit = dict(config.get("solver_seed_identity_audit") or {})
    if (
        cohort
        != {
            "role": "fixed_tasks_fresh_solver_seed_general_model_screen",
            "source_config": "four_map_source_config",
            "solver_seed": SOLVER_SEED,
            "task_count": 4,
            "paired_key_count": 4,
            "episode_count": 12,
            "result_based_filtering": False,
            "known_seed25_outcomes_used_for_selection": False,
        }
        or seed_audit
        != {
            "candidate_seed": SOLVER_SEED,
            "selected_solver_seed": SOLVER_SEED,
            "audit_scope": (
                "four_exact_task_ids_registered_configs_and_manifest_rows_only"
            ),
            "global_freshness_scan": False,
            "outcome_fields_read": False,
            "registered_config_match_count": 0,
            "completed_manifest_match_count": 0,
            "completed_before_registration": True,
        }
    ):
        raise ValueError("G1 fixed task / fresh solver seed cohort changed")

    gates = dict(config.get("performance_gates") or {})
    if (
        dict(gates.get("per_map") or {}) != _EXPECTED_PER_MAP_GATES
        or dict(gates.get("batch_a") or {}) != _EXPECTED_BATCH_A_GATES
        or dict(gates.get("batch_b") or {}) != _EXPECTED_BATCH_B_GATES
    ):
        raise ValueError("G1 gates changed")

    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "screen_only": True,
        "single_fresh_solver_seed": True,
        "formal_promotion_allowed": False,
        "bootstrap": False,
        "auc_gate": False,
        "official_adaptive_arm": False,
        "warehouse_specialist_tuning_allowed": False,
        "gcbs_allowed": False,
    }:
        raise ValueError("G1 claim boundary changed")

    inputs = dict(config.get("inputs") or {})
    source_path = registered_input(
        root,
        dict(inputs.get("four_map_source_config") or {}),
        label="G1 four-map source config",
    )
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="G1 V2 controller manifest",
    )
    if controller_manifest.parent != (
        root / str(config.get("controller_bundle"))
    ).resolve():
        raise ValueError("G1 controller bundle changed")
    _source_path, _source_root, source = load_four_map_source_config(source_path)
    by_id = {str(group["id"]): dict(group) for group in source["cohort"]["groups"]}
    expected_ids = set(_STAGE_GROUPS["batch_a"] + _STAGE_GROUPS["batch_b"])
    if set(by_id) != expected_ids:
        raise ValueError("G1 source group identity changed")
    config = dict(config)
    config["_groups"] = by_id
    return path, root, config


def schedule(config: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in _STAGE_GROUPS:
        raise ValueError(f"unknown G1 stage: {stage}")
    all_ids = _STAGE_GROUPS["batch_a"] + _STAGE_GROUPS["batch_b"]
    groups = dict(config["_groups"])
    rows: list[dict[str, Any]] = []
    for group_id in _STAGE_GROUPS[stage]:
        global_index = all_ids.index(group_id)
        order = CONTROLLERS[global_index % len(CONTROLLERS) :] + CONTROLLERS[
            : global_index % len(CONTROLLERS)
        ]
        group = groups[group_id]
        for position, controller in enumerate(order):
            rows.append(
                {
                    "stage": stage,
                    "group_index": global_index,
                    "within_key_position": position,
                    "key_id": f"{group_id}@{SOLVER_SEED}",
                    "group_id": group_id,
                    "map_id": str(group["map_id"]),
                    "family": str(group["family"]),
                    "task_id": str(group["task"]),
                    "solver_seed": SOLVER_SEED,
                    "controller": controller,
                }
            )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    batch_a = schedule(config, "batch_a")
    batch_b = schedule(config, "batch_b")
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    fuse = float(config["runtime"]["episode_process_timeout_seconds"])
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "solver_seed": SOLVER_SEED,
        "controllers": list(CONTROLLERS),
        "batch_a": {
            "ordered_group_ids": list(_STAGE_GROUPS["batch_a"]),
            "schedule_entry_count": len(batch_a),
            "schedule_sha256": json_fingerprint(batch_a),
            "maximum_registered_ttf_seconds": len(batch_a) * wall,
        },
        "batch_b": {
            "ordered_group_ids": list(_STAGE_GROUPS["batch_b"]),
            "schedule_entry_count": len(batch_b),
            "schedule_sha256": json_fingerprint(batch_b),
            "maximum_registered_ttf_seconds": len(batch_b) * wall,
            "blocked_until_batch_a_passes": True,
        },
        "wall_time_budget_seconds_per_episode": wall,
        "episode_process_fuse_seconds": fuse,
        "qualification": {
            "workers": int(config["runtime"]["qualification"]["workers"]),
            "reset_only": True,
            "included_in_ttf": False,
        },
        "timed_episodes": {
            "workers": int(config["runtime"]["timed_episodes"]["workers"]),
            "strict_serial": True,
            "included_in_ttf": True,
        },
        "maximum_all_stage_ttf_seconds": (len(batch_a) + len(batch_b)) * wall,
        "maximum_all_stage_process_fuse_seconds": (
            (len(batch_a) + len(batch_b)) * fuse
        ),
        "stop_suffix_on_map_gate_failure": True,
        "solver_or_controller_invoked": False,
        "bootstrap": False,
        "auc_gate": False,
        "official_adaptive": False,
        "gcbs": False,
        "global_scan": False,
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    result: dict[str, Any] = {
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
        "controller": "v2-full",
        "controller_bundle": str(
            (root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if controller == "v2_only":
        return result
    if controller == "component16_rescue":
        augmentation = v2_first_single_family_rescue_augmentation(
            "conflict_component", 16
        )
    elif controller == "hotspot16_rescue":
        augmentation = v2_first_single_family_rescue_augmentation("hotspot", 16)
    else:
        raise ValueError(f"unknown G1 controller: {controller}")
    result["hybridstructpool_augmentation"] = (
        validate_v2_first_single_family_rescue_augmentation(augmentation)
    )
    return result


def _group(config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return dict(dict(config["_groups"])[group_id])


def _stage_root(output: Path, stage: str) -> Path:
    return output / stage


def _runtime_config_path(
    output: Path,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Path:
    payload = read_json(Path(str(group["_runtime_path"])).resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"{group['id']} runtime config must be an object")
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    fuse = float(config["runtime"]["episode_process_timeout_seconds"])
    payload["split"] = str(group["split"])
    payload["solver_seeds"] = [SOLVER_SEED]
    payload["wall_time_budget_seconds"] = wall
    payload["episode_process_timeout_seconds"] = fuse
    # This is the timed runtime file.  Qualification passes its independent
    # worker count explicitly and is outside the TTF clock.
    payload["workers"] = 1
    environment = dict(payload.get("environment") or {})
    environment["time_limit"] = wall
    payload["environment"] = environment
    destination = (
        output
        / "runtime_configs"
        / f"{group['id']}__seed_{SOLVER_SEED:04d}__wall_0200.json"
    )
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError(f"{group['id']} materialized G1 runtime changed")
    else:
        write_json(destination, payload)
    return destination


def _controller_root(output: Path, stage: str, item: Mapping[str, Any]) -> Path:
    return (
        _stage_root(output, stage)
        / "maps"
        / str(item["group_id"])
        / str(item["controller"])
    )


def _manifest_path(output: Path, stage: str, item: Mapping[str, Any]) -> Path:
    return _controller_root(output, stage, item) / "realized_dynamic_manifest.jsonl"


def _manifest_row(
    output: Path, stage: str, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _manifest_path(output, stage, item)
    rows = read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("G1 manifest is ambiguous")
    return matches[0] if matches else None


def _qualification_root(output: Path, group_id: str) -> Path:
    return output / "qualification" / group_id


def _qualify_group(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    task_id = str(group["task"])
    key = {(task_id, SOLVER_SEED)}
    qualification = _qualification_root(output, str(group["id"]))
    run_closed_loop_collection(
        root / str(group["dataset"]),
        _runtime_config_path(output, config, group),
        qualification,
        phase="qualify",
        workers=int(config["runtime"]["qualification"]["workers"]),
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        task_ids=[task_id],
        cohort_job_keys=key,
        job_keys=key,
        qualification_process_timeout_seconds=float(
            config["runtime"]["episode_process_timeout_seconds"]
        ),
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "v2_only"),
    )
    report = read_json(qualification / "qualification_report.json")
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or int(report.get("valid_count", -1)) != 1
        or int(report.get("incomplete_reset_count", -1)) != 0
    ):
        raise RuntimeError(f"{group['id']} G1 reset qualification failed")
    return qualification


def _run_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    stage: str,
    item: Mapping[str, Any],
    qualification: Path,
) -> dict[str, Any]:
    group = _group(config, str(item["group_id"]))
    task_id = str(item["task_id"])
    key = {(task_id, SOLVER_SEED)}
    collection = _controller_root(output, stage, item)
    runtime = _runtime_config_path(output, config, group)
    common = {
        "workers": int(config["runtime"]["timed_episodes"]["workers"]),
        "task_ids": [task_id],
        "cohort_job_keys": key,
        "job_keys": key,
        "qualification_source": qualification,
        "qualification_process_timeout_seconds": float(
            config["runtime"]["episode_process_timeout_seconds"]
        ),
        "use_global_collection_lock": False,
    }
    kwargs = controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        resume=collection.joinpath("run_config.json").is_file(),
        **common,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="realized_dynamic",
        resume=True,
        **common,
        **kwargs,
    )
    row = _manifest_row(output, stage, item)
    if row is None:
        raise RuntimeError("G1 timed episode produced no manifest")
    return row


def _valid_summary(
    manifest: Mapping[str, Any], wall_time_seconds: float
) -> tuple[dict[str, Any] | None, str | None]:
    summary = manifest.get("summary")
    if manifest.get("status") != "ok" or not isinstance(summary, dict):
        return None, f"manifest status is {manifest.get('status')}"
    if (
        float(summary.get("wall_time_budget_seconds", -1.0)) != wall_time_seconds
        or summary.get("ttf_clock_schema") != "lns2.ttf.reset_inclusive_wall.v1"
        or summary.get("capped_wall_time_to_feasible") is None
        or int(summary.get("invalid_action_count", -1)) != 0
        or int(summary.get("fingerprint_mismatch_count", -1)) != 0
    ):
        return None, "invalid reset-inclusive bounded TTF summary"
    return dict(summary), None


def _is_exact_rollback(row: Mapping[str, Any]) -> bool:
    metrics = dict(row["actual_metrics"])
    return bool(
        metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
        and metrics.get("replan_success") is False
        and metrics.get("pp_rolled_back") is True
        and str(row["before_platform_signature"])
        == str(row["after_platform_signature"])
        and int(row["before_conflicts"]) == int(row["after_conflicts"])
    )


def _rescue_event(
    collection_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    decisions = _decision_rows(collection_root, manifest)
    offered_rows: list[dict[str, Any]] = []
    for row in decisions:
        record = dict(
            dict(row["controller"]).get("v2_first_single_family_rescue") or {}
        )
        selection = dict(record.get("selection") or {})
        generation = dict(record.get("generation") or {})
        execution = dict(record.get("execution") or {})
        observation = dict(record.get("observation") or {})
        phase = str(selection.get("selection_phase") or "")
        offered = bool(
            selection.get("offered", False)
            or generation.get("offered", False)
            or observation.get("offered", False)
            or generation.get("attempted", False)
            or phase in {"single_family_rescue_due", "fresh_v2_after_unavailable"}
        )
        if offered:
            enriched = dict(row)
            enriched["_rescue_fields"] = {
                "offered": True,
                "challenger_present": bool(
                    execution.get("challenger_present", False)
                    or selection.get("challenger_present", False)
                    or generation.get("challenger_present", False)
                    or generation.get("available", False)
                ),
                "structural_selected": bool(
                    execution.get("structural_selected", False)
                    or selection.get("structural_selected", False)
                ),
                "consumed": bool(
                    execution.get("consumed", False)
                    or selection.get("consumed", False)
                    or generation.get("consumed", False)
                    or observation.get("consumed", False)
                ),
            }
            offered_rows.append(enriched)
    if len(offered_rows) > 1:
        raise ValueError("G1 episode offered more than one structural rescue")
    offered = offered_rows[0] if offered_rows else None
    fields = dict(offered.get("_rescue_fields") or {}) if offered else {}
    if offered is not None and fields.get("consumed") is not True:
        raise ValueError("G1 rescue offer did not consume the episode opportunity")
    executed = bool(fields.get("structural_selected", False))
    unavailable = bool(
        offered is not None
        and not fields.get("challenger_present", False)
        and not executed
    )
    if executed and not fields.get("challenger_present", False):
        raise ValueError("G1 selected a missing structural challenger")
    return {
        "decision_count": len(decisions),
        "offered": offered is not None,
        "offer_consumed": bool(fields.get("consumed", False)),
        "generation_attempted": offered is not None,
        "challenger_present": bool(fields.get("challenger_present", False)),
        "unavailable": unavailable,
        "executed": executed,
        "decision_index": (
            int(offered["decision_index"]) if offered is not None else None
        ),
        "before_platform_signature": (
            str(offered["before_platform_signature"])
            if offered is not None
            else None
        ),
        "repeat_exact_rollback": (
            _is_exact_rollback(offered) if executed and offered is not None else None
        ),
    }


def _action_parity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    action = dict(row["actual_action"])
    metrics = dict(row["actual_metrics"])
    controller = dict(row["controller"])
    return {
        "before_platform_signature": str(row["before_platform_signature"]),
        "after_platform_signature": str(row["after_platform_signature"]),
        "before_conflicts": int(row["before_conflicts"]),
        "after_conflicts": int(row["after_conflicts"]),
        "candidate_id": str(
            controller.get(
                "selected_candidate_id",
                controller.get("base_selected_candidate_id", ""),
            )
        ),
        "mode": str(action.get("mode") or ""),
        "agents": list(map(int, action.get("agents") or ())),
        "random_seed": int(action.get("random_seed", -1)),
        "pp_random_seed": int(action.get("pp_random_seed", -1)),
        "requested_pp_random_seed": int(
            metrics.get("requested_pp_random_seed", -1)
        ),
        "applied_pp_random_seed": int(metrics.get("applied_pp_random_seed", -1)),
        "repair_order": list(map(int, metrics.get("repair_order") or ())),
    }


def _seed_parity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    action = dict(row["actual_action"])
    metrics = dict(row["actual_metrics"])
    controller = dict(row["controller"])
    return {
        "action_random_seed": int(action.get("random_seed", -1)),
        "action_pp_random_seed": int(action.get("pp_random_seed", -1)),
        "requested_random_seed": int(metrics.get("requested_random_seed", -1)),
        "requested_pp_random_seed": int(
            metrics.get("requested_pp_random_seed", -1)
        ),
        "applied_pp_random_seed": int(metrics.get("applied_pp_random_seed", -1)),
        "repair_seed_draw_index": controller.get("repair_seed_draw_index"),
    }


def _candidate_pool_projection(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    controller = dict(row["controller"])
    return [
        {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "agents": list(map(int, candidate.get("agents") or ())),
            "actual_size": int(candidate.get("actual_size", -1)),
            "selection_families": list(
                map(str, candidate.get("selection_families") or ())
            ),
            "hybridstructpool_provenance": list(
                map(str, candidate.get("hybridstructpool_provenance") or ())
            ),
            "retained": bool(candidate.get("retained", False)),
            "score": candidate.get("score"),
        }
        for candidate in list(controller.get("candidate_pool") or ())
    ]


def _v2_first_trace_audit(
    decisions: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    expected_profile: str,
) -> dict[str, Any]:
    """Validate the one-shot rescue state machine in one linear trace pass."""

    profile = str(expected_profile)
    family_contract = {
        "conflict_component": (
            "structpool-conflict-component:16",
            "conflict_component",
        ),
        "hotspot": (
            "structpool-spatiotemporal-hotspot:16",
            "spatiotemporal_hotspot",
        ),
    }.get(profile)
    if family_contract is None:
        raise ValueError(f"unsupported G1 trace profile: {profile}")
    family_label, family_group = family_contract
    if not decisions:
        return {
            "passed": True,
            "decision_count": 0,
            "offered": False,
            "executed": False,
            "unavailable": False,
            "offer_decision_index": None,
            "pre_intervention_v2_parity": {
                "checked_decision_count": 0,
                "passed": True,
                "first_mismatch_decision": None,
            },
            "baseline_counterfactual": None,
            "immediate_rescue_repeat_exact_rollback": None,
        }
    if any(int(row["decision_index"]) != index for index, row in enumerate(decisions)):
        raise ValueError("G1 rescue trace decision indices are not contiguous")

    offer_index: int | None = None
    offer_record: dict[str, Any] | None = None
    streak_fingerprint: str | None = None
    streak = 0
    required_offer_index: int | None = None
    for index, row in enumerate(decisions):
        controller = dict(row["controller"])
        raw_record = controller.get("v2_first_single_family_rescue")
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"G1 decision {index} lacks rescue trace record")
        record = dict(raw_record)
        if not {"selection", "generation", "execution", "observation"} <= set(
            record
        ):
            raise ValueError(f"G1 decision {index} rescue trace is incomplete")
        selection = record.get("selection")
        observation = record.get("observation")
        if not isinstance(selection, Mapping) or not isinstance(observation, Mapping):
            raise ValueError(
                f"G1 decision {index} selection/observation trace is incomplete"
            )
        selection = dict(selection)
        observation = dict(observation)
        if (
            int(selection.get("decision_index", -1)) != index
            or int(observation.get("decision_index", -1)) != index
            or str(selection.get("before_repair_fingerprint") or "")
            != str(row["before_platform_signature"])
            or str(observation.get("before_repair_fingerprint") or "")
            != str(row["before_platform_signature"])
            or str(observation.get("after_repair_fingerprint") or "")
            != str(row["after_platform_signature"])
            or str(selection.get("structural_profile") or "") != profile
            or int(selection.get("nominal_size", -1)) != 16
        ):
            raise ValueError(f"G1 decision {index} trace identity changed")

        offered = bool(selection.get("offered", False))
        if offered:
            if offer_index is not None:
                raise ValueError("G1 episode offered structural rescue more than once")
            offer_index = index
            offer_record = record
        if required_offer_index == index and not offered:
            raise ValueError("G1 trace missed the decision immediately after trigger")
        phase = str(selection.get("selection_phase") or "")
        if offer_index is None:
            if phase != "v2_only" or record["generation"] is not None or record[
                "execution"
            ] is not None:
                raise ValueError(f"G1 decision {index} offered rescue early")
            if str(observation.get("decision_mode") or "") != "v2_only":
                raise ValueError(f"G1 decision {index} prefix is not V2-only")
            if str(
                dict(controller.get("proposal") or {}).get(
                    "v2_first_rescue_mode"
                )
                or ""
            ) != "v2_only":
                raise ValueError(f"G1 decision {index} prefix proposal is not V2")
            if any(
                "v2_first_single_family_rescue"
                in set(
                    map(
                        str,
                        candidate.get("hybridstructpool_provenance") or (),
                    )
                )
                for candidate in list(controller.get("candidate_pool") or ())
            ):
                raise ValueError(f"G1 decision {index} prefix mixed rescue candidate")
        elif index > offer_index:
            if (
                phase != "v2_only_after_offer"
                or offered
                or record["generation"] is not None
                or record["execution"] is not None
                or str(observation.get("decision_mode") or "") != "v2_only"
            ):
                raise ValueError("G1 rescue did not latch permanent V2-only mode")
            if str(
                dict(controller.get("proposal") or {}).get(
                    "v2_first_rescue_mode"
                )
                or ""
            ) != "v2_only_after_offer":
                raise ValueError("G1 post-offer proposal did not stay V2-only")
            for candidate in list(controller.get("candidate_pool") or ()):
                provenance = set(
                    map(str, candidate.get("hybridstructpool_provenance") or ())
                )
                if "v2_first_single_family_rescue" in provenance:
                    raise ValueError("G1 post-offer V2 pool retained a rescue candidate")

        exact = _is_exact_rollback(row)
        if exact:
            fingerprint = str(row["before_platform_signature"])
            streak = streak + 1 if streak_fingerprint == fingerprint else 1
            streak_fingerprint = fingerprint
        else:
            streak = 0
            streak_fingerprint = str(row["after_platform_signature"])
        if offer_index is None and streak == 3 and index + 1 < len(decisions):
            required_offer_index = index + 1

    parity = _pre_intervention_parity(
        baseline=baseline,
        challenger=decisions,
        offer_index=offer_index,
    )
    if parity["passed"] is not True:
        raise ValueError("G1 pre-intervention V2 parity failed")
    if offer_index is None or offer_record is None:
        return {
            "passed": True,
            "decision_count": len(decisions),
            "offered": False,
            "executed": False,
            "unavailable": False,
            "offer_decision_index": None,
            "pre_intervention_v2_parity": parity,
            "baseline_counterfactual": None,
            "immediate_rescue_repeat_exact_rollback": None,
        }
    if offer_index < 3:
        raise ValueError("G1 rescue offer has fewer than three trigger rollbacks")
    trigger_rows = decisions[offer_index - 3 : offer_index]
    trigger_fingerprint = str(decisions[offer_index]["before_platform_signature"])
    if not all(
        _is_exact_rollback(row)
        and str(row["before_platform_signature"]) == trigger_fingerprint
        and str(row["after_platform_signature"]) == trigger_fingerprint
        for row in trigger_rows
    ):
        raise ValueError("G1 rescue trigger is not three same-fingerprint exact rollbacks")
    for offset, row in enumerate(trigger_rows):
        record = dict(
            dict(row["controller"])["v2_first_single_family_rescue"]
        )
        trigger_selection = dict(record["selection"])
        trigger_observation = dict(record["observation"])
        if (
            str(trigger_selection.get("selection_phase") or "") != "v2_only"
            or bool(trigger_selection.get("offered", False))
            or int(
                trigger_selection.get("consecutive_v2_exact_rollbacks", -1)
            )
            != offset
            or trigger_observation.get("v2_exact_conflict_bound_rollback")
            is not True
            or int(
                trigger_observation.get("consecutive_v2_exact_rollbacks", -1)
            )
            != offset + 1
            or bool(
                trigger_observation.get(
                    "rescue_scheduled_for_next_decision", False
                )
            )
            != (offset == 2)
        ):
            raise ValueError("G1 three-step trigger counter trace changed")
    if offer_index >= 4:
        previous = decisions[offer_index - 4]
        if (
            _is_exact_rollback(previous)
            and str(previous["before_platform_signature"]) == trigger_fingerprint
            and str(previous["after_platform_signature"]) == trigger_fingerprint
        ):
            raise ValueError("G1 rescue was not offered immediately after rollback three")
    selection = dict(offer_record["selection"])
    generation = offer_record["generation"]
    execution = offer_record["execution"]
    observation = dict(offer_record["observation"])
    if (
        str(selection.get("selection_phase") or "")
        != "single_family_rescue_due"
        or int(selection.get("consecutive_v2_exact_rollbacks", -1)) != 3
        or not isinstance(generation, Mapping)
    ):
        raise ValueError("G1 offer transition does not match the trigger contract")
    generation = dict(generation)
    if (
        generation.get("attempted") is not True
        or generation.get("offered") is not True
        or generation.get("consumed") is not True
        or int(generation.get("decision_index", -1)) != offer_index
        or str(generation.get("before_repair_fingerprint") or "")
        != trigger_fingerprint
    ):
        raise ValueError("G1 offer was not atomically generated and consumed")

    if offer_index >= len(baseline):
        raise ValueError("G1 paired V2 trace lacks the offer decision")
    baseline_row = baseline[offer_index]
    offer_row = decisions[offer_index]
    if (
        str(baseline_row["before_platform_signature"]) != trigger_fingerprint
        or int(baseline_row["before_conflicts"]) != int(offer_row["before_conflicts"])
    ):
        raise ValueError("G1 offer lacks a paired V2 counterfactual state")
    counterfactual = {
        "decision_index": offer_index,
        "before_platform_signature": trigger_fingerprint,
        "repeat_exact_rollback": _is_exact_rollback(baseline_row),
        "seed_parity": _seed_parity_projection(baseline_row)
        == _seed_parity_projection(offer_row),
    }

    challenger_present = bool(generation.get("challenger_present", False))
    if challenger_present:
        if not isinstance(execution, Mapping):
            raise ValueError("G1 available rescue has no execution record")
        execution = dict(execution)
        candidate_pool = list(dict(offer_row["controller"]).get("candidate_pool") or ())
        if len(candidate_pool) != 1:
            raise ValueError("G1 direct rescue pool is not unique")
        candidate = dict(candidate_pool[0])
        candidate_id = str(candidate.get("candidate_id") or "")
        if (
            generation.get("structural_selected") is not False
            or execution.get("structural_selected") is not True
            or execution.get("consumed") is not True
            or str(generation.get("candidate_id") or "") != candidate_id
            or str(execution.get("candidate_id") or "") != candidate_id
            or str(dict(offer_row["controller"]).get("selected_candidate_id") or "")
            != candidate_id
            or list(map(str, candidate.get("selection_families") or ()))
            != [family_label]
            or list(map(str, candidate.get("structpool_family_groups") or ()))
            != [family_group]
            or set(map(str, dict(candidate.get("proposal_count_by_family") or {})))
            != {family_label}
            or set(map(str, dict(candidate.get("selection_rank_by_family") or {})))
            != {family_label}
            or list(map(str, candidate.get("hybridstructpool_provenance") or ()))
            != ["structshell_equal_four_size", "v2_first_single_family_rescue"]
            or int(candidate.get("actual_size", -1))
            != len(list(candidate.get("agents") or ()))
            or not 1 <= int(candidate.get("actual_size", -1)) <= 16
            or list(map(int, candidate.get("agents") or ()))
            != list(map(int, offer_row["actual_action"].get("agents") or ()))
            or float(dict(offer_row["controller"]).get("inference_seconds", -1.0))
            != 0.0
            or str(observation.get("decision_mode") or "")
            != "single_family_rescue"
        ):
            raise ValueError("G1 direct rescue family/size/provenance contract changed")
        if counterfactual["seed_parity"] is not True:
            raise ValueError("G1 rescue/V2 counterfactual random or PP seed mismatch")
        executed = True
        unavailable = False
    else:
        if (
            execution is not None
            or generation.get("structural_selected") is not False
            or generation.get("available") is not False
            or str(observation.get("decision_mode") or "") != "v2_only"
            or str(
                dict(dict(offer_row["controller"]).get("proposal") or {}).get(
                    "v2_first_rescue_mode"
                )
                or ""
            )
            != "fresh_v2_after_unavailable"
            or _action_parity_projection(baseline_row)
            != _action_parity_projection(offer_row)
            or _candidate_pool_projection(baseline_row)
            != _candidate_pool_projection(offer_row)
        ):
            raise ValueError("G1 unavailable offer did not run fresh V2 parity")
        executed = False
        unavailable = True
    return {
        "passed": True,
        "decision_count": len(decisions),
        "offered": True,
        "offer_consumed": True,
        "executed": executed,
        "unavailable": unavailable,
        "offer_decision_index": offer_index,
        "trigger_fingerprint": trigger_fingerprint,
        "trigger_exact_rollback_count": 3,
        "pre_intervention_v2_parity": parity,
        "baseline_counterfactual": counterfactual,
        "immediate_rescue_repeat_exact_rollback": (
            _is_exact_rollback(offer_row) if executed else None
        ),
    }


def _pre_intervention_parity(
    baseline: list[dict[str, Any]],
    challenger: list[dict[str, Any]],
    offer_index: int | None,
) -> dict[str, Any]:
    limit = len(challenger) if offer_index is None else int(offer_index)
    if len(baseline) < limit or len(challenger) < limit:
        return {
            "checked_decision_count": limit,
            "passed": False,
            "first_mismatch_decision": min(len(baseline), len(challenger)),
        }
    for index in range(limit):
        if _action_parity_projection(baseline[index]) != _action_parity_projection(
            challenger[index]
        ):
            return {
                "checked_decision_count": limit,
                "passed": False,
                "first_mismatch_decision": index,
            }
    return {
        "checked_decision_count": limit,
        "passed": True,
        "first_mismatch_decision": None,
    }


def _baseline_counterfactual(
    collection_root: Path,
    manifest: Mapping[str, Any],
    rescue: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not bool(rescue.get("executed")):
        return None
    index = int(rescue["decision_index"])
    matches = [
        row
        for row in _decision_rows(collection_root, manifest)
        if int(row["decision_index"]) == index
    ]
    if len(matches) != 1:
        raise ValueError("G1 paired V2 trace lacks the rescue decision index")
    row = matches[0]
    if str(row["before_platform_signature"]) != str(
        rescue["before_platform_signature"]
    ):
        raise ValueError("G1 V2/rescue pre-intervention fingerprint mismatch")
    return {
        "decision_index": index,
        "before_platform_signature": str(row["before_platform_signature"]),
        "repeat_exact_rollback": _is_exact_rollback(row),
    }


def _relative_regression(baseline: float, challenger: float) -> float:
    if baseline <= 0.0:
        return 0.0 if challenger <= baseline else float("inf")
    return (challenger - baseline) / baseline


def evaluate_map_gate(
    *,
    baseline_success: bool,
    baseline_ttf: float,
    challenger_success: bool,
    challenger_ttf: float,
    maximum_regression: float = 0.20,
) -> dict[str, Any]:
    regression = _relative_regression(float(baseline_ttf), float(challenger_ttf))
    success_noninferior = bool(challenger_success or not baseline_success)
    ttf_within_gate = bool(regression <= float(maximum_regression))
    return {
        "success_noninferior": success_noninferior,
        "restricted_ttf_regression": regression,
        "restricted_ttf_within_gate": ttf_within_gate,
        "passed": bool(success_noninferior and ttf_within_gate),
    }


def evaluate_batch_a_arm(
    rows: Iterable[Mapping[str, Any]], gates: Mapping[str, Any]
) -> dict[str, Any]:
    selected = [dict(row) for row in rows]
    baseline_success = sum(bool(row["baseline_success"]) for row in selected)
    challenger_success = sum(bool(row["challenger_success"]) for row in selected)
    baseline_mean = statistics.fmean(float(row["baseline_ttf"]) for row in selected)
    challenger_mean = statistics.fmean(
        float(row["challenger_ttf"]) for row in selected
    )
    regressions = [float(row["map_gate"]["restricted_ttf_regression"]) for row in selected]
    triggered = [row for row in selected if bool(row["rescue"]["executed"])]
    treatment_repeats = [
        bool(row["rescue"]["repeat_exact_rollback"]) for row in triggered
    ]
    baseline_repeats = [
        bool(row["baseline_counterfactual"]["repeat_exact_rollback"])
        for row in triggered
    ]
    treatment_rate = (
        statistics.fmean(treatment_repeats) if treatment_repeats else None
    )
    baseline_rate = statistics.fmean(baseline_repeats) if baseline_repeats else None
    reduction = (
        baseline_rate - treatment_rate
        if baseline_rate is not None and treatment_rate is not None
        else None
    )
    checks = {
        "success_noninferior": challenger_success >= baseline_success,
        "mean_ttf_within_gate": _relative_regression(
            baseline_mean, challenger_mean
        )
        <= float(gates["maximum_mean_restricted_ttf_regression"]),
        "all_maps_within_gate": max(regressions, default=float("inf"))
        <= float(gates["maximum_any_map_restricted_ttf_regression"]),
        "minimum_triggered_maps": len(triggered)
        >= int(gates["minimum_rescue_triggered_map_count"]),
        "repeat_rollback_reduction": reduction is not None
        and reduction
        >= float(gates["minimum_post_trigger_repeat_rollback_rate_reduction"]),
    }
    return {
        "map_count": len(selected),
        "baseline_success_count": baseline_success,
        "challenger_success_count": challenger_success,
        "baseline_mean_restricted_ttf": baseline_mean,
        "challenger_mean_restricted_ttf": challenger_mean,
        "mean_restricted_ttf_regression": _relative_regression(
            baseline_mean, challenger_mean
        ),
        "maximum_map_restricted_ttf_regression": max(
            regressions, default=None
        ),
        "rescue_triggered_map_count": len(triggered),
        "rescue_unavailable_map_count": sum(
            bool(row["rescue"]["unavailable"]) for row in selected
        ),
        "baseline_post_trigger_repeat_rollback_rate": baseline_rate,
        "challenger_post_trigger_repeat_rollback_rate": treatment_rate,
        "post_trigger_repeat_rollback_rate_reduction": reduction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _group_result(
    config: Mapping[str, Any], output: Path, stage: str, group_id: str
) -> tuple[dict[str, Any] | None, list[str]]:
    items = [
        item
        for item in schedule(config, stage)
        if str(item["group_id"]) == group_id
    ]
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    manifests: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for item in items:
        controller = str(item["controller"])
        manifest = _manifest_row(output, stage, item)
        if manifest is None:
            errors.append(f"{group_id}/{controller}: missing manifest")
            continue
        summary, error = _valid_summary(manifest, wall)
        if error is not None or summary is None:
            errors.append(f"{group_id}/{controller}: {error}")
            continue
        manifests[controller] = manifest
        summaries[controller] = summary
    if len(summaries) != len(CONTROLLERS):
        return None, errors
    if (
        len({str(row.get("initial_fingerprint")) for row in summaries.values()})
        != 1
        or len({int(row.get("initial_conflicts", -1)) for row in summaries.values()})
        != 1
    ):
        errors.append(f"{group_id}: paired reset mismatch")

    baseline = summaries["v2_only"]
    baseline_item = next(
        item for item in items if item["controller"] == "v2_only"
    )
    baseline_root = _controller_root(output, stage, baseline_item)
    baseline_decisions = _decision_rows(baseline_root, manifests["v2_only"])
    if len(baseline_decisions) != int(baseline.get("repair_iterations", -1)):
        errors.append(f"{group_id}/v2_only: trace/repair count mismatch")

    arms: dict[str, dict[str, Any]] = {}
    for arm in RESCUE_ARMS:
        item = next(item for item in items if item["controller"] == arm)
        collection = _controller_root(output, stage, item)
        decisions = _decision_rows(collection, manifests[arm])
        if len(decisions) != int(summaries[arm].get("repair_iterations", -1)):
            errors.append(f"{group_id}/{arm}: trace/repair count mismatch")
        expected_profile = (
            "conflict_component" if arm == "component16_rescue" else "hotspot"
        )
        try:
            trace_audit = _v2_first_trace_audit(
                decisions, baseline_decisions, expected_profile
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{group_id}/{arm}: trace audit failed: {error}")
            return None, errors
        rescue = {
            "offered": bool(trace_audit["offered"]),
            "offer_consumed": bool(trace_audit.get("offer_consumed", False)),
            "generation_attempted": bool(trace_audit["offered"]),
            "challenger_present": bool(trace_audit["executed"]),
            "unavailable": bool(trace_audit["unavailable"]),
            "executed": bool(trace_audit["executed"]),
            "decision_index": trace_audit["offer_decision_index"],
            "before_platform_signature": trace_audit.get("trigger_fingerprint"),
            "repeat_exact_rollback": trace_audit[
                "immediate_rescue_repeat_exact_rollback"
            ],
        }
        parity = dict(trace_audit["pre_intervention_v2_parity"])
        counterfactual = trace_audit["baseline_counterfactual"]
        totals = dict(summaries[arm].get("controller_totals") or {})
        expected_offer = int(rescue["offered"])
        expected_executed = int(rescue["executed"])
        expected_unavailable = int(rescue["unavailable"])
        total_expectations = {
            "v2_first_rescue_trigger_count": expected_offer,
            "v2_first_rescue_offer_count": expected_offer,
            "v2_first_rescue_consumed_count": expected_offer,
            "v2_first_rescue_generation_attempt_count": expected_offer,
            "v2_first_rescue_unavailable_count": expected_unavailable,
            "v2_first_rescue_executed_count": expected_executed,
            "v2_first_rescue_structural_selected_count": expected_executed,
        }
        for field, expected in total_expectations.items():
            if int(totals.get(field, 0)) != expected:
                errors.append(
                    f"{group_id}/{arm}: {field} trace/total mismatch"
                )
        gate = evaluate_map_gate(
            baseline_success=bool(baseline["success"]),
            baseline_ttf=float(baseline["capped_wall_time_to_feasible"]),
            challenger_success=bool(summaries[arm]["success"]),
            challenger_ttf=float(
                summaries[arm]["capped_wall_time_to_feasible"]
            ),
            maximum_regression=float(
                config["performance_gates"]["per_map"][
                    "maximum_restricted_ttf_regression"
                ]
            ),
        )
        arms[arm] = {
            "baseline_success": bool(baseline["success"]),
            "challenger_success": bool(summaries[arm]["success"]),
            "baseline_ttf": float(baseline["capped_wall_time_to_feasible"]),
            "challenger_ttf": float(
                summaries[arm]["capped_wall_time_to_feasible"]
            ),
            "map_gate": gate,
            "rescue": rescue,
            "baseline_counterfactual": counterfactual,
            "pre_intervention_v2_parity": parity,
            "trace_audit": trace_audit,
            "candidate_generation_seconds": float(
                totals.get("candidate_generation_seconds", 0.0)
            ),
            "neighborhood_selection_seconds": float(
                totals.get("neighborhood_selection_seconds", 0.0)
            ),
            "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
        }
    return {
        "group_id": group_id,
        "family": str(_group(config, group_id)["family"]),
        "task_id": str(_group(config, group_id)["task"]),
        "solver_seed": SOLVER_SEED,
        "initial_fingerprint": str(baseline["initial_fingerprint"]),
        "initial_conflicts": int(baseline["initial_conflicts"]),
        "v2_only": {
            "success": bool(baseline["success"]),
            "restricted_ttf": float(baseline["capped_wall_time_to_feasible"]),
            "repair_iterations": int(baseline.get("repair_iterations", 0)),
        },
        "rescue_arms": arms,
    }, errors


def _completed_schedule_count(
    output: Path, stage: str, rows: Iterable[Mapping[str, Any]]
) -> int:
    return sum(_manifest_row(output, stage, item) is not None for item in rows)


def _batch_a_analysis(
    path: Path,
    config: Mapping[str, Any],
    output: Path,
    *,
    producer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    per_map: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    surviving = set(RESCUE_ARMS)
    cancelled: list[str] = []
    stop_group: str | None = None
    for index, group_id in enumerate(_STAGE_GROUPS["batch_a"]):
        result, group_errors = _group_result(config, output, "batch_a", group_id)
        if result is None:
            if stop_group is not None:
                cancelled.extend(_STAGE_GROUPS["batch_a"][index:])
                break
            errors.extend(group_errors)
            continue
        errors.extend(group_errors)
        per_map[group_id] = result
        surviving = {
            arm
            for arm in surviving
            if bool(result["rescue_arms"][arm]["map_gate"]["passed"])
        }
        if not surviving:
            stop_group = group_id
            cancelled.extend(_STAGE_GROUPS["batch_a"][index + 1 :])
            break

    arm_summaries: dict[str, dict[str, Any]] = {}
    independently_passing: list[str] = []
    if len(per_map) == len(_STAGE_GROUPS["batch_a"]):
        for arm in RESCUE_ARMS:
            summary = evaluate_batch_a_arm(
                [per_map[group]["rescue_arms"][arm] for group in _STAGE_GROUPS["batch_a"]],
                config["performance_gates"]["batch_a"],
            )
            arm_summaries[arm] = summary
            if summary["passed"]:
                independently_passing.append(arm)
    passed = bool(
        not errors
        and len(independently_passing)
        >= int(
            config["performance_gates"]["batch_a"][
                "minimum_independently_passing_arm_count"
            ]
        )
    )
    if passed:
        decision = "continue_batch_b_maze"
    elif stop_group is not None:
        decision = "stop_mainline_map_gate_failure"
    elif len(per_map) == len(_STAGE_GROUPS["batch_a"]):
        decision = "stop_mainline_batch_a_gate_failure"
    else:
        decision = "incomplete_or_integrity_failure"
    return {
        "schema": BATCH_A_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "fresh_seed_bounded_ttf_screen_batch_a",
        "integrity_passed": not errors,
        "errors": errors,
        "batch_a_passed": passed,
        "decision": decision,
        "stop_group_id": stop_group,
        "completed_group_count": len(per_map),
        "cancelled_group_ids": cancelled,
        "surviving_map_gate_arms": sorted(surviving),
        "independently_passing_arms": independently_passing,
        "cross_arm_best_selected": False,
        "per_map": per_map,
        "arm_summaries": arm_summaries,
        "solver_seed": SOLVER_SEED,
        "timed_workers": 1,
        "qualification_workers": 16,
        "qualification_included_in_ttf": False,
        "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
        "bootstrap": False,
        "auc_gate": False,
        "formal_promotion_allowed": False,
        "inputs": {"config_sha256": sha256_file(path)},
        "producer_identity": dict(producer) if producer is not None else None,
    }


def analyze_batch_a(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output = Path(output).resolve()
    stage_root = _stage_root(output, "batch_a")
    completed = load_completed_report(
        stage_root,
        status_filename=STATUS_FILENAME,
        report_filename=BATCH_A_REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=BATCH_A_REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    if stage_root.joinpath(STATUS_FILENAME).is_file():
        status = read_json(stage_root / STATUS_FILENAME)
        if isinstance(status, dict):
            terminal = _trusted_terminal_report(
                stage_root, status, BATCH_A_REPORT_FILENAME, BATCH_A_REPORT_SCHEMA
            )
            if terminal is not None:
                return terminal
    report = _batch_a_analysis(path, config, output, producer=producer)
    write_json(stage_root / BATCH_A_REPORT_FILENAME, report)
    return report


def _load_trusted_batch_a(
    path: Path,
    root: Path,
    output: Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_root = _stage_root(output, "batch_a")
    report = load_completed_report(
        stage_root,
        status_filename=STATUS_FILENAME,
        report_filename=BATCH_A_REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=BATCH_A_REPORT_SCHEMA,
        config_path=path,
    )
    if report is None:
        raise ValueError("G1 Batch A has no hash-verified completed report")
    status = read_json(stage_root / STATUS_FILENAME)
    if not isinstance(status, dict):
        raise ValueError("G1 Batch A status is not an object")
    current_producer = dict(producer) if producer is not None else _producer(root)
    report_path = stage_root / BATCH_A_REPORT_FILENAME
    config_hash = sha256_file(path)
    if (
        status.get("complete") is not True
        or int(status.get("total_schedule_entries", -1)) != 9
        or int(status.get("completed_schedule_entries", -1)) != 9
        or status.get("config_sha256") != config_hash
        or status.get("producer_identity") != current_producer
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("producer_identity") != current_producer
        or dict(report.get("inputs") or {}).get("config_sha256") != config_hash
        or report.get("batch_a_passed") is not True
        or int(report.get("completed_group_count", -1)) != 3
        or list(report.get("cancelled_group_ids") or ())
    ):
        raise ValueError("G1 Batch A trust chain or passing gate changed")
    return report, status


def _final_analysis(
    path: Path,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    *,
    producer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    batch_a, batch_a_status = _load_trusted_batch_a(
        path, root, output, producer=producer
    )
    batch_a_path = _stage_root(output, "batch_a") / BATCH_A_REPORT_FILENAME
    group_id = _STAGE_GROUPS["batch_b"][0]
    maze, errors = _group_result(config, output, "batch_b", group_id)
    independently_retained: list[str] = []
    arm_checks: dict[str, dict[str, Any]] = {}
    if maze is not None:
        for arm in map(str, batch_a["independently_passing_arms"]):
            batch_rows = [
                dict(batch_a["per_map"])[name]["rescue_arms"][arm]
                for name in _STAGE_GROUPS["batch_a"]
            ]
            maze_row = maze["rescue_arms"][arm]
            combined = batch_rows + [maze_row]
            baseline_mean = statistics.fmean(
                float(row["baseline_ttf"]) for row in combined
            )
            challenger_mean = statistics.fmean(
                float(row["challenger_ttf"]) for row in combined
            )
            checks = {
                "passed_batch_a": True,
                "maze_success_noninferior": bool(
                    maze_row["map_gate"]["success_noninferior"]
                ),
                "maze_ttf_within_gate": float(
                    maze_row["map_gate"]["restricted_ttf_regression"]
                )
                <= float(
                    config["performance_gates"]["batch_b"][
                        "maximum_maze_restricted_ttf_regression"
                    ]
                ),
                "all_map_mean_ttf_within_gate": _relative_regression(
                    baseline_mean, challenger_mean
                )
                <= float(
                    config["performance_gates"]["batch_b"][
                        "maximum_all_map_mean_restricted_ttf_regression"
                    ]
                ),
            }
            arm_checks[arm] = {
                "checks": checks,
                "baseline_all_map_mean_restricted_ttf": baseline_mean,
                "challenger_all_map_mean_restricted_ttf": challenger_mean,
                "all_map_mean_restricted_ttf_regression": _relative_regression(
                    baseline_mean, challenger_mean
                ),
                "passed": all(checks.values()),
            }
            if arm_checks[arm]["passed"]:
                independently_retained.append(arm)
    passed = bool(not errors and maze is not None and independently_retained)
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "fresh_seed_bounded_ttf_screen_complete",
        "integrity_passed": not errors,
        "errors": errors,
        "mainline_passed": passed,
        "decision": (
            "freeze_general_rescue_interface_then_start_fresh_warehouse_transfer"
            if passed
            else "stop_general_rescue_and_continue_warehouse_specialist"
        ),
        "batch_a_report_sha256": sha256_file(batch_a_path),
        "batch_a_status_sha256": sha256_file(
            _stage_root(output, "batch_a") / STATUS_FILENAME
        ),
        "batch_a_run_fingerprint": str(batch_a_status["run_fingerprint"]),
        "batch_a_independently_passing_arms": list(
            map(str, batch_a["independently_passing_arms"])
        ),
        "cross_arm_best_selected": False,
        "independently_retained_arms": independently_retained,
        "maze": maze,
        "arm_checks": arm_checks,
        "solver_seed": SOLVER_SEED,
        "timed_workers": 1,
        "qualification_workers": 16,
        "bootstrap": False,
        "auc_gate": False,
        "formal_promotion_allowed": False,
        "inputs": {"config_sha256": sha256_file(path)},
        "producer_identity": dict(producer) if producer is not None else None,
    }


def analyze_final(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    stage_root = _stage_root(output, "batch_b")
    completed = load_completed_report(
        stage_root,
        status_filename=STATUS_FILENAME,
        report_filename=FINAL_REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=FINAL_REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    effective_producer = dict(producer) if producer is not None else _producer(root)
    report = _final_analysis(
        path, root, config, output, producer=effective_producer
    )
    write_json(stage_root / FINAL_REPORT_FILENAME, report)
    return report


def _status_payload(
    base: Mapping[str, Any],
    output: Path,
    stage: str,
    rows: list[dict[str, Any]],
    *,
    complete: bool = False,
    terminal_scientific_stop: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = _completed_schedule_count(output, stage, rows)
    result = {
        **dict(base),
        "stage": stage,
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
        "terminal_scientific_stop": bool(terminal_scientific_stop),
        "timed_worker_count": 1,
        "qualification_worker_count": 16,
        "qualification_included_in_ttf": False,
    }
    if terminal_failure is not None:
        result["terminal_failure"] = dict(terminal_failure)
    return result


def _trusted_terminal_report(
    stage_root: Path,
    status: Mapping[str, Any],
    report_filename: str,
    report_schema: str,
) -> dict[str, Any] | None:
    if status.get("terminal_scientific_stop") is not True:
        return None
    report_path = stage_root / report_filename
    expected = status.get("report_sha256")
    if (
        not report_path.is_file()
        or not isinstance(expected, str)
        or sha256_file(report_path) != expected
    ):
        raise ValueError("G1 terminal scientific-stop report changed")
    report = read_json(report_path)
    if not isinstance(report, dict) or report.get("schema") != report_schema:
        raise ValueError("G1 terminal scientific-stop report schema changed")
    return report


def _run_stage(
    config_path: str | Path,
    output: str | Path,
    stage: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    if stage not in _STAGE_GROUPS:
        raise ValueError(f"unknown G1 stage: {stage}")
    if dry_run:
        full = plan(path)
        return dict(full[stage]) | {
            "schema": STATUS_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "solver_or_controller_invoked": False,
            "qualification_workers": 16,
            "timed_workers": 1,
        }
    output = Path(output).resolve()
    producer = _producer(root)
    if stage == "batch_b":
        try:
            _load_trusted_batch_a(
                path, root, output, producer=producer
            )
        except (KeyError, TypeError, ValueError) as error:
            return {
                "schema": STATUS_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "stage": stage,
                "blocked": True,
                "terminal_failure": "batch_a_trust_failed",
                "error": str(error),
            }

    rows = schedule(config, stage)
    stage_root = _stage_root(output, stage)
    report_filename = (
        BATCH_A_REPORT_FILENAME if stage == "batch_a" else FINAL_REPORT_FILENAME
    )
    report_schema = (
        BATCH_A_REPORT_SCHEMA if stage == "batch_a" else FINAL_REPORT_SCHEMA
    )
    prepared = prepare_resumable_output(
        stage_root,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=rows,
        producer=producer,
        resume=resume,
        report_filename=report_filename,
        report_schema=report_schema,
        label=f"G1 {stage}",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.resumed:
        terminal = _trusted_terminal_report(
            stage_root, prepared.status, report_filename, report_schema
        )
        if terminal is not None:
            return terminal
        if prepared.status.get("terminal_failure") is not None:
            return dict(prepared.status)

    # Each three-arm map block is atomic for scientific stopping: no map gate
    # is read until all three paired timed episodes have completed.
    surviving = set(RESCUE_ARMS)
    for group_id in _STAGE_GROUPS[stage]:
        already, existing_errors = _group_result(config, output, stage, group_id)
        if already is not None:
            surviving = {
                arm
                for arm in surviving
                if bool(already["rescue_arms"][arm]["map_gate"]["passed"])
            }
            if not surviving and stage == "batch_a":
                report = analyze_batch_a(path, output, producer=producer)
                status = _status_payload(
                    prepared.base_status,
                    output,
                    stage,
                    rows,
                    terminal_scientific_stop=True,
                )
                status["report_sha256"] = sha256_file(
                    stage_root / report_filename
                )
                write_json(stage_root / STATUS_FILENAME, status)
                return report
            continue
        if existing_errors and any("missing manifest" not in row for row in existing_errors):
            status = _status_payload(
                prepared.base_status,
                output,
                stage,
                rows,
                terminal_failure={
                    "phase": "existing_map_integrity",
                    "group_id": group_id,
                    "errors": existing_errors,
                },
            )
            write_json(stage_root / STATUS_FILENAME, status)
            return status
        group = _group(config, group_id)
        try:
            qualification = _qualify_group(
                root, output, config, group, resume=prepared.resumed
            )
        except Exception as error:
            status = _status_payload(
                prepared.base_status,
                output,
                stage,
                rows,
                terminal_failure={
                    "phase": "reset_only_qualification",
                    "group_id": group_id,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            write_json(stage_root / STATUS_FILENAME, status)
            return status

        for item in [row for row in rows if row["group_id"] == group_id]:
            if _manifest_row(output, stage, item) is not None:
                continue
            try:
                manifest = _run_episode(
                    root, output, config, stage, item, qualification
                )
            except Exception as error:
                status = _status_payload(
                    prepared.base_status,
                    output,
                    stage,
                    rows,
                    terminal_failure={
                        "phase": "timed_episode",
                        "item": dict(item),
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                write_json(stage_root / STATUS_FILENAME, status)
                return status
            if manifest.get("status") in {"error", "timeout"}:
                status = _status_payload(
                    prepared.base_status,
                    output,
                    stage,
                    rows,
                    terminal_failure={
                        "phase": "timed_episode",
                        "item": dict(item),
                        "error": str(manifest.get("error") or manifest.get("status")),
                    },
                )
                write_json(stage_root / STATUS_FILENAME, status)
                return status
            write_json(
                stage_root / STATUS_FILENAME,
                _status_payload(prepared.base_status, output, stage, rows),
            )

        completed, group_errors = _group_result(config, output, stage, group_id)
        if completed is None or group_errors:
            status = _status_payload(
                prepared.base_status,
                output,
                stage,
                rows,
                terminal_failure={
                    "phase": "atomic_map_gate",
                    "group_id": group_id,
                    "errors": group_errors,
                },
            )
            write_json(stage_root / STATUS_FILENAME, status)
            return status
        surviving = {
            arm
            for arm in surviving
            if bool(completed["rescue_arms"][arm]["map_gate"]["passed"])
        }
        if not surviving and stage == "batch_a":
            report = analyze_batch_a(path, output, producer=producer)
            status = _status_payload(
                prepared.base_status,
                output,
                stage,
                rows,
                terminal_scientific_stop=True,
            )
            status["report_sha256"] = sha256_file(stage_root / report_filename)
            write_json(stage_root / STATUS_FILENAME, status)
            return report

    report = (
        analyze_batch_a(path, output, producer=producer)
        if stage == "batch_a"
        else analyze_final(path, output, producer=producer)
    )
    status = _status_payload(
        prepared.base_status, output, stage, rows, complete=True
    )
    status["report_sha256"] = sha256_file(stage_root / report_filename)
    write_json(stage_root / STATUS_FILENAME, status)
    return report


def run_batch_a(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _run_stage(
        config_path, output, "batch_a", resume=resume, dry_run=dry_run
    )


def run_batch_b(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _run_stage(
        config_path, output, "batch_b", resume=resume, dry_run=dry_run
    )


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return plan(config_path)
    first = run_batch_a(config_path, output, resume=resume)
    if first.get("batch_a_passed") is not True:
        return first
    return run_batch_b(config_path, output, resume=resume)


__all__ = [
    "BATCH_A_REPORT_SCHEMA",
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "FINAL_REPORT_SCHEMA",
    "RESCUE_ARMS",
    "SOLVER_SEED",
    "STATUS_SCHEMA",
    "analyze_batch_a",
    "analyze_final",
    "controller_kwargs",
    "evaluate_batch_a_arm",
    "evaluate_map_gate",
    "load_config",
    "plan",
    "run",
    "run_batch_a",
    "run_batch_b",
    "schedule",
]
