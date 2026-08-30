from __future__ import annotations

import statistics
from collections import Counter
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
from experiments.stride_v2first_family_rescue_g1 import (
    _action_parity_projection,
    _candidate_pool_projection,
    _is_exact_rollback,
    _pre_intervention_parity,
    _seed_parity_projection,
    _valid_summary,
    controller_kwargs as _g1_controller_kwargs,
    evaluate_map_gate,
)


CONFIG_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_config.v1"
STATUS_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_status.v1"
STAGE_A_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_stage_a_report.v1"
)
FINAL_REPORT_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_final_report.v1"
EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-v1"
PRE_REGISTRATION_PARENT_COMMIT = "4b0ffa7dabd341044230d5ad988b9d60a89d0a1c"
CONTROLLERS = ("v2_only", "consensus16_rescue")
TASK_IDS = (
    "w1020a__oe__t0233__n0600",
    "w1020a__oe__t0277__n0600",
    "w1020a__ur__t0233__n0600",
    "w1020a__ur__t0277__n0600",
)
STAGE_SEEDS = {"stage_a": (27,), "stage_b": (28,)}
STATUS_FILENAME = "collection_status.json"
STAGE_A_REPORT_FILENAME = "stage_a_report.json"
FINAL_REPORT_FILENAME = "final_report.json"

_EXPECTED_RUNTIME = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 60.0,
    "environment_time_limit_seconds": 60.0,
    "episode_process_timeout_seconds": 90.0,
    "outer_job_timeout_seconds": 120.0,
    "timing_boundary": "reset_inclusive_ttf",
    "execution_order": "rotating_strict_two_controller_serial",
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

_EXPECTED_STAGE_A_GATES = {
    "minimum_executed_consensus_keys": 2,
    "minimum_executed_opposite_exchange_keys": 1,
    "minimum_executed_uniform_random_keys": 1,
    "success_count_must_not_be_below_v2": True,
    "mean_restricted_ttf_must_be_strictly_lower": True,
    "maximum_any_key_restricted_ttf_regression": 0.20,
    "minimum_post_trigger_repeat_rollback_rate_reduction": 0.15,
}

_EXPECTED_COMBINED_GATES = {
    "minimum_executed_consensus_keys": 4,
    "minimum_executed_consensus_keys_per_seed": 1,
    "minimum_executed_opposite_exchange_keys": 2,
    "minimum_executed_uniform_random_keys": 2,
    "success_count_must_not_be_below_v2": True,
    "mean_restricted_ttf_must_be_strictly_lower": True,
    "minimum_paired_faster_key_count": 5,
    "maximum_per_task_two_seed_mean_restricted_ttf_regression": 0.10,
    "maximum_any_key_restricted_ttf_regression": 0.20,
    "minimum_post_trigger_repeat_rollback_rate_reduction": 0.15,
    "maximum_any_seed_mean_restricted_ttf_regression": 0.10,
}


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_v2first_consensus16_warehouse.py",
            "scripts/run_stride_v2first_consensus16_warehouse.py",
            "experiments/stride_v2first_family_rescue_g1.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/runtime/v2_first_consensus_rescue.py",
            "lns2_selector/runtime/v2_first_single_family_rescue.py",
            "lns2_selector/runtime/structshell_single_family.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
        native_required=native_required,
    )


def _validate_controller_contract(config: Mapping[str, Any]) -> None:
    contract = dict(config.get("controller_contract") or {})
    trigger = dict(contract.get("trigger") or {})
    consensus = dict(contract.get("consensus16_rescue") or {})
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
            "offer_consumed_without_consensus": True,
            "post_offer_policy": "permanent_v2_only",
            "time_limit_triggers_rescue": False,
        }
        or consensus
        != {
            "component_family": "conflict_component",
            "hotspot_family": "spatiotemporal_hotspot",
            "nominal_size": 16,
            "generation": (
                "generate_component16_and_hotspot16_on_the_due_decision_only"
            ),
            "agreement": "nonempty_sorted_agent_set_exact_equality",
            "selection": "direct_unique_consensus_neighborhood_without_copeland",
            "no_consensus_fallback": "same_decision_fresh_v2_only",
            "maximum_pp_calls_per_decision": 1,
        }
        or contract.get("full_v2_pool_unchanged_before_trigger") is not True
        or contract.get("family_candidates_never_enter_v2_copeland_pool") is not True
        or contract.get("gcbs_retry_plateau_router_or_new_ranker") is not False
    ):
        raise ValueError("Warehouse consensus16 controller contract changed")


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Warehouse consensus16 config must be an object")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != (
            "preregistered_w1020a_existing_task_fresh_solver_seed_"
            "consensus_rescue_screen"
        )
        or config.get("pre_registration_parent_commit")
        != PRE_REGISTRATION_PARENT_COMMIT
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or dict(config.get("runtime") or {}) != _EXPECTED_RUNTIME
    ):
        raise ValueError("Warehouse consensus16 experiment identity changed")
    _validate_controller_contract(config)

    stages = dict(config.get("stages") or {})
    for stage, seeds in STAGE_SEEDS.items():
        value = dict(stages.get(stage) or {})
        if (
            tuple(map(int, value.get("solver_seeds") or ())) != seeds
            or int(value.get("paired_key_count", -1)) != 4
            or int(value.get("episode_count", -1)) != 8
            or bool(value.get("requires_prior_stage_pass")) != (stage == "stage_b")
        ):
            raise ValueError(f"Warehouse consensus16 {stage} schedule changed")

    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("role")
        != "existing_w1020a_tasks_fresh_solver_seed_warehouse_specialist_screen"
        or cohort.get("dataset") != "build/wh-f16-v2-r2/dataset"
        or cohort.get("split") != "balanced_wall_clock"
        or cohort.get("map_id") != "warehouse-10-20-10-2-1"
        or cohort.get("map_code") != "w1020a"
        or int(cohort.get("agent_count", -1)) != 600
        or dict(cohort.get("fresh_solver_seed_halves") or {})
        != {"A": [27], "B": [28]}
        or cohort.get("result_based_task_filtering") is not False
        or cohort.get("task_replacement_after_low_trigger_coverage") is not False
    ):
        raise ValueError("Warehouse consensus16 cohort changed")
    tasks = [dict(row) for row in cohort.get("tasks") or ()]
    if tuple(str(row.get("id")) for row in tasks) != TASK_IDS:
        raise ValueError("Warehouse consensus16 task order or identity changed")
    variants = tuple(str(row.get("variant")) for row in tasks)
    seeds = tuple(int(row.get("task_seed", -1)) for row in tasks)
    if variants != (
        "opposite_exchange",
        "opposite_exchange",
        "uniform_random",
        "uniform_random",
    ) or seeds != (233, 277, 233, 277):
        raise ValueError("Warehouse consensus16 task factors changed")
    for row in tasks:
        registered_input(
            root,
            dict(row.get("task_file") or {}),
            label=f"Warehouse consensus16 task {row['id']}",
        )
        registered_input(
            root,
            dict(row.get("scenario_file") or {}),
            label=f"Warehouse consensus16 scenario {row['id']}",
        )

    audit = dict(config.get("solver_seed_identity_audit") or {})
    if audit != {
        "candidate_seeds": [27, 28],
        "selected_seeds": [27, 28],
        "audit_scope": (
            "four_exact_w1020a_task_ids_registered_configs_and_manifest_rows_only"
        ),
        "global_freshness_scan": False,
        "outcome_fields_read": False,
        "registered_config_match_count": 0,
        "completed_manifest_match_count": 0,
        "completed_before_registration": True,
    }:
        raise ValueError("Warehouse consensus16 solver-seed audit changed")

    gates = dict(config.get("performance_gates") or {})
    if (
        dict(gates.get("stage_a") or {}) != _EXPECTED_STAGE_A_GATES
        or dict(gates.get("combined") or {}) != _EXPECTED_COMBINED_GATES
        or gates.get("insufficient_trigger_coverage_status") != "INCONCLUSIVE"
    ):
        raise ValueError("Warehouse consensus16 performance gates changed")

    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "screen_only": True,
        "existing_tasks_fresh_solver_seeds": True,
        "formal_promotion_allowed": False,
        "warehouse_generalization_allowed": False,
        "maximum_claim": "w1020a_four_task_two_fresh_solver_seed_narrow_slice",
        "bootstrap": False,
        "auc_gate": False,
        "official_adaptive_arm": False,
        "gcbs_allowed": False,
        "compactcut_allowed_before_combined_pass": False,
    }:
        raise ValueError("Warehouse consensus16 claim boundary changed")

    inputs = dict(config.get("inputs") or {})
    runtime_path = registered_input(
        root,
        dict(inputs.get("runtime_config") or {}),
        label="Warehouse consensus16 runtime source",
    )
    manifest_path = registered_input(
        root,
        dict(inputs.get("dataset_manifest") or {}),
        label="Warehouse consensus16 dataset manifest",
    )
    registered_input(
        root,
        dict(inputs.get("map") or {}),
        label="Warehouse consensus16 map",
    )
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="Warehouse consensus16 V2 controller manifest",
    )
    if controller_manifest.parent != (root / str(config["controller_bundle"])).resolve():
        raise ValueError("Warehouse consensus16 controller bundle changed")
    runtime_source = read_json(runtime_path)
    if (
        not isinstance(runtime_source, dict)
        or runtime_source.get("split") != "balanced_wall_clock"
        or int(dict(runtime_source.get("dataset_design") or {}).get("instance_count", -1))
        != 4
    ):
        raise ValueError("Warehouse consensus16 runtime source changed")
    manifest = read_jsonl(manifest_path)
    if (
        len(manifest) != 4
        or tuple(str(row.get("task_id")) for row in manifest) != TASK_IDS
        or any(int(row.get("agent_count", -1)) != 600 for row in manifest)
        or any(str(row.get("map_id")) != "warehouse-10-20-10-2-1" for row in manifest)
    ):
        raise ValueError("Warehouse consensus16 dataset manifest changed")
    result = dict(config)
    result["_runtime_source_path"] = str(runtime_path)
    result["_task_by_id"] = {str(row["id"]): row for row in tasks}
    return path, root, result


def schedule(config: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in STAGE_SEEDS:
        raise ValueError(f"unknown Warehouse consensus16 stage: {stage}")
    task_by_id = dict(config["_task_by_id"])
    rows: list[dict[str, Any]] = []
    all_keys = [
        (task_id, seed)
        for seed in STAGE_SEEDS["stage_a"] + STAGE_SEEDS["stage_b"]
        for task_id in TASK_IDS
    ]
    for seed in STAGE_SEEDS[stage]:
        for task_id in TASK_IDS:
            global_index = all_keys.index((task_id, seed))
            # Four keys per seed is an even block, so a plain global rotation
            # would give each task the same arm order under both seeds.  Flip
            # Stage B once more to deconfound per-task cross-seed comparisons
            # from warmup or system drift without adding episodes.
            rotation_index = global_index + int(stage == "stage_b")
            order = CONTROLLERS[rotation_index % 2 :] + CONTROLLERS[
                : rotation_index % 2
            ]
            task = task_by_id[task_id]
            for position, controller in enumerate(order):
                rows.append(
                    {
                        "stage": stage,
                        "paired_key_index": global_index,
                        "within_key_position": position,
                        "key_id": f"{task_id}@{seed}",
                        "map_id": "warehouse-10-20-10-2-1",
                        "family": "warehouse",
                        "task_id": task_id,
                        "task_variant": str(task["variant"]),
                        "task_seed": int(task["task_seed"]),
                        "solver_seed": seed,
                        "controller": controller,
                    }
                )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    fuse = float(config["runtime"]["episode_process_timeout_seconds"])
    stages: dict[str, Any] = {}
    for stage in STAGE_SEEDS:
        rows = schedule(config, stage)
        stages[stage] = {
            "solver_seeds": list(STAGE_SEEDS[stage]),
            "schedule_entry_count": len(rows),
            "schedule_sha256": json_fingerprint(rows),
            "maximum_registered_ttf_seconds": len(rows) * wall,
            "maximum_process_fuse_seconds": len(rows) * fuse,
            "blocked_until_stage_a_passes": stage == "stage_b",
        }
    return {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "controllers": list(CONTROLLERS),
        "task_ids": list(TASK_IDS),
        "stages": stages,
        "qualification": {"workers": 16, "included_in_ttf": False},
        "timed_episodes": {
            "workers": 1,
            "strict_serial": True,
            "included_in_ttf": True,
        },
        "maximum_all_stage_ttf_seconds": 16 * wall,
        "maximum_all_stage_process_fuse_seconds": 16 * fuse,
        "solver_or_controller_invoked": False,
        "bootstrap": False,
        "auc_gate": False,
        "global_scan": False,
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    result = _g1_controller_kwargs(root, config, "v2_only")
    if controller == "v2_only":
        return result
    if controller != "consensus16_rescue":
        raise ValueError(f"unknown Warehouse consensus16 controller: {controller}")
    from lns2_selector.runtime.v2_first_consensus_rescue import (
        validate_v2_first_consensus_rescue_augmentation,
        v2_first_consensus_rescue_augmentation,
    )

    augmentation = v2_first_consensus_rescue_augmentation(
        nominal_size=16,
        minimum_consecutive_v2_exact_rollbacks=3,
    )
    result["hybridstructpool_augmentation"] = (
        validate_v2_first_consensus_rescue_augmentation(augmentation)
    )
    return result


def _stage_root(output: Path, stage: str) -> Path:
    return output / stage


def _runtime_config_path(output: Path, config: Mapping[str, Any]) -> Path:
    payload = read_json(Path(str(config["_runtime_source_path"])).resolve())
    if not isinstance(payload, dict):
        raise ValueError("Warehouse consensus16 runtime source must be an object")
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    fuse = float(config["runtime"]["episode_process_timeout_seconds"])
    payload["split"] = "balanced_wall_clock"
    payload["solver_seeds"] = [27, 28]
    payload["wall_time_budget_seconds"] = wall
    payload["episode_process_timeout_seconds"] = fuse
    payload["workers"] = 1
    environment = dict(payload.get("environment") or {})
    environment["time_limit"] = wall
    payload["environment"] = environment
    destination = output / "runtime_configs" / "warehouse_seed27_28_wall0060.json"
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError("materialized Warehouse consensus16 runtime changed")
    else:
        write_json(destination, payload)
    return destination


def _stage_keys(stage: str) -> set[tuple[str, int]]:
    if stage not in STAGE_SEEDS:
        raise ValueError(f"unknown Warehouse consensus16 stage: {stage}")
    return {(task, seed) for task in TASK_IDS for seed in STAGE_SEEDS[stage]}


def _qualification_root(output: Path, stage: str) -> Path:
    return output / "qualification" / stage


def _controller_root(output: Path, stage: str, controller: str) -> Path:
    return _stage_root(output, stage) / "controllers" / controller


def _manifest_path(output: Path, stage: str, controller: str) -> Path:
    return _controller_root(output, stage, controller) / "realized_dynamic_manifest.jsonl"


def _manifest_row(
    output: Path, stage: str, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _manifest_path(output, stage, str(item["controller"]))
    rows = read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("Warehouse consensus16 manifest is ambiguous")
    return matches[0] if matches else None


def _qualify(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    stage: str,
    *,
    resume: bool,
) -> Path:
    qualification = _qualification_root(output, stage)
    dataset = root / str(config["cohort"]["dataset"])
    runtime = _runtime_config_path(output, config)
    keys = _stage_keys(stage)
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification,
        phase="qualify",
        workers=16,
        resume=resume and qualification.joinpath("run_config.json").is_file(),
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
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
        or int(report.get("valid_count", -1)) != 4
        or int(report.get("incomplete_reset_count", -1)) != 0
    ):
        raise RuntimeError("Warehouse consensus16 reset qualification failed")
    return qualification


def _prepare_controller_roots(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    stage: str,
    qualification: Path,
) -> None:
    dataset = root / str(config["cohort"]["dataset"])
    runtime = _runtime_config_path(output, config)
    stage_keys = _stage_keys(stage)
    for controller in CONTROLLERS:
        collection = _controller_root(output, stage, controller)
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            # This controller-local metadata pass must retain the same run
            # identity as its later timed calls, whose worker count is one.
            # The actual reset-only qualification above remains 16-worker.
            workers=1,
            resume=collection.joinpath("run_config.json").is_file(),
            task_ids=list(TASK_IDS),
            cohort_job_keys=stage_keys,
            job_keys=stage_keys,
            qualification_source=qualification,
            qualification_process_timeout_seconds=float(
                config["runtime"]["episode_process_timeout_seconds"]
            ),
            use_global_collection_lock=False,
            **controller_kwargs(root, config, controller),
        )


def _run_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    stage: str,
    item: Mapping[str, Any],
    qualification: Path,
) -> dict[str, Any]:
    key = {(str(item["task_id"]), int(item["solver_seed"]))}
    stage_keys = _stage_keys(stage)
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config),
        _controller_root(output, stage, str(item["controller"])),
        phase="realized_dynamic",
        workers=1,
        resume=True,
        task_ids=list(TASK_IDS),
        cohort_job_keys=stage_keys,
        job_keys=key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=float(
            config["runtime"]["episode_process_timeout_seconds"]
        ),
        use_global_collection_lock=False,
        **controller_kwargs(root, config, str(item["controller"])),
    )
    row = _manifest_row(output, stage, item)
    if row is None:
        raise RuntimeError("Warehouse consensus16 timed episode produced no manifest")
    return row


def _trace_record(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row["controller"]).get("v2_first_consensus_rescue")
    if not isinstance(value, Mapping):
        raise ValueError("decision lacks consensus-rescue trace record")
    result = dict(value)
    if not {"selection", "generation", "execution", "observation"} <= set(result):
        raise ValueError("consensus-rescue trace transition is incomplete")
    return result


def _consensus_trace_audit(
    decisions: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed on the one-shot C16/H16 exact-agent consensus contract."""

    if any(int(row["decision_index"]) != index for index, row in enumerate(decisions)):
        raise ValueError("consensus trace decision indices are not contiguous")
    offer_index: int | None = None
    offer_record: dict[str, Any] | None = None
    streak = 0
    streak_fingerprint: str | None = None
    required_offer_index: int | None = None
    for index, row in enumerate(decisions):
        record = _trace_record(row)
        selection = record.get("selection")
        observation = record.get("observation")
        if not isinstance(selection, Mapping) or not isinstance(observation, Mapping):
            raise ValueError("consensus selection/observation trace is incomplete")
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
            or str(selection.get("structural_profile") or "")
            != "component_hotspot_consensus"
            or int(selection.get("nominal_size", -1)) != 16
        ):
            raise ValueError(f"consensus decision {index} trace identity changed")
        offered = bool(selection.get("offered", False))
        if offered:
            if offer_index is not None:
                raise ValueError("consensus rescue was offered more than once")
            offer_index = index
            offer_record = record
        if required_offer_index == index and not offered:
            raise ValueError("consensus rescue missed the decision after rollback three")
        phase = str(selection.get("selection_phase") or "")
        if offer_index is None:
            if (
                phase != "v2_only"
                or record["generation"] is not None
                or record["execution"] is not None
                or str(observation.get("decision_mode") or "") != "v2_only"
                or str(
                    dict(dict(row["controller"]).get("proposal") or {}).get(
                        "v2_first_rescue_mode"
                    )
                    or ""
                )
                != "v2_only"
            ):
                raise ValueError(f"consensus decision {index} altered the V2 prefix")
        elif index > offer_index:
            if (
                phase != "v2_only_after_offer"
                or offered
                or record["generation"] is not None
                or record["execution"] is not None
                or str(observation.get("decision_mode") or "") != "v2_only"
                or str(
                    dict(dict(row["controller"]).get("proposal") or {}).get(
                        "v2_first_rescue_mode"
                    )
                    or ""
                )
                != "v2_only_after_offer"
            ):
                raise ValueError("consensus rescue did not latch permanent V2-only mode")
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

    parity = _pre_intervention_parity(baseline, decisions, offer_index)
    if parity["passed"] is not True:
        raise ValueError("consensus pre-intervention V2 parity failed")
    if offer_index is None or offer_record is None:
        return {
            "passed": True,
            "decision_count": len(decisions),
            "offered": False,
            "executed": False,
            "no_consensus": False,
            "offer_decision_index": None,
            "pre_intervention_v2_parity": parity,
            "baseline_counterfactual": None,
            "immediate_rescue_repeat_exact_rollback": None,
        }
    if offer_index < 3 or offer_index >= len(baseline):
        raise ValueError("consensus offer lacks three triggers or paired V2 decision")
    trigger_fingerprint = str(decisions[offer_index]["before_platform_signature"])
    trigger_rows = decisions[offer_index - 3 : offer_index]
    if not all(
        _is_exact_rollback(row)
        and str(row["before_platform_signature"]) == trigger_fingerprint
        and str(row["after_platform_signature"]) == trigger_fingerprint
        for row in trigger_rows
    ):
        raise ValueError("consensus trigger is not three same-fingerprint exact rollbacks")
    for offset, row in enumerate(trigger_rows):
        record = _trace_record(row)
        selection = dict(record["selection"])
        observation = dict(record["observation"])
        if (
            int(selection.get("consecutive_v2_exact_rollbacks", -1)) != offset
            or observation.get("v2_exact_conflict_bound_rollback") is not True
            or int(observation.get("consecutive_v2_exact_rollbacks", -1))
            != offset + 1
            or bool(observation.get("rescue_scheduled_for_next_decision", False))
            != (offset == 2)
        ):
            raise ValueError("consensus three-step trigger counter changed")
    selection = dict(offer_record["selection"])
    generation_raw = offer_record["generation"]
    observation = dict(offer_record["observation"])
    execution_raw = offer_record["execution"]
    if (
        str(selection.get("selection_phase") or "") != "consensus_rescue_due"
        or int(selection.get("consecutive_v2_exact_rollbacks", -1)) != 3
        or not isinstance(generation_raw, Mapping)
    ):
        raise ValueError("consensus offer transition changed")
    generation = dict(generation_raw)
    component_agents = tuple(map(int, generation.get("component_agents") or ()))
    hotspot_agents = tuple(map(int, generation.get("hotspot_agents") or ()))
    exact_consensus = bool(generation.get("exact_agent_consensus", False))
    component_available = bool(generation.get("component_available", False))
    hotspot_available = bool(generation.get("hotspot_available", False))
    component_candidate_id = generation.get("component_candidate_id")
    hotspot_candidate_id = generation.get("hotspot_candidate_id")
    consensus_candidate_id = generation.get("consensus_candidate_id")
    generated_count = int(generation.get("generated_candidate_count", -1))
    if (
        generation.get("attempted") is not True
        or generation.get("offered") is not True
        or generation.get("consumed") is not True
        or int(generation.get("decision_index", -1)) != offer_index
        or str(generation.get("before_repair_fingerprint") or "")
        != trigger_fingerprint
        or component_available != bool(component_candidate_id is not None)
        or hotspot_available != bool(hotspot_candidate_id is not None)
        or component_available != bool(component_agents)
        or hotspot_available != bool(hotspot_agents)
        or not 0 <= generated_count <= 2
        or exact_consensus != bool(component_agents and component_agents == hotspot_agents)
        or (consensus_candidate_id is not None) != exact_consensus
        or generation.get("challenger_present") is not exact_consensus
        or generation.get("available") is not exact_consensus
        or generation.get("structural_selected") is not False
        or generation.get("consumed") is not True
    ):
        raise ValueError("consensus generation/equality contract changed")
    baseline_row = baseline[offer_index]
    offer_row = decisions[offer_index]
    if (
        str(baseline_row["before_platform_signature"]) != trigger_fingerprint
        or int(baseline_row["before_conflicts"]) != int(offer_row["before_conflicts"])
    ):
        raise ValueError("consensus offer lacks paired V2 counterfactual state")
    counterfactual = {
        "decision_index": offer_index,
        "before_platform_signature": trigger_fingerprint,
        "repeat_exact_rollback": _is_exact_rollback(baseline_row),
        "seed_parity": _seed_parity_projection(baseline_row)
        == _seed_parity_projection(offer_row),
    }
    if exact_consensus:
        if not isinstance(execution_raw, Mapping):
            raise ValueError("consensus neighborhood has no execution record")
        execution = dict(execution_raw)
        candidate_pool = list(dict(offer_row["controller"]).get("candidate_pool") or ())
        candidate = dict(candidate_pool[0]) if len(candidate_pool) == 1 else {}
        candidate_id = str(candidate.get("candidate_id") or "")
        if (
            len(candidate_pool) != 1
            or not candidate_id
            or str(consensus_candidate_id or "") != candidate_id
            or str(component_candidate_id or "") != candidate_id
            or str(hotspot_candidate_id or "") != candidate_id
            or str(generation.get("candidate_id") or "") != candidate_id
            or str(execution.get("candidate_id") or "") != candidate_id
            or str(
                dict(offer_row["controller"]).get("selected_candidate_id") or ""
            )
            != candidate_id
            or execution.get("structural_selected") is not True
            or execution.get("consumed") is not True
            or generation.get("fallback") is not None
            or sorted(map(int, offer_row["actual_action"].get("agents") or ()))
            != list(component_agents)
            or sorted(map(int, candidate.get("agents") or ()))
            != list(component_agents)
            or int(candidate.get("actual_size", -1))
            != len(list(candidate.get("agents") or ()))
            or not 1 <= int(candidate.get("actual_size", -1)) <= 16
            or set(map(str, candidate.get("selection_families") or ()))
            != {
                "structpool-conflict-component:16",
                "structpool-spatiotemporal-hotspot:16",
            }
            or set(map(str, candidate.get("structpool_family_groups") or ()))
            != {"conflict_component", "spatiotemporal_hotspot"}
            or list(map(str, candidate.get("hybridstructpool_provenance") or ()))
            != ["structshell_equal_four_size", "v2_first_consensus_rescue"]
            or counterfactual["seed_parity"] is not True
            or float(dict(offer_row["controller"]).get("inference_seconds", -1.0))
            != 0.0
            or str(observation.get("decision_mode") or "")
            != "consensus_rescue"
        ):
            raise ValueError("consensus direct execution contract changed")
        executed = True
        no_consensus = False
    else:
        if (
            execution_raw is not None
            or generation.get("candidate_id") is not None
            or generation.get("fallback") != "fresh_v2_after_no_consensus"
            or str(observation.get("decision_mode") or "") != "v2_only"
            or str(
                dict(dict(offer_row["controller"]).get("proposal") or {}).get(
                    "v2_first_rescue_mode"
                )
                or ""
            )
            != "fresh_v2_after_no_consensus"
            or _action_parity_projection(baseline_row)
            != _action_parity_projection(offer_row)
            or _candidate_pool_projection(baseline_row)
            != _candidate_pool_projection(offer_row)
        ):
            raise ValueError("no-consensus offer did not fall back to fresh V2 parity")
        executed = False
        no_consensus = True
    return {
        "passed": True,
        "decision_count": len(decisions),
        "offered": True,
        "offer_consumed": True,
        "executed": executed,
        "no_consensus": no_consensus,
        "offer_decision_index": offer_index,
        "trigger_fingerprint": trigger_fingerprint,
        "component_agents": list(component_agents),
        "hotspot_agents": list(hotspot_agents),
        "pre_intervention_v2_parity": parity,
        "baseline_counterfactual": counterfactual,
        "immediate_rescue_repeat_exact_rollback": (
            _is_exact_rollback(offer_row) if executed else None
        ),
    }


def _key_result(
    config: Mapping[str, Any], output: Path, stage: str, task_id: str, seed: int
) -> tuple[dict[str, Any] | None, list[str]]:
    items = [
        row
        for row in schedule(config, stage)
        if row["task_id"] == task_id and int(row["solver_seed"]) == seed
    ]
    wall = float(config["runtime"]["wall_time_budget_seconds"])
    manifests: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for item in items:
        controller = str(item["controller"])
        manifest = _manifest_row(output, stage, item)
        if manifest is None:
            errors.append(f"{task_id}@{seed}/{controller}: missing manifest")
            continue
        summary, error = _valid_summary(manifest, wall)
        if summary is None or error is not None:
            errors.append(f"{task_id}@{seed}/{controller}: {error}")
            continue
        manifests[controller] = manifest
        summaries[controller] = summary
    if len(summaries) != 2:
        return None, errors
    if (
        len({str(row.get("initial_fingerprint")) for row in summaries.values()}) != 1
        or len({int(row.get("initial_conflicts", -1)) for row in summaries.values()})
        != 1
    ):
        errors.append(f"{task_id}@{seed}: paired reset mismatch")

    baseline_root = _controller_root(output, stage, "v2_only")
    baseline_decisions = _decision_rows(baseline_root, manifests["v2_only"])
    challenger_root = _controller_root(output, stage, "consensus16_rescue")
    challenger_decisions = _decision_rows(
        challenger_root, manifests["consensus16_rescue"]
    )
    if len(baseline_decisions) != int(summaries["v2_only"].get("repair_iterations", -1)):
        errors.append(f"{task_id}@{seed}/v2_only: trace/repair count mismatch")
    if len(challenger_decisions) != int(
        summaries["consensus16_rescue"].get("repair_iterations", -1)
    ):
        errors.append(f"{task_id}@{seed}/consensus16_rescue: trace/repair mismatch")
    try:
        trace_audit = _consensus_trace_audit(challenger_decisions, baseline_decisions)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"{task_id}@{seed}: consensus trace audit failed: {error}")
        return None, errors
    totals = dict(summaries["consensus16_rescue"].get("controller_totals") or {})
    offered = int(bool(trace_audit["offered"]))
    executed = int(bool(trace_audit["executed"]))
    no_consensus = int(bool(trace_audit["no_consensus"]))
    for field, expected in {
        "v2_first_rescue_trigger_count": offered,
        "v2_first_rescue_offer_count": offered,
        "v2_first_rescue_consumed_count": offered,
        "v2_first_rescue_generation_attempt_count": offered,
        "v2_first_rescue_unavailable_count": no_consensus,
        "v2_first_rescue_executed_count": executed,
        "v2_first_rescue_structural_selected_count": executed,
    }.items():
        if int(totals.get(field, 0)) != expected:
            errors.append(f"{task_id}@{seed}: {field} trace/total mismatch")
    baseline = summaries["v2_only"]
    challenger = summaries["consensus16_rescue"]
    gate = evaluate_map_gate(
        baseline_success=bool(baseline["success"]),
        baseline_ttf=float(baseline["capped_wall_time_to_feasible"]),
        challenger_success=bool(challenger["success"]),
        challenger_ttf=float(challenger["capped_wall_time_to_feasible"]),
        maximum_regression=0.20,
    )
    task = dict(config["_task_by_id"])[task_id]
    return {
        "key_id": f"{task_id}@{seed}",
        "task_id": task_id,
        "task_variant": str(task["variant"]),
        "task_seed": int(task["task_seed"]),
        "solver_seed": seed,
        "initial_fingerprint": str(baseline["initial_fingerprint"]),
        "initial_conflicts": int(baseline["initial_conflicts"]),
        "baseline_success": bool(baseline["success"]),
        "challenger_success": bool(challenger["success"]),
        "baseline_ttf": float(baseline["capped_wall_time_to_feasible"]),
        "challenger_ttf": float(challenger["capped_wall_time_to_feasible"]),
        "key_gate": gate,
        "rescue": {
            "offered": bool(trace_audit["offered"]),
            "executed": bool(trace_audit["executed"]),
            "no_consensus": bool(trace_audit["no_consensus"]),
            "decision_index": trace_audit["offer_decision_index"],
            "repeat_exact_rollback": trace_audit[
                "immediate_rescue_repeat_exact_rollback"
            ],
        },
        "baseline_counterfactual": trace_audit["baseline_counterfactual"],
        "trace_audit": trace_audit,
        "candidate_generation_seconds": float(
            totals.get("candidate_generation_seconds", 0.0)
        ),
        "neighborhood_selection_seconds": float(
            totals.get("neighborhood_selection_seconds", 0.0)
        ),
        "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
    }, errors


def _repeat_rollback_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    executed = [dict(row) for row in rows if bool(row["rescue"]["executed"])]
    baseline_values = [
        bool(row["baseline_counterfactual"]["repeat_exact_rollback"])
        for row in executed
    ]
    challenger_values = [bool(row["rescue"]["repeat_exact_rollback"]) for row in executed]
    baseline_rate = statistics.fmean(baseline_values) if baseline_values else None
    challenger_rate = (
        statistics.fmean(challenger_values) if challenger_values else None
    )
    return {
        "executed_count": len(executed),
        "baseline_post_trigger_repeat_rollback_rate": baseline_rate,
        "challenger_post_trigger_repeat_rollback_rate": challenger_rate,
        "post_trigger_repeat_rollback_rate_reduction": (
            baseline_rate - challenger_rate
            if baseline_rate is not None and challenger_rate is not None
            else None
        ),
    }


def _relative_regression(baseline: float, challenger: float) -> float:
    if baseline <= 0.0:
        return 0.0 if challenger <= baseline else float("inf")
    return (challenger - baseline) / baseline


def evaluate_stage_a(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in rows]
    repeat = _repeat_rollback_metrics(selected)
    executed_by_variant = Counter(
        str(row["task_variant"])
        for row in selected
        if bool(row["rescue"]["executed"])
    )
    baseline_mean = statistics.fmean(float(row["baseline_ttf"]) for row in selected)
    challenger_mean = statistics.fmean(
        float(row["challenger_ttf"]) for row in selected
    )
    max_regression = max(
        float(row["key_gate"]["restricted_ttf_regression"]) for row in selected
    )
    reduction = repeat["post_trigger_repeat_rollback_rate_reduction"]
    coverage_checks = {
        "minimum_executed_consensus_keys": repeat["executed_count"] >= 2,
        "minimum_executed_opposite_exchange_keys": executed_by_variant[
            "opposite_exchange"
        ]
        >= 1,
        "minimum_executed_uniform_random_keys": executed_by_variant["uniform_random"]
        >= 1,
    }
    performance_checks = {
        "success_noninferior": sum(bool(row["challenger_success"]) for row in selected)
        >= sum(bool(row["baseline_success"]) for row in selected),
        "mean_ttf_strictly_lower": challenger_mean < baseline_mean,
        "all_keys_within_regression_gate": max_regression <= 0.20,
        "repeat_rollback_reduction": reduction is not None and reduction >= 0.15,
    }
    coverage_passed = all(coverage_checks.values())
    passed = coverage_passed and all(performance_checks.values())
    return {
        "paired_key_count": len(selected),
        "baseline_success_count": sum(bool(row["baseline_success"]) for row in selected),
        "challenger_success_count": sum(
            bool(row["challenger_success"]) for row in selected
        ),
        "baseline_mean_restricted_ttf": baseline_mean,
        "challenger_mean_restricted_ttf": challenger_mean,
        "mean_restricted_ttf_regression": _relative_regression(
            baseline_mean, challenger_mean
        ),
        "maximum_key_restricted_ttf_regression": max_regression,
        "executed_consensus_by_variant": dict(executed_by_variant),
        **repeat,
        "coverage_checks": coverage_checks,
        "performance_checks": performance_checks,
        "coverage_passed": coverage_passed,
        "passed": passed,
        "decision_status": "PASS" if passed else ("FAIL" if coverage_passed else "INCONCLUSIVE"),
    }


def evaluate_combined(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in rows]
    repeat = _repeat_rollback_metrics(selected)
    executed = [row for row in selected if bool(row["rescue"]["executed"])]
    executed_by_variant = Counter(str(row["task_variant"]) for row in executed)
    executed_by_seed = Counter(int(row["solver_seed"]) for row in executed)
    baseline_mean = statistics.fmean(float(row["baseline_ttf"]) for row in selected)
    challenger_mean = statistics.fmean(
        float(row["challenger_ttf"]) for row in selected
    )
    key_regressions = [
        float(row["key_gate"]["restricted_ttf_regression"]) for row in selected
    ]
    per_seed_regressions: dict[str, float] = {}
    for seed in (27, 28):
        subset = [row for row in selected if int(row["solver_seed"]) == seed]
        per_seed_regressions[str(seed)] = _relative_regression(
            statistics.fmean(float(row["baseline_ttf"]) for row in subset),
            statistics.fmean(float(row["challenger_ttf"]) for row in subset),
        )
    per_task_regressions: dict[str, float] = {}
    for task in TASK_IDS:
        subset = [row for row in selected if str(row["task_id"]) == task]
        per_task_regressions[task] = _relative_regression(
            statistics.fmean(float(row["baseline_ttf"]) for row in subset),
            statistics.fmean(float(row["challenger_ttf"]) for row in subset),
        )
    coverage_checks = {
        "minimum_executed_consensus_keys": len(executed) >= 4,
        "minimum_executed_consensus_keys_per_seed": all(
            executed_by_seed[seed] >= 1 for seed in (27, 28)
        ),
        "minimum_executed_opposite_exchange_keys": executed_by_variant[
            "opposite_exchange"
        ]
        >= 2,
        "minimum_executed_uniform_random_keys": executed_by_variant["uniform_random"]
        >= 2,
    }
    reduction = repeat["post_trigger_repeat_rollback_rate_reduction"]
    performance_checks = {
        "success_noninferior": sum(bool(row["challenger_success"]) for row in selected)
        >= sum(bool(row["baseline_success"]) for row in selected),
        "mean_ttf_strictly_lower": challenger_mean < baseline_mean,
        "minimum_paired_faster_keys": sum(
            float(row["challenger_ttf"]) < float(row["baseline_ttf"])
            for row in selected
        )
        >= 5,
        "per_task_two_seed_means_within_gate": max(per_task_regressions.values())
        <= 0.10,
        "all_keys_within_regression_gate": max(key_regressions) <= 0.20,
        "repeat_rollback_reduction": reduction is not None and reduction >= 0.15,
        "all_seed_means_within_gate": max(per_seed_regressions.values()) <= 0.10,
    }
    coverage_passed = all(coverage_checks.values())
    passed = coverage_passed and all(performance_checks.values())
    return {
        "paired_key_count": len(selected),
        "baseline_success_count": sum(bool(row["baseline_success"]) for row in selected),
        "challenger_success_count": sum(
            bool(row["challenger_success"]) for row in selected
        ),
        "baseline_mean_restricted_ttf": baseline_mean,
        "challenger_mean_restricted_ttf": challenger_mean,
        "mean_restricted_ttf_regression": _relative_regression(
            baseline_mean, challenger_mean
        ),
        "paired_faster_key_count": sum(
            float(row["challenger_ttf"]) < float(row["baseline_ttf"])
            for row in selected
        ),
        "maximum_key_restricted_ttf_regression": max(key_regressions),
        "per_seed_mean_restricted_ttf_regression": per_seed_regressions,
        "per_task_two_seed_mean_restricted_ttf_regression": per_task_regressions,
        "executed_consensus_by_seed": {
            str(seed): executed_by_seed[seed] for seed in (27, 28)
        },
        "executed_consensus_by_variant": dict(executed_by_variant),
        **repeat,
        "coverage_checks": coverage_checks,
        "performance_checks": performance_checks,
        "coverage_passed": coverage_passed,
        "passed": passed,
        "decision_status": "PASS" if passed else ("FAIL" if coverage_passed else "INCONCLUSIVE"),
    }


def _stage_rows(
    config: Mapping[str, Any], output: Path, stage: str
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for seed in STAGE_SEEDS[stage]:
        for task_id in TASK_IDS:
            result, current = _key_result(config, output, stage, task_id, seed)
            errors.extend(current)
            if result is not None:
                results.append(result)
    return results, errors


def _completed_schedule_count(
    output: Path, stage: str, rows: Iterable[Mapping[str, Any]]
) -> int:
    return sum(_manifest_row(output, stage, row) is not None for row in rows)


def _status_payload(
    base: Mapping[str, Any],
    output: Path,
    stage: str,
    rows: list[dict[str, Any]],
    *,
    complete: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = _completed_schedule_count(output, stage, rows)
    result = {
        **dict(base),
        "stage": stage,
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
        "timed_worker_count": 1,
        "qualification_worker_count": 16,
        "qualification_included_in_ttf": False,
    }
    if terminal_failure is not None:
        result["terminal_failure"] = dict(terminal_failure)
    return result


def analyze_stage_a(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output = Path(output).resolve()
    stage_root = _stage_root(output, "stage_a")
    completed = load_completed_report(
        stage_root,
        status_filename=STATUS_FILENAME,
        report_filename=STAGE_A_REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=STAGE_A_REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    rows, errors = _stage_rows(config, output, "stage_a")
    summary = evaluate_stage_a(rows) if len(rows) == 4 else None
    passed = bool(not errors and summary is not None and summary["passed"])
    report = {
        "schema": STAGE_A_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "fresh_seed27_w1020a_consensus16_stage_a",
        "integrity_passed": not errors,
        "errors": errors,
        "completed_paired_key_count": len(rows),
        "stage_a_passed": passed,
        "decision_status": (
            summary["decision_status"] if summary is not None and not errors else "INVALID"
        ),
        "decision": "continue_stage_b_seed28" if passed else "stop_before_stage_b",
        "per_key": {str(row["key_id"]): row for row in rows},
        "summary": summary,
        "solver_seed": 27,
        "timed_workers": 1,
        "qualification_workers": 16,
        "formal_promotion_allowed": False,
        "inputs": {"config_sha256": sha256_file(path)},
        "producer_identity": dict(producer) if producer is not None else None,
    }
    write_json(stage_root / STAGE_A_REPORT_FILENAME, report)
    return report


def _load_trusted_stage_a(
    path: Path,
    root: Path,
    output: Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_root = _stage_root(output, "stage_a")
    report = load_completed_report(
        stage_root,
        status_filename=STATUS_FILENAME,
        report_filename=STAGE_A_REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=STAGE_A_REPORT_SCHEMA,
        config_path=path,
    )
    if report is None:
        raise ValueError("Warehouse consensus16 Stage A has no completed report")
    status = read_json(stage_root / STATUS_FILENAME)
    if not isinstance(status, dict):
        raise ValueError("Warehouse consensus16 Stage A status is invalid")
    current_producer = dict(producer) if producer is not None else _producer(root)
    report_path = stage_root / STAGE_A_REPORT_FILENAME
    config_hash = sha256_file(path)
    if (
        status.get("complete") is not True
        or int(status.get("total_schedule_entries", -1)) != 8
        or int(status.get("completed_schedule_entries", -1)) != 8
        or status.get("config_sha256") != config_hash
        or status.get("producer_identity") != current_producer
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("producer_identity") != current_producer
        or dict(report.get("inputs") or {}).get("config_sha256") != config_hash
        or report.get("integrity_passed") is not True
        or report.get("stage_a_passed") is not True
        or report.get("decision_status") != "PASS"
        or int(report.get("completed_paired_key_count", -1)) != 4
        or dict(report.get("summary") or {}).get("passed") is not True
    ):
        raise ValueError("Warehouse consensus16 Stage A trust chain changed")
    return report, status


def analyze_final(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    stage_root = _stage_root(output, "stage_b")
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
    stage_a, stage_a_status = _load_trusted_stage_a(
        path, root, output, producer=effective_producer
    )
    stage_b_rows, errors = _stage_rows(config, output, "stage_b")
    stage_a_rows = [dict(row) for row in dict(stage_a["per_key"]).values()]
    combined_rows = stage_a_rows + stage_b_rows
    combined = evaluate_combined(combined_rows) if len(combined_rows) == 8 else None
    passed = bool(not errors and combined is not None and combined["passed"])
    decision_status = (
        combined["decision_status"] if combined is not None and not errors else "INVALID"
    )
    report = {
        "schema": FINAL_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "fresh_seed27_28_w1020a_consensus16_complete",
        "integrity_passed": not errors,
        "errors": errors,
        "warehouse_consensus_screen_passed": passed,
        "decision_status": decision_status,
        "decision": (
            "freeze_w1020a_consensus16_then_consider_compactcut_stage_c"
            if passed
            else (
                "stop_without_task_replacement_due_to_low_trigger_coverage"
                if decision_status == "INCONCLUSIVE"
                else "stop_warehouse_consensus16"
            )
        ),
        "stage_a_report_sha256": sha256_file(
            _stage_root(output, "stage_a") / STAGE_A_REPORT_FILENAME
        ),
        "stage_a_status_sha256": sha256_file(
            _stage_root(output, "stage_a") / STATUS_FILENAME
        ),
        "stage_a_run_fingerprint": str(stage_a_status["run_fingerprint"]),
        "stage_b_per_key": {str(row["key_id"]): row for row in stage_b_rows},
        "combined_summary": combined,
        "timed_workers": 1,
        "qualification_workers": 16,
        "claim_scope": "w1020a_four_task_two_fresh_solver_seed_narrow_slice",
        "warehouse_generalization_allowed": False,
        "formal_promotion_allowed": False,
        "inputs": {"config_sha256": sha256_file(path)},
        "producer_identity": effective_producer,
    }
    write_json(stage_root / FINAL_REPORT_FILENAME, report)
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
    if stage not in STAGE_SEEDS:
        raise ValueError(f"unknown Warehouse consensus16 stage: {stage}")
    if dry_run:
        return {
            **dict(plan(path)["stages"][stage]),
            "schema": STATUS_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "solver_or_controller_invoked": False,
            "qualification_workers": 16,
            "timed_workers": 1,
        }
    output = Path(output).resolve()
    producer = _producer(root)
    if stage == "stage_b":
        try:
            _load_trusted_stage_a(path, root, output, producer=producer)
        except (KeyError, TypeError, ValueError) as error:
            return {
                "schema": STATUS_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "stage": stage,
                "blocked": True,
                "terminal_failure": "stage_a_trust_or_gate_failed",
                "error": str(error),
            }
    rows = schedule(config, stage)
    stage_root = _stage_root(output, stage)
    report_filename = (
        STAGE_A_REPORT_FILENAME if stage == "stage_a" else FINAL_REPORT_FILENAME
    )
    report_schema = STAGE_A_REPORT_SCHEMA if stage == "stage_a" else FINAL_REPORT_SCHEMA
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
        label=f"Warehouse consensus16 {stage}",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.status.get("terminal_failure") is not None:
        return dict(prepared.status)
    try:
        qualification = _qualify(
            root, output, config, stage, resume=prepared.resumed
        )
        _prepare_controller_roots(root, output, config, stage, qualification)
    except Exception as error:
        status = _status_payload(
            prepared.base_status,
            output,
            stage,
            rows,
            terminal_failure={
                "phase": "reset_only_qualification",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        write_json(stage_root / STATUS_FILENAME, status)
        return status
    for item in rows:
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
    report = (
        analyze_stage_a(path, output, producer=producer)
        if stage == "stage_a"
        else analyze_final(path, output, producer=producer)
    )
    status = _status_payload(
        prepared.base_status, output, stage, rows, complete=True
    )
    status["report_sha256"] = sha256_file(stage_root / report_filename)
    write_json(stage_root / STATUS_FILENAME, status)
    return report


def run_stage_a(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _run_stage(
        config_path, output, "stage_a", resume=resume, dry_run=dry_run
    )


def run_stage_b(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _run_stage(
        config_path, output, "stage_b", resume=resume, dry_run=dry_run
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
    first = run_stage_a(config_path, output, resume=resume)
    if first.get("stage_a_passed") is not True:
        return first
    return run_stage_b(config_path, output, resume=resume)


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "FINAL_REPORT_SCHEMA",
    "STAGE_A_REPORT_SCHEMA",
    "STATUS_SCHEMA",
    "TASK_IDS",
    "analyze_final",
    "analyze_stage_a",
    "controller_kwargs",
    "evaluate_combined",
    "evaluate_stage_a",
    "load_config",
    "plan",
    "run",
    "run_stage_a",
    "run_stage_b",
    "schedule",
]
