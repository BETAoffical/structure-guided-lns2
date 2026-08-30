from __future__ import annotations

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
from experiments.stride_v2first_consensus16_warehouse_n800_supply import (
    MAXIMUM_SCREEN_TRACE_DECISIONS,
    PREFIX_V2_OUTCOMES,
    _screen_preaction_certificate,
    controller_kwargs as _n800_controller_kwargs,
)


CONFIG_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_config.v1"
STATUS_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_status.v1"
REPORT_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_report.v1"
EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-n700-supply-v1"
SCIENTIFIC_STATUS = "post_hoc_non_nested_exploratory_n700_state_supply_only"
V2_CONFIG_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_overlay.v2"
V2_STATUS_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_status.v2"
V2_REPORT_SCHEMA = "lns2.stride.v2first_consensus16_warehouse_n700_supply_report.v2"
V2_EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-n700-supply-v2"
V2_SCIENTIFIC_STATUS = (
    "post_v1_invalid_runner_resume_fix_preregistered_before_v2_controller"
)
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "screen_report.json"
TASK_IDS = (
    "w1020a__oe__t0233__n0700",
    "w1020a__oe__t0277__n0700",
    "w1020a__ur__t0233__n0700",
    "w1020a__ur__t0277__n0700",
)
SOLVER_SEEDS = (20, 21, 22)
SEED_PRIORITY = {
    TASK_IDS[0]: (20, 21, 22),
    TASK_IDS[1]: (21, 22, 20),
    TASK_IDS[2]: (22, 20, 21),
    TASK_IDS[3]: (20, 21, 22),
}
SCREEN_RUNTIME = {
    "wall_time_safety_fuse_seconds": 200.0,
    "environment_time_safety_fuse_seconds": 200.0,
    "episode_process_timeout_seconds": 300.0,
    "workers": 16,
    "included_in_ttf": False,
}
DEFAULT_OUTPUT = "build/stride-v2first-consensus16-warehouse-n700-supply-v1"
V2_DEFAULT_OUTPUT = "build/stride-v2first-consensus16-warehouse-n700-supply-v2"


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_v2first_consensus16_warehouse_n700_supply.py",
            "scripts/run_stride_v2first_consensus16_warehouse_n700_supply.py",
            "experiments/stride_v2first_consensus16_warehouse_n800_supply.py",
            "experiments/stride_v2first_consensus16_warehouse.py",
            "experiments/stride_v2first_family_rescue_g1.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "lns2_selector/runtime/v2_first_consensus_rescue.py",
            "lns2_selector/runtime/v2_first_single_family_rescue.py",
            "lns2_selector/runtime/topology_candidates.py",
        ),
        native_required=native_required,
    )


def _registered(root: Path, value: Any, label: str) -> Path:
    return registered_input(root, dict(value or {}), label=label)


def _is_v2(config: Mapping[str, Any]) -> bool:
    return str(dict(config.get("_profile") or {}).get("profile") or "v1") == "v2"


def _experiment_id(config: Mapping[str, Any]) -> str:
    return V2_EXPERIMENT_ID if _is_v2(config) else EXPERIMENT_ID


def _scientific_status(config: Mapping[str, Any]) -> str:
    return V2_SCIENTIFIC_STATUS if _is_v2(config) else SCIENTIFIC_STATUS


def _status_schema(config: Mapping[str, Any]) -> str:
    return V2_STATUS_SCHEMA if _is_v2(config) else STATUS_SCHEMA


def _report_schema(config: Mapping[str, Any]) -> str:
    return V2_REPORT_SCHEMA if _is_v2(config) else REPORT_SCHEMA


def _load_v2_overlay(
    path: Path, root: Path, overlay: Mapping[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    if (
        overlay.get("schema") != V2_CONFIG_SCHEMA
        or overlay.get("experiment_id") != V2_EXPERIMENT_ID
        or overlay.get("scientific_status") != V2_SCIENTIFIC_STATUS
        or overlay.get("pre_registration_parent_commit")
        != "4b0ffa7dabd341044230d5ad988b9d60a89d0a1c"
        or dict(overlay.get("output_identity") or {})
        != {
            "default_output": V2_DEFAULT_OUTPUT,
            "v1_output_resumable": False,
            "foreign_or_nested_output_allowed": False,
        }
        or dict(overlay.get("scientific_contract") or {})
        != {
            "base_cohort_controller_budget_selection_and_gate_changed": False,
            "runner_change": "realized_dynamic_recomputes_resume_after_same_root_qualification",
            "v1_controller_episode_count": 0,
            "v1_state_or_cache_reused": False,
            "v2_screen_fresh_reset_required": True,
            "formal_or_ttf_added": False,
        }
    ):
        raise ValueError("N700 supply v2 overlay identity changed")
    base_path = _registered(root, overlay.get("base_config"), "N700 v1 base config")
    if base_path != (root / "configs/stride_v2first_consensus16_warehouse_n700_supply_v1.json").resolve():
        raise ValueError("N700 supply v2 base config changed")
    _base_path, _base_root, base = load_config(base_path)
    evidence = dict(overlay.get("v1_invalid_evidence") or {})
    status_path = _registered(root, evidence.get("status"), "N700 v1 INVALID status")
    expected_status = (
        root
        / "build/stride-v2first-consensus16-warehouse-n700-supply-v1/screen/collection_status.json"
    ).resolve()
    if (
        status_path != expected_status
        or evidence.get("decision_status") != "INVALID"
        or int(evidence.get("completed_schedule_entries", -1)) != 0
        or int(evidence.get("controller_episode_count", -1)) != 0
        or evidence.get("resumption_allowed") is not False
    ):
        raise ValueError("N700 supply v1 INVALID evidence identity changed")
    status = read_json(status_path)
    terminal = dict(status.get("terminal") or {}) if isinstance(status, Mapping) else {}
    if (
        not isinstance(status, dict)
        or status.get("schema") != STATUS_SCHEMA
        or status.get("decision_status") != "INVALID"
        or status.get("complete") is not False
        or int(status.get("completed_schedule_entries", -1)) != 0
        or str(status.get("config_sha256") or "")
        != str(dict(overlay["base_config"])["sha256"])
        or terminal.get("phase") != "prefix_collection"
        or "output already exists; pass resume to continue"
        not in str(terminal.get("error") or "")
    ):
        raise ValueError("N700 supply v1 INVALID status changed")
    realized_manifest = (
        root
        / "build/stride-v2first-consensus16-warehouse-n700-supply-v1/screen/controller/consensus16_probe/realized_dynamic_manifest.jsonl"
    )
    if realized_manifest.exists():
        raise ValueError("N700 supply v1 unexpectedly contains controller outcomes")
    result = dict(base)
    result.update(
        {
            "schema": V2_CONFIG_SCHEMA,
            "experiment_id": V2_EXPERIMENT_ID,
            "scientific_status": V2_SCIENTIFIC_STATUS,
            "output_identity": dict(overlay["output_identity"]),
            "v1_invalid_evidence": evidence,
            "scientific_contract": dict(overlay["scientific_contract"]),
            "_profile": {"profile": "v2", "base_config_path": str(base_path)},
        }
    )
    return path, root, result


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("N700 supply config must be an object")
    if config.get("schema") == V2_CONFIG_SCHEMA:
        return _load_v2_overlay(path, root, config)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or tuple(map(str, config.get("controllers") or ()))
        != ("v2_only", "consensus16_rescue")
        or dict(config.get("screen") or {})
        != {
            "role": "one_shot_non_ttf_state_supply_screen",
            "solver_seeds": list(SOLVER_SEEDS),
            "v2_repair_outcome_prefix_k": PREFIX_V2_OUTCOMES,
            "maximum_trace_decisions": MAXIMUM_SCREEN_TRACE_DECISIONS,
            **SCREEN_RUNTIME,
            "stopping_rule": "historical",
            "selection": {
                "maximum_selected_keys_per_task": 1,
                "required_selected_task_count": 4,
                "seed_priority_by_task": {
                    task: list(seeds) for task, seeds in SEED_PRIORITY.items()
                },
                "cross_task_backfill": False,
                "seed_replacement": False,
                "task_replacement": False,
                "prefix_extension": False,
                "timeout_extension": False,
            },
            "terminal_classification": {
                "timeout_or_incomplete_prefix": "INCONCLUSIVE_STATE_SUPPLY",
                "reset_or_trace_integrity_failure": "INVALID",
            },
        }
    ):
        raise ValueError("N700 supply experiment identity changed")

    controller_contract = dict(config.get("controller_contract") or {})
    if controller_contract != {
        "base": "frozen_v2_full_native_features_optimized_copeland",
        "trigger": {
            "minimum_consecutive_v2_exact_rollbacks": 3,
            "fingerprint_scope": "same_repair_fingerprint",
            "offer_on_next_decision": True,
            "maximum_rescue_offers_per_episode": 1,
            "maximum_executed_rescues_per_episode": 1,
            "post_offer_policy": "permanent_v2_only",
            "time_limit_triggers_rescue": False,
        },
        "consensus16_rescue": {
            "component_family": "conflict_component",
            "hotspot_family": "spatiotemporal_hotspot",
            "nominal_size": 16,
            "agreement": "nonempty_sorted_agent_set_exact_equality",
            "selection": "direct_unique_consensus_neighborhood_without_copeland",
            "no_consensus_fallback": "same_decision_fresh_v2_only",
            "generated_candidate_count": "unique_nonempty_family_candidate_ids",
            "maximum_pp_calls_per_decision": 1,
        },
    }:
        raise ValueError("N700 supply controller contract changed")

    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("dataset") != "build/stride-warehouse-n700-supply-v1/dataset"
        or cohort.get("split") != "balanced_wall_clock"
        or cohort.get("map_id") != "warehouse-10-20-10-2-1"
        or cohort.get("map_code") != "w1020a"
        or int(cohort.get("agent_count", -1)) != 700
        or tuple(map(int, cohort.get("candidate_solver_seeds") or ()))
        != SOLVER_SEEDS
        or cohort.get("result_based_task_or_seed_replacement") is not False
    ):
        raise ValueError("N700 supply cohort changed")
    tasks = [dict(row) for row in cohort.get("tasks") or ()]
    if tuple(str(row.get("id")) for row in tasks) != TASK_IDS:
        raise ValueError("N700 supply task order changed")
    if tuple(str(row.get("variant")) for row in tasks) != (
        "opposite_exchange",
        "opposite_exchange",
        "uniform_random",
        "uniform_random",
    ):
        raise ValueError("N700 supply task variants changed")
    if tuple(int(row.get("task_seed", -1)) for row in tasks) != (233, 277, 233, 277):
        raise ValueError("N700 supply task seeds changed")
    for task in tasks:
        _registered(root, task.get("task_file"), f"N700 task {task['id']}")
        _registered(root, task.get("scenario_file"), f"N700 scenario {task['id']}")

    inputs = dict(config.get("inputs") or {})
    dataset_config_path = _registered(
        root, inputs.get("dataset_config"), "N700 dataset generation config"
    )
    dataset_summary_path = _registered(
        root, inputs.get("dataset_summary"), "N700 dataset summary"
    )
    manifest_path = _registered(
        root, inputs.get("dataset_manifest"), "N700 dataset manifest"
    )
    _registered(root, inputs.get("dataset_q0"), "N700 geometry-only Q0 audit")
    _registered(root, inputs.get("map"), "N700 map")
    runtime_path = _registered(root, inputs.get("runtime_config"), "N700 runtime source")
    controller_manifest = _registered(
        root, inputs.get("controller_manifest"), "N700 V2 controller manifest"
    )
    if controller_manifest.parent != (root / str(config["controller_bundle"])).resolve():
        raise ValueError("N700 controller bundle changed")
    if dataset_config_path.name != "stride_warehouse_n700_supply_dataset_v1.json":
        raise ValueError("N700 dataset generation identity changed")
    summary = read_json(dataset_summary_path)
    split_summary = (
        dict(summary.get("splits") or {}).get("balanced_wall_clock")
        if isinstance(summary, Mapping)
        else None
    )
    if (
        not isinstance(summary, dict)
        or summary.get("schema")
        != "lns2.stride.warehouse_n700_supply_dataset.v1"
        or summary.get("experiment_id")
        != "stride-warehouse-n700-supply-dataset-v1"
        or summary.get("solver_or_controller_invoked") is not False
        or summary.get("task_semantics")
        != "geometry_only_derived_not_official_mapf_scenarios"
        or not isinstance(split_summary, Mapping)
        or int(dict(split_summary).get("instance_count", -1)) != 4
        or int(dict(split_summary).get("map_count", -1)) != 1
        or str(summary.get("config_sha256") or "")
        != str(dict(inputs["dataset_config"])["sha256"])
    ):
        raise ValueError("N700 dataset summary changed")
    manifest = [dict(row) for row in read_jsonl(manifest_path)]
    if (
        len(manifest) != 4
        or tuple(str(row.get("task_id")) for row in manifest) != TASK_IDS
        or any(
            int(row.get("agent_count", -1)) != 700
            or str(row.get("map_id")) != "warehouse-10-20-10-2-1"
            or str(row.get("instance_origin"))
            != "warehouse_n700_supply_derived_od"
            for row in manifest
        )
    ):
        raise ValueError("N700 dataset manifest changed")

    claim = dict(config.get("claim_boundary") or {})
    if claim != {
        "maximum_claim": "state_supply_feasibility_on_one_fixed_non_nested_post_hoc_w1020a_n700_slice",
        "non_nested_post_hoc_exploratory_slice": True,
        "load_curve_or_causal_interpolation_claim_allowed": False,
        "ttf_or_performance_claim_allowed": False,
        "formal_run_in_this_experiment_allowed": False,
        "future_formal_registration_requires_screen_4_of_4": True,
        "warehouse_generalization_allowed": False,
        "default_promotion_allowed": False,
    }:
        raise ValueError("N700 supply claim boundary changed")
    stop = dict(config.get("one_shot_stop_policy") or {})
    if stop != {
        "on_less_than_four_selected_tasks": "stop_intermediate_load_route",
        "n650_followup_allowed": False,
        "n750_followup_allowed": False,
        "additional_solver_seeds_allowed": False,
        "automatic_formal_allowed": False,
    }:
        raise ValueError("N700 supply one-shot stop policy changed")
    if dict(config.get("output_identity") or {}) != {
        "default_output": DEFAULT_OUTPUT,
        "n800_or_dataset_output_reuse_forbidden": True,
    }:
        raise ValueError("N700 supply output identity changed")

    result = dict(config)
    result["_runtime_source_path"] = str(runtime_path)
    result["_task_by_id"] = {str(row["id"]): row for row in tasks}
    result["_dataset_manifest_path"] = str(manifest_path)
    return path, root, result


def screen_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_by_id = dict(config["_task_by_id"])
    return [
        {
            "phase": "screen",
            "key_id": f"{task_id}@{seed}",
            "task_id": task_id,
            "task_variant": str(task_by_id[task_id]["variant"]),
            "task_seed": int(task_by_id[task_id]["task_seed"]),
            "solver_seed": seed,
            "controller": "consensus16_rescue",
            "selection_inputs": "pre_action_trigger_and_exact_consensus_only",
        }
        for task_id in TASK_IDS
        for seed in SOLVER_SEEDS
    ]


def _selection_rule() -> dict[str, Any]:
    return {
        "prefix_v2_outcomes": PREFIX_V2_OUTCOMES,
        "next_pre_action_offer_decision_allowed": True,
        "eligibility": (
            "three_same_fingerprint_exact_v2_rollbacks_then_exact_nonempty_"
            "component16_hotspot16_agent_consensus"
        ),
        "generated_candidate_count": "unique_nonempty_family_candidate_ids",
        "seed_priority_by_task": {
            task: list(seeds) for task, seeds in SEED_PRIORITY.items()
        },
        "maximum_selected_keys_per_task": 1,
        "cross_task_backfill": False,
        "seed_replacement": False,
        "task_replacement": False,
        "prefix_extension": False,
        "timeout_extension": False,
        "selection_fields_only": True,
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    schedule = screen_schedule(config)
    return {
        "schema": _status_schema(config),
        "experiment_id": _experiment_id(config),
        "scientific_status": _scientific_status(config),
        "task_ids": list(TASK_IDS),
        "solver_seeds": list(SOLVER_SEEDS),
        "candidate_key_count": len(schedule),
        "schedule_sha256": json_fingerprint(schedule),
        "v2_repair_outcome_prefix_k": PREFIX_V2_OUTCOMES,
        "maximum_trace_decisions": MAXIMUM_SCREEN_TRACE_DECISIONS,
        "wall_time_safety_fuse_seconds": 200.0,
        "environment_time_safety_fuse_seconds": 200.0,
        "episode_process_timeout_seconds": 300.0,
        "workers": 16,
        "included_in_ttf": False,
        "formal_or_ttf_available": False,
        "future_formal_requires_new_registration": True,
        "solver_or_controller_invoked": False,
        "non_nested_post_hoc_exploratory_slice": True,
        "runner_resume_fix_profile": _is_v2(config),
        "v1_controller_episode_count": 0 if _is_v2(config) else None,
        "default_output": str(config["output_identity"]["default_output"]),
    }


def controller_kwargs(
    root: Path, config: Mapping[str, Any], controller: str
) -> dict[str, Any]:
    return _n800_controller_kwargs(root, config, controller, phase="screen")


def _runtime_config_path(output: Path, config: Mapping[str, Any]) -> Path:
    payload = read_json(Path(str(config["_runtime_source_path"])).resolve())
    if not isinstance(payload, dict):
        raise ValueError("N700 runtime source must be an object")
    payload["split"] = "balanced_wall_clock"
    payload["solver_seeds"] = list(SOLVER_SEEDS)
    payload["repair_seed_policy"] = "episode_stream"
    payload["deterministic_pp_replay"] = False
    payload["max_decisions"] = MAXIMUM_SCREEN_TRACE_DECISIONS
    payload["metric_iteration_budget"] = MAXIMUM_SCREEN_TRACE_DECISIONS
    payload["wall_time_budget_seconds"] = 200.0
    payload["episode_process_timeout_seconds"] = 300.0
    payload["workers"] = 16
    payload["formal"] = False
    payload["dataset_design"] = {
        "mode": "balanced_wall_clock",
        "map_count": 1,
        "instance_count": 4,
        "source_counts": {"movingai": 4},
        "layout_counts": {"warehouse": 4},
    }
    environment = dict(payload.get("environment") or {})
    environment["time_limit"] = 200.0
    payload["environment"] = environment
    destination = output / "runtime_configs" / "screen_n700_seed20_22_wall0200_k64.json"
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError("materialized N700 screen runtime changed")
    else:
        write_json(destination, payload)
    return destination


def _screen_root(output: Path) -> Path:
    return output / "screen"


def _validate_output_identity(
    root: Path, config: Mapping[str, Any], output: Path
) -> None:
    expected = (root / str(config["output_identity"]["default_output"])).resolve()
    if output.resolve() != expected:
        raise ValueError(
            "N700 supply screen requires its registered isolated output root: "
            f"{config['output_identity']['default_output']}"
        )


def _qualification_root(output: Path) -> Path:
    return _screen_root(output) / "reset_qualification"


def _controller_root(output: Path) -> Path:
    return _screen_root(output) / "controller" / "consensus16_probe"


def _job_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in rows
    }


def _manifest_row(
    collection: Path, task_id: str, solver_seed: int
) -> dict[str, Any] | None:
    path = collection / "realized_dynamic_manifest.jsonl"
    rows = read_jsonl(path) if path.is_file() else []
    matches = [
        dict(row)
        for row in rows
        if str(row.get("task_id")) == task_id
        and int(row.get("solver_seed", -1)) == solver_seed
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous N700 manifest row for {task_id}@{solver_seed}")
    return matches[0] if matches else None


def _completed_count(output: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    collection = _controller_root(output)
    return sum(
        _manifest_row(collection, str(row["task_id"]), int(row["solver_seed"]))
        is not None
        for row in rows
    )


def _reset_qualification(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    keys: set[tuple[str, int]],
    *,
    resume: bool,
) -> Path:
    destination = _qualification_root(output)
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config),
        destination,
        phase="qualify",
        workers=16,
        resume=resume and destination.joinpath("run_config.json").is_file(),
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_process_timeout_seconds=300.0,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "v2_only"),
    )
    report = read_json(destination / "qualification_report.json")
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or int(report.get("valid_count", -1)) != 12
        or int(report.get("incomplete_reset_count", -1)) != 0
    ):
        raise RuntimeError("N700 fresh reset qualification failed")
    return destination


def _run_screen_jobs(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    keys: set[tuple[str, int]],
    qualification: Path,
) -> None:
    collection = _controller_root(output)
    runtime_config = _runtime_config_path(output, config)
    run_config = collection / "run_config.json"
    common = dict(
        workers=16,
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        qualification_process_timeout_seconds=300.0,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "consensus16_rescue"),
    )
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        runtime_config,
        collection,
        phase="qualify",
        resume=run_config.is_file(),
        **common,
    )
    resume_realized = run_config.is_file()
    if not resume_realized:
        raise RuntimeError(
            "N700 qualification completed without materializing run_config.json"
        )
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        runtime_config,
        collection,
        phase="realized_dynamic",
        resume=resume_realized,
        **common,
    )


def _candidate_result(
    config: Mapping[str, Any], output: Path, item: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    manifest = _manifest_row(_controller_root(output), task_id, seed)
    if manifest is None:
        return None, f"{task_id}@{seed}: missing screen manifest", None
    status = str(manifest.get("status") or "")
    if status == "timeout":
        return None, f"{task_id}@{seed}: process timeout", None
    if status != "ok":
        return None, None, f"{task_id}@{seed}: manifest status is {status}"
    try:
        decisions = _decision_rows(_controller_root(output), manifest)
        certificate = _screen_preaction_certificate(decisions)
    except (KeyError, TypeError, ValueError) as error:
        return None, None, f"{task_id}@{seed}: trace integrity failed: {error}"
    if (
        not bool(certificate["triggered_within_k"])
        and len(decisions) < MAXIMUM_SCREEN_TRACE_DECISIONS
    ):
        summary = manifest.get("summary")
        complete_early_feasible = (
            isinstance(summary, Mapping)
            and str(dict(summary).get("stop_reason") or "") == "success"
            and dict(summary).get("truncated") is False
            and dict(summary).get("external_timeout") is False
        )
        if not complete_early_feasible:
            return (
                None,
                f"{task_id}@{seed}: incomplete K64 prefix before a pre-action offer",
                None,
            )
    task = dict(config["_task_by_id"])[task_id]
    return (
        {
            "key_id": f"{task_id}@{seed}",
            "task_id": task_id,
            "task_variant": str(task["variant"]),
            "task_seed": int(task["task_seed"]),
            "solver_seed": seed,
            **certificate,
        },
        None,
        None,
    )


def select_screen_keys(
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_key = {
        (str(row["task_id"]), int(row["solver_seed"])): dict(row)
        for row in candidates
    }
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in TASK_IDS:
        chosen = next(
            (
                by_key[(task_id, seed)]
                for seed in SEED_PRIORITY[task_id]
                if (task_id, seed) in by_key
                and bool(by_key[(task_id, seed)].get("eligible"))
            ),
            None,
        )
        if chosen is None:
            missing.append(task_id)
        else:
            selected.append(chosen)
    return selected, missing


def analyze_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    _validate_output_identity(root, config, output)
    stage = _screen_root(output)
    completed = load_completed_report(
        stage,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=_status_schema(config),
        report_schema=_report_schema(config),
        config_path=path,
    )
    if completed is not None:
        return completed

    integrity_errors: list[str] = []
    incomplete_keys: list[str] = []
    qualification_path = _qualification_root(output) / "qualification_report.json"
    qualification_sha256: str | None = None
    if not qualification_path.is_file():
        integrity_errors.append("fresh reset qualification report is missing")
    else:
        qualification_sha256 = sha256_file(qualification_path)
        qualification = read_json(qualification_path)
        if (
            not isinstance(qualification, dict)
            or qualification.get("passed") is not True
            or int(qualification.get("valid_count", -1)) != 12
            or int(qualification.get("incomplete_reset_count", -1)) != 0
        ):
            integrity_errors.append("fresh reset qualification is invalid")

    candidates: list[dict[str, Any]] = []
    for item in screen_schedule(config):
        candidate, incomplete, integrity = _candidate_result(config, output, item)
        if candidate is not None:
            candidates.append(candidate)
        if incomplete is not None:
            incomplete_keys.append(incomplete)
        if integrity is not None:
            integrity_errors.append(integrity)
    selected, missing = select_screen_keys(candidates)
    integrity_passed = not integrity_errors
    supply_passed = (
        integrity_passed
        and not incomplete_keys
        and not missing
        and len(selected) == 4
    )
    decision_status = (
        "INVALID"
        if not integrity_passed
        else "PASS_STATE_SUPPLY"
        if supply_passed
        else "INCONCLUSIVE_STATE_SUPPLY"
    )
    report = {
        "schema": _report_schema(config),
        "experiment_id": _experiment_id(config),
        "scientific_status": _scientific_status(config),
        "integrity_passed": integrity_passed,
        "integrity_errors": integrity_errors,
        "incomplete_key_count": len(incomplete_keys),
        "incomplete_keys": incomplete_keys,
        "decision_status": decision_status,
        "future_formal_registration_allowed": supply_passed,
        "formal_or_ttf_run_by_this_experiment": False,
        "candidate_key_count": len(candidates),
        "eligible_key_count": sum(bool(row["eligible"]) for row in candidates),
        "selected_key_count": len(selected),
        "missing_task_ids": missing,
        "candidates": candidates,
        "selected_keys": selected,
        "selection_rule": _selection_rule(),
        "selection_rule_sha256": json_fingerprint(_selection_rule()),
        "selected_keys_sha256": json_fingerprint(selected),
        "screen_schedule_sha256": json_fingerprint(screen_schedule(config)),
        "screen_workers": 16,
        "fresh_reset": True,
        "state_or_cache_reused": False,
        "included_in_ttf": False,
        "selection_fields_only": True,
        "task_or_seed_replacement": False,
        "non_nested_post_hoc_exploratory_slice": True,
        "load_curve_or_causal_interpolation_claim_allowed": False,
        "one_shot_stop_policy": dict(config["one_shot_stop_policy"]),
        "next_step": (
            "future_formal_requires_separate_preregistration"
            if supply_passed
            else "stop_without_n650_n750_or_additional_seeds"
        ),
        "inputs": {
            "config_sha256": sha256_file(path),
            "dataset_manifest_sha256": str(
                config["inputs"]["dataset_manifest"]["sha256"]
            ),
            "qualification_report_sha256": qualification_sha256,
        },
        "producer_identity": dict(producer) if producer is not None else None,
        "runner_resume_fix_profile": _is_v2(config),
        "v1_invalid_status_sha256": (
            str(config["v1_invalid_evidence"]["status"]["sha256"])
            if _is_v2(config)
            else None
        ),
    }
    write_json(stage / REPORT_FILENAME, report)
    return report


def _status(
    base: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    *,
    complete: bool = False,
    terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = _completed_count(output, rows)
    result = {
        **dict(base),
        "phase": "screen",
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
        "workers": 16,
        "included_in_ttf": False,
        "formal_or_ttf_available": False,
    }
    if terminal is not None:
        result["decision_status"] = str(terminal["decision_status"])
        result["terminal"] = dict(terminal)
    return result


def _timeout_like(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "timeout" in text or "time out" in text


def run_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    _validate_output_identity(root, config, output)
    if dry_run:
        return {**plan(path), "solver_or_controller_invoked": False}
    rows = screen_schedule(config)
    producer = _producer(root)
    stage = _screen_root(output)
    prepared = prepare_resumable_output(
        stage,
        status_filename=STATUS_FILENAME,
        status_schema=_status_schema(config),
        config_path=path,
        schedule=rows,
        producer=producer,
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=_report_schema(config),
        label="one-shot exploratory N700 non-TTF state-supply screen",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.status.get("terminal") is not None:
        return dict(prepared.status)
    keys = _job_keys(rows)
    try:
        qualification = _reset_qualification(
            root, output, config, keys, resume=prepared.resumed
        )
    except Exception as error:
        result = _status(
            prepared.base_status,
            output,
            rows,
            terminal={
                "decision_status": "INVALID",
                "phase": "fresh_reset_qualification",
                "error": f"{type(error).__name__}: {error}",
                "replacement_forbidden": True,
            },
        )
        write_json(stage / STATUS_FILENAME, result)
        return result
    try:
        _run_screen_jobs(root, output, config, keys, qualification)
    except Exception as error:
        decision = (
            "INCONCLUSIVE_STATE_SUPPLY" if _timeout_like(error) else "INVALID"
        )
        result = _status(
            prepared.base_status,
            output,
            rows,
            terminal={
                "decision_status": decision,
                "phase": "prefix_collection",
                "error": f"{type(error).__name__}: {error}",
                "replacement_or_extension_forbidden": True,
            },
        )
        write_json(stage / STATUS_FILENAME, result)
        return result
    report = analyze_screen(path, output, producer=producer)
    result = _status(prepared.base_status, output, rows, complete=True)
    result["decision_status"] = str(report["decision_status"])
    result["report_sha256"] = sha256_file(stage / REPORT_FILENAME)
    write_json(stage / STATUS_FILENAME, result)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_screen(config_path, output, resume=resume, dry_run=dry_run)


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "SCIENTIFIC_STATUS",
    "SEED_PRIORITY",
    "SOLVER_SEEDS",
    "STATUS_SCHEMA",
    "TASK_IDS",
    "analyze_screen",
    "load_config",
    "plan",
    "run",
    "run_screen",
    "screen_schedule",
    "select_screen_keys",
]
