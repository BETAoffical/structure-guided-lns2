from __future__ import annotations

import copy
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
from experiments.stride_v2first_consensus16_warehouse import (
    _consensus_trace_audit,
    _relative_regression,
    _repeat_rollback_metrics,
    _valid_summary,
    controller_kwargs as _warehouse_controller_kwargs,
)


CONFIG_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_config.v1"
)
STATUS_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_status.v1"
)
SCREEN_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_screen_report.v1"
)
FINAL_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_final_report.v1"
)
EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-n800-supply-v1"
V2_CONFIG_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_overlay.v2"
)
V2_STATUS_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_status.v2"
)
V2_SCREEN_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_screen_report.v2"
)
V2_FINAL_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_final_report.v2"
)
V2_EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-n800-supply-v2"
V3_CONFIG_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_overlay.v3"
)
V3_STATUS_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_status.v3"
)
V3_SCREEN_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_screen_report.v3"
)
V3_FINAL_REPORT_SCHEMA = (
    "lns2.stride.v2first_consensus16_warehouse_n800_supply_final_report.v3"
)
V3_EXPERIMENT_ID = "stride-v2first-consensus16-warehouse-n800-supply-v3"
PRE_REGISTRATION_PARENT_COMMIT = "4b0ffa7dabd341044230d5ad988b9d60a89d0a1c"
CONTROLLERS = ("v2_only", "consensus16_rescue")
TASK_IDS = (
    "w1020a__oe__t0233__n0800",
    "w1020a__oe__t0277__n0800",
    "w1020a__ur__t0233__n0800",
    "w1020a__ur__t0277__n0800",
)
CANDIDATE_SEEDS = (29, 30, 31, 32)
PREFIX_V2_OUTCOMES = 64
MAXIMUM_SCREEN_TRACE_DECISIONS = 65
STATUS_FILENAME = "collection_status.json"
SCREEN_REPORT_FILENAME = "screen_report.json"
FINAL_REPORT_FILENAME = "final_report.json"

_SEED_PRIORITY = {
    TASK_IDS[0]: (29, 30, 31, 32),
    TASK_IDS[1]: (30, 31, 32, 29),
    TASK_IDS[2]: (31, 32, 29, 30),
    TASK_IDS[3]: (32, 29, 30, 31),
}
_ARM_ORDER = {
    TASK_IDS[0]: ("v2_only", "consensus16_rescue"),
    TASK_IDS[1]: ("consensus16_rescue", "v2_only"),
    TASK_IDS[2]: ("v2_only", "consensus16_rescue"),
    TASK_IDS[3]: ("consensus16_rescue", "v2_only"),
}
_V2_CANDIDATE_SEEDS = (20, 21, 22)
_V2_SEED_PRIORITY = {
    TASK_IDS[0]: (20, 21, 22),
    TASK_IDS[1]: (21, 22, 20),
    TASK_IDS[2]: (22, 20, 21),
    TASK_IDS[3]: (20, 21, 22),
}
_V3_SCREEN_RUNTIME = {
    "wall_time_safety_fuse_seconds": 200.0,
    "environment_time_safety_fuse_seconds": 200.0,
    "episode_process_timeout_seconds": 300.0,
}
_V3_FORMAL_RUNTIME = {
    "wall_time_budget_seconds": 200.0,
    "environment_time_limit_seconds": 200.0,
    "episode_process_timeout_seconds": 300.0,
}
_EXPECTED_SCREEN = {
    "role": "non_ttf_state_supply_qualification",
    "solver_seeds": [29, 30, 31, 32],
    "v2_repair_outcome_prefix_k": 64,
    "maximum_trace_decisions": 65,
    "offer_observation_rule": (
        "read_first_pre_action_offer_at_decision_index_at_most_64_after_"
        "only_earlier_v2_outcomes"
    ),
    "workers": 16,
    "stopping_rule": "historical",
    "wall_time_safety_fuse_seconds": 60.0,
    "environment_time_safety_fuse_seconds": 60.0,
    "episode_process_timeout_seconds": 90.0,
    "included_in_ttf": False,
    "selection_read_whitelist": [
        "trace_integrity",
        "v2_first_consensus_rescue.selection",
        "v2_first_consensus_rescue.generation",
    ],
    "selection_read_blacklist": [
        "success",
        "capped_wall_time_to_feasible",
        "final_conflicts",
        "pp_replan_seconds",
        "offer_action_outcome",
        "post_offer_decisions",
    ],
    "eligibility": {
        "requires_three_same_fingerprint_exact_v2_rollbacks": True,
        "requires_offer_within_prefix_boundary": True,
        "requires_nonempty_exact_component_hotspot_agent_consensus": True,
    },
    "selection": {
        "maximum_selected_keys_per_task": 1,
        "required_selected_task_count": 4,
        "seed_priority_by_task": {
            task: list(seeds) for task, seeds in _SEED_PRIORITY.items()
        },
        "cross_task_backfill": False,
        "seed_replacement": False,
        "prefix_extension": False,
        "incomplete_state_supply_status": "INCONCLUSIVE_STATE_SUPPLY",
        "any_invalid_reset_status": "INVALID",
    },
}
_EXPECTED_FORMAL = {
    "role": "prefix_screened_ttf_unseen_conditional_effect_test",
    "paired_key_count": 4,
    "episode_count": 8,
    "wall_time_budget_seconds": 120.0,
    "environment_time_limit_seconds": 120.0,
    "episode_process_timeout_seconds": 150.0,
    "workers": 1,
    "strict_serial": True,
    "included_in_ttf": True,
    "fresh_reset_qualification": {
        "workers": 16,
        "reuse_screen_state_or_cache": False,
        "included_in_ttf": False,
    },
    "task_order": list(TASK_IDS),
    "arm_order_by_task": {
        task: list(controllers) for task, controllers in _ARM_ORDER.items()
    },
}
_EXPECTED_GATES = {
    "required_formal_trigger_and_exact_consensus_replay_keys": 4,
    "success_count_must_not_be_below_v2": True,
    "minimum_mean_restricted_ttf_reduction": 0.05,
    "minimum_paired_faster_key_count": 3,
    "maximum_any_key_restricted_ttf_regression": 0.20,
    "minimum_immediate_exact_rollback_rate_reduction": 0.25,
}


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_v2first_consensus16_warehouse_n800_supply.py",
            "scripts/run_stride_v2first_consensus16_warehouse_n800_supply.py",
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


def _validate_controller_contract(config: Mapping[str, Any]) -> None:
    contract = dict(config.get("controller_contract") or {})
    if contract != {
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
            "maximum_pp_calls_per_decision": 1,
        },
    }:
        raise ValueError("N800 supply controller contract changed")


def _materialize_config(
    path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("N800 supply config must be an object")
    if raw.get("schema") not in {V2_CONFIG_SCHEMA, V3_CONFIG_SCHEMA}:
        return dict(raw), {
            "profile": "v1",
            "config_schema": CONFIG_SCHEMA,
            "scientific_status": (
                "preregistered_trigger_qualified_n800_warehouse_conditional_effect_screen"
            ),
            "experiment_id": EXPERIMENT_ID,
            "status_schema": STATUS_SCHEMA,
            "screen_report_schema": SCREEN_REPORT_SCHEMA,
            "final_report_schema": FINAL_REPORT_SCHEMA,
            "candidate_seeds": CANDIDATE_SEEDS,
            "seed_priority": _SEED_PRIORITY,
        }

    if raw.get("schema") == V3_CONFIG_SCHEMA:
        expected_runtime = {
            "screen": {
                **_V3_SCREEN_RUNTIME,
                "workers": 16,
                "included_in_ttf": False,
            },
            "formal": {
                **_V3_FORMAL_RUNTIME,
                "workers": 1,
                "fresh_qualification_workers": 16,
                "included_in_ttf": True,
            },
        }
        if (
            raw.get("experiment_id") != V3_EXPERIMENT_ID
            or raw.get("scientific_status")
            != "post_v2_invalid_budget_corrected_preregistered_before_v3_controller"
            or tuple(map(int, raw.get("candidate_solver_seeds") or ()))
            != _V2_CANDIDATE_SEEDS
            or {
                str(task): tuple(map(int, seeds))
                for task, seeds in dict(
                    raw.get("seed_priority_by_task") or {}
                ).items()
            }
            != _V2_SEED_PRIORITY
            or dict(raw.get("runtime_budget") or {}) != expected_runtime
            or dict(raw.get("screen_qualification_policy") or {})
            != {
                "fresh_reset": True,
                "candidate_job_seeds": [20, 21, 22],
                "diagnostic_used_only_to_correct_registered_budget": True,
                "diagnostic_state_or_cache_reused": False,
                "reset_failure_keys_retained": True,
                "post_reset_task_or_seed_filtering": False,
            }
            or dict(raw.get("output_identity") or {})
            != {
                "required_new_output": True,
                "previous_v1_output_resumable": False,
                "previous_v2_output_resumable": False,
            }
        ):
            raise ValueError("N800 supply v3 overlay identity changed")
        base_path = registered_input(
            root,
            dict(raw.get("base_config") or {}),
            label="N800 supply v3 base config",
        )
        base, base_profile = _materialize_config(base_path, root)
        if (
            base_profile.get("profile") != "v2"
            or base.get("schema") != V2_CONFIG_SCHEMA
            or base.get("experiment_id") != V2_EXPERIMENT_ID
        ):
            raise ValueError("N800 supply v3 base config changed")
        config = copy.deepcopy(base)
        config["schema"] = V3_CONFIG_SCHEMA
        config["experiment_id"] = V3_EXPERIMENT_ID
        config["scientific_status"] = str(raw["scientific_status"])
        config["screen"].update(_V3_SCREEN_RUNTIME)
        config["formal"].update(_V3_FORMAL_RUNTIME)
        config["runtime_budget"] = copy.deepcopy(expected_runtime)
        config["screen_qualification_policy"] = dict(
            raw["screen_qualification_policy"]
        )
        config["v2_invalid_evidence"] = dict(
            raw.get("v2_invalid_evidence") or {}
        )
        config["budget_diagnostic_evidence"] = dict(
            raw.get("budget_diagnostic_evidence") or {}
        )
        config["claim_boundary"] = dict(raw.get("claim_boundary") or {})
        config["output_identity"] = dict(raw["output_identity"])
        config.pop("pre_registration_parent_commit", None)
        return config, {
            "profile": "v3",
            "config_schema": V3_CONFIG_SCHEMA,
            "scientific_status": str(raw["scientific_status"]),
            "experiment_id": V3_EXPERIMENT_ID,
            "status_schema": V3_STATUS_SCHEMA,
            "screen_report_schema": V3_SCREEN_REPORT_SCHEMA,
            "final_report_schema": V3_FINAL_REPORT_SCHEMA,
            "candidate_seeds": _V2_CANDIDATE_SEEDS,
            "seed_priority": _V2_SEED_PRIORITY,
        }

    if (
        raw.get("experiment_id") != V2_EXPERIMENT_ID
        or raw.get("scientific_status")
        != "post_v1_invalid_preregistered_before_v2_controller"
        or tuple(map(int, raw.get("candidate_solver_seeds") or ()))
        != _V2_CANDIDATE_SEEDS
        or {
            str(task): tuple(map(int, seeds))
            for task, seeds in dict(raw.get("seed_priority_by_task") or {}).items()
        }
        != _V2_SEED_PRIORITY
        or dict(raw.get("output_identity") or {})
        != {"required_new_output": True, "previous_v1_output_resumable": False}
        or dict(raw.get("screen_qualification_policy") or {})
        != {
            "fresh_reset": True,
            "historical_source_solver_seeds": [19, 20, 21, 22],
            "candidate_job_seeds": [20, 21, 22],
            "historical_source_environment_time_limit_seconds": 180.0,
            "historical_rows_used_only_to_freeze_candidate_seed_universe": True,
            "historical_state_or_cache_reused": False,
        }
    ):
        raise ValueError("N800 supply v2 overlay identity changed")
    base_path = registered_input(
        root,
        dict(raw.get("base_config") or {}),
        label="N800 supply v2 base config",
    )
    base = read_json(base_path)
    if (
        not isinstance(base, dict)
        or base.get("schema") != CONFIG_SCHEMA
        or base.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("N800 supply v2 base config changed")
    config = copy.deepcopy(base)
    config["schema"] = V2_CONFIG_SCHEMA
    config["experiment_id"] = V2_EXPERIMENT_ID
    config["scientific_status"] = str(raw["scientific_status"])
    config["screen"]["solver_seeds"] = list(_V2_CANDIDATE_SEEDS)
    config["screen"]["selection"]["seed_priority_by_task"] = {
        task: list(seeds) for task, seeds in _V2_SEED_PRIORITY.items()
    }
    config["cohort"]["candidate_solver_seeds"] = list(_V2_CANDIDATE_SEEDS)
    config["claim_boundary"] = dict(raw.get("claim_boundary") or {})
    config["historical_reset_qualification"] = dict(
        raw.get("historical_reset_qualification") or {}
    )
    config["output_identity"] = dict(raw["output_identity"])
    config["screen_qualification_policy"] = dict(raw["screen_qualification_policy"])
    config["v1_invalid_evidence"] = dict(raw.get("v1_invalid_evidence") or {})
    config.pop("pre_registration_parent_commit", None)
    return config, {
        "profile": "v2",
        "config_schema": V2_CONFIG_SCHEMA,
        "scientific_status": "post_v1_invalid_preregistered_before_v2_controller",
        "experiment_id": V2_EXPERIMENT_ID,
        "status_schema": V2_STATUS_SCHEMA,
        "screen_report_schema": V2_SCREEN_REPORT_SCHEMA,
        "final_report_schema": V2_FINAL_REPORT_SCHEMA,
        "candidate_seeds": _V2_CANDIDATE_SEEDS,
        "seed_priority": _V2_SEED_PRIORITY,
    }


def _profile_value(config: Mapping[str, Any], field: str) -> Any:
    return dict(config["_profile"])[field]


def _candidate_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(map(int, _profile_value(config, "candidate_seeds")))


def _seed_priority(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(task): tuple(map(int, seeds))
        for task, seeds in dict(_profile_value(config, "seed_priority")).items()
    }


def _experiment_id(config: Mapping[str, Any]) -> str:
    return str(_profile_value(config, "experiment_id"))


def _status_schema(config: Mapping[str, Any]) -> str:
    return str(_profile_value(config, "status_schema"))


def _screen_report_schema(config: Mapping[str, Any]) -> str:
    return str(_profile_value(config, "screen_report_schema"))


def _final_report_schema(config: Mapping[str, Any]) -> str:
    return str(_profile_value(config, "final_report_schema"))


def _profile_name(config: Mapping[str, Any]) -> str:
    return str(_profile_value(config, "profile"))


def _is_v2(config: Mapping[str, Any]) -> bool:
    return _profile_name(config) == "v2"


def _is_v3(config: Mapping[str, Any]) -> bool:
    return _profile_name(config) == "v3"


def _is_overlay(config: Mapping[str, Any]) -> bool:
    return _profile_name(config) in {"v2", "v3"}


def load_config(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    config, profile = _materialize_config(path, root)
    expected_screen = copy.deepcopy(_EXPECTED_SCREEN)
    expected_screen["solver_seeds"] = list(profile["candidate_seeds"])
    expected_screen["selection"]["seed_priority_by_task"] = {
        task: list(seeds)
        for task, seeds in dict(profile["seed_priority"]).items()
    }
    expected_formal = copy.deepcopy(_EXPECTED_FORMAL)
    if profile["profile"] == "v3":
        expected_screen.update(_V3_SCREEN_RUNTIME)
        expected_formal.update(_V3_FORMAL_RUNTIME)
    if (
        config.get("schema") != profile["config_schema"]
        or config.get("experiment_id") != profile["experiment_id"]
        or config.get("scientific_status") != profile["scientific_status"]
        or (
            profile["profile"] == "v1"
            and config.get("pre_registration_parent_commit")
            != PRE_REGISTRATION_PARENT_COMMIT
        )
        or (
            profile["profile"] in {"v2", "v3"}
            and "pre_registration_parent_commit" in config
        )
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or dict(config.get("screen") or {}) != expected_screen
        or dict(config.get("formal") or {}) != expected_formal
        or dict(config.get("performance_gates") or {}) != _EXPECTED_GATES
    ):
        raise ValueError("N800 supply experiment identity changed")
    _validate_controller_contract(config)

    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("dataset") != "build/wh-f16-v1-r2/dataset"
        or cohort.get("split") != "balanced_wall_clock"
        or cohort.get("map_id") != "warehouse-10-20-10-2-1"
        or cohort.get("map_code") != "w1020a"
        or int(cohort.get("agent_count", -1)) != 800
        or tuple(map(int, cohort.get("candidate_solver_seeds") or ()))
        != tuple(profile["candidate_seeds"])
        or cohort.get("result_based_task_or_seed_replacement") is not False
    ):
        raise ValueError("N800 supply cohort changed")
    tasks = [dict(row) for row in cohort.get("tasks") or ()]
    if tuple(str(row.get("id")) for row in tasks) != TASK_IDS:
        raise ValueError("N800 supply task order or identity changed")
    expected_variants = (
        "opposite_exchange",
        "opposite_exchange",
        "uniform_random",
        "uniform_random",
    )
    if tuple(str(row.get("variant")) for row in tasks) != expected_variants:
        raise ValueError("N800 supply task variants changed")
    if tuple(int(row.get("task_seed", -1)) for row in tasks) != (233, 277, 233, 277):
        raise ValueError("N800 supply task seeds changed")
    for task in tasks:
        registered_input(
            root,
            dict(task.get("task_file") or {}),
            label=f"N800 supply task {task['id']}",
        )
        registered_input(
            root,
            dict(task.get("scenario_file") or {}),
            label=f"N800 supply scenario {task['id']}",
        )

    inputs = dict(config.get("inputs") or {})
    runtime_path = registered_input(
        root,
        dict(inputs.get("runtime_config") or {}),
        label="N800 supply runtime source",
    )
    manifest_path = registered_input(
        root,
        dict(inputs.get("dataset_manifest") or {}),
        label="N800 supply dataset manifest",
    )
    registered_input(root, dict(inputs.get("map") or {}), label="N800 supply map")
    controller_manifest = registered_input(
        root,
        dict(inputs.get("controller_manifest") or {}),
        label="N800 supply V2 controller manifest",
    )
    if controller_manifest.parent != (root / str(config["controller_bundle"])).resolve():
        raise ValueError("N800 supply controller bundle changed")
    manifest_rows = {
        str(row.get("task_id")): row for row in read_jsonl(manifest_path)
    }
    selected_manifest = [manifest_rows.get(task_id) for task_id in TASK_IDS]
    if any(row is None for row in selected_manifest):
        raise ValueError("N800 supply manifest is missing a registered task")
    if any(
        int(dict(row).get("agent_count", -1)) != 800
        or str(dict(row).get("map_id")) != "warehouse-10-20-10-2-1"
        for row in selected_manifest
    ):
        raise ValueError("N800 supply manifest task factors changed")

    boundary = dict(config.get("claim_boundary") or {})
    if (
        boundary.get("prefix_screened") is not True
        or boundary.get("ttf_unseen_at_selection") is not True
        or boundary.get("unbiased_fresh_seed_confirmation") is not False
        or boundary.get("trigger_prevalence_claim_allowed") is not False
        or boundary.get("warehouse_generalization_allowed") is not False
        or boundary.get("default_promotion_allowed") is not False
    ):
        raise ValueError("N800 supply claim boundary changed")
    if profile["profile"] in {"v2", "v3"}:
        expected_boundary = (
            {
                "maximum_claim": (
                    "conditional_effect_on_four_k64_trigger_and_exact_consensus_"
                    "enriched_w1020a_n800_task_seed_pairs_from_historically_reset_"
                    "qualified_ttf_unseen_candidates"
                ),
                "historically_reset_qualified": True,
                "historical_reset_state_reused": False,
                "ttf_unseen_at_selection": True,
                "prefix_screened": True,
                "unbiased_fresh_seed_confirmation": False,
                "trigger_prevalence_claim_allowed": False,
                "warehouse_generalization_allowed": False,
                "n600_generalization_allowed": False,
                "default_promotion_allowed": False,
            }
            if profile["profile"] == "v2"
            else {
                "maximum_claim": (
                    "conditional_effect_on_four_k64_trigger_and_exact_consensus_"
                    "enriched_w1020a_n800_task_seed_pairs_under_the_preregistered_"
                    "200_second_budget"
                ),
                "historically_reset_qualified": True,
                "budget_diagnostic_conditioned": True,
                "budget_diagnostic_state_reused": False,
                "current_screen_fresh_reset": True,
                "post_reset_task_or_seed_filtering": False,
                "ttf_unseen_at_selection": True,
                "prefix_screened": True,
                "unbiased_fresh_seed_confirmation": False,
                "trigger_prevalence_claim_allowed": False,
                "warehouse_generalization_allowed": False,
                "n600_generalization_allowed": False,
                "absolute_ttf_comparison_to_v1_or_v2_allowed": False,
                "default_promotion_allowed": False,
            }
        )
        if boundary != expected_boundary:
            raise ValueError(
                f"N800 supply {profile['profile']} claim boundary changed"
            )
        if profile["profile"] == "v2":
            invalid_field = "v1_invalid_evidence"
            expected_invalid = {
                "config": {
                    "path": "configs/stride_v2first_consensus16_warehouse_n800_supply_v1.json",
                    "sha256": "4c4ed554d53f00e4dfc5e63a4f843aedd426d0cb3e69564ef775f03e5e321ed4",
                },
                "status": {
                    "path": "build/stride-v2first-consensus16-warehouse-n800-supply-v1/screen/collection_status.json",
                    "sha256": "7eca529b7a564fd4dda42f4d5aea339b45055b652c594950a14dc9c66591c04a",
                },
                "decision_status": "INVALID",
                "completed_schedule_entries": 0,
                "resumption_allowed": False,
            }
            invalid_label = "v1"
        else:
            invalid_field = "v2_invalid_evidence"
            expected_invalid = {
                "config": {
                    "path": "configs/stride_v2first_consensus16_warehouse_n800_supply_v2.json",
                    "sha256": "484f1944135106782c8355712fc7393c5825827bb25ccf3b6a635be48b16ca3b",
                },
                "status": {
                    "path": "build/stride-v2first-consensus16-warehouse-n800-supply-v2/screen/collection_status.json",
                    "sha256": "c8ba1529c04e091a865b2d6e0f5be110baef0cf3b474b63cd0ef36cd0716a51d",
                },
                "decision_status": "INVALID",
                "completed_schedule_entries": 0,
                "resumption_allowed": False,
            }
            invalid_label = "v2"
        invalid_evidence = dict(config.get(invalid_field) or {})
        if invalid_evidence != expected_invalid:
            raise ValueError(
                f"N800 supply {profile['profile']} prior-invalid identity changed"
            )
        registered_input(
            root,
            dict(invalid_evidence["config"]),
            label=f"N800 supply {invalid_label} invalid config",
        )
        invalid_status_path = registered_input(
            root,
            dict(invalid_evidence["status"]),
            label=f"N800 supply {invalid_label} invalid status",
        )
        invalid_status = read_json(invalid_status_path)
        if (
            not isinstance(invalid_status, dict)
            or invalid_status.get("decision_status") != "INVALID"
            or int(invalid_status.get("completed_schedule_entries", -1)) != 0
            or invalid_status.get("complete") is not False
        ):
            raise ValueError(
                f"N800 supply {invalid_label} invalid evidence changed"
            )
        historical = dict(config.get("historical_reset_qualification") or {})
        expected_historical = {
            "role": "registered_reset_validity_evidence_only_not_screen_or_ttf_reuse",
            "qualification_manifest": {
                "path": "build/wh-f16-v1-r2/qualification/qualification_manifest.jsonl",
                "sha256": "89b6d877fb9cc64f6dcf7e86ad903d109e95313b510d74e625234bbdabaafc50",
            },
            "qualification_report": {
                "path": "build/wh-f16-v1-r2/qualification/qualification_report.json",
                "sha256": "725ba67b95bb263440e897cc900a0a59637f696facf8796f6867610ab12cfbfc",
            },
            "required_exact_task_seed_rows": 12,
            "required_row_status": "ok",
            "required_initial_complete": True,
            "required_initial_feasible": False,
            "overall_report_pass_required": False,
            "registered_n800_controller_or_ttf_manifest_match_count": 0,
            "outcome_fields_read_for_seed_selection": False,
        }
        if historical != expected_historical:
            raise ValueError("N800 supply v2 historical reset contract changed")
        historical_manifest = registered_input(
            root,
            dict(historical["qualification_manifest"]),
            label="N800 supply v2 historical qualification manifest",
        )
        registered_input(
            root,
            dict(historical["qualification_report"]),
            label="N800 supply v2 historical qualification report",
        )
        exact_keys = {
            (task_id, seed)
            for task_id in TASK_IDS
            for seed in _V2_CANDIDATE_SEEDS
        }
        matches = [
            dict(row)
            for row in read_jsonl(historical_manifest)
            if (str(row.get("task_id")), int(row.get("solver_seed", -1)))
            in exact_keys
        ]
        if (
            len(matches) != 12
            or {
                (str(row.get("task_id")), int(row.get("solver_seed", -1)))
                for row in matches
            }
            != exact_keys
            or any(
                row.get("status") != "ok"
                or row.get("error") is not None
                or row.get("initial_complete") is not True
                or row.get("initial_feasible") is not False
                or row.get("repairable") is not True
                or int(row.get("agent_count", -1)) != 800
                for row in matches
            )
        ):
            raise ValueError("N800 supply v2 historical reset rows changed")
        forbidden_historical_fields = {
            "controller",
            "policy",
            "actual_action",
            "actual_metrics",
            "success",
            "capped_wall_time_to_feasible",
            "wall_time_to_feasible",
            "repair_iterations",
            "trace_file",
        }
        if any(forbidden_historical_fields & set(row) for row in matches):
            raise ValueError("N800 supply v2 historical rows contain controller/TTF data")
        historical_report = read_json(
            root / str(historical["qualification_report"]["path"])
        )
        if (
            not isinstance(historical_report, dict)
            or historical_report.get("passed") is not False
        ):
            raise ValueError("N800 supply v2 source report status changed")
        if profile["profile"] == "v3":
            diagnostic = dict(config.get("budget_diagnostic_evidence") or {})
            expected_diagnostic_keys = (
                (TASK_IDS[0], 20),
                (TASK_IDS[0], 22),
                (TASK_IDS[1], 20),
                (TASK_IDS[1], 22),
            )
            expected_diagnostic_rows = {
                f"{TASK_IDS[0]}@20": {
                    "initial_conflicts": 1065,
                    "state_fingerprint": (
                        "5475761580f19c18397ad3d42eb682390cfcabc46203d380c2b43d9be67e778d"
                    ),
                },
                f"{TASK_IDS[0]}@22": {
                    "initial_conflicts": 2140,
                    "state_fingerprint": (
                        "e3862fb699bbe1c0b74f978941830abbfc42de883558fa94ab175ca15ef78bc3"
                    ),
                },
                f"{TASK_IDS[1]}@20": {
                    "initial_conflicts": 1651,
                    "state_fingerprint": (
                        "ee08a194c2d2f5fd38af5f0905ed3f823444dbe9368f9216354d5a1fafd8d3ee"
                    ),
                },
                f"{TASK_IDS[1]}@22": {
                    "initial_conflicts": 1050,
                    "state_fingerprint": (
                        "bd7f2d006f297fc3add9e772b4142e70b819fa51765427971d34262582d26f16"
                    ),
                },
            }
            if diagnostic != {
                "role": (
                    "reset_only_budget_sufficiency_evidence_not_screen_or_ttf_reuse"
                ),
                "run_config": {
                    "path": (
                        "build/stride-n800-reset-budget-diagnostic-v1/screen/"
                        "reset_qualification/run_config.json"
                    ),
                    "sha256": (
                        "28f9b337e0daaa48b4a0f9b9b347242722e746de5b573e6d2f51563fa3760519"
                    ),
                },
                "qualification_manifest": {
                    "path": (
                        "build/stride-n800-reset-budget-diagnostic-v1/screen/"
                        "reset_qualification/qualification_manifest.jsonl"
                    ),
                    "sha256": (
                        "441f719183978f4b674609c642cc7fef587462f1e11e1e75fd1d650c20e2d731"
                    ),
                },
                "qualification_report": {
                    "path": (
                        "build/stride-n800-reset-budget-diagnostic-v1/screen/"
                        "reset_qualification/qualification_report.json"
                    ),
                    "sha256": (
                        "61a8e82618f293c934ef7b8ee177c65522f2cbd75e987527235303094bcf9f76"
                    ),
                },
                "exact_reset_keys": [list(key) for key in expected_diagnostic_keys],
                "expected_rows": expected_diagnostic_rows,
                "wall_time_budget_seconds": 200.0,
                "environment_time_limit_seconds": 200.0,
                "episode_process_timeout_seconds": 300.0,
                "workers": 16,
                "maximum_repair_iterations": 0,
                "required_valid_count": 4,
                "required_incomplete_reset_count": 0,
                "historical_state_fingerprint_and_conflict_mismatch_count": 0,
                "native_module_sha256": (
                    "1231dbc0bbca39cd5d1d2aba49bc68f8c16889ff22e16d5d7eb9afe49ec09221"
                ),
                "controller_or_ttf_outcomes_read": False,
                "diagnostic_state_or_cache_reused": False,
            }:
                raise ValueError("N800 supply v3 budget diagnostic identity changed")
            diagnostic_run_path = registered_input(
                root,
                dict(diagnostic["run_config"]),
                label="N800 supply v3 budget diagnostic run config",
            )
            diagnostic_manifest_path = registered_input(
                root,
                dict(diagnostic["qualification_manifest"]),
                label="N800 supply v3 budget diagnostic qualification manifest",
            )
            diagnostic_report_path = registered_input(
                root,
                dict(diagnostic["qualification_report"]),
                label="N800 supply v3 budget diagnostic qualification report",
            )
            diagnostic_run = read_json(diagnostic_run_path)
            if not isinstance(diagnostic_run, dict):
                raise ValueError("N800 supply v3 diagnostic run config is invalid")
            diagnostic_runtime = dict(diagnostic_run.get("configuration") or {})
            diagnostic_environment = dict(
                diagnostic_runtime.get("environment") or {}
            )
            diagnostic_native = dict(
                dict(diagnostic_run.get("controller_implementation") or {}).get(
                    "native_module"
                )
                or {}
            )
            configured_keys = tuple(
                (str(key[0]), int(key[1]))
                for key in diagnostic_runtime.get("cohort_job_keys_override") or ()
            )
            if (
                diagnostic_run.get("formal") is not False
                or diagnostic_runtime.get("formal") is not False
                or configured_keys != expected_diagnostic_keys
                or float(diagnostic_runtime.get("wall_time_budget_seconds", -1.0))
                != 200.0
                or float(diagnostic_environment.get("time_limit", -1.0)) != 200.0
                or int(diagnostic_environment.get("max_repair_iterations", -1)) != 0
                or float(
                    diagnostic_runtime.get("episode_process_timeout_seconds", -1.0)
                )
                != 300.0
                or float(
                    diagnostic_runtime.get(
                        "qualification_process_timeout_seconds", -1.0
                    )
                )
                != 300.0
                or int(diagnostic_runtime.get("workers", -1)) != 16
                or diagnostic_native.get("sha256")
                != diagnostic["native_module_sha256"]
            ):
                raise ValueError("N800 supply v3 diagnostic runtime changed")
            diagnostic_rows = [dict(row) for row in read_jsonl(diagnostic_manifest_path)]
            diagnostic_report = read_json(diagnostic_report_path)
            if not isinstance(diagnostic_report, dict):
                raise ValueError("N800 supply v3 diagnostic report is invalid")
            report_rows = [
                dict(row)
                for row in dict(diagnostic_report.get("natural_distribution") or {}).get(
                    "tasks", ()
                )
            ]
            diagnostic_by_key = {
                (str(row.get("task_id")), int(row.get("solver_seed", -1))): row
                for row in diagnostic_rows
            }
            report_by_key = {
                (str(row.get("task_id")), int(row.get("solver_seed", -1))): row
                for row in report_rows
            }
            if (
                diagnostic_report.get("passed") is not True
                or int(diagnostic_report.get("valid_count", -1)) != 4
                or int(diagnostic_report.get("incomplete_reset_count", -1)) != 0
                or diagnostic_report.get("errors") != []
                or set(diagnostic_by_key) != set(expected_diagnostic_keys)
                or set(report_by_key) != set(expected_diagnostic_keys)
            ):
                raise ValueError("N800 supply v3 diagnostic reset evidence changed")
            forbidden_diagnostic_fields = forbidden_historical_fields | {
                "actual_action",
                "actual_metrics",
            }
            historical_by_key = {
                (str(row.get("task_id")), int(row.get("solver_seed", -1))): row
                for row in matches
            }
            for key in expected_diagnostic_keys:
                row = diagnostic_by_key[key]
                report_row = report_by_key[key]
                registered_row = expected_diagnostic_rows[f"{key[0]}@{key[1]}"]
                historical_row = historical_by_key[key]
                if (
                    forbidden_diagnostic_fields & set(row)
                    or row.get("status") != "ok"
                    or row.get("error") is not None
                    or row.get("initial_complete") is not True
                    or row.get("initial_feasible") is not False
                    or row.get("repairable") is not True
                    or report_row.get("initial_complete") is not True
                    or report_row.get("initial_state_consistent") is not True
                    or report_row.get("initial_feasible") is not False
                    or int(row.get("initial_conflicts", -1))
                    != int(registered_row["initial_conflicts"])
                    or row.get("state_fingerprint")
                    != registered_row["state_fingerprint"]
                    or int(report_row.get("initial_conflicts", -1))
                    != int(registered_row["initial_conflicts"])
                    or report_row.get("state_fingerprint")
                    != registered_row["state_fingerprint"]
                    or int(historical_row.get("initial_conflicts", -1))
                    != int(registered_row["initial_conflicts"])
                    or historical_row.get("state_fingerprint")
                    != registered_row["state_fingerprint"]
                ):
                    raise ValueError(
                        "N800 supply v3 diagnostic/historical reset parity changed"
                    )
    result = dict(config)
    result["_runtime_source_path"] = str(runtime_path)
    result["_task_by_id"] = {str(row["id"]): row for row in tasks}
    result["_profile"] = profile
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
        for seed in _candidate_seeds(config)
    ]


def formal_schedule(
    config: Mapping[str, Any], selected_keys: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selected = {str(row["task_id"]): dict(row) for row in selected_keys}
    if set(selected) != set(TASK_IDS):
        raise ValueError("formal schedule requires exactly one selected key per task")
    task_by_id = dict(config["_task_by_id"])
    rows: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(TASK_IDS):
        seed = int(selected[task_id]["solver_seed"])
        if seed not in _candidate_seeds(config):
            raise ValueError("formal schedule contains an unregistered seed")
        for position, controller in enumerate(_ARM_ORDER[task_id]):
            rows.append(
                {
                    "phase": "formal",
                    "paired_key_index": task_index,
                    "within_key_position": position,
                    "key_id": f"{task_id}@{seed}",
                    "task_id": task_id,
                    "task_variant": str(task_by_id[task_id]["variant"]),
                    "task_seed": int(task_by_id[task_id]["task_seed"]),
                    "solver_seed": seed,
                    "controller": controller,
                }
            )
    return rows


def plan(config_path: str | Path) -> dict[str, Any]:
    _path, _root, config = load_config(config_path)
    screen = screen_schedule(config)
    formal = dict(config["formal"])
    return {
        "schema": _status_schema(config),
        "experiment_id": _experiment_id(config),
        "task_ids": list(TASK_IDS),
        "candidate_solver_seeds": list(_candidate_seeds(config)),
        "screen": {
            "candidate_key_count": len(screen),
            "schedule_sha256": json_fingerprint(screen),
            "v2_repair_outcome_prefix_k": PREFIX_V2_OUTCOMES,
            "maximum_trace_decisions": MAXIMUM_SCREEN_TRACE_DECISIONS,
            "workers": 16,
            "included_in_ttf": False,
            "wall_time_safety_fuse_seconds": float(
                config["screen"]["wall_time_safety_fuse_seconds"]
            ),
            "environment_time_safety_fuse_seconds": float(
                config["screen"]["environment_time_safety_fuse_seconds"]
            ),
            "episode_process_timeout_seconds": float(
                config["screen"]["episode_process_timeout_seconds"]
            ),
            "maximum_process_fuse_seconds": len(screen)
            * float(config["screen"]["episode_process_timeout_seconds"]),
            "selection_reads": "pre_action_trigger_and_exact_consensus_only",
            "any_invalid_reset_is_terminal": True,
        },
        "formal": {
            "blocked_until_screen_selects_four_tasks": True,
            "paired_key_count": 4,
            "episode_count": 8,
            "timed_workers": 1,
            "strict_serial": True,
            "fresh_qualification_workers": 16,
            "wall_time_budget_seconds": float(formal["wall_time_budget_seconds"]),
            "environment_time_limit_seconds": float(
                formal["environment_time_limit_seconds"]
            ),
            "episode_process_timeout_seconds": float(
                formal["episode_process_timeout_seconds"]
            ),
            "maximum_registered_ttf_seconds": 8
            * float(formal["wall_time_budget_seconds"]),
            "maximum_process_fuse_seconds": 8
            * float(formal["episode_process_timeout_seconds"]),
            "automatic_timeout_extension": False,
        },
        "solver_or_controller_invoked": False,
        "screen_ttf_or_outcome_ranking": False,
        "task_or_seed_replacement": False,
        "historically_reset_qualified": _is_overlay(config),
        "screen_fresh_reset": True,
        "ttf_unseen_at_selection": True,
        "required_new_output": _is_overlay(config),
        "budget_corrected_profile": _is_v3(config),
        "diagnostic_state_or_cache_reused": False,
        "post_reset_task_or_seed_filtering": False,
    }


def controller_kwargs(
    root: Path,
    config: Mapping[str, Any],
    controller: str,
    *,
    phase: str,
) -> dict[str, Any]:
    if phase == "screen":
        runtime = dict(config["screen"])
        shim_runtime = {
            "wall_time_budget_seconds": float(runtime["wall_time_safety_fuse_seconds"]),
            "environment_time_limit_seconds": float(
                runtime["environment_time_safety_fuse_seconds"]
            ),
            "episode_process_timeout_seconds": float(
                runtime["episode_process_timeout_seconds"]
            ),
        }
    elif phase == "formal":
        runtime = dict(config["formal"])
        shim_runtime = {
            "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
            "environment_time_limit_seconds": float(
                runtime["environment_time_limit_seconds"]
            ),
            "episode_process_timeout_seconds": float(
                runtime["episode_process_timeout_seconds"]
            ),
        }
    else:
        raise ValueError(f"unknown N800 supply phase: {phase}")
    kwargs = _warehouse_controller_kwargs(
        root,
        {"runtime": shim_runtime, "controller_bundle": config["controller_bundle"]},
        controller,
    )
    kwargs["stopping_rule"] = "historical" if phase == "screen" else "wall-clock"
    return kwargs


def _runtime_config_path(
    output: Path, config: Mapping[str, Any], phase: str
) -> Path:
    payload = read_json(Path(str(config["_runtime_source_path"])).resolve())
    if not isinstance(payload, dict):
        raise ValueError("N800 supply runtime source must be an object")
    payload["split"] = "balanced_wall_clock"
    payload["solver_seeds"] = list(_candidate_seeds(config))
    payload["repair_seed_policy"] = "episode_stream"
    payload["deterministic_pp_replay"] = False
    environment = dict(payload.get("environment") or {})
    if phase == "screen":
        runtime = dict(config["screen"])
        payload["max_decisions"] = MAXIMUM_SCREEN_TRACE_DECISIONS
        payload["metric_iteration_budget"] = MAXIMUM_SCREEN_TRACE_DECISIONS
        payload["wall_time_budget_seconds"] = float(
            runtime["wall_time_safety_fuse_seconds"]
        )
        payload["episode_process_timeout_seconds"] = float(
            runtime["episode_process_timeout_seconds"]
        )
        payload["workers"] = 16
        environment["time_limit"] = float(
            runtime["environment_time_safety_fuse_seconds"]
        )
        filename = (
            "screen_seed20_22_fresh_reset_wall0200_k64_plus_offer.json"
            if _is_v3(config)
            else "screen_seed20_22_fresh_reset_k64_plus_offer.json"
            if _is_v2(config)
            else "screen_seed29_32_k64_plus_offer.json"
        )
    elif phase == "formal":
        runtime = dict(config["formal"])
        payload["max_decisions"] = 0
        payload["wall_time_budget_seconds"] = float(
            runtime["wall_time_budget_seconds"]
        )
        payload["episode_process_timeout_seconds"] = float(
            runtime["episode_process_timeout_seconds"]
        )
        payload["workers"] = 1
        environment["time_limit"] = float(runtime["environment_time_limit_seconds"])
        filename = (
            "formal_seed20_22_wall0200.json"
            if _is_v3(config)
            else "formal_seed20_22_wall0120.json"
            if _is_v2(config)
            else "formal_seed29_32_wall0120.json"
        )
    else:
        raise ValueError(f"unknown N800 supply phase: {phase}")
    payload["environment"] = environment
    destination = output / "runtime_configs" / filename
    if destination.is_file():
        if read_json(destination) != payload:
            raise ValueError(f"materialized N800 supply {phase} runtime changed")
    else:
        write_json(destination, payload)
    return destination


def _screen_root(output: Path) -> Path:
    return output / "screen"


def _formal_root(output: Path) -> Path:
    return output / "formal"


def _reject_v1_output_identity(
    root: Path, output: Path, config: Mapping[str, Any]
) -> None:
    previous_v1 = (
        root / "build" / "stride-v2first-consensus16-warehouse-n800-supply-v1"
    ).resolve()
    previous_v2 = (
        root / "build" / "stride-v2first-consensus16-warehouse-n800-supply-v2"
    ).resolve()
    if _is_v2(config) and output.resolve() == previous_v1:
        raise ValueError("N800 supply v2 requires a new output; v1 INVALID is immutable")
    if _is_v3(config) and output.resolve() in {previous_v1, previous_v2}:
        raise ValueError(
            "N800 supply v3 requires a new output; prior INVALID outputs are immutable"
        )


def _screen_qualification_root(output: Path) -> Path:
    return _screen_root(output) / "reset_qualification"


def _screen_controller_root(output: Path) -> Path:
    return _screen_root(output) / "controller" / "consensus16_probe"


def _formal_qualification_root(output: Path) -> Path:
    return _formal_root(output) / "fresh_reset_qualification"


def _formal_controller_root(output: Path, controller: str) -> Path:
    return _formal_root(output) / "controllers" / controller


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
        raise ValueError(f"ambiguous manifest row for {task_id}@{solver_seed}")
    return matches[0] if matches else None


def _completed_screen_count(output: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    collection = _screen_controller_root(output)
    return sum(
        _manifest_row(collection, str(row["task_id"]), int(row["solver_seed"]))
        is not None
        for row in rows
    )


def _completed_formal_count(output: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        _manifest_row(
            _formal_controller_root(output, str(row["controller"])),
            str(row["task_id"]),
            int(row["solver_seed"]),
        )
        is not None
        for row in rows
    )


def _reset_qualification(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    *,
    phase: str,
    keys: set[tuple[str, int]],
    resume: bool,
) -> Path:
    if phase == "screen":
        destination = _screen_qualification_root(output)
        workers = 16
        timeout = float(config["screen"]["episode_process_timeout_seconds"])
    elif phase == "formal":
        destination = _formal_qualification_root(output)
        workers = 16
        timeout = float(config["formal"]["episode_process_timeout_seconds"])
    else:
        raise ValueError(f"unknown N800 supply phase: {phase}")
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config, phase),
        destination,
        phase="qualify",
        workers=workers,
        resume=resume and destination.joinpath("run_config.json").is_file(),
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_process_timeout_seconds=timeout,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, "v2_only", phase=phase),
    )
    report = read_json(destination / "qualification_report.json")
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or int(report.get("valid_count", -1)) != len(keys)
        or int(report.get("incomplete_reset_count", -1)) != 0
    ):
        raise RuntimeError(
            f"N800 supply {phase} reset qualification failed; replacement is forbidden"
        )
    return destination


def _prepare_collection_root(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    *,
    phase: str,
    controller: str,
    keys: set[tuple[str, int]],
    qualification: Path,
) -> Path:
    if phase == "screen":
        collection = _screen_controller_root(output)
        workers = 16
        timeout = float(config["screen"]["episode_process_timeout_seconds"])
    elif phase == "formal":
        collection = _formal_controller_root(output, controller)
        workers = 1
        timeout = float(config["formal"]["episode_process_timeout_seconds"])
    else:
        raise ValueError(f"unknown N800 supply phase: {phase}")
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config, phase),
        collection,
        phase="qualify",
        workers=workers,
        resume=collection.joinpath("run_config.json").is_file(),
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        qualification_process_timeout_seconds=timeout,
        use_global_collection_lock=False,
        **controller_kwargs(root, config, controller, phase=phase),
    )
    return collection


def _run_screen_jobs(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    rows: list[dict[str, Any]],
    qualification: Path,
) -> None:
    keys = _job_keys(rows)
    collection = _prepare_collection_root(
        root,
        output,
        config,
        phase="screen",
        controller="consensus16_rescue",
        keys=keys,
        qualification=qualification,
    )
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config, "screen"),
        collection,
        phase="realized_dynamic",
        workers=16,
        resume=True,
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        qualification_process_timeout_seconds=float(
            config["screen"]["episode_process_timeout_seconds"]
        ),
        use_global_collection_lock=False,
        **controller_kwargs(
            root, config, "consensus16_rescue", phase="screen"
        ),
    )


def _screen_preaction_certificate(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read only trace identity plus selection/generation fields used by the screen.

    The first offer may occur after any three-rollbacks trigger within K=64;
    row 64 exists so a trigger completed by outcome 63 remains observable.
    No offer observation, later row, success, TTF, conflicts, or timing field
    participates in eligibility.
    """

    if len(decisions) > MAXIMUM_SCREEN_TRACE_DECISIONS:
        raise ValueError("screen trace exceeds K64 plus the next offer decision")
    if any(int(row.get("decision_index", -1)) != index for index, row in enumerate(decisions)):
        raise ValueError("screen trace decision indices are not contiguous")
    offers: list[dict[str, Any]] = []
    for index, row in enumerate(decisions):
        record_raw = dict(row.get("controller") or {}).get(
            "v2_first_consensus_rescue"
        )
        if not isinstance(record_raw, Mapping):
            raise ValueError("screen decision lacks consensus-rescue trace")
        record = dict(record_raw)
        selection_raw = record.get("selection")
        if not isinstance(selection_raw, Mapping):
            raise ValueError("screen decision lacks pre-action selection trace")
        selection = dict(selection_raw)
        before_fingerprint = str(row.get("before_platform_signature") or "")
        if (
            int(selection.get("decision_index", -1)) != index
            or not before_fingerprint
            or str(selection.get("before_repair_fingerprint") or "")
            != before_fingerprint
            or str(selection.get("repair_fingerprint") or "")
            != before_fingerprint
            or str(selection.get("structural_profile") or "")
            != "component_hotspot_consensus"
            or int(selection.get("nominal_size", -1)) != 16
            or int(selection.get("minimum_consecutive_v2_exact_rollbacks", -1))
            != 3
            or selection.get("consensus_required") is not True
        ):
            raise ValueError("screen pre-action selection identity changed")
        generation_raw = record.get("generation")
        offered = bool(selection.get("offered", False))
        phase = str(selection.get("selection_phase") or "")
        if offered or phase == "consensus_rescue_due" or generation_raw is not None:
            if not offered or phase != "consensus_rescue_due":
                raise ValueError("screen offer phase is internally inconsistent")
            if not isinstance(generation_raw, Mapping):
                raise ValueError("screen offer lacks pre-action consensus generation")
            generation = dict(generation_raw)
            fingerprint = str(selection.get("before_repair_fingerprint") or "")
            component = tuple(map(int, generation.get("component_agents") or ()))
            hotspot = tuple(map(int, generation.get("hotspot_agents") or ()))
            exact = bool(generation.get("exact_agent_consensus", False))
            component_available = bool(generation.get("component_available", False))
            hotspot_available = bool(generation.get("hotspot_available", False))
            component_id = generation.get("component_candidate_id")
            hotspot_id = generation.get("hotspot_candidate_id")
            consensus_id = generation.get("consensus_candidate_id")
            generated_count = int(generation.get("generated_candidate_count", -1))
            if (
                index < 3
                or index > PREFIX_V2_OUTCOMES
                or int(selection.get("consecutive_v2_exact_rollbacks", -1)) != 3
                or selection.get("rescue_due") is not True
                or not fingerprint
                or generation.get("attempted") is not True
                or generation.get("offered") is not True
                or generation.get("consumed") is not True
                or int(generation.get("decision_index", -1)) != index
                or str(generation.get("before_repair_fingerprint") or "")
                != fingerprint
                or not 0 <= generated_count <= 2
                or component_available != bool(component_id is not None)
                or hotspot_available != bool(hotspot_id is not None)
                or bool(component) != bool(component_id is not None)
                or bool(hotspot) != bool(hotspot_id is not None)
                or component != tuple(sorted(set(component)))
                or hotspot != tuple(sorted(set(hotspot)))
                or any(
                    not isinstance(candidate_id, str) or not candidate_id
                    for candidate_id in (component_id, hotspot_id)
                    if candidate_id is not None
                )
                or generated_count
                != len(
                    {
                        candidate_id
                        for candidate_id in (component_id, hotspot_id)
                        if candidate_id is not None
                    }
                )
                or exact != bool(component and component == hotspot)
                or bool(consensus_id is not None) != exact
                or bool(generation.get("challenger_present", False)) != exact
                or bool(generation.get("available", False)) != exact
                or generation.get("structural_selected") is not False
                or (
                    exact
                    and not (
                        1 <= len(component) <= 16
                        and all(
                            isinstance(value, str) and bool(value)
                            for value in (
                                component_id,
                                hotspot_id,
                                consensus_id,
                                generation.get("candidate_id"),
                            )
                        )
                        and component_id == hotspot_id == consensus_id
                        and generation.get("candidate_id") == consensus_id
                        and generation.get("fallback") is None
                    )
                )
                or (
                    not exact
                    and (
                        generation.get("candidate_id") is not None
                        or generation.get("fallback")
                        != "fresh_v2_after_no_consensus"
                    )
                )
            ):
                raise ValueError("screen pre-action trigger/consensus contract changed")
            offers.append(
                {
                    "offer_decision_index": index,
                    "trigger_completion_decision_index": index - 1,
                    "trigger_outcome_count": index,
                    "trigger_fingerprint": fingerprint,
                    "exact_agent_consensus": exact,
                    "consensus_agent_count": len(component) if exact else 0,
                    "consensus_agents": list(component) if exact else [],
                    "consensus_agents_sha256": (
                        json_fingerprint(list(component)) if exact else None
                    ),
                }
            )
            # Eligibility is fully determined before this action executes.
            # Deliberately do not inspect this row's outcome or any later row.
            break
        if phase != "v2_only" or offered or generation_raw is not None:
            raise ValueError("screen prefix changed before its first rescue offer")
        if selection.get("rescue_due") is not False:
            raise ValueError("screen prefix marked rescue due before its first offer")
    if len(offers) > 1:
        raise ValueError("screen contains more than one rescue offer")
    if not offers:
        return {
            "triggered_within_k": False,
            "offer_decision_index": None,
            "trigger_completion_decision_index": None,
            "trigger_outcome_count": None,
            "trigger_fingerprint": None,
            "exact_agent_consensus": False,
            "consensus_agent_count": 0,
            "consensus_agents": [],
            "consensus_agents_sha256": None,
            "eligible": False,
            "selection_fields_only": True,
        }
    offer = offers[0]
    return {
        "triggered_within_k": True,
        **offer,
        "eligible": bool(offer["exact_agent_consensus"]),
        "selection_fields_only": True,
    }


def select_screen_keys(
    candidates: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    priority = _SEED_PRIORITY if config is None else _seed_priority(config)
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
                for seed in priority[task_id]
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


def _selection_rule_payload(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    priority = _SEED_PRIORITY if config is None else _seed_priority(config)
    return {
        "prefix_v2_outcomes": PREFIX_V2_OUTCOMES,
        "next_pre_action_offer_decision_allowed": True,
        "eligibility": "three_same_fingerprint_exact_v2_rollbacks_then_exact_nonempty_component16_hotspot16_agent_consensus",
        "seed_priority_by_task": {
            task: list(seeds) for task, seeds in priority.items()
        },
        "maximum_selected_keys_per_task": 1,
        "cross_task_backfill": False,
        "seed_replacement": False,
        "prefix_extension": False,
        "selection_fields_only": True,
    }


def _screen_candidate_result(
    config: Mapping[str, Any], output: Path, item: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    task_id = str(item["task_id"])
    seed = int(item["solver_seed"])
    collection = _screen_controller_root(output)
    manifest = _manifest_row(collection, task_id, seed)
    if manifest is None:
        return None, f"{task_id}@{seed}: missing screen manifest"
    if manifest.get("status") != "ok":
        return None, f"{task_id}@{seed}: screen manifest status is {manifest.get('status')}"
    try:
        decisions = _decision_rows(collection, manifest)
        certificate = _screen_preaction_certificate(decisions)
    except (KeyError, TypeError, ValueError) as error:
        return None, f"{task_id}@{seed}: screen trace audit failed: {error}"
    if (
        not bool(certificate["triggered_within_k"])
        and len(decisions) < MAXIMUM_SCREEN_TRACE_DECISIONS
    ):
        # This is an integrity classification only.  `stop_reason` is never
        # retained, reported, or used to rank/select among eligible keys.
        summary = manifest.get("summary")
        if (
            not isinstance(summary, Mapping)
            or str(dict(summary).get("stop_reason") or "") != "success"
            or dict(summary).get("truncated") is not False
            or dict(summary).get("external_timeout") is not False
        ):
            return None, (
                f"{task_id}@{seed}: prefix ended before K64 without a complete "
                "early-feasible termination"
            )
    task = dict(config["_task_by_id"])[task_id]
    return {
        "key_id": f"{task_id}@{seed}",
        "task_id": task_id,
        "task_variant": str(task["variant"]),
        "task_seed": int(task["task_seed"]),
        "solver_seed": seed,
        **certificate,
    }, None


def analyze_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    output = Path(output).resolve()
    stage = _screen_root(output)
    completed = load_completed_report(
        stage,
        status_filename=STATUS_FILENAME,
        report_filename=SCREEN_REPORT_FILENAME,
        status_schema=_status_schema(config),
        report_schema=_screen_report_schema(config),
        config_path=path,
    )
    if completed is not None:
        return completed

    errors: list[str] = []
    qualification_report_path = (
        _screen_qualification_root(output) / "qualification_report.json"
    )
    expected_candidate_count = len(TASK_IDS) * len(_candidate_seeds(config))
    if not qualification_report_path.is_file():
        errors.append("screen reset qualification report is missing")
        qualification_report_sha256 = None
    else:
        qualification_report_sha256 = sha256_file(qualification_report_path)
        qualification = read_json(qualification_report_path)
        if (
            not isinstance(qualification, dict)
            or qualification.get("passed") is not True
            or int(qualification.get("valid_count", -1)) != expected_candidate_count
            or int(qualification.get("incomplete_reset_count", -1)) != 0
        ):
            errors.append(
                "one or more registered screen qualification rows is invalid"
            )

    candidates: list[dict[str, Any]] = []
    for item in screen_schedule(config):
        candidate, error = _screen_candidate_result(config, output, item)
        if error is not None:
            errors.append(error)
        elif candidate is not None:
            candidates.append(candidate)
    selected, missing = select_screen_keys(candidates, config)
    integrity_passed = not errors and len(candidates) == expected_candidate_count
    supply_passed = integrity_passed and not missing and len(selected) == 4
    decision_status = (
        "PASS_STATE_SUPPLY"
        if supply_passed
        else "INVALID"
        if not integrity_passed
        else "INCONCLUSIVE_STATE_SUPPLY"
    )
    report = {
        "schema": _screen_report_schema(config),
        "experiment_id": _experiment_id(config),
        "scientific_status": (
            str(config["scientific_status"])
            if _is_overlay(config)
            else "n800_k64_preaction_trigger_consensus_supply_screen"
        ),
        "integrity_passed": integrity_passed,
        "errors": errors,
        "decision_status": decision_status,
        "formal_ttf_allowed": supply_passed,
        "candidate_key_count": len(candidates),
        "eligible_key_count": sum(bool(row["eligible"]) for row in candidates),
        "selected_key_count": len(selected),
        "missing_task_ids": missing,
        "candidates": candidates,
        "selected_keys": selected,
        "selection_rule": _selection_rule_payload(config),
        "selection_rule_sha256": json_fingerprint(_selection_rule_payload(config)),
        "selected_keys_sha256": json_fingerprint(selected),
        "screen_schedule_sha256": json_fingerprint(screen_schedule(config)),
        "screen_workers": 16,
        "historically_reset_qualified": _is_overlay(config),
        "screen_fresh_reset": True,
        "historical_reset_state_or_cache_reused": False,
        "budget_corrected_profile": _is_v3(config),
        "diagnostic_state_or_cache_reused": False,
        "post_reset_task_or_seed_filtering": False,
        "included_in_ttf": False,
        "selection_fields_only": True,
        "screen_ttf_or_outcome_ranking": False,
        "task_or_seed_replacement": False,
        "claim_scope": "state_supply_only_not_performance",
        "inputs": {
            "config_sha256": sha256_file(path),
            "qualification_report_sha256": qualification_report_sha256,
            "historical_qualification_manifest_sha256": (
                str(
                    config["historical_reset_qualification"]
                    ["qualification_manifest"]["sha256"]
                )
                if _is_overlay(config)
                else None
            ),
            "historical_qualification_report_sha256": (
                str(
                    config["historical_reset_qualification"]
                    ["qualification_report"]["sha256"]
                )
                if _is_overlay(config)
                else None
            ),
            "budget_diagnostic_run_config_sha256": (
                str(config["budget_diagnostic_evidence"]["run_config"]["sha256"])
                if _is_v3(config)
                else None
            ),
            "budget_diagnostic_qualification_manifest_sha256": (
                str(
                    config["budget_diagnostic_evidence"]
                    ["qualification_manifest"]["sha256"]
                )
                if _is_v3(config)
                else None
            ),
            "budget_diagnostic_qualification_report_sha256": (
                str(
                    config["budget_diagnostic_evidence"]
                    ["qualification_report"]["sha256"]
                )
                if _is_v3(config)
                else None
            ),
        },
        "producer_identity": dict(producer) if producer is not None else None,
    }
    write_json(stage / SCREEN_REPORT_FILENAME, report)
    return report


def _screen_status(
    base: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    *,
    complete: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = _completed_screen_count(output, rows)
    result = {
        **dict(base),
        "phase": "screen",
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
        "workers": 16,
        "included_in_ttf": False,
        "selection_fields_only": True,
    }
    if terminal_failure is not None:
        result["decision_status"] = "INVALID"
        result["terminal_failure"] = dict(terminal_failure)
    return result


def run_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    if dry_run:
        return {
            **dict(plan(path)["screen"]),
            "schema": _status_schema(config),
            "experiment_id": _experiment_id(config),
            "phase": "screen",
            "solver_or_controller_invoked": False,
        }
    output = Path(output).resolve()
    _reject_v1_output_identity(root, output, config)
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
        report_filename=SCREEN_REPORT_FILENAME,
        report_schema=_screen_report_schema(config),
        label="N800 trigger-qualified supply screen",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.status.get("terminal_failure") is not None:
        return dict(prepared.status)
    keys = _job_keys(rows)
    try:
        qualification = _reset_qualification(
            root,
            output,
            config,
            phase="screen",
            keys=keys,
            resume=prepared.resumed,
        )
        _run_screen_jobs(root, output, config, rows, qualification)
    except Exception as error:
        status = _screen_status(
            prepared.base_status,
            output,
            rows,
            terminal_failure={
                "phase": "screen_qualification_or_prefix_collection",
                "error": f"{type(error).__name__}: {error}",
                "sample_replacement_forbidden": True,
            },
        )
        write_json(stage / STATUS_FILENAME, status)
        return status
    report = analyze_screen(path, output, producer=producer)
    status = _screen_status(
        prepared.base_status,
        output,
        rows,
        complete=True,
    )
    status["decision_status"] = str(report["decision_status"])
    status["report_sha256"] = sha256_file(stage / SCREEN_REPORT_FILENAME)
    write_json(stage / STATUS_FILENAME, status)
    return report


def _load_trusted_screen(
    path: Path,
    root: Path,
    output: Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _unused, _unused_root, config = load_config(path)
    stage = _screen_root(output)
    report = load_completed_report(
        stage,
        status_filename=STATUS_FILENAME,
        report_filename=SCREEN_REPORT_FILENAME,
        status_schema=_status_schema(config),
        report_schema=_screen_report_schema(config),
        config_path=path,
    )
    if report is None:
        raise ValueError("N800 supply screen has no completed report")
    status = read_json(stage / STATUS_FILENAME)
    if not isinstance(status, dict):
        raise ValueError("N800 supply screen status is invalid")
    current_producer = dict(producer) if producer is not None else _producer(root)
    schedule_hash = json_fingerprint(screen_schedule(config))
    rule_hash = json_fingerprint(_selection_rule_payload(config))
    candidates = [dict(row) for row in report.get("candidates") or ()]
    selected = [dict(row) for row in report.get("selected_keys") or ()]
    mechanical, missing = select_screen_keys(candidates, config)
    qualification_path = (
        _screen_qualification_root(output) / "qualification_report.json"
    )
    if not qualification_path.is_file():
        raise ValueError("N800 supply screen qualification report is missing")
    if (
        status.get("complete") is not True
        or int(status.get("total_schedule_entries", -1))
        != len(TASK_IDS) * len(_candidate_seeds(config))
        or int(status.get("completed_schedule_entries", -1))
        != len(TASK_IDS) * len(_candidate_seeds(config))
        or status.get("config_sha256") != sha256_file(path)
        or status.get("producer_identity") != current_producer
        or status.get("schedule_sha256") != schedule_hash
        or status.get("report_sha256")
        != sha256_file(stage / SCREEN_REPORT_FILENAME)
        or status.get("decision_status") != "PASS_STATE_SUPPLY"
        or report.get("producer_identity") != current_producer
        or (
            _is_overlay(config)
            and report.get("scientific_status") != config["scientific_status"]
        )
        or dict(report.get("inputs") or {}).get("config_sha256")
        != sha256_file(path)
        or dict(report.get("inputs") or {}).get("qualification_report_sha256")
        != sha256_file(qualification_path)
        or report.get("integrity_passed") is not True
        or report.get("formal_ttf_allowed") is not True
        or report.get("decision_status") != "PASS_STATE_SUPPLY"
        or report.get("selection_fields_only") is not True
        or report.get("screen_ttf_or_outcome_ranking") is not False
        or bool(report.get("historically_reset_qualified")) != _is_overlay(config)
        or report.get("screen_fresh_reset") is not True
        or report.get("historical_reset_state_or_cache_reused") is not False
        or bool(report.get("budget_corrected_profile")) != _is_v3(config)
        or report.get("diagnostic_state_or_cache_reused") is not False
        or report.get("post_reset_task_or_seed_filtering") is not False
        or (
            _is_overlay(config)
            and dict(report.get("inputs") or {}).get(
                "historical_qualification_manifest_sha256"
            )
            != str(
                config["historical_reset_qualification"]
                ["qualification_manifest"]["sha256"]
            )
        )
        or (
            _is_overlay(config)
            and dict(report.get("inputs") or {}).get(
                "historical_qualification_report_sha256"
            )
            != str(
                config["historical_reset_qualification"]
                ["qualification_report"]["sha256"]
            )
        )
        or (
            _is_v3(config)
            and dict(report.get("inputs") or {}).get(
                "budget_diagnostic_run_config_sha256"
            )
            != str(config["budget_diagnostic_evidence"]["run_config"]["sha256"])
        )
        or (
            _is_v3(config)
            and dict(report.get("inputs") or {}).get(
                "budget_diagnostic_qualification_manifest_sha256"
            )
            != str(
                config["budget_diagnostic_evidence"]
                ["qualification_manifest"]["sha256"]
            )
        )
        or (
            _is_v3(config)
            and dict(report.get("inputs") or {}).get(
                "budget_diagnostic_qualification_report_sha256"
            )
            != str(
                config["budget_diagnostic_evidence"]
                ["qualification_report"]["sha256"]
            )
        )
        or int(report.get("candidate_key_count", -1))
        != len(TASK_IDS) * len(_candidate_seeds(config))
        or len(candidates) != len(TASK_IDS) * len(_candidate_seeds(config))
        or len(selected) != 4
        or missing
        or mechanical != selected
        or report.get("screen_schedule_sha256") != schedule_hash
        or report.get("selection_rule") != _selection_rule_payload(config)
        or report.get("selection_rule_sha256") != rule_hash
        or report.get("selected_keys_sha256") != json_fingerprint(selected)
        or tuple(str(row.get("task_id")) for row in selected) != TASK_IDS
        or any(not bool(row.get("eligible")) for row in selected)
    ):
        raise ValueError("N800 supply screen trust chain changed")
    qualification = read_json(qualification_path)
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or int(qualification.get("valid_count", -1))
        != len(TASK_IDS) * len(_candidate_seeds(config))
        or int(qualification.get("incomplete_reset_count", -1)) != 0
    ):
        raise ValueError("N800 supply screen reset trust changed")
    return report, status


def _run_formal_episode(
    root: Path,
    output: Path,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    keys: set[tuple[str, int]],
    qualification: Path,
) -> dict[str, Any]:
    controller = str(item["controller"])
    collection = _formal_controller_root(output, controller)
    key = {(str(item["task_id"]), int(item["solver_seed"]))}
    run_closed_loop_collection(
        root / str(config["cohort"]["dataset"]),
        _runtime_config_path(output, config, "formal"),
        collection,
        phase="realized_dynamic",
        workers=1,
        resume=True,
        task_ids=list(TASK_IDS),
        cohort_job_keys=keys,
        job_keys=key,
        qualification_source=qualification,
        qualification_process_timeout_seconds=float(
            config["formal"]["episode_process_timeout_seconds"]
        ),
        use_global_collection_lock=False,
        **controller_kwargs(root, config, controller, phase="formal"),
    )
    manifest = _manifest_row(
        collection, str(item["task_id"]), int(item["solver_seed"])
    )
    if manifest is None:
        raise RuntimeError("N800 supply formal episode produced no manifest")
    return manifest


def _formal_key_result(
    config: Mapping[str, Any],
    output: Path,
    schedule_rows: list[dict[str, Any]],
    screen_key: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    task_id = str(screen_key["task_id"])
    seed = int(screen_key["solver_seed"])
    items = [
        row
        for row in schedule_rows
        if str(row["task_id"]) == task_id and int(row["solver_seed"]) == seed
    ]
    errors: list[str] = []
    manifests: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    wall = float(config["formal"]["wall_time_budget_seconds"])
    for item in items:
        controller = str(item["controller"])
        manifest = _manifest_row(
            _formal_controller_root(output, controller), task_id, seed
        )
        if manifest is None:
            errors.append(f"{task_id}@{seed}/{controller}: missing formal manifest")
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
        or len(
            {int(row.get("initial_conflicts", -1)) for row in summaries.values()}
        )
        != 1
    ):
        errors.append(f"{task_id}@{seed}: paired formal reset mismatch")

    baseline_root = _formal_controller_root(output, "v2_only")
    challenger_root = _formal_controller_root(output, "consensus16_rescue")
    try:
        baseline_decisions = _decision_rows(baseline_root, manifests["v2_only"])
        challenger_decisions = _decision_rows(
            challenger_root, manifests["consensus16_rescue"]
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"{task_id}@{seed}: formal trace loading failed: {error}")
        return None, errors
    if len(baseline_decisions) != int(
        summaries["v2_only"].get("repair_iterations", -1)
    ):
        errors.append(f"{task_id}@{seed}/v2_only: trace/repair count mismatch")
    if len(challenger_decisions) != int(
        summaries["consensus16_rescue"].get("repair_iterations", -1)
    ):
        errors.append(f"{task_id}@{seed}/consensus16_rescue: trace/repair mismatch")
    trace_cardinality_matches = bool(
        len(baseline_decisions)
        == int(summaries["v2_only"].get("repair_iterations", -1))
        and len(challenger_decisions)
        == int(summaries["consensus16_rescue"].get("repair_iterations", -1))
    )
    try:
        audit = _consensus_trace_audit(challenger_decisions, baseline_decisions)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"{task_id}@{seed}: formal consensus trace audit failed: {error}")
        return None, errors

    totals = dict(summaries["consensus16_rescue"].get("controller_totals") or {})
    offered = int(bool(audit.get("offered")))
    executed = int(bool(audit.get("executed")))
    no_consensus = int(bool(audit.get("no_consensus")))
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
    forbidden_counter_fields = (
        "bounded_retry_trigger_count",
        "failure_rescue_trigger_count",
        "signature_rescue_executed_count",
    )
    forbidden_retry_count = sum(
        int(totals.get(field, 0)) for field in forbidden_counter_fields
    )
    forbidden_trace_keys = (
        "bounded_native_retry",
        "failure_informed_rescue",
        "signature_scoped_rescue",
    )
    forbidden_trace_record_count = sum(
        any(key in dict(row.get("controller") or {}) for key in forbidden_trace_keys)
        for row in baseline_decisions + challenger_decisions
    )
    augmentation = dict(
        controller_kwargs(
            Path(__file__).resolve().parents[1],
            config,
            "consensus16_rescue",
            phase="formal",
        ).get("hybridstructpool_augmentation")
        or {}
    )
    rescue_contract = dict(augmentation.get("v2_first_rescue") or {})
    one_pp_contract = bool(
        int(rescue_contract.get("maximum_pp_calls_per_decision", -1)) == 1
        and rescue_contract.get("same_decision_retry") is False
        and rescue_contract.get("repairer") == "PP"
        and int(augmentation.get("maximum_total_candidates", -1)) == 1
    )
    one_pp_audited = bool(
        trace_cardinality_matches
        and one_pp_contract
        and forbidden_retry_count == 0
        and forbidden_trace_record_count == 0
    )
    if not one_pp_audited:
        errors.append(f"{task_id}@{seed}: more than one PP route entered a decision")

    replayed = bool(
        audit.get("offered")
        and audit.get("executed")
        and int(audit.get("offer_decision_index", -1))
        == int(screen_key["offer_decision_index"])
        and str(audit.get("trigger_fingerprint") or "")
        == str(screen_key["trigger_fingerprint"])
        and list(audit.get("component_agents") or ())
        == list(audit.get("hotspot_agents") or ())
        and bool(audit.get("component_agents"))
        and list(audit.get("component_agents") or ())
        == list(screen_key.get("consensus_agents") or ())
        and json_fingerprint(list(audit.get("component_agents") or ()))
        == str(screen_key.get("consensus_agents_sha256") or "")
    )
    baseline = summaries["v2_only"]
    challenger = summaries["consensus16_rescue"]
    baseline_ttf = float(baseline["capped_wall_time_to_feasible"])
    challenger_ttf = float(challenger["capped_wall_time_to_feasible"])
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
        "baseline_ttf": baseline_ttf,
        "challenger_ttf": challenger_ttf,
        "restricted_ttf_regression": _relative_regression(
            baseline_ttf, challenger_ttf
        ),
        "paired_faster": challenger_ttf < baseline_ttf,
        "screen_replay": {
            "passed": replayed,
            "screen_offer_decision_index": int(screen_key["offer_decision_index"]),
            "formal_offer_decision_index": audit.get("offer_decision_index"),
            "screen_trigger_fingerprint": str(screen_key["trigger_fingerprint"]),
            "formal_trigger_fingerprint": audit.get("trigger_fingerprint"),
            "exact_consensus_executed": bool(audit.get("executed")),
        },
        "rescue": {
            "offered": bool(audit.get("offered")),
            "executed": bool(audit.get("executed")),
            "no_consensus": bool(audit.get("no_consensus")),
            "decision_index": audit.get("offer_decision_index"),
            "repeat_exact_rollback": audit.get(
                "immediate_rescue_repeat_exact_rollback"
            ),
        },
        "baseline_counterfactual": audit.get("baseline_counterfactual"),
        "trace_audit": audit,
        "maximum_pp_calls_per_decision_audited": one_pp_audited,
        "one_pp_per_decision_certificate": {
            "one_trace_transition_per_repair_iteration": trace_cardinality_matches,
            "augmentation_maximum_pp_calls": rescue_contract.get(
                "maximum_pp_calls_per_decision"
            ),
            "same_decision_retry": rescue_contract.get("same_decision_retry"),
            "maximum_total_candidates": augmentation.get("maximum_total_candidates"),
            "forbidden_retry_counter_sum": forbidden_retry_count,
            "forbidden_retry_trace_record_count": forbidden_trace_record_count,
            "passed": one_pp_audited,
        },
        "candidate_generation_seconds": float(
            totals.get("candidate_generation_seconds", 0.0)
        ),
        "neighborhood_selection_seconds": float(
            totals.get("neighborhood_selection_seconds", 0.0)
        ),
        "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
    }, errors


def evaluate_formal(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in rows]
    if len(selected) != 4:
        raise ValueError("formal evaluation requires exactly four paired keys")
    baseline_mean = statistics.fmean(float(row["baseline_ttf"]) for row in selected)
    challenger_mean = statistics.fmean(
        float(row["challenger_ttf"]) for row in selected
    )
    mean_reduction = (
        (baseline_mean - challenger_mean) / baseline_mean
        if baseline_mean > 0.0
        else 0.0 if challenger_mean <= baseline_mean else float("-inf")
    )
    replay_count = sum(bool(row["screen_replay"]["passed"]) for row in selected)
    baseline_success = sum(bool(row["baseline_success"]) for row in selected)
    challenger_success = sum(bool(row["challenger_success"]) for row in selected)
    paired_faster = sum(bool(row["paired_faster"]) for row in selected)
    maximum_regression = max(
        float(row["restricted_ttf_regression"]) for row in selected
    )
    repeat = _repeat_rollback_metrics(selected)
    repeat_reduction = repeat["post_trigger_repeat_rollback_rate_reduction"]
    gates = {
        "formal_trigger_and_exact_consensus_replay_4_of_4": replay_count == 4,
        "success_count_not_below_v2": challenger_success >= baseline_success,
        "mean_restricted_ttf_reduction_at_least_5pct": mean_reduction >= 0.05,
        "paired_faster_at_least_3_of_4": paired_faster >= 3,
        "any_key_restricted_ttf_regression_at_most_20pct": (
            maximum_regression <= 0.20
        ),
        "immediate_exact_rollback_rate_reduction_at_least_25pp": (
            repeat_reduction is not None and float(repeat_reduction) >= 0.25
        ),
        "one_pp_call_route_per_decision": all(
            bool(row["maximum_pp_calls_per_decision_audited"]) for row in selected
        ),
    }
    passed = all(gates.values())
    decision_status = (
        "PASS"
        if passed
        else "INCONCLUSIVE_FORMAL_REPLAY"
        if replay_count < 4
        else "FAIL"
    )
    return {
        "passed": passed,
        "decision_status": decision_status,
        "gates": gates,
        "formal_replay_key_count": replay_count,
        "baseline_success_count": baseline_success,
        "challenger_success_count": challenger_success,
        "baseline_mean_restricted_ttf": baseline_mean,
        "challenger_mean_restricted_ttf": challenger_mean,
        "mean_restricted_ttf_reduction": mean_reduction,
        "paired_faster_key_count": paired_faster,
        "maximum_any_key_restricted_ttf_regression": maximum_regression,
        "immediate_exact_rollback": repeat,
    }


def _formal_qualification_sha256(output: Path) -> str:
    path = _formal_qualification_root(output) / "qualification_report.json"
    if not path.is_file():
        raise ValueError("formal fresh-reset qualification report is missing")
    report = read_json(path)
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or int(report.get("valid_count", -1)) != 4
        or int(report.get("incomplete_reset_count", -1)) != 0
    ):
        raise ValueError("formal fresh-reset qualification is invalid")
    return sha256_file(path)


def analyze_final(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    stage = _formal_root(output)
    completed = load_completed_report(
        stage,
        status_filename=STATUS_FILENAME,
        report_filename=FINAL_REPORT_FILENAME,
        status_schema=_status_schema(config),
        report_schema=_final_report_schema(config),
        config_path=path,
    )
    if completed is not None:
        qualification_sha = _formal_qualification_sha256(output)
        completed_status = read_json(stage / STATUS_FILENAME)
        if (
            dict(completed.get("inputs") or {}).get(
                "fresh_reset_qualification_report_sha256"
            )
            != qualification_sha
            or not isinstance(completed_status, dict)
            or completed_status.get("fresh_reset_qualification_report_sha256")
            != qualification_sha
        ):
            raise ValueError("completed formal qualification trust changed")
        return completed
    effective_producer = dict(producer) if producer is not None else _producer(root)
    screen_report, screen_status = _load_trusted_screen(
        path, root, output, producer=effective_producer
    )
    selected = [dict(row) for row in screen_report["selected_keys"]]
    rows = formal_schedule(config, selected)
    paired: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        formal_qualification_sha256 = _formal_qualification_sha256(output)
    except (OSError, TypeError, ValueError) as error:
        formal_qualification_sha256 = None
        errors.append(str(error))
    selected_by_task = {str(row["task_id"]): row for row in selected}
    for task_id in TASK_IDS:
        result, key_errors = _formal_key_result(
            config, output, rows, selected_by_task[task_id]
        )
        errors.extend(key_errors)
        if result is not None:
            paired.append(result)
    summary = evaluate_formal(paired) if len(paired) == 4 else None
    integrity_passed = not errors and len(paired) == 4
    passed = bool(integrity_passed and summary is not None and summary["passed"])
    decision_status = (
        "INVALID"
        if not integrity_passed
        else str(summary["decision_status"])
        if summary is not None
        else "INVALID"
    )
    report = {
        "schema": _final_report_schema(config),
        "experiment_id": _experiment_id(config),
        "scientific_status": (
            str(config["scientific_status"])
            if _is_overlay(config)
            else "prefix_screened_ttf_unseen_n800_conditional_effect_test"
        ),
        "integrity_passed": integrity_passed,
        "errors": errors,
        "passed": passed,
        "decision_status": decision_status,
        "decision": (
            "retain_as_conditional_n800_warehouse_evidence_only"
            if passed
            else "stop_without_replenishment_or_timeout_extension"
        ),
        "completed_paired_key_count": len(paired),
        "per_key": {str(row["key_id"]): row for row in paired},
        "summary": summary,
        "screen_report_sha256": sha256_file(
            _screen_root(output) / SCREEN_REPORT_FILENAME
        ),
        "screen_status_sha256": sha256_file(
            _screen_root(output) / STATUS_FILENAME
        ),
        "screen_run_fingerprint": str(screen_status["run_fingerprint"]),
        "selected_keys_sha256": str(screen_report["selected_keys_sha256"]),
        "formal_schedule_sha256": json_fingerprint(rows),
        "timed_workers": 1,
        "strict_serial_ttf": True,
        "fresh_qualification_workers": 16,
        "screen_prefix_state_or_cache_reused": False,
        "historically_reset_qualified_candidates": _is_overlay(config),
        "budget_corrected_profile": _is_v3(config),
        "registered_wall_time_budget_seconds": float(
            config["formal"]["wall_time_budget_seconds"]
        ),
        "registered_environment_time_limit_seconds": float(
            config["formal"]["environment_time_limit_seconds"]
        ),
        "registered_episode_process_timeout_seconds": float(
            config["formal"]["episode_process_timeout_seconds"]
        ),
        "diagnostic_state_or_cache_reused": False,
        "post_reset_task_or_seed_filtering": False,
        "absolute_ttf_comparison_to_v1_or_v2_allowed": False,
        "automatic_timeout_extension": False,
        "claim_scope": str(config["claim_boundary"]["maximum_claim"]),
        "trigger_prevalence_claim_allowed": False,
        "warehouse_generalization_allowed": False,
        "default_promotion_allowed": False,
        "inputs": {
            "config_sha256": sha256_file(path),
            "fresh_reset_qualification_report_sha256": (
                formal_qualification_sha256
            ),
            "budget_diagnostic_qualification_report_sha256": (
                str(
                    config["budget_diagnostic_evidence"]
                    ["qualification_report"]["sha256"]
                )
                if _is_v3(config)
                else None
            ),
        },
        "producer_identity": effective_producer,
    }
    write_json(stage / FINAL_REPORT_FILENAME, report)
    return report


def _formal_status(
    base: Mapping[str, Any],
    output: Path,
    rows: list[dict[str, Any]],
    *,
    complete: bool = False,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = _completed_formal_count(output, rows)
    result = {
        **dict(base),
        "phase": "formal",
        "completed_schedule_entries": completed,
        "complete": bool(complete and completed == len(rows)),
        "timed_workers": 1,
        "strict_serial_ttf": True,
        "fresh_qualification_workers": 16,
        "screen_prefix_state_or_cache_reused": False,
        "automatic_timeout_extension": False,
    }
    if terminal_failure is not None:
        result["decision_status"] = "INVALID"
        result["terminal_failure"] = dict(terminal_failure)
    return result


def run_formal(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    if dry_run:
        return {
            **dict(plan(path)["formal"]),
            "schema": _status_schema(config),
            "experiment_id": _experiment_id(config),
            "phase": "formal",
            "solver_or_controller_invoked": False,
        }
    output = Path(output).resolve()
    _reject_v1_output_identity(root, output, config)
    producer = _producer(root)
    try:
        screen_report, _screen_status_payload = _load_trusted_screen(
            path, root, output, producer=producer
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema": _status_schema(config),
            "experiment_id": _experiment_id(config),
            "phase": "formal",
            "blocked": True,
            "decision_status": "BLOCKED_SCREEN_NOT_PASSED",
            "error": str(error),
            "solver_or_controller_invoked": False,
        }
    selected = [dict(row) for row in screen_report["selected_keys"]]
    rows = formal_schedule(config, selected)
    stage = _formal_root(output)
    prepared = prepare_resumable_output(
        stage,
        status_filename=STATUS_FILENAME,
        status_schema=_status_schema(config),
        config_path=path,
        schedule=rows,
        producer=producer,
        resume=resume,
        report_filename=FINAL_REPORT_FILENAME,
        report_schema=_final_report_schema(config),
        label="N800 trigger-qualified formal TTF",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.status.get("terminal_failure") is not None:
        return dict(prepared.status)
    keys = _job_keys(rows)
    try:
        qualification = _reset_qualification(
            root,
            output,
            config,
            phase="formal",
            keys=keys,
            resume=prepared.resumed,
        )
        for controller in CONTROLLERS:
            _prepare_collection_root(
                root,
                output,
                config,
                phase="formal",
                controller=controller,
                keys=keys,
                qualification=qualification,
            )
    except Exception as error:
        status = _formal_status(
            prepared.base_status,
            output,
            rows,
            terminal_failure={
                "phase": "fresh_reset_qualification",
                "error": f"{type(error).__name__}: {error}",
                "screen_state_or_cache_reused": False,
            },
        )
        write_json(stage / STATUS_FILENAME, status)
        return status

    for item in rows:
        collection = _formal_controller_root(output, str(item["controller"]))
        if _manifest_row(
            collection, str(item["task_id"]), int(item["solver_seed"])
        ) is not None:
            continue
        try:
            manifest = _run_formal_episode(
                root, output, config, item, keys, qualification
            )
        except Exception as error:
            status = _formal_status(
                prepared.base_status,
                output,
                rows,
                terminal_failure={
                    "phase": "strict_serial_ttf_episode",
                    "item": dict(item),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            write_json(stage / STATUS_FILENAME, status)
            return status
        if manifest.get("status") in {"error", "timeout"}:
            status = _formal_status(
                prepared.base_status,
                output,
                rows,
                terminal_failure={
                    "phase": "strict_serial_ttf_episode",
                    "item": dict(item),
                    "error": str(manifest.get("error") or manifest.get("status")),
                },
            )
            write_json(stage / STATUS_FILENAME, status)
            return status
        write_json(
            stage / STATUS_FILENAME,
            _formal_status(prepared.base_status, output, rows),
        )
    report = analyze_final(path, output, producer=producer)
    status = _formal_status(
        prepared.base_status, output, rows, complete=True
    )
    status["decision_status"] = str(report["decision_status"])
    status["fresh_reset_qualification_report_sha256"] = (
        _formal_qualification_sha256(output)
    )
    status["report_sha256"] = sha256_file(stage / FINAL_REPORT_FILENAME)
    write_json(stage / STATUS_FILENAME, status)
    return report


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return plan(config_path)
    screened = run_screen(config_path, output, resume=resume)
    if screened.get("formal_ttf_allowed") is not True:
        return screened
    return run_formal(config_path, output, resume=resume)


__all__ = [
    "CANDIDATE_SEEDS",
    "CONFIG_SCHEMA",
    "CONTROLLERS",
    "EXPERIMENT_ID",
    "FINAL_REPORT_SCHEMA",
    "MAXIMUM_SCREEN_TRACE_DECISIONS",
    "PREFIX_V2_OUTCOMES",
    "SCREEN_REPORT_SCHEMA",
    "STATUS_SCHEMA",
    "TASK_IDS",
    "V2_EXPERIMENT_ID",
    "V2_STATUS_SCHEMA",
    "V3_CONFIG_SCHEMA",
    "V3_EXPERIMENT_ID",
    "V3_FINAL_REPORT_SCHEMA",
    "V3_SCREEN_REPORT_SCHEMA",
    "V3_STATUS_SCHEMA",
    "_screen_preaction_certificate",
    "analyze_final",
    "analyze_screen",
    "controller_kwargs",
    "evaluate_formal",
    "formal_schedule",
    "load_config",
    "plan",
    "run",
    "run_formal",
    "run_screen",
    "screen_schedule",
    "select_screen_keys",
]
