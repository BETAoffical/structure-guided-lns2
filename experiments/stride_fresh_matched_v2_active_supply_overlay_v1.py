from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_fresh_matched_v2_c16_h16_v1 import (
    ARMS,
    STRUCTURAL_FAMILIES,
    TRIAL_INDICES,
    _h1_failure_result as _v1_h1_failure_result,
    _h1_state_artifact_valid,
    _h1_state_worker,
    _preflight_decision,
    analyze_h1_payloads,
)
from experiments.trace_replay import result_blind_decision_rows


CONFIG_SCHEMA = "lns2.stride.fresh_matched_v2_active_supply_overlay_config.v1"
PLAN_SCHEMA = "lns2.stride.fresh_matched_v2_active_supply_overlay_plan.v1"
PREFLIGHT_EPISODE_SCHEMA = (
    "lns2.stride.fresh_matched_v2_active_supply_overlay_preflight_episode.v1"
)
SELECTION_SCHEMA = "lns2.stride.fresh_matched_v2_active_supply_overlay_selection.v1"
H1_STATE_ENVELOPE_SCHEMA = (
    "lns2.stride.fresh_matched_v2_active_supply_overlay_h1_state_envelope.v1"
)
H1_REPORT_SCHEMA = "lns2.stride.fresh_matched_v2_active_supply_overlay_h1_report.v1"
EXPERIMENT_ID = "stride_fresh_matched_v2_active_supply_overlay_v1"
PROFILE = "realized_dynamic"
V1_EXPERIMENT_ID = "stride_fresh_matched_v2_c16_h16_v1"
FOLDS = {
    "fold0": (
        "den520d",
        "maze-128-128-10",
        "warehouse-10-20-10-2-1",
        "maze-128-128-1",
    ),
    "fold1": (
        "lak303d",
        "random-32-32-10",
        "room-32-32-4",
        "random-32-32-20",
    ),
    "fold2": (
        "maze-32-32-2",
        "random-64-64-10",
        "warehouse-20-40-10-2-1",
        "maze-32-32-4",
    ),
    "fold3": (
        "random-64-64-20",
        "room-64-64-8",
        "warehouse-20-40-10-2-2",
        "room-64-64-16",
    ),
}
WAVE_TASK_IDS = {
    "balanced_wall_clock": {
        "random-32-32-20__random_01__agents_0400",
        "random-32-32-20__random_02__agents_0400",
    },
    "movingai_ood": {
        "maze-128-128-1__random_04__agents_0600",
        "maze-128-128-1__random_05__agents_0600",
        "maze-32-32-4__random_04__agents_0200",
        "maze-32-32-4__random_05__agents_0200",
        "random-64-64-10__random_04__agents_0600",
        "random-64-64-10__random_05__agents_0600",
        "room-64-64-16__random_04__agents_0600",
        "room-64-64-16__random_05__agents_0600",
    },
}
WAVE_TASK_IDENTITIES = {
    "random-32-32-20__random_01__agents_0400": (
        "5fc8509b1c2415a88ec021e6a225f84b79b7b027ef98ed3fad6bf1d058999b54",
        "random-32-32-20",
        "random",
        "8c5a83498ab92a2579aeef91c5f42d9ecf9019ef038cf98e623061a15beb6f56",
    ),
    "random-32-32-20__random_02__agents_0400": (
        "c9cff3c1d5dff1e423f6892a5f72611b424fd23f16ad1c4827457770fc226c1f",
        "random-32-32-20",
        "random",
        "8c5a83498ab92a2579aeef91c5f42d9ecf9019ef038cf98e623061a15beb6f56",
    ),
    "maze-128-128-1__random_04__agents_0600": (
        "c8422d1ddac049f9a777d30acc4b72ea06276e60759c107b6cca2bbe64693494",
        "maze-128-128-1",
        "maze",
        "9ef42dc6c43a2b07364c9678b7501a6a7ff1fc7e61215abae6d34000ba8e70c9",
    ),
    "maze-128-128-1__random_05__agents_0600": (
        "d6030daa5ab2458179f096de25ded739ebb6dbf23e9b63e8436e6bec7e384ca4",
        "maze-128-128-1",
        "maze",
        "9ef42dc6c43a2b07364c9678b7501a6a7ff1fc7e61215abae6d34000ba8e70c9",
    ),
    "maze-32-32-4__random_04__agents_0200": (
        "c441ca5a2f7e5178bba952d80ec8769c6e123df88d772a1c1293213733c4abd7",
        "maze-32-32-4",
        "maze",
        "7ff67aa59f71933b8cf2605e12631b8a28d9ebcfb9b941de3afdc7dce3123fee",
    ),
    "maze-32-32-4__random_05__agents_0200": (
        "f6cd1843c9f29c6e7cebe20dba0477da5b1e6497fa148c26e258343799ea7e75",
        "maze-32-32-4",
        "maze",
        "7ff67aa59f71933b8cf2605e12631b8a28d9ebcfb9b941de3afdc7dce3123fee",
    ),
    "random-64-64-10__random_04__agents_0600": (
        "347c2c9e4a1f9d5748cf6115fe90bbaac664832d84883ccf7f8480ea5e25db52",
        "random-64-64-10",
        "random",
        "b31c671228f884a113ca11c41b83630dc042e58e07f9b36da74ec508f82a5659",
    ),
    "random-64-64-10__random_05__agents_0600": (
        "1f2af6fcb522e595a44c722dca75aa0917bcd36af72a70b6102fe5be6bdd4405",
        "random-64-64-10",
        "random",
        "b31c671228f884a113ca11c41b83630dc042e58e07f9b36da74ec508f82a5659",
    ),
    "room-64-64-16__random_04__agents_0600": (
        "0e4a14b9ba0a801f40f932e64d15b2c55bc6e26ee7bbbf50957e31e3d6e3297c",
        "room-64-64-16",
        "room",
        "983df5c9bf0c59799daa107feb1b2d3ed81c5d32bf4b23d019f06161d0be6092",
    ),
    "room-64-64-16__random_05__agents_0600": (
        "a2303cd059b6bbc7040bf658de6b57dd21b5c5b3012a853c317d6e01a9175b77",
        "room-64-64-16",
        "room",
        "983df5c9bf0c59799daa107feb1b2d3ed81c5d32bf4b23d019f06161d0be6092",
    ),
}
WAVE_SOURCE_INPUTS = {
    "balanced_wall_clock": {
        "dataset_root": "build/stride-stage2-movingai-v1/balanced_wall_clock",
        "manifest": {
            "path": "build/stride-stage2-movingai-v1/balanced_wall_clock/manifest.jsonl",
            "sha256": "efe524b6397642f85964bc47124c348ed8280dd083170e0b33b5c2741573b132",
        },
        "runtime": {
            "path": "configs/stride_stage4r_high_load_runtime.json",
            "sha256": "5ff9264ca8ef3cfca7bb344ca6ece8ebc34ac454561ca4558d485b24bd671aca",
        },
        "source_split": "balanced_wall_clock",
        "dataset_design_from_v1_source": "balanced_wall_clock",
    },
    "movingai_ood": {
        "dataset_root": "build/initlns-movingai-ood-dataset-v1/movingai_ood",
        "manifest": {
            "path": "build/initlns-movingai-ood-dataset-v1/movingai_ood/manifest.jsonl",
            "sha256": "cab89325c4ca62c7eefd7a0c5de38e1a700bf8384b618c399eafab876dd05742",
        },
        "runtime": {
            "path": "configs/stride_hybridstructpool_result_blind_movingai_runtime.json",
            "sha256": "27c6cc29cc59c1cf00fc29d6220a790d212d7634c139afbdc476c69768e2a936",
        },
        "source_split": "movingai_ood",
        "dataset_design_from_v1_source": "movingai_ood",
    },
}
FORBIDDEN_Q0_OUTCOME_KEYS = {
    "after",
    "after_conflicts",
    "conflicts_after",
    "final_success",
    "native_step_seconds",
    "outcome",
    "pp_seconds",
    "pp_time",
    "repair_outcome",
    "repair_status",
    "replan_success",
    "success",
    "target",
    "target_outcome",
    "target_result",
    "trials",
    "ttf",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _registered(root: Path, spec: dict[str, Any], label: str) -> Path:
    return registered_input(root, spec, label=label)


def _wave_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = config.get("wave_a_sources")
    if not isinstance(sources, dict):
        raise ValueError("Wave-A source registry is missing")
    return {str(key): dict(value) for key, value in sources.items()}


def _trace_registry(
    root: Path, manifests: dict[str, Path]
) -> tuple[list[dict[str, Any]], int]:
    registry = []
    for source_id, manifest_path in sorted(manifests.items()):
        source_root = manifest_path.parent
        for row in _read_jsonl(manifest_path):
            trace_path = contained_file(
                source_root, row.get("trace_file"), field="read-only source trace"
            )
            observed = sha256_file(trace_path)
            if observed != str(row.get("trace_sha256")):
                raise ValueError(f"source trace changed: {source_id}:{row.get('episode_id')}")
            registry.append(
                {
                    "source_id": source_id,
                    "episode_id": str(row["episode_id"]),
                    "trace_file": str(row["trace_file"]),
                    "trace_sha256": observed,
                }
            )
    registry.sort(key=lambda row: (row["source_id"], row["episode_id"]))
    return registry, len(registry)


def validate_config(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_active_supply_overlay_h1_opportunity_collection"
        or config.get("freshness")
        != "fresh_exact_task_solver_state_cohort_with_read_only_v1_source_reuse"
        or config.get("research_split") != "fresh_matched_development"
    ):
        raise ValueError("active-supply overlay identity changed")
    v1 = dict(config.get("v1_read_only") or {})
    if (
        set(v1)
        != {
            "config",
            "output_root",
            "source_report",
            "preflight_report",
            "source_manifests",
            "trace_registry_sha256",
            "expected_episode_count",
            "expected_zero_prefix_inactive_episode_count",
            "mutation_allowed",
        }
        or int(v1.get("expected_episode_count", -1)) != 48
        or int(v1.get("expected_zero_prefix_inactive_episode_count", -1)) != 25
        or v1.get("mutation_allowed") is not False
        or v1.get("output_root") != "build/stride-fresh-matched-v2-c16-h16-v1"
        or v1.get("config")
        != {
            "path": "configs/stride_fresh_matched_v2_c16_h16_v1.json",
            "sha256": "7ccef2c91853f8fcc90ac1f94f721269f31d0f9af16c83fb95eab61a81f6cdad",
        }
        or v1.get("source_report")
        != {
            "path": "build/stride-fresh-matched-v2-c16-h16-v1/source_collection_report.json",
            "sha256": "513ba5c9ecb88c58b683ef144ce2e0653381d96d0d75cd058ef4a2ae38f14a8c",
        }
        or v1.get("preflight_report")
        != {
            "path": "build/stride-fresh-matched-v2-c16-h16-v1/preflight_report.json",
            "sha256": "da45e154c1140c9e0e2e9f4f1b8c76f222f19687ce706bc2f1f684dcc995a8a4",
        }
        or v1.get("source_manifests")
        != {
            "balanced_wall_clock": {
                "path": "build/stride-fresh-matched-v2-c16-h16-v1/source/balanced_wall_clock/realized_dynamic_manifest.jsonl",
                "sha256": "19e0e7f73074187aed311462a6bc746250da4cc4080381853b0932ceef463dc6",
            },
            "movingai_ood": {
                "path": "build/stride-fresh-matched-v2-c16-h16-v1/source/movingai_ood/realized_dynamic_manifest.jsonl",
                "sha256": "a1ef41f2bb2d4f4747f090d2c7c2e0996ce13339493a4fd5b89fbcfcde17bd2e",
            },
        }
        or v1.get("trace_registry_sha256")
        != "c8b3b9f9713d8220264dfab12667ed22fa7604fccf6feabb86391fffc144e796"
    ):
        raise ValueError("V1 read-only reuse contract changed")
    bundle = dict(config.get("controller_bundle") or {})
    if bundle != {
        "controller": "v2-full",
        "manifest": {
            "path": "artifacts/initlns-closed-loop-controller-v2/controller_manifest.json",
            "sha256": "1b699182f9890148d0030e691b457c82ac7054760534665ad65484afb0ac82a8",
        },
    }:
        raise ValueError("active-supply V2 controller bundle changed")
    sources = _wave_sources(config)
    if set(sources) != {"balanced_wall_clock", "movingai_ood"}:
        raise ValueError("Wave-A source identities changed")
    tasks = []
    for source_id, source in sources.items():
        if set(source) != {
            "dataset_root",
            "manifest",
            "runtime",
            "source_split",
            "dataset_design_from_v1_source",
            "tasks",
        }:
            raise ValueError(f"Wave-A source fields changed: {source_id}")
        if source["dataset_design_from_v1_source"] != source_id:
            raise ValueError("Wave-A dataset-design inheritance changed")
        if {key: value for key, value in source.items() if key != "tasks"} != WAVE_SOURCE_INPUTS[source_id]:
            raise ValueError(f"Wave-A registered input changed: {source_id}")
        if {str(task["task_id"]) for task in source["tasks"]} != WAVE_TASK_IDS[source_id]:
            raise ValueError(f"Wave-A exact task IDs changed: {source_id}")
        for task_value in source["tasks"]:
            task = dict(task_value)
            if set(task) != {
                "task_id",
                "task_sha256",
                "map_id",
                "map_family",
                "map_sha256",
            }:
                raise ValueError("Wave-A task identity fields changed")
            task_id = str(task["task_id"])
            if (
                str(task["task_sha256"]),
                str(task["map_id"]),
                str(task["map_family"]),
                str(task["map_sha256"]),
            ) != WAVE_TASK_IDENTITIES[task_id]:
                raise ValueError(f"Wave-A frozen task identity changed: {task_id}")
        tasks.extend((source_id, dict(task)) for task in source["tasks"])
    if (
        len(tasks) != 10
        or len({task["task_id"] for _, task in tasks}) != 10
        or len({task["map_id"] for _, task in tasks}) != 5
        or Counter(task["map_family"] for _, task in tasks)
        != Counter({"random": 4, "maze": 4, "room": 2})
    ):
        raise ValueError("Wave-A exact 10-task/5-block cohort changed")
    source_contract = dict(config.get("wave_a_source_collection") or {})
    if source_contract != {
        "policy": "v2-full",
        "solver_seeds": [41, 42],
        "expected_episode_count": 20,
        "max_decisions": 12,
        "workers": 16,
        "environment_time_limit_seconds": 200.0,
        "wall_time_budget_seconds": 200.0,
        "episode_process_timeout_seconds": 240.0,
        "deterministic_pp_replay": True,
        "fresh_reset_qualification_required": True,
    }:
        raise ValueError("Wave-A source execution contract changed")
    selection = dict(config.get("state_selection") or {})
    if selection != {
        "target_state_count": 96,
        "states_per_fold": 24,
        "maximum_states_per_episode": 4,
        "maximum_states_per_map": 16,
        "minimum_selected_map_count": 10,
        "minimum_selected_family_count": 4,
        "structural_required_actual_size": 16,
        "required_pairwise_distinct_arms": True,
        "rank_rule": "sha256_preaction_identity_ascending_with_frozen_quota_fill",
        "depth_bands": {"d0": [0, 0], "d1_3": [1, 3], "d4_plus": [4, 11]},
        "minimum_states_per_depth_band_per_fold": 4,
        "failure_action": "STATE_SUPPLY_FAIL_h1_zero_no_reserve",
        "target_outcome_fields_read": False,
        "final_success_or_pp_outcomes_read": False,
    }:
        raise ValueError("active-supply Q0 selection contract changed")
    folds = {key: tuple(value) for key, value in dict(config.get("map_folds") or {}).items()}
    if folds != FOLDS or len({map_id for values in folds.values() for map_id in values}) != 16:
        raise ValueError("active-supply map folds changed")
    h1 = dict(config.get("h1") or {})
    if h1 != {
        "arms": list(ARMS),
        "trial_indices": list(TRIAL_INDICES),
        "first_fixed_half": list(range(8)),
        "second_fixed_half": list(range(8, 16)),
        "logical_trial_count": 4608,
        "workers": 16,
        "per_action_time_limit_seconds": 5.0,
        "per_state_process_fuse_seconds": 420.0,
        "same_state_trial_seed_across_arms": True,
        "runtime_or_pp_seconds_used_in_label": False,
    }:
        raise ValueError("active-supply H1 execution contract changed")
    label = dict(config.get("h1_label") or {})
    if label != {
        "per_seed_score": "normalized_current_step_conflict_reduction",
        "minimum_strict_paired_wins": 12,
        "minimum_mean_delta": 0.02,
        "require_positive_first_half_mean_delta": True,
        "require_positive_second_half_mean_delta": True,
        "require_no_progress_rate_noninferiority": True,
        "require_rollback_rate_noninferiority": True,
        "require_time_limit_rate_noninferiority": True,
        "tie_epsilon": 1e-12,
        "duplicates_or_unavailable_label": None,
    }:
        raise ValueError("active-supply H1 label contract changed")
    gates = dict(config.get("h1_gates") or {})
    expected_gates = {
        "expected_state_count": 96,
        "minimum_opportunity_state_count": 20,
        "minimum_opportunity_map_count": 8,
        "minimum_opportunity_family_count": 4,
        "each_fold_requires_opportunity_and_nonopportunity": True,
        "valid_candidate_labels_require_both_classes": True,
        "all_integrity_gates_required": True,
        "map_folds": {key: list(value) for key, value in FOLDS.items()},
    }
    if gates != expected_gates:
        raise ValueError("active-supply H1 gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "active_supply_overlay_only": True,
        "h1_opportunity_only": True,
        "h8_executed": False,
        "training_allowed": False,
        "ttf_or_speed_claim_allowed": False,
        "v1_artifact_mutation_allowed": False,
        "reserve_backfill_allowed": False,
        "outcome_based_state_deletion_allowed": False,
    }:
        raise ValueError("active-supply claim boundary changed")
    if project_root is None:
        return
    root = project_root.resolve()
    v1_config_path = _registered(root, dict(v1["config"]), "V1 config")
    v1_config = _read_json(v1_config_path)
    if v1_config.get("experiment_id") != V1_EXPERIMENT_ID:
        raise ValueError("registered V1 config identity changed")
    source_report_path = _registered(root, dict(v1["source_report"]), "V1 source report")
    preflight_report_path = _registered(
        root, dict(v1["preflight_report"]), "V1 preflight report"
    )
    source_report = _read_json(source_report_path)
    preflight_report = _read_json(preflight_report_path)
    if (
        source_report.get("complete") is not True
        or int(source_report.get("observed_episode_count", -1)) != 48
        or source_report.get("config_sha256") != sha256_file(v1_config_path)
        or preflight_report
        != {
            "expected_source_episode_count": 48,
            "reason": "source_episode_has_no_replayable_preaction_rows",
            "schema": "lns2.stride.fresh_matched_v2_c16_h16_preflight_report.v1",
            "source_episode_count": 23,
            "status": "STATE_SUPPLY_FAIL",
        }
    ):
        raise ValueError("V1 read-only source evidence changed")
    manifest_paths = {
        source_id: _registered(root, dict(spec), f"V1 {source_id} manifest")
        for source_id, spec in dict(v1["source_manifests"]).items()
    }
    registry, count = _trace_registry(root, manifest_paths)
    if count != 48 or _fingerprint(registry) != str(v1["trace_registry_sha256"]):
        raise ValueError("V1 trace registry changed")
    _registered(root, dict(bundle["manifest"]), "V2 controller bundle")
    for source_id, source in sources.items():
        manifest = _registered(root, dict(source["manifest"]), f"Wave-A {source_id} manifest")
        _registered(root, dict(source["runtime"]), f"Wave-A {source_id} runtime")
        dataset_root = (root / str(source["dataset_root"])).resolve()
        rows = {str(row["task_id"]): row for row in _read_jsonl(manifest)}
        for task in source["tasks"]:
            row = rows.get(str(task["task_id"]))
            if row is None:
                raise ValueError(f"Wave-A task missing: {task['task_id']}")
            task_path = contained_file(dataset_root, row["task_file"], field="Wave-A task")
            map_path = contained_file(dataset_root, row["map_file"], field="Wave-A map")
            if (
                sha256_file(task_path) != task["task_sha256"]
                or sha256_file(map_path) != task["map_sha256"]
                or row["map_id"] != task["map_id"]
                or row["layout_mode"] != task["map_family"]
            ):
                raise ValueError(f"Wave-A task provenance changed: {task['task_id']}")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_config(config, project_root=root)
    return path, root, config


def _overlay_output_root(
    root: Path, config: dict[str, Any], output: str | Path
) -> Path:
    output_root = Path(output).resolve()
    v1_output_root = (root / str(config["v1_read_only"]["output_root"])).resolve()
    if output_root == v1_output_root or v1_output_root in output_root.parents:
        raise ValueError("V2 overlay output must remain outside the read-only V1 output")
    return output_root


def wave_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "source_id": source_id,
            "task_id": str(task["task_id"]),
            "map_id": str(task["map_id"]),
            "map_family": str(task["map_family"]),
            "solver_seed": int(seed),
        }
        for source_id, source in _wave_sources(config).items()
        for task in source["tasks"]
        for seed in config["wave_a_source_collection"]["solver_seeds"]
    ]
    rows.sort(key=lambda row: (row["source_id"], row["task_id"], row["solver_seed"]))
    if len(rows) != 20:
        raise ValueError("Wave-A schedule is not exactly 20 jobs")
    return rows


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, _root, config = load_config(config_path)
    schedule = wave_schedule(config)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "v1_read_only_episode_count": 48,
        "v1_zero_prefix_inactive_episode_count": 25,
        "wave_a_episode_count": len(schedule),
        "combined_source_episode_count": 68,
        "wave_a_workers": 16,
        "target_state_count": 96,
        "states_per_fold": 24,
        "logical_h1_trial_count": 4608,
        "h1_workers": 16,
        "h1_executed_by_dry_run": False,
        "reserve_backfill_allowed": False,
        "v1_mutation_allowed": False,
        "wave_a_schedule_sha256": _fingerprint(schedule),
        "wave_a_schedule": schedule,
    }


def _wave_runtime(
    base: dict[str, Any], source_id: str, config: dict[str, Any], v1_config: dict[str, Any]
) -> dict[str, Any]:
    source = _wave_sources(config)[source_id]
    contract = config["wave_a_source_collection"]
    return {
        **base,
        "split": source["source_split"],
        "solver_seeds": list(contract["solver_seeds"]),
        "policies": ["official_adaptive", PROFILE],
        "environment": {
            **dict(base["environment"]),
            "time_limit": 200.0,
            "unlimited_time": False,
        },
        "dataset_design": dict(v1_config["sources"][source_id]["dataset_design"]),
        "max_decisions": 12,
        "metric_iteration_budget": 12,
        "wall_time_budget_seconds": 200.0,
        "episode_process_timeout_seconds": 240.0,
        "workers": 16,
        "deterministic_pp_replay": True,
    }


def _source_call(
    dataset_root: Path,
    runtime_path: Path,
    output: Path,
    bundle_root: Path,
    task_ids: list[str],
    keys: set[tuple[str, int]],
    *,
    phase: str,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    return run_closed_loop_collection(
        dataset_root,
        runtime_path,
        output,
        phase=phase,
        workers=16,
        resume=resume,
        dry_run=dry_run,
        task_ids=task_ids,
        controller="v2-full",
        feature_backend="native",
        controller_bundle=bundle_root,
        controller_runtime="optimized",
        verification_profile="deployment",
        job_keys=keys,
        cohort_job_keys=keys,
        wall_time_budget_seconds=200.0,
        episode_process_timeout_seconds=240.0,
        environment_time_limit_seconds=200.0,
        qualification_process_timeout_seconds=240.0,
        stopping_rule="historical",
        qualification_source=None,
        deterministic_pp_replay=True,
    )


def run_wave_a_source(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = _overlay_output_root(root, config, output)
    v1_config = _read_json(_registered(root, config["v1_read_only"]["config"], "V1 config"))
    bundle = _registered(root, config["controller_bundle"]["manifest"], "controller")
    summaries = {}
    for source_id, source in sorted(_wave_sources(config).items()):
        dataset_split = (root / source["dataset_root"]).resolve()
        runtime = _wave_runtime(
            _read_json(_registered(root, source["runtime"], f"{source_id} runtime")),
            source_id,
            config,
            v1_config,
        )
        tasks = [str(row["task_id"]) for row in source["tasks"]]
        keys = {(task, seed) for task in tasks for seed in (41, 42)}
        source_output = output_root / "wave_a" / "source" / source_id
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="stride-active-supply-") as temporary:
                runtime_path = Path(temporary) / "runtime.json"
                _write_json(runtime_path, runtime)
                calls = {
                    phase: _source_call(
                        dataset_split.parent,
                        runtime_path,
                        source_output,
                        bundle.parent,
                        tasks,
                        keys,
                        phase=phase,
                        dry_run=True,
                        resume=False,
                    )
                    for phase in ("qualify", PROFILE)
                }
        else:
            source_output.mkdir(parents=True, exist_ok=True)
            runtime_path = source_output / "wave_a_runtime.json"
            if runtime_path.is_file() and _read_json(runtime_path) != runtime:
                raise ValueError(f"Wave-A runtime changed: {source_id}")
            _write_json(runtime_path, runtime)
            preview = _source_call(
                dataset_split.parent,
                runtime_path,
                source_output,
                bundle.parent,
                tasks,
                keys,
                phase="qualify",
                dry_run=True,
                resume=False,
            )
            calls = {
                "preview": preview,
                "qualify": _source_call(
                    dataset_split.parent,
                    runtime_path,
                    source_output,
                    bundle.parent,
                    tasks,
                    keys,
                    phase="qualify",
                    dry_run=False,
                    resume=resume,
                ),
                PROFILE: _source_call(
                    dataset_split.parent,
                    runtime_path,
                    source_output,
                    bundle.parent,
                    tasks,
                    keys,
                    phase=PROFILE,
                    dry_run=False,
                    resume=True,
                ),
            }
        for call in calls.values():
            if "dataset_design" in call and not bool(call["dataset_design"]["passed"]):
                raise ValueError(f"Wave-A dataset design failed: {source_id}")
        summaries[source_id] = calls
    report = {
        "schema": "lns2.stride.fresh_matched_v2_active_supply_overlay_source.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "dry_run": dry_run,
        "v1_artifacts_mutated": False,
        "fresh_qualification_reused": False,
        "expected_wave_a_episode_count": 20,
        "workers": 16,
        "summaries": summaries,
    }
    if not dry_run:
        manifests = {
            source_id: output_root
            / "wave_a"
            / "source"
            / source_id
            / "realized_dynamic_manifest.jsonl"
            for source_id in _wave_sources(config)
        }
        schedule_by_source: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for row in wave_schedule(config):
            schedule_by_source[row["source_id"]].add(
                (row["task_id"], int(row["solver_seed"]))
            )
        for source_id, manifest_path in manifests.items():
            rows = _read_jsonl(manifest_path)
            keys = [
                (str(row.get("task_id")), int(row.get("solver_seed", -1)))
                for row in rows
            ]
            if (
                len(keys) != len(schedule_by_source[source_id])
                or set(keys) != schedule_by_source[source_id]
                or len(keys) != len(set(keys))
                or any(
                    str(row.get("status")) != "ok" or not row.get("trace_file")
                    for row in rows
                )
            ):
                raise RuntimeError(f"Wave-A exact product changed: {source_id}")
        registry, count = _trace_registry(root, manifests)
        if count != 20:
            raise RuntimeError("Wave-A source product is not exactly 20 episodes")
        report.update(
            {
                "complete": True,
                "observed_wave_a_episode_count": count,
                "manifest_sha256": {
                    key: sha256_file(value) for key, value in manifests.items()
                },
                "trace_registry_sha256": _fingerprint(registry),
            }
        )
        _atomic_json(output_root / "wave_a_source_report.json", report)
    return report


def _task_registry(sources: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (source_id, str(task["task_id"])): dict(task)
        for source_id, source in sources.items()
        for task in source["tasks"]
    }


def _read_source_product(
    config: dict[str, Any], output_root: Path, *, origin: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if origin == "v1_reused":
        v1_path = (
            Path(__file__).resolve().parents[1]
            / config["v1_read_only"]["config"]["path"]
        ).resolve()
        v1_config = _read_json(v1_path)
        sources = {key: dict(value) for key, value in v1_config["sources"].items()}
        roots = {
            key: (Path(__file__).resolve().parents[1] / config["v1_read_only"]["output_root"] / "source" / key).resolve()
            for key in sources
        }
        expected = 48
    else:
        sources = _wave_sources(config)
        roots = {
            key: (output_root / "wave_a" / "source" / key).resolve()
            for key in sources
        }
        expected = 20
    registry = _task_registry(sources)
    decisions = []
    audits = []
    manifest_count = 0
    for source_id, source in sorted(sources.items()):
        manifest_path = roots[source_id] / "realized_dynamic_manifest.jsonl"
        manifests = _read_jsonl(manifest_path)
        expected_keys = {
            (str(task["task_id"]), int(seed))
            for task in source["tasks"]
            for seed in (41, 42)
        }
        observed_keys = [
            (str(row.get("task_id")), int(row.get("solver_seed", -1)))
            for row in manifests
        ]
        if (
            len(observed_keys) != len(expected_keys)
            or set(observed_keys) != expected_keys
            or len(observed_keys) != len(set(observed_keys))
        ):
            raise ValueError(f"{origin} exact task/seed product changed: {source_id}")
        for manifest in manifests:
            task_id = str(manifest["task_id"])
            if (
                (source_id, task_id) not in registry
                or str(manifest.get("status")) != "ok"
                or not manifest.get("trace_file")
            ):
                raise ValueError(f"{origin} source episode is incomplete")
            manifest_count += 1
            trace_path = contained_file(roots[source_id], manifest["trace_file"], field="source trace")
            trace_sha = sha256_file(trace_path)
            if trace_sha != str(manifest.get("trace_sha256")):
                raise ValueError("combined source trace changed")
            rows, _events = result_blind_decision_rows(roots[source_id], manifest)
            audits.append(
                {
                    "origin": origin,
                    "source_id": source_id,
                    "episode_id": str(manifest["episode_id"]),
                    "task_id": task_id,
                    "solver_seed": int(manifest["solver_seed"]),
                    "preaction_prefix_count": len(rows),
                    "zero_prefix_inactive": not rows,
                }
            )
            task = registry[(source_id, task_id)]
            for row in rows:
                if int(row["decision_index"]) >= 12:
                    continue
                identity = {
                    "origin": origin,
                    "source_id": source_id,
                    "episode_id": str(manifest["episode_id"]),
                    "task_id": task_id,
                    "solver_seed": int(manifest["solver_seed"]),
                    "decision_index": int(row["decision_index"]),
                    "before_fingerprint": str(row["before_fingerprint"]),
                }
                decisions.append(
                    {
                        **row,
                        **identity,
                        "state_id": "active-supply-preaction-" + _fingerprint(identity)[:24],
                        "source_root": str(roots[source_id]),
                        "source_policy": "v2-full",
                        "split": source["source_split"],
                        "research_split": config["research_split"],
                        "map_id": task["map_id"],
                        "map_family": task["map_family"],
                        "layout_mode": task["map_family"],
                        "agent_count": int(manifest.get("agent_count", 0)),
                        "source_trace_sha256": trace_sha,
                        "target_outcome_fields_read": False,
                    }
                )
    if manifest_count != expected:
        raise ValueError(f"{origin} manifest product count changed: {manifest_count}")
    return decisions, audits


def _depth_band(index: int) -> str:
    if index == 0:
        return "d0"
    if index <= 3:
        return "d1_3"
    return "d4_plus"


def _forbidden_q0_outcome_paths(
    value: Any, path: tuple[str, ...] = ()
) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key_value, nested in value.items():
            key = str(key_value)
            nested_path = (*path, key)
            if key in FORBIDDEN_Q0_OUTCOME_KEYS:
                found.append(".".join(nested_path))
            found.extend(_forbidden_q0_outcome_paths(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_q0_outcome_paths(nested, (*path, str(index))))
    return found


def _q0_row_eligibility(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Recompute active-cohort eligibility from pre-action data only."""

    reasons = [
        f"forbidden_outcome_field:{path}"
        for path in _forbidden_q0_outcome_paths(row)
    ]
    if any(
        not isinstance(row.get(field), str) or not str(row[field])
        for field in (
            "origin",
            "source_id",
            "episode_id",
            "task_id",
            "map_id",
            "map_family",
            "before_fingerprint",
        )
    ):
        reasons.append("invalid_preaction_identity")
    if (
        type(row.get("solver_seed")) is not int
        or type(row.get("decision_index")) is not int
        or not 0 <= int(row["decision_index"]) < 12
    ):
        reasons.append("invalid_preaction_index_or_seed")
    known_maps = {map_id for map_ids in FOLDS.values() for map_id in map_ids}
    if str(row.get("map_id")) not in known_maps:
        reasons.append("map_outside_frozen_folds")
    if row.get("target_outcome_fields_read") is not False:
        reasons.append("target_outcome_read_or_unspecified")
    if row.get("candidate_repair_actions_executed") is not False:
        reasons.append("candidate_repair_action_executed_or_unspecified")
    if row.get("repair_fingerprint_preserved") is not True:
        reasons.append("candidate_generation_repair_fingerprint_unverified")
    if type(row.get("before_conflicts")) is not int or int(row["before_conflicts"]) <= 0:
        reasons.append("inactive_or_conflict_free")
    agent_count = row.get("agent_count")
    if type(agent_count) is not int or int(agent_count) <= 0:
        reasons.append("invalid_agent_count")
        agent_count = 0
    arms = row.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        reasons.append("missing_or_extra_arms")
        arms = {}
    agent_sets: list[tuple[int, ...]] = []
    for arm in ARMS:
        candidate = arms.get(arm)
        if not isinstance(candidate, dict):
            reasons.append(f"{arm}_unavailable")
            continue
        agents = candidate.get("agents")
        if not isinstance(agents, list) or not agents:
            reasons.append(f"{arm}_invalid_agents")
            continue
        if any(type(agent) is not int for agent in agents):
            reasons.append(f"{arm}_noninteger_agents")
            continue
        members = tuple(map(int, agents))
        if (
            len(set(members)) != len(members)
            or any(agent < 0 or agent >= int(agent_count) for agent in members)
            or type(candidate.get("actual_size")) is not int
            or int(candidate["actual_size"]) != len(members)
            or str(candidate.get("candidate_id") or "") == ""
            or candidate.get("role") != arm
        ):
            reasons.append(f"{arm}_illegal")
            continue
        if arm in STRUCTURAL_FAMILIES:
            families = set(map(str, candidate.get("selection_families") or ()))
            if len(members) != 16:
                reasons.append(f"{arm}_not_size16")
            if STRUCTURAL_FAMILIES[arm] not in families:
                reasons.append(f"{arm}_family_mismatch")
        agent_sets.append(tuple(sorted(members)))
    if len(agent_sets) != 3 or len(set(agent_sets)) != 3:
        reasons.append("arm_agent_sets_not_pairwise_distinct")
    if row.get("eligible") is not True or list(row.get("ineligibility_reasons") or ()):
        reasons.append("preflight_declared_ineligible")
    return not reasons, sorted(set(reasons))


def _selection_rank(row: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "namespace": "stride-active-supply-q0-rank-v1",
            "origin": row["origin"],
            "source_id": row["source_id"],
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "solver_seed": row["solver_seed"],
            "decision_index": row["decision_index"],
            "before_fingerprint": row["before_fingerprint"],
        }
    )


def select_active_supply_states(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = [dict(row) for row in rows]
    eligibility = [_q0_row_eligibility(row) for row in all_rows]
    eligible = [
        row for row, (is_eligible, _reasons) in zip(all_rows, eligibility) if is_eligible
    ]
    map_to_fold = {
        map_id: fold for fold, map_ids in FOLDS.items() for map_id in map_ids
    }
    episode_counts: Counter[tuple[str, str, str]] = Counter()
    map_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    preaction_identities = [
        (
            str(row.get("origin")),
            str(row.get("source_id")),
            str(row.get("episode_id")),
            str(row.get("decision_index")),
            str(row.get("before_fingerprint")),
        )
        for row in all_rows
    ]
    failures = [
        reason
        for _is_eligible, reasons in eligibility
        for reason in reasons
        if reason.startswith("forbidden_outcome_field:")
    ]
    if len(preaction_identities) != len(set(preaction_identities)):
        failures.append("duplicate_preaction_identity")

    def episode_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row["origin"]), str(row["source_id"]), str(row["episode_id"]))

    def can_add(row: dict[str, Any]) -> bool:
        return (
            episode_counts[episode_key(row)] < 4
            and map_counts[str(row["map_id"])] < 16
            and row not in selected
        )

    def add(row: dict[str, Any]) -> None:
        selected.append(row)
        episode_counts[episode_key(row)] += 1
        map_counts[str(row["map_id"])] += 1

    for fold in FOLDS:
        candidates = sorted(
            (row for row in eligible if map_to_fold.get(str(row["map_id"])) == fold),
            key=lambda row: (_selection_rank(row), int(row["decision_index"])),
        )
        fold_selected: list[dict[str, Any]] = []
        represented: set[str] = set()

        def add_fold(row: dict[str, Any]) -> None:
            add(row)
            fold_selected.append(row)
            represented.add(str(row["map_id"]))

        for band in ("d0", "d1_3", "d4_plus"):
            band_rows = [row for row in candidates if _depth_band(int(row["decision_index"])) == band]
            for prefer_new_map in (True, False):
                for row in band_rows:
                    if sum(
                        _depth_band(int(item["decision_index"])) == band
                        for item in fold_selected
                    ) >= 4:
                        break
                    if prefer_new_map and str(row["map_id"]) in represented:
                        continue
                    if can_add(row):
                        add_fold(row)
            if sum(
                _depth_band(int(item["decision_index"])) == band
                for item in fold_selected
            ) < 4:
                failures.append(f"{fold}:{band}_supply")
        for map_id in FOLDS[fold]:
            if map_id in represented:
                continue
            replacement = next(
                (
                    row
                    for row in candidates
                    if str(row["map_id"]) == map_id and can_add(row)
                ),
                None,
            )
            if replacement is not None:
                add_fold(replacement)
        for row in candidates:
            if len(fold_selected) >= 24:
                break
            if can_add(row):
                add_fold(row)
        if len(fold_selected) != 24:
            failures.append(f"{fold}:exact_24")
    selected_maps = {str(row["map_id"]) for row in selected}
    selected_families = {str(row["map_family"]) for row in selected}
    if len(selected_maps) < 10:
        failures.append("minimum_10_maps")
    if len(selected_families) < 4:
        failures.append("minimum_4_families")
    if len(selected) != 96:
        failures.append("exact_96")
    if max(episode_counts.values(), default=0) > 4:
        failures.append("episode_cap")
    if max(map_counts.values(), default=0) > 16:
        failures.append("map_cap")
    if failures:
        frozen = []
    else:
        frozen = []
        for row in selected:
            identity = {
                "origin": row["origin"],
                "source_id": row["source_id"],
                "episode_id": row["episode_id"],
                "decision_index": row["decision_index"],
                "before_fingerprint": row["before_fingerprint"],
            }
            frozen.append(
                {
                    **row,
                    "schema": SELECTION_SCHEMA,
                    "state_occurrence_id": "active-supply-" + _fingerprint(identity)[:24],
                    "selection_rank_sha256": _selection_rank(row),
                    "depth_band": _depth_band(int(row["decision_index"])),
                    "target_outcome_fields_read": False,
                }
            )
        frozen.sort(key=lambda row: str(row["state_occurrence_id"]))
    fold_counts = Counter(map_to_fold.get(str(row["map_id"])) for row in selected)
    depth_counts = Counter(
        (map_to_fold.get(str(row["map_id"])), _depth_band(int(row["decision_index"])))
        for row in selected
    )
    return frozen, {
        "status": "ok" if not failures else "STATE_SUPPLY_FAIL",
        "failure_reasons": sorted(set(failures)),
        "preaction_state_count": len(all_rows),
        "eligible_preaction_state_count": len(eligible),
        "eligible_preaction_fraction": len(eligible) / len(all_rows) if all_rows else 0.0,
        "ineligibility_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for is_eligible, reasons in eligibility
                    if not is_eligible
                    for reason in reasons
                ).items()
            )
        ),
        "selected_state_count": len(frozen),
        "h1_logical_trial_count": 4608 if not failures else 0,
        "selected_map_count": len(selected_maps),
        "selected_family_count": len(selected_families),
        "maximum_episode_count": max(episode_counts.values(), default=0),
        "maximum_map_count": max(map_counts.values(), default=0),
        "fold_counts": dict(sorted(fold_counts.items())),
        "fold_depth_counts": {
            fold: {band: depth_counts[(fold, band)] for band in ("d0", "d1_3", "d4_plus")}
            for fold in FOLDS
        },
    }


def _preflight_episode(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["output_path"])
    identity = str(job["identity"])
    if bool(job["resume"]) and path.is_file():
        payload = _read_json(path)
        rows = payload.get("rows")
        expected_rows = {
            (
                str(row["origin"]),
                str(row["source_id"]),
                str(row["episode_id"]),
                int(row["decision_index"]),
                str(row["before_fingerprint"]),
            )
            for row in job["decisions"]
        }
        observed_rows = {
            (
                str(row.get("origin")),
                str(row.get("source_id")),
                str(row.get("episode_id")),
                int(row.get("decision_index", -1)),
                str(row.get("before_fingerprint")),
            )
            for row in rows
        } if isinstance(rows, list) else set()
        if (
            payload.get("schema") == PREFLIGHT_EPISODE_SCHEMA
            and payload.get("identity") == identity
            and payload.get("complete") is True
            and payload.get("job_id") == job["job_id"]
            and isinstance(rows, list)
            and len(rows) == len(expected_rows)
            and observed_rows == expected_rows
            and payload.get("target_final_success_or_pp_outcomes_read") is False
        ):
            return {"status": "resumed", "job_id": job["job_id"], "output_path": str(path)}
        raise ValueError("invalid active-supply preflight resume artifact")
    rows = []
    for decision in job["decisions"]:
        row = _preflight_decision(dict(decision), bundle_root=job["bundle_root"])
        row["schema"] = PREFLIGHT_EPISODE_SCHEMA
        row["origin"] = decision["origin"]
        rows.append(row)
    payload = {
        "schema": PREFLIGHT_EPISODE_SCHEMA,
        "identity": identity,
        "complete": True,
        "job_id": job["job_id"],
        "rows": rows,
        "target_final_success_or_pp_outcomes_read": False,
    }
    _atomic_json(path, payload)
    return {"status": "ok", "job_id": job["job_id"], "output_path": str(path)}


def _preflight_failure(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "error": message,
        "job_id": job["job_id"],
        "output_path": job["output_path"],
    }


def _validate_wave_source_report(
    config: dict[str, Any], config_path: Path, output_root: Path, report: dict[str, Any]
) -> None:
    if (
        report.get("schema")
        != "lns2.stride.fresh_matched_v2_active_supply_overlay_source.v1"
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("config_sha256") != sha256_file(config_path)
        or report.get("dry_run") is not False
        or report.get("complete") is not True
        or report.get("v1_artifacts_mutated") is not False
        or report.get("fresh_qualification_reused") is not False
        or int(report.get("expected_wave_a_episode_count", -1)) != 20
        or int(report.get("observed_wave_a_episode_count", -1)) != 20
        or int(report.get("workers", -1)) != 16
    ):
        raise ValueError("Wave-A source trust report changed")
    manifests = {
        source_id: output_root
        / "wave_a"
        / "source"
        / source_id
        / "realized_dynamic_manifest.jsonl"
        for source_id in _wave_sources(config)
    }
    current_manifest_hashes = {
        source_id: sha256_file(manifest_path)
        for source_id, manifest_path in manifests.items()
    }
    registry, count = _trace_registry(output_root, manifests)
    if (
        report.get("manifest_sha256") != current_manifest_hashes
        or count != 20
        or report.get("trace_registry_sha256") != _fingerprint(registry)
    ):
        raise ValueError("Wave-A manifest/trace trust chain changed")


def run_preflight(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = _overlay_output_root(root, config, output)
    wave_report_path = output_root / "wave_a_source_report.json"
    if not wave_report_path.is_file():
        raise ValueError("active-supply Q0 requires completed Wave-A source")
    wave_report = _read_json(wave_report_path)
    _validate_wave_source_report(config, path, output_root, wave_report)
    v1_decisions, v1_audits = _read_source_product(config, output_root, origin="v1_reused")
    wave_decisions, wave_audits = _read_source_product(config, output_root, origin="wave_a")
    if sum(row["zero_prefix_inactive"] for row in v1_audits) != 25:
        raise ValueError("V1 inactive audit count changed")
    decisions = v1_decisions + wave_decisions
    audits = v1_audits + wave_audits
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(row["origin"], row["source_id"], row["episode_id"])].append(row)
    bundle = _registered(root, config["controller_bundle"]["manifest"], "controller")
    identity = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "wave_report_sha256": sha256_file(wave_report_path),
            "decisions": [
                (row["origin"], row["source_id"], row["episode_id"], row["decision_index"], row["before_fingerprint"])
                for row in decisions
            ],
        }
    )
    jobs = []
    for key, episode_rows in sorted(grouped.items()):
        job_id = ":".join(key)
        jobs.append(
            {
                "job_id": job_id,
                "identity": identity,
                "bundle_root": str(bundle.parent),
                "decisions": sorted(episode_rows, key=lambda row: int(row["decision_index"])),
                "output_path": str(output_root / "preflight_episodes" / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")),
                "resume": resume,
            }
        )
    if dry_run:
        return {
            "schema": "lns2.stride.fresh_matched_v2_active_supply_overlay_preflight_report.v1",
            "status": "dry_run",
            "source_episode_audit_count": len(audits),
            "v1_inactive_episode_count": 25,
            "preflight_episode_job_count": len(jobs),
            "workers": 16,
            "h1_executed": False,
        }
    results = _run_jobs(
        _preflight_episode,
        jobs,
        16,
        phase="active-supply-preflight",
        output_root=output_root / "preflight_progress",
        run_fingerprint=identity,
        timeout_seconds=420.0,
        failure_result=_preflight_failure,
    )
    successful = [row for row in results if row.get("status") in {"ok", "resumed"}]
    if len(successful) != len(jobs):
        raise RuntimeError("active-supply preflight worker failed")
    preflight_rows = [
        item
        for result in successful
        for item in _read_json(Path(result["output_path"]))["rows"]
    ]
    selected, report = select_active_supply_states(preflight_rows, config)
    report.update(
        {
            "schema": "lns2.stride.fresh_matched_v2_active_supply_overlay_preflight_report.v1",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(path),
            "wave_a_source_report_sha256": sha256_file(wave_report_path),
            "v1_inactive_episode_count": 25,
            "source_episode_audit_count": len(audits),
            "outcome_fields_read": False,
            "reserve_backfill_used": False,
        }
    )
    _write_jsonl(output_root / "source_episode_audit.jsonl", audits)
    _write_jsonl(output_root / "preflight_rows.jsonl", preflight_rows)
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    report["source_episode_audit_sha256"] = sha256_file(output_root / "source_episode_audit.jsonl")
    report["preflight_rows_sha256"] = sha256_file(output_root / "preflight_rows.jsonl")
    report["selected_states_sha256"] = sha256_file(output_root / "selected_states.jsonl")
    _atomic_json(output_root / "preflight_report.json", report)
    return report


def _h1_envelope_worker(job: dict[str, Any]) -> dict[str, Any]:
    core_job = {
        "job_id": job["job_id"],
        "state_row": job["state_row"],
        "identity": job["identity"],
        "per_action_time_limit_seconds": job["per_action_time_limit_seconds"],
        "output_path": job["core_output_path"],
        "resume": job["resume"],
    }
    result = _h1_state_worker(core_job)
    core_path = Path(job["core_output_path"])
    payload = _read_json(core_path)
    envelope = {
        "schema": H1_STATE_ENVELOPE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "identity": job["identity"],
        "complete": True,
        "state_occurrence_id": job["state_row"]["state_occurrence_id"],
        "core_schema": payload["schema"],
        "core_file": str(core_path),
        "core_sha256": sha256_file(core_path),
        "trial_count": len(payload["trials"]),
    }
    _atomic_json(Path(job["output_path"]), envelope)
    return {
        **result,
        "output_path": job["output_path"],
        "core_output_path": str(core_path),
    }


def _h1_failure(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    result = _v1_h1_failure_result(
        {
            "job_id": job["job_id"],
            "state_row": job["state_row"],
            "output_path": job["output_path"],
        },
        status,
        message,
    )
    result["core_output_path"] = job["core_output_path"]
    return result


def run_h1(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = _overlay_output_root(root, config, output)
    report_path = output_root / "preflight_report.json"
    selection_path = output_root / "selected_states.jsonl"
    preflight_rows_path = output_root / "preflight_rows.jsonl"
    source_audit_path = output_root / "source_episode_audit.jsonl"
    wave_report_path = output_root / "wave_a_source_report.json"
    if not all(
        artifact.is_file()
        for artifact in (
            report_path,
            selection_path,
            preflight_rows_path,
            source_audit_path,
            wave_report_path,
        )
    ):
        raise ValueError("active-supply H1 requires completed Q0")
    preflight = _read_json(report_path)
    selected = _read_jsonl(selection_path)
    preflight_rows = _read_jsonl(preflight_rows_path)
    source_audits = _read_jsonl(source_audit_path)
    if (
        preflight.get("schema")
        != "lns2.stride.fresh_matched_v2_active_supply_overlay_preflight_report.v1"
        or preflight.get("experiment_id") != EXPERIMENT_ID
        or preflight.get("status") != "ok"
        or preflight.get("config_sha256") != sha256_file(path)
        or preflight.get("selected_states_sha256") != sha256_file(selection_path)
        or preflight.get("preflight_rows_sha256") != sha256_file(preflight_rows_path)
        or preflight.get("source_episode_audit_sha256") != sha256_file(source_audit_path)
        or preflight.get("wave_a_source_report_sha256") != sha256_file(wave_report_path)
        or preflight.get("outcome_fields_read") is not False
        or preflight.get("reserve_backfill_used") is not False
        or int(preflight.get("source_episode_audit_count", -1)) != 68
        or int(preflight.get("v1_inactive_episode_count", -1)) != 25
        or len(source_audits) != 68
        or sum(row.get("origin") == "v1_reused" for row in source_audits) != 48
        or sum(row.get("origin") == "wave_a" for row in source_audits) != 20
        or sum(
            bool(row.get("zero_prefix_inactive"))
            for row in source_audits
            if row.get("origin") == "v1_reused"
        )
        != 25
        or len(selected) != 96
        or int(preflight.get("h1_logical_trial_count", -1)) != 4608
    ):
        raise ValueError("active-supply Q0 did not authorize H1")
    _validate_wave_source_report(
        config, path, output_root, _read_json(wave_report_path)
    )
    recomputed, recomputed_report = select_active_supply_states(preflight_rows, config)
    if (
        recomputed_report.get("status") != "ok"
        or recomputed != selected
        or len({str(row.get("state_occurrence_id")) for row in selected}) != 96
    ):
        raise ValueError("active-supply selected-state product changed")
    identity = _fingerprint(
        {
            "config_sha256": sha256_file(path),
            "preflight_report_sha256": sha256_file(report_path),
            "selection_sha256": sha256_file(selection_path),
            "h1": config["h1"],
        }
    )
    jobs = []
    for row in selected:
        state_id = str(row["state_occurrence_id"])
        jobs.append(
            {
                "job_id": state_id,
                "state_row": row,
                "identity": identity,
                "per_action_time_limit_seconds": float(
                    config["h1"]["per_action_time_limit_seconds"]
                ),
                "core_output_path": str(output_root / "h1_core_v1" / f"{state_id}.json"),
                "output_path": str(output_root / "h1_states" / f"{state_id}.json"),
                "resume": resume,
            }
        )
    if dry_run:
        return {
            "schema": H1_REPORT_SCHEMA,
            "status": "dry_run",
            "state_job_count": 96,
            "logical_trial_count": 4608,
            "workers": int(config["h1"]["workers"]),
            "h1_executed": False,
            "reserve_backfill_used": False,
        }
    results = _run_jobs(
        _h1_envelope_worker,
        jobs,
        int(config["h1"]["workers"]),
        phase="active-supply-h1",
        output_root=output_root / "h1_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_h1_failure,
    )
    successful = [row for row in results if row.get("status") in {"ok", "resumed"}]
    payloads = []
    invalid = 0
    for result in successful:
        expected_state_id = str(result["job_id"])
        core_path = Path(result["core_output_path"])
        envelope_path = Path(result["output_path"])
        try:
            payload = _read_json(core_path)
            envelope = _read_json(envelope_path)
        except (OSError, ValueError):
            invalid += 1
            continue
        envelope_valid = (
            envelope.get("schema") == H1_STATE_ENVELOPE_SCHEMA
            and envelope.get("experiment_id") == EXPERIMENT_ID
            and envelope.get("identity") == identity
            and envelope.get("complete") is True
            and envelope.get("state_occurrence_id") == expected_state_id
            and envelope.get("core_file") == str(core_path)
            and envelope.get("core_sha256") == sha256_file(core_path)
            and int(envelope.get("trial_count", -1)) == 48
        )
        core_valid = _h1_state_artifact_valid(
            payload,
            identity=identity,
            state_occurrence_id=expected_state_id,
        )
        if not envelope_valid or not core_valid:
            invalid += 1
            continue
        payloads.append(payload)
    report = analyze_h1_payloads(
        config, payloads, error_count=len(results) - len(successful) + invalid
    )
    report.update(
        {
            "schema": H1_REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "run_identity": identity,
            "config_sha256": sha256_file(path),
            "preflight_report_sha256": sha256_file(report_path),
            "selected_states_sha256": sha256_file(selection_path),
            "worker_results": results,
            "reserve_backfill_used": False,
            "h8_authorized_by_h1": bool(report.get("passed")),
            "h8_executed": False,
            "training_authorized": False,
        }
    )
    _atomic_json(output_root / "h1_report.json", report)
    _write_jsonl(output_root / "h1_state_results.jsonl", report["state_results"])
    return report
