from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.balanced_wall_clock import (
    _map_metrics,
    prepare_movingai_map_derived_dataset,
)
from experiments.repair_collection import _fingerprint, _read_json, _write_json


DESIGN_SCHEMA = "lns2.stride.robustaction_expansion_design.v1"
STRUCTPOOL_DESIGN_SCHEMA = "lns2.stride.robustaction_structpool_data_design.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_static_audit.v1"
TOPOLOGY_GROUPS = (
    "dao_high_topology",
    "dao_mid_topology",
    "dao_low_topology_control",
)
FORBIDDEN_SELECTION_FIELDS = {
    "candidate_repair_outcome",
    "candidate_conflicts_after",
    "candidate_runtime",
    "controller_action",
    "controller_ttf",
    "future_trajectory",
    "repair_runtime",
}


def _projected_counts(config: dict[str, Any]) -> dict[str, int]:
    map_count = len(config["benchmarks"])
    episodes = (
        map_count
        * int(config["selected_tasks_per_map"])
        * len(config["solver_seeds"])
        * len(config["source_policies"])
    )
    states = episodes * int(config["maximum_states_per_episode"])
    return {
        "map_count": map_count,
        "independent_episode_count": episodes,
        "selected_state_count": states,
        "combined_state_count": int(config["existing_state_count"]) + states,
    }


def topology_group(low_degree_cell_ratio: float, thresholds: list[float]) -> str:
    mid, high = map(float, thresholds)
    if low_degree_cell_ratio >= high:
        return TOPOLOGY_GROUPS[0]
    if low_degree_cell_ratio >= mid:
        return TOPOLOGY_GROUPS[1]
    return TOPOLOGY_GROUPS[2]


def robustaction_source_adapter(config: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen generic map-derived dataset input for this design."""

    validate_robustaction_expansion_design(config)
    task_design = dict(config["task_design"])
    return {
        "schema_version": 1,
        "dataset_revision": str(
            config.get("preflight_dataset_revision")
            or "stride-robustaction-preflight-v2"
        ),
        "source": str(config["map_archive"]["source"]),
        "task_semantics": str(task_design["semantics"]),
        "master_seed": int(task_design["master_seed"]),
        "task_seeds": list(map(int, task_design["task_seeds"])),
        "task_variants": list(map(str, task_design["task_variants"])),
        "map_archive": {
            "url": str(config["map_archive"]["url"]),
            "sha256": str(config["map_archive"]["sha256"]),
        },
        "expected_map_count": len(config["benchmarks"]),
        "expected_instance_count": int(config["expected_preflight_task_count"]),
        "benchmarks": [
            {
                "id": str(row["id"]),
                "layout_family": str(row["topology_group"]),
                "member": str(row["member"]),
                "member_sha256": str(row["member_sha256"]),
                "agent_counts": list(map(int, row["agent_counts"])),
            }
            for row in config["benchmarks"]
        ],
    }


def _validate_robustaction_expansion_v1(config: dict[str, Any]) -> None:
    if (
        config.get("scientific_status")
        != "preregistered_static_only_before_new_initial_pp_or_repair_outcomes"
        or config.get("data_line_id") != "stride-robustaction-data-v1"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("pre_registration_git_commit")
        != "63346bd6c9d8b48ed3f42bf55012d604cf93dba3"
    ):
        raise ValueError("robust-action expansion identity changed")
    if list(config.get("design_amendments") or ()) != [
        {
            "parent_commit": "f550c6e",
            "reason": "deterministic_opposite_exchange_requires_even_agent_counts",
            "change": "round_each_odd_registered_agent_count_up_by_one",
            "initial_pp_or_repair_outcomes_read": False,
        }
    ]:
        raise ValueError("robust-action deterministic design amendment changed")

    if (
        int(config.get("existing_state_count", -1)) != 303
        or int(config.get("minimum_combined_state_count", -1)) != 600
        or list(config.get("source_policies") or ())
        != ["official_adaptive", "v2-full"]
        or int(config.get("selected_tasks_per_map", -1)) != 2
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("maximum_states_per_episode", -1)) != 2
        or int(config.get("repair_trials_per_candidate", -1)) != 16
    ):
        raise ValueError("robust-action expansion dimensions changed")

    task_design = dict(config.get("task_design") or {})
    if (
        list(task_design.get("task_seeds") or ()) != [307]
        or list(task_design.get("task_variants") or ())
        != ["uniform_random", "opposite_exchange"]
        or int(task_design.get("candidate_tasks_per_map", -1)) != 6
        or int(task_design.get("low_topology_candidate_tasks_per_map", -1)) != 8
        or bool(task_design.get("uses_official_scenarios"))
    ):
        raise ValueError("robust-action task design changed")

    thresholds = list(map(float, config.get("topology_thresholds") or ()))
    if thresholds != [0.035, 0.06]:
        raise ValueError("robust-action topology thresholds changed")

    benchmarks = [dict(row) for row in config.get("benchmarks") or ()]
    ids = [str(row.get("id")) for row in benchmarks]
    if len(benchmarks) != 20 or len(set(ids)) != 20:
        raise ValueError("robust-action map registry changed")
    locked_lists = {
        field: list(map(str, config.get(field) or ()))
        for field in (
            "existing_label_map_ids",
            "formal_ood_map_ids",
            "fresh_evidence_map_ids",
        )
    }
    if [len(locked_lists[field]) for field in locked_lists] != [30, 12, 6] or any(
        len(values) != len(set(values)) for values in locked_lists.values()
    ):
        raise ValueError("robust-action locked map registry changed")
    locked = {value for values in locked_lists.values() for value in values}
    if locked & set(ids):
        raise ValueError("robust-action maps overlap locked evidence")

    group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    preflight_tasks = 0
    for row in benchmarks:
        group = str(row.get("topology_group"))
        ratio = float(row.get("static_low_degree_cell_ratio", -1.0))
        counts = list(map(int, row.get("agent_counts") or ()))
        if (
            group not in TOPOLOGY_GROUPS
            or group != topology_group(ratio, thresholds)
            or not 0.0 <= float(row.get("static_obstacle_ratio", -1.0)) <= 1.0
            or int(row.get("free_cell_count", 0)) <= 0
            or len(str(row.get("member_sha256", ""))) != 64
            or str(row.get("member")) != f"{row['id']}.map"
            or counts != sorted(set(counts))
            or any(value % 2 for value in counts)
            or len(counts) != (4 if group == TOPOLOGY_GROUPS[2] else 3)
            or counts[0] <= 0
            or counts[-1] > min(1500, int(row["free_cell_count"]))
        ):
            raise ValueError(f"invalid robust-action map registration: {row.get('id')}")
        group_counts[group] += 1
        family_counts[str(row.get("map_family"))] += 1
        preflight_tasks += (
            len(counts)
            * len(task_design["task_seeds"])
            * len(task_design["task_variants"])
        )
    if dict(group_counts) != {
        TOPOLOGY_GROUPS[0]: 7,
        TOPOLOGY_GROUPS[1]: 7,
        TOPOLOGY_GROUPS[2]: 6,
    } or len(family_counts) < 7:
        raise ValueError("robust-action topology or family balance changed")
    if preflight_tasks != 132 or int(config.get("expected_preflight_job_count", -1)) != 264:
        raise ValueError("robust-action preflight dimensions changed")

    projected = _projected_counts(config)
    if (
        projected["independent_episode_count"] != 160
        or projected["selected_state_count"] != 320
        or projected["combined_state_count"] < int(config["minimum_combined_state_count"])
        or int(config.get("projected_new_independent_episode_count", -1)) != 160
        or int(config.get("projected_new_selected_state_count", -1)) != 320
        or int(config.get("projected_combined_state_count", -1)) != 623
        or list(config.get("paired_repair_seed_indices") or ()) != list(range(16))
    ):
        raise ValueError("robust-action projected sample count changed")

    selection = dict(config.get("selection_boundary") or {})
    allowed = set(map(str, selection.get("allowed_inputs") or ()))
    forbidden = set(map(str, selection.get("forbidden_inputs") or ()))
    if (
        forbidden != FORBIDDEN_SELECTION_FIELDS
        or allowed & forbidden
        or not bool(selection.get("outcome_blind"))
    ):
        raise ValueError("robust-action outcome boundary changed")

    power = dict(config.get("power_state_policy") or {})
    if power != {
        "source": "user_reported_only",
        "additional_performance_preflight": False,
        "slow_power_allowed_work": [
            "static_design",
            "label_processing",
            "offline_training",
            "functional_and_semantic_tests",
        ],
        "slow_power_forbidden_work": ["wall_clock_ttf", "shadow_timing"],
        "timing_pooling_across_power_states": False,
    }:
        raise ValueError("robust-action power-state policy changed")
    opportunity = dict(config.get("post_collection_opportunity_gates") or {})
    if opportunity != {
        "minimum_unique_state_ids": 320,
        "minimum_unique_episode_ids": 160,
        "maximum_states_per_episode": 2,
        "minimum_states_with_nonanchor_robust_win_fraction": 0.25,
        "minimum_robust_positive_action_fraction": 0.04,
        "minimum_positive_opportunity_maps_per_topology_group": 3,
        "failure_action": (
            "register_a_second_outcome_blind_map_task_expansion_not_"
            "outcome_filter_existing_states"
        ),
    } or bool(config.get("formal_speed_claim")):
        raise ValueError("robust-action post-collection evidence boundary changed")


def _validate_structpool_data_design(config: dict[str, Any]) -> None:
    if (
        config.get("scientific_status")
        != "preregistered_outcome_blind_static_rebalance_after_structpool_headroom_pass"
        or config.get("data_line_id") != "stride-robustaction-structpool-data-v1"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("candidate_pool_id") != "v2-plus-stride-structpool-v1"
        or config.get("pre_registration_git_commit")
        != "d05deaf26ec01beb3c4315b35594e5ce14a6b429"
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("StructPool robust-action data identity changed")

    predecessor = dict(config.get("predecessor_reports") or {})
    expected_predecessor = {
        "base_data_design": {
            "path": "configs/stride_robustaction_expansion_design.json",
            "sha256": "6a1a870ea061c233abd570c83ad523b7cbc7e1b4ced46ef8a625128d690519ea",
        },
        "structpool_coverage": {
            "path": "build/stride-structpool-coverage-v3/structpool_coverage_report.json",
            "sha256": "7eebceb22cdefab8eabdd72d1236befe074bad2329e3b386b1a91dadd77fc1dc",
        },
        "structpool_headroom": {
            "path": "build/stride-structpool-headroom-v2/structpool_headroom_report.json",
            "sha256": "a063be4f7837fbe0f60b9db363ed01a2b2002ed19deafd6146f20610b4001a56",
        },
    }
    if predecessor != expected_predecessor:
        raise ValueError("StructPool robust-action predecessor registry changed")

    if (
        int(config.get("existing_state_count", -1)) != 303
        or int(config.get("minimum_combined_state_count", -1)) != 600
        or list(config.get("source_policies") or ())
        != ["official_adaptive", "v2-full"]
        or int(config.get("selected_tasks_per_map", -1)) != 2
        or list(config.get("solver_seeds") or ()) != [1, 2]
        or int(config.get("maximum_states_per_episode", -1)) != 2
        or int(config.get("repair_trials_per_candidate", -1)) != 16
        or list(config.get("paired_repair_seed_indices") or ()) != list(range(16))
    ):
        raise ValueError("StructPool robust-action data dimensions changed")

    task_design = dict(config.get("task_design") or {})
    if (
        int(task_design.get("master_seed", -1)) != 20260809
        or list(task_design.get("task_seeds") or ()) != [311]
        or list(task_design.get("task_variants") or ())
        != ["uniform_random", "opposite_exchange"]
        or int(task_design.get("candidate_tasks_per_map", -1)) != 6
        or int(task_design.get("low_topology_candidate_tasks_per_map", -1)) != 8
        or bool(task_design.get("uses_official_scenarios"))
        or task_design.get("new_map_agent_count_rule")
        != "ceil_to_even_free_cell_fractions_0.02_0.04_0.07_with_1000_cap"
    ):
        raise ValueError("StructPool robust-action task design changed")

    thresholds = list(map(float, config.get("topology_thresholds") or ()))
    targets = dict(config.get("topology_group_targets") or {})
    if thresholds != [0.035, 0.06] or targets != {
        TOPOLOGY_GROUPS[0]: 12,
        TOPOLOGY_GROUPS[1]: 6,
        TOPOLOGY_GROUPS[2]: 2,
    }:
        raise ValueError("StructPool robust-action topology design changed")

    benchmarks = [dict(row) for row in config.get("benchmarks") or ()]
    ids = [str(row.get("id")) for row in benchmarks]
    if len(benchmarks) != 20 or len(set(ids)) != 20:
        raise ValueError("StructPool robust-action map registry changed")
    locked_lists = {
        field: list(map(str, config.get(field) or ()))
        for field in (
            "existing_label_map_ids",
            "formal_ood_map_ids",
            "fresh_evidence_map_ids",
        )
    }
    if [len(locked_lists[field]) for field in locked_lists] != [30, 12, 6] or any(
        len(values) != len(set(values)) for values in locked_lists.values()
    ):
        raise ValueError("StructPool robust-action locked map registry changed")
    locked = {value for values in locked_lists.values() for value in values}
    if locked & set(ids):
        raise ValueError("StructPool robust-action maps overlap locked evidence")

    group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    preflight_tasks = 0
    for row in benchmarks:
        group = str(row.get("topology_group"))
        ratio = float(row.get("static_low_degree_cell_ratio", -1.0))
        counts = list(map(int, row.get("agent_counts") or ()))
        if (
            group not in TOPOLOGY_GROUPS
            or group != topology_group(ratio, thresholds)
            or not 0.0 <= float(row.get("static_obstacle_ratio", -1.0)) <= 1.0
            or int(row.get("free_cell_count", 0)) <= 0
            or len(str(row.get("member_sha256", ""))) != 64
            or str(row.get("member")) != f"{row['id']}.map"
            or counts != sorted(set(counts))
            or any(value % 2 for value in counts)
            or len(counts) != (4 if group == TOPOLOGY_GROUPS[2] else 3)
            or counts[0] <= 0
            or counts[-1] > min(1500, int(row["free_cell_count"]))
        ):
            raise ValueError(
                f"invalid StructPool robust-action map registration: {row.get('id')}"
            )
        group_counts[group] += 1
        family_counts[str(row.get("map_family"))] += 1
        preflight_tasks += (
            len(counts)
            * len(task_design["task_seeds"])
            * len(task_design["task_variants"])
        )
    if dict(group_counts) != targets or len(family_counts) < 8:
        raise ValueError("StructPool robust-action topology or family balance changed")
    if (
        preflight_tasks != 124
        or int(config.get("expected_preflight_task_count", -1)) != 124
        or int(config.get("expected_preflight_job_count", -1)) != 248
    ):
        raise ValueError("StructPool robust-action preflight dimensions changed")

    projected = _projected_counts(config)
    if (
        projected["independent_episode_count"] != 160
        or projected["selected_state_count"] != 320
        or projected["combined_state_count"] != 623
        or int(config.get("projected_new_independent_episode_count", -1)) != 160
        or int(config.get("projected_new_selected_state_count", -1)) != 320
        or int(config.get("projected_combined_state_count", -1)) != 623
    ):
        raise ValueError("StructPool robust-action projected sample count changed")

    selection = dict(config.get("selection_boundary") or {})
    allowed = set(map(str, selection.get("allowed_inputs") or ()))
    forbidden = set(map(str, selection.get("forbidden_inputs") or ()))
    if (
        forbidden != FORBIDDEN_SELECTION_FIELDS
        or allowed & forbidden
        or not bool(selection.get("outcome_blind"))
        or selection.get("map_selection_rule")
        != "static_metrics_only_12_high_6_mid_2_low_locked_evidence_disjoint"
    ):
        raise ValueError("StructPool robust-action outcome boundary changed")

    opportunity = dict(config.get("post_collection_opportunity_gates") or {})
    if opportunity != {
        "minimum_unique_state_ids": 320,
        "minimum_unique_episode_ids": 160,
        "maximum_states_per_episode": 2,
        "minimum_states_with_nonanchor_robust_win_fraction": 0.25,
        "minimum_robust_positive_action_fraction": 0.04,
        "minimum_positive_opportunity_maps_by_topology_group": {
            TOPOLOGY_GROUPS[0]: 3,
            TOPOLOGY_GROUPS[1]: 3,
            TOPOLOGY_GROUPS[2]: 2,
        },
        "failure_action": (
            "register_a_second_outcome_blind_map_task_expansion_not_"
            "outcome_filter_existing_states"
        ),
    }:
        raise ValueError("StructPool robust-action opportunity gates changed")

    power = dict(config.get("power_state_policy") or {})
    if power != {
        "source": "user_reported_only",
        "additional_performance_preflight": False,
        "slow_power_allowed_work": [
            "static_design",
            "dataset_generation",
            "non_timing_paired_repair_labels",
            "label_processing",
            "offline_training",
            "functional_and_semantic_tests",
        ],
        "slow_power_forbidden_work": ["wall_clock_ttf", "shadow_timing"],
        "timing_pooling_across_power_states": False,
    }:
        raise ValueError("StructPool robust-action power-state policy changed")


def validate_robustaction_expansion_design(config: dict[str, Any]) -> None:
    schema = config.get("schema")
    if schema == DESIGN_SCHEMA:
        _validate_robustaction_expansion_v1(config)
        return
    if schema == STRUCTPOOL_DESIGN_SCHEMA:
        _validate_structpool_data_design(config)
        return
    raise ValueError("unexpected robust-action expansion design")


def audit_robustaction_expansion(
    *, config_path: str | Path, archive: str | Path, map_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    archive = Path(archive).resolve()
    map_root = Path(map_root).resolve()
    config = _read_json(config_path)
    validate_robustaction_expansion_design(config)
    project_root = config_path.parents[1]
    expected_archive_sha = str(config["map_archive"]["sha256"])
    archive_sha = sha256_file(archive)

    registered_files = {
        "formal_ood_config": dict(config["formal_ood_config"]),
        "fresh_evidence_config": dict(config["fresh_evidence_config"]),
        **{
            f"predecessor_{name}": dict(value)
            for name, value in dict(config["predecessor_reports"]).items()
        },
    }
    registered_file_gates = {}
    for name, spec in registered_files.items():
        path = project_root / str(spec["path"])
        registered_file_gates[f"{name}_sha_matches"] = (
            path.is_file() and sha256_file(path) == str(spec["sha256"])
        )
    if config.get("schema") == STRUCTPOOL_DESIGN_SCHEMA:
        headroom = _read_json(
            project_root
            / str(dict(config["predecessor_reports"])["structpool_headroom"]["path"])
        )
        registered_file_gates["structpool_headroom_passed"] = (
            headroom.get("passed") is True
            and int(headroom.get("state_count", -1)) == 16
            and int(headroom.get("novel_trial_count", -1)) == 1344
        )

    map_rows = []
    errors: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for raw in config["benchmarks"]:
            row = dict(raw)
            member = str(row["member"])
            path = map_root / member
            if member not in names:
                errors.append(f"archive_missing:{member}")
                continue
            archive_member_sha = hashlib.sha256(bundle.read(member)).hexdigest()
            if not path.is_file():
                errors.append(f"extracted_missing:{member}")
                continue
            extracted_sha = sha256_file(path)
            metrics = _map_metrics(path)
            metric_matches = (
                int(metrics["free_cell_count"]) == int(row["free_cell_count"])
                and abs(
                    float(metrics["low_degree_cell_ratio"])
                    - float(row["static_low_degree_cell_ratio"])
                )
                <= 1e-15
                and abs(
                    float(metrics["obstacle_ratio"])
                    - float(row["static_obstacle_ratio"])
                )
                <= 1e-15
            )
            checksum_matches = (
                archive_member_sha == extracted_sha == str(row["member_sha256"])
            )
            if not metric_matches:
                errors.append(f"metric_mismatch:{row['id']}")
            if not checksum_matches:
                errors.append(f"checksum_mismatch:{row['id']}")
            map_rows.append(
                {
                    "map_id": str(row["id"]),
                    "map_family": str(row["map_family"]),
                    "topology_group": str(row["topology_group"]),
                    "free_cell_count": int(metrics["free_cell_count"]),
                    "static_low_degree_cell_ratio": float(
                        metrics["low_degree_cell_ratio"]
                    ),
                    "static_obstacle_ratio": float(metrics["obstacle_ratio"]),
                    "checksum_matches": checksum_matches,
                    "metric_matches": metric_matches,
                }
            )

    projected = _projected_counts(config)
    expected_groups = Counter(
        {
            str(name): int(count)
            for name, count in dict(config["topology_group_targets"]).items()
        }
    )
    minimum_families = int(config["minimum_distinct_map_families"])
    gates = {
        "archive_sha_matches": archive_sha == expected_archive_sha,
        **registered_file_gates,
        "all_twenty_maps_verified": len(map_rows) == 20 and not errors,
        "topology_balance_exact": Counter(
            row["topology_group"] for row in map_rows
        )
        == expected_groups,
        "minimum_map_family_count": len({row["map_family"] for row in map_rows})
        >= minimum_families,
        "locked_evidence_disjoint": not (
            {row["map_id"] for row in map_rows}
            & {
                str(value)
                for field in (
                    "existing_label_map_ids",
                    "formal_ood_map_ids",
                    "fresh_evidence_map_ids",
                )
                for value in config[field]
            }
        ),
        "minimum_combined_state_count": projected["combined_state_count"]
        >= int(config["minimum_combined_state_count"]),
        "no_performance_measurement_run": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "data_line_id": str(config["data_line_id"]),
        "planned_model_id": str(config["planned_model_id"]),
        "scientific_status": "static_registration_audited_before_new_solver_outcomes",
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "archive_sha256": archive_sha,
        },
        "projected_counts": projected,
        "map_rows": sorted(map_rows, key=lambda row: row["map_id"]),
        "errors": errors,
        "gates": gates,
        "passed": all(gates.values()),
        "performance_measurements_run": False,
        "next_decision": str(
            config.get("next_decision_on_static_audit_pass")
            or "defer_initial_pp_qualification_until_user_reports_comparable_power"
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "static_audit_report.json", report)
    return report


def prepare_robustaction_preflight_dataset(
    *, config_path: str | Path, fetched: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Prepare deterministic map/OD tasks without running PP or a controller."""

    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    adapter = robustaction_source_adapter(config)
    adapter_path = output.parent / f"{output.name}.source_adapter.json"
    summary_path = output / "dataset_summary.json"
    adapter_fingerprint = _fingerprint(adapter)
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("configuration_fingerprint") != adapter_fingerprint:
            raise ValueError("robust-action output belongs to another source adapter")
    elif output.is_dir() and any(output.iterdir()):
        raise ValueError("robust-action output is non-empty but incomplete")
    if adapter_path.is_file() and _read_json(adapter_path) != adapter:
        raise ValueError("robust-action source adapter path belongs to another run")
    _write_json(adapter_path, adapter)
    summary = prepare_movingai_map_derived_dataset(
        Path(fetched).resolve(), adapter_path, output
    )
    manifest = output / "balanced_wall_clock" / "manifest.jsonl"
    return {
        "schema": "lns2.stride.robustaction_preflight_dataset.v1",
        "data_line_id": str(config["data_line_id"]),
        "design_sha256": sha256_file(config_path),
        "source_adapter_sha256": sha256_file(adapter_path),
        "manifest_sha256": sha256_file(manifest),
        "map_count": int(summary["splits"]["balanced_wall_clock"]["map_count"]),
        "task_count": int(
            summary["splits"]["balanced_wall_clock"]["instance_count"]
        ),
        "solver_or_controller_run": False,
        "performance_measurement_run": False,
        "summary": summary,
    }


__all__ = [
    "DESIGN_SCHEMA",
    "REPORT_SCHEMA",
    "STRUCTPOOL_DESIGN_SCHEMA",
    "TOPOLOGY_GROUPS",
    "audit_robustaction_expansion",
    "prepare_robustaction_preflight_dataset",
    "robustaction_source_adapter",
    "topology_group",
    "validate_robustaction_expansion_design",
]
