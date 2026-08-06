from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.trace_replay import result_blind_decision_rows


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_recovery_source_design.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_recovery_source_report.v1"
)
SPLIT = "balanced_wall_clock"
POLICY_MANIFESTS = {
    "official_adaptive": "official_adaptive_manifest.jsonl",
    "realized_dynamic": "realized_dynamic_manifest.jsonl",
}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"DA2 recovery source input changed: {spec['path']}")
    return path


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def validate_da2_recovery_source_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_after_primary_recovery_pass_before_source_episode_outcomes"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v1"
        or config.get("pre_registration_git_commit")
        != "1851facd5f33b6de211f501b153054d3b6c49c88"
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("DA2 recovery source identity changed")

    expected_inputs = {
        "recovery_design",
        "recovery_qualification_report",
        "recovery_dataset_manifest",
        "recovery_dataset_summary",
        "source_runtime",
        "source_v4_qualification_manifest",
        "source_v4_official_adaptive_manifest",
        "source_v4_realized_dynamic_manifest",
        "source_v4_run_config",
        "source_v4_collection_summary",
    }
    inputs = dict(config.get("inputs") or {})
    if set(inputs) != expected_inputs:
        raise ValueError("DA2 recovery source input registry changed")
    paths = {
        name: _registered(project_root, dict(spec))
        for name, spec in inputs.items()
    }

    recovery = _read_json(paths["recovery_qualification_report"])
    dataset = _read_json(paths["recovery_dataset_summary"])
    if (
        recovery.get("passed") is not True
        or list(recovery.get("selected_recovery_task_ids") or ())
        != [
            "ca_caverns2__derived_opposite_exchange__task_seed_0337__agents_0686",
            "ca_caverns2__derived_uniform_random__task_seed_0337__agents_0960",
        ]
        or int(dataset.get("map_count", -1)) != 8
        or int(dataset.get("task_count", -1)) != 16
        or dict(dataset.get("layout_counts") or {})
        != {
            "dao_high_topology": 6,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        }
        or dataset.get("scientific_status")
        != "user_authorized_outcome_informed_recovery_cohort"
    ):
        raise ValueError("DA2 recovery source cohort evidence changed")

    boundary = dict(config.get("outcome_boundary") or {})
    expected_fields = {
        "status",
        "error",
        "episode_id",
        "map_id",
        "task_id",
        "solver_seed",
        "policy",
        "initial_fingerprint",
        "repair_iterations",
        "invalid_action_count",
        "external_timeout",
        "decision_index",
        "before_fingerprint",
        "before_conflicts",
    }
    if (
        not bool(boundary.get("recovery_task_selection_is_outcome_informed"))
        or bool(boundary.get("candidate_repair_outcomes_read"))
        or bool(boundary.get("ttf_read_for_capacity"))
        or set(
            map(
                str,
                boundary.get("source_episode_fields_read_for_integrity_and_capacity")
                or (),
            )
        )
        != expected_fields
        or bool(
            boundary.get("source_episode_outcomes_used_to_filter_tasks_or_episodes")
        )
        or boundary.get("claim_boundary")
        != "recovery_training_capacity_only_not_clean_evaluation_or_ttf"
    ):
        raise ValueError("DA2 recovery source outcome boundary changed")

    contract = dict(config.get("source_contract") or {})
    if contract != {
        "map_count": 8,
        "task_count": 16,
        "layout_counts": {
            "dao_high_topology": 6,
            "dao_mid_topology": 6,
            "dao_low_topology_control": 4,
        },
        "solver_seeds": [1, 2],
        "source_policies": ["official_adaptive", "realized_dynamic"],
        "expected_qualification_rows": 32,
        "expected_episode_rows_per_policy": 32,
        "expected_total_episode_rows": 64,
        "stopping_rule": "historical",
        "max_decisions": 12,
        "max_repair_iterations": 12,
        "metric_iteration_budget": 12,
        "deterministic_pp_replay": True,
        "require_zero_errors": True,
        "require_zero_timeouts": True,
        "require_zero_invalid_actions": True,
        "require_initial_fingerprint_match_across_qualification_and_policies": True,
        "failure_action": "preserve_complete_product_and_stop_without_success_filtering",
    }:
        raise ValueError("DA2 recovery source contract changed")

    frozen = dict(config.get("source_v4_frozen_capacity") or {})
    if frozen != {
        "raw_positive_pre_action_state_count": 1054,
        "maximum_obtainable_unique_state_ids": 297,
        "maximum_obtainable_unique_episode_ids": 154,
        "maximum_states_per_episode": 2,
        "must_be_recomputed_from_all_pinned_source_v4_traces": True,
        "must_remain_unfiltered": True,
    }:
        raise ValueError("DA2 recovery frozen source-v4 capacity changed")
    gate = dict(config.get("combined_capacity_gate") or {})
    if gate != {
        "minimum_unique_state_ids": 320,
        "minimum_unique_episode_ids": 160,
        "maximum_states_per_episode": 2,
        "capacity_inputs": [
            "episode_id",
            "source_policy",
            "decision_index",
            "before_fingerprint",
            "before_conflicts",
        ],
        "failure_action": (
            "report_exact_capacity_and_reassess_without_outcome_filtering"
        ),
    }:
        raise ValueError("DA2 recovery combined-capacity gate changed")
    if (
        config.get("next_decision_on_pass")
        != "deterministically_sample_combined_source_states_before_candidate_labels"
        or config.get("next_decision_on_failure")
        != "stop_before_candidate_labels_and_report_exact_capacity"
    ):
        raise ValueError("DA2 recovery source decision sequence changed")

    runtime = _read_json(paths["source_runtime"])
    if (
        runtime.get("split") != SPLIT
        or list(runtime.get("solver_seeds") or ()) != [1, 2]
        or list(runtime.get("policies") or ())
        != ["official_adaptive", "realized_dynamic"]
        or int(runtime.get("max_decisions", -1)) != 12
        or int(runtime.get("metric_iteration_budget", -1)) != 12
        or int(dict(runtime.get("environment") or {}).get("max_repair_iterations", -1))
        != 12
        or not bool(runtime.get("deterministic_pp_replay"))
        or dict(runtime.get("dataset_design") or {}).get("map_count") != 8
        or dict(runtime.get("dataset_design") or {}).get("instance_count") != 16
        or dict(runtime.get("qualification") or {}).get("minimum_nonzero_states")
        != 32
        or dict(runtime.get("qualification") or {}).get(
            "minimum_nonzero_states_per_solver_seed"
        )
        != 16
        or dict(runtime.get("qualification") or {}).get(
            "minimum_nonzero_states_per_agent_band"
        )
        != {"high": 32}
    ):
        raise ValueError("DA2 recovery source runtime changed")


def summarize_episode_capacity(
    counts_by_policy: dict[str, dict[str, int]], *, maximum_per_episode: int
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    all_episode_ids: set[str] = set()
    total_raw = 0
    total_capacity = 0
    total_eligible = 0
    for policy in sorted(counts_by_policy):
        counts = {str(key): int(value) for key, value in counts_by_policy[policy].items()}
        overlap = all_episode_ids & set(counts)
        if overlap:
            raise ValueError(f"duplicate source episode ids: {sorted(overlap)}")
        all_episode_ids.update(counts)
        distribution = Counter(
            "zero" if value <= 0 else "one" if value == 1 else "two_plus"
            for value in counts.values()
        )
        raw = sum(max(value, 0) for value in counts.values())
        capacity = sum(
            min(max(value, 0), int(maximum_per_episode))
            for value in counts.values()
        )
        eligible = sum(value > 0 for value in counts.values())
        policies[policy] = {
            "episode_count": len(counts),
            "raw_positive_pre_action_state_count": raw,
            "maximum_obtainable_unique_state_ids": capacity,
            "maximum_obtainable_unique_episode_ids": eligible,
            "episode_state_count_bands": {
                name: int(distribution[name])
                for name in ("zero", "one", "two_plus")
            },
        }
        total_raw += raw
        total_capacity += capacity
        total_eligible += eligible
    return {
        "maximum_states_per_episode": int(maximum_per_episode),
        "episode_count": len(all_episode_ids),
        "raw_positive_pre_action_state_count": total_raw,
        "maximum_obtainable_unique_state_ids": total_capacity,
        "maximum_obtainable_unique_episode_ids": total_eligible,
        "policies": policies,
    }


def _trace_capacity(
    source_root: Path,
    manifests: dict[str, list[dict[str, Any]]],
    *,
    namespace: str,
    maximum_per_episode: int,
    max_decisions: int,
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    counts: dict[str, dict[str, int]] = {}
    state_ids: set[str] = set()
    trace_errors: list[dict[str, Any]] = []
    for policy, rows in manifests.items():
        policy_counts: dict[str, int] = {}
        for row in rows:
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in policy_counts:
                trace_errors.append(
                    {"policy": policy, "episode_id": episode_id, "error": "duplicate_or_empty_episode_id"}
                )
                continue
            if str(row.get("status")) != "ok":
                policy_counts[episode_id] = 0
                continue
            try:
                decisions, _ = result_blind_decision_rows(source_root, row)
                eligible = [
                    decision
                    for decision in decisions
                    if int(decision["before_conflicts"]) > 0
                    and 0 <= int(decision["decision_index"]) < int(max_decisions)
                ]
                indices = [int(decision["decision_index"]) for decision in eligible]
                if len(indices) != len(set(indices)):
                    raise ValueError("duplicate decision index")
                policy_counts[episode_id] = len(eligible)
                for decision in eligible:
                    identity = {
                        "namespace": namespace,
                        "source_policy": policy,
                        "episode_id": episode_id,
                        "decision_index": int(decision["decision_index"]),
                        "before_fingerprint": str(decision["before_fingerprint"]),
                        "before_conflicts": int(decision["before_conflicts"]),
                    }
                    state_id = _fingerprint(identity)
                    if state_id in state_ids:
                        raise ValueError("duplicate result-blind state identity")
                    state_ids.add(state_id)
            except (OSError, ValueError, KeyError, TypeError) as error:
                policy_counts[episode_id] = 0
                trace_errors.append(
                    {"policy": policy, "episode_id": episode_id, "error": f"{type(error).__name__}: {error}"}
                )
        counts[policy] = policy_counts
    return (
        summarize_episode_capacity(
            counts, maximum_per_episode=maximum_per_episode
        ),
        state_ids,
        trace_errors,
    )


def _job_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id", "")), int(row.get("solver_seed", -1))


def analyze_da2_recovery_source(
    *, config_path: str | Path, source_root: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_da2_recovery_source_design(config, project_root=project_root)
    inputs = dict(config["inputs"])
    input_paths = {
        name: _registered(project_root, dict(spec))
        for name, spec in inputs.items()
    }
    contract = dict(config["source_contract"])
    maximum = int(config["combined_capacity_gate"]["maximum_states_per_episode"])

    dataset_rows = _read_jsonl(input_paths["recovery_dataset_manifest"])
    task_ids = {str(row["task_id"]) for row in dataset_rows}
    expected_keys = {
        (task_id, seed)
        for task_id in task_ids
        for seed in map(int, contract["solver_seeds"])
    }
    qualification_path = source_root / "qualification_manifest.jsonl"
    qualification_report_path = source_root / "qualification_report.json"
    run_config_path = source_root / "run_config.json"
    collection_summary_path = source_root / "collection_summary.json"
    qualification = _read_jsonl(qualification_path)
    qualification_report = _read_json(qualification_report_path)
    run_config = _read_json(run_config_path)
    collection_summary = _read_json(collection_summary_path)
    registered_runtime = _read_json(input_paths["source_runtime"])
    manifests = {
        policy: _read_jsonl(source_root / filename)
        for policy, filename in POLICY_MANIFESTS.items()
    }

    row_errors: list[dict[str, Any]] = []
    timeout_count = 0
    invalid_action_count = 0
    iteration_excess_count = 0
    fingerprint_mismatches: list[dict[str, Any]] = []
    qualification_by_key = {_job_key(row): row for row in qualification}
    if len(qualification_by_key) != len(qualification):
        row_errors.append({"source": "qualification", "error": "duplicate job key"})
    for key, row in qualification_by_key.items():
        if (
            key not in expected_keys
            or str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
        ):
            row_errors.append(
                {"source": "qualification", "job_key": list(key), "error": row.get("error") or "invalid row"}
            )

    episode_ids: set[str] = set()
    for policy, rows in manifests.items():
        by_key = {_job_key(row): row for row in rows}
        if len(by_key) != len(rows):
            row_errors.append({"source": policy, "error": "duplicate job key"})
        for key, row in by_key.items():
            summary = dict(row.get("summary") or {})
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in episode_ids:
                row_errors.append({"source": policy, "job_key": list(key), "error": "duplicate or empty episode id"})
            episode_ids.add(episode_id)
            if (
                key not in expected_keys
                or str(row.get("policy")) != policy
                or str(row.get("status")) != "ok"
                or row.get("error") is not None
            ):
                row_errors.append(
                    {"source": policy, "job_key": list(key), "error": row.get("error") or "invalid row"}
                )
            if bool(summary.get("external_timeout")):
                timeout_count += 1
            invalid_action_count += int(summary.get("invalid_action_count", 0))
            if int(summary.get("repair_iterations", 0)) > int(contract["max_decisions"]):
                iteration_excess_count += 1
            qualified = qualification_by_key.get(key)
            expected_fingerprint = str(
                (qualified or {}).get("state_fingerprint", "")
            )
            observed_fingerprint = str(summary.get("initial_fingerprint", ""))
            if not expected_fingerprint or observed_fingerprint != expected_fingerprint:
                fingerprint_mismatches.append(
                    {
                        "source": policy,
                        "job_key": list(key),
                        "expected": expected_fingerprint,
                        "observed": observed_fingerprint,
                    }
                )

    expected_runtime = dict(registered_runtime)
    expected_runtime["stopping_rule"] = str(contract["stopping_rule"])
    observed_runtime = dict(run_config.get("configuration") or {})

    base_root = input_paths["source_v4_official_adaptive_manifest"].parent
    base_manifests = {
        "official_adaptive": _read_jsonl(
            input_paths["source_v4_official_adaptive_manifest"]
        ),
        "realized_dynamic": _read_jsonl(
            input_paths["source_v4_realized_dynamic_manifest"]
        ),
    }
    base_capacity, base_state_ids, base_trace_errors = _trace_capacity(
        base_root,
        base_manifests,
        namespace="source-v4",
        maximum_per_episode=maximum,
        max_decisions=int(contract["max_decisions"]),
    )
    recovery_capacity, recovery_state_ids, recovery_trace_errors = _trace_capacity(
        source_root,
        manifests,
        namespace="da2-recovery",
        maximum_per_episode=maximum,
        max_decisions=int(contract["max_decisions"]),
    )
    combined_state_ids = base_state_ids | recovery_state_ids
    frozen = dict(config["source_v4_frozen_capacity"])
    combined_capacity = {
        "raw_positive_pre_action_state_count": (
            int(base_capacity["raw_positive_pre_action_state_count"])
            + int(recovery_capacity["raw_positive_pre_action_state_count"])
        ),
        "maximum_obtainable_unique_state_ids": (
            int(base_capacity["maximum_obtainable_unique_state_ids"])
            + int(recovery_capacity["maximum_obtainable_unique_state_ids"])
        ),
        "maximum_obtainable_unique_episode_ids": (
            int(base_capacity["maximum_obtainable_unique_episode_ids"])
            + int(recovery_capacity["maximum_obtainable_unique_episode_ids"])
        ),
        "materialized_result_blind_state_identity_count": len(combined_state_ids),
    }
    gates = {
        "dataset_dimensions": (
            len(dataset_rows) == int(contract["task_count"])
            and len(task_ids) == int(contract["task_count"])
            and len({str(row["map_id"]) for row in dataset_rows})
            == int(contract["map_count"])
        ),
        "qualification_complete": (
            len(qualification) == int(contract["expected_qualification_rows"])
            and set(qualification_by_key) == expected_keys
            and bool(qualification_report.get("passed"))
        ),
        "episode_dimensions": (
            all(
                len(rows) == int(contract["expected_episode_rows_per_policy"])
                and {_job_key(row) for row in rows} == expected_keys
                for rows in manifests.values()
            )
            and sum(map(len, manifests.values()))
            == int(contract["expected_total_episode_rows"])
        ),
        "all_episode_rows_valid": not row_errors,
        "zero_timeouts": timeout_count == 0,
        "zero_invalid_actions": invalid_action_count == 0,
        "historical_iteration_limit": iteration_excess_count == 0,
        "initial_fingerprints_match": not fingerprint_mismatches,
        "registered_runtime_exact": _contains(observed_runtime, expected_runtime),
        "deterministic_pp_replay": bool(
            observed_runtime.get("deterministic_pp_replay")
        ),
        "source_v4_traces_valid": not base_trace_errors,
        "source_v4_capacity_reproduced": (
            int(base_capacity["raw_positive_pre_action_state_count"])
            == int(frozen["raw_positive_pre_action_state_count"])
            and int(base_capacity["maximum_obtainable_unique_state_ids"])
            == int(frozen["maximum_obtainable_unique_state_ids"])
            and int(base_capacity["maximum_obtainable_unique_episode_ids"])
            == int(frozen["maximum_obtainable_unique_episode_ids"])
        ),
        "recovery_traces_valid": not recovery_trace_errors,
        "combined_state_id_capacity": (
            int(combined_capacity["maximum_obtainable_unique_state_ids"])
            >= int(config["combined_capacity_gate"]["minimum_unique_state_ids"])
        ),
        "combined_episode_id_capacity": (
            int(combined_capacity["maximum_obtainable_unique_episode_ids"])
            >= int(config["combined_capacity_gate"]["minimum_unique_episode_ids"])
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "outcome_informed_recovery_source_capacity_audit",
        "data_line_id": str(config["data_line_id"]),
        "config_sha256": sha256_file(config_path),
        "run_config_sha256": sha256_file(run_config_path),
        "qualification_manifest_sha256": sha256_file(qualification_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "collection_summary_sha256": sha256_file(collection_summary_path),
        "source_manifest_sha256": {
            policy: sha256_file(source_root / POLICY_MANIFESTS[policy])
            for policy in POLICY_MANIFESTS
        },
        "candidate_repair_outcomes_read": False,
        "ttf_read_for_capacity": False,
        "source_episode_outcomes_used_to_filter_tasks_or_episodes": False,
        "formal_speed_claim": False,
        "counts": {
            "map_count": len({str(row["map_id"]) for row in dataset_rows}),
            "task_count": len(task_ids),
            "qualification_row_count": len(qualification),
            "episode_row_count": sum(map(len, manifests.values())),
            "row_error_count": len(row_errors),
            "timeout_count": timeout_count,
            "invalid_action_count": invalid_action_count,
            "iteration_excess_count": iteration_excess_count,
            "fingerprint_mismatch_count": len(fingerprint_mismatches),
        },
        "source_v4_capacity": base_capacity,
        "recovery_capacity": recovery_capacity,
        "combined_capacity": combined_capacity,
        "row_errors": row_errors,
        "fingerprint_mismatches": fingerprint_mismatches,
        "source_v4_trace_errors": base_trace_errors,
        "recovery_trace_errors": recovery_trace_errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "source_capacity_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_da2_recovery_source",
    "summarize_episode_capacity",
    "validate_da2_recovery_source_design",
]
