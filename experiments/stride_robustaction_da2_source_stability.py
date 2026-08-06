from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_robustaction_da2_recovery_source import (
    _contains,
    _job_key,
    _trace_capacity,
)
from experiments.stride_robustaction_load_extension import _copy
from experiments.stride_robustaction_qualification import FORBIDDEN_FIELDS


CONFIG_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_source_stability_design.v1"
)
QUALIFICATION_REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_source_stability_qualification_report.v1"
)
SOURCE_REPORT_SCHEMA = (
    "lns2.stride.robustaction_structpool_da2_source_stability_report.v1"
)
SPLIT = "balanced_wall_clock"
POLICY_MANIFESTS = {
    "official_adaptive": "official_adaptive_manifest.jsonl",
    "realized_dynamic": "realized_dynamic_manifest.jsonl",
}
REPLACEMENTS = {
    "lt_undercitydungeon__derived_opposite_exchange__task_seed_0331__agents_1788": (
        "lt_undercitydungeon__derived_opposite_exchange__task_seed_0331__agents_1342"
    ),
    "lt_undercityserialkiller__derived_uniform_random__task_seed_0331__agents_1816": (
        "lt_undercityserialkiller__derived_uniform_random__task_seed_0331__agents_1362"
    ),
}
EXPECTED_FAILED_KEYS = {
    (
        "lt_undercitydungeon__derived_opposite_exchange__task_seed_0331__agents_1788",
        1,
    ),
    (
        "lt_undercitydungeon__derived_opposite_exchange__task_seed_0331__agents_1788",
        2,
    ),
    (
        "lt_undercityserialkiller__derived_uniform_random__task_seed_0331__agents_1816",
        1,
    ),
}
EXPECTED_LAYOUT_COUNTS = {
    "dao_high_topology": 6,
    "dao_mid_topology": 6,
    "dao_low_topology_control": 4,
}


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"DA2 source stability input changed: {spec['path']}")
    return path


def _input_paths(
    config: dict[str, Any], project_root: Path
) -> dict[str, Path]:
    return {
        name: _registered(project_root, dict(spec))
        for name, spec in dict(config["inputs"]).items()
    }


def validate_da2_source_stability_design(
    config: dict[str, Any], *, project_root: Path
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_after_source_v1_qualification_failure_before_stability_v2_resets"
        or config.get("data_line_id")
        != "stride-robustaction-structpool-recovery-data-v2"
        or config.get("pre_registration_git_commit")
        != "f5f7ccde2d6a49dbfb8f3bef630d35fda0466f4a"
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("DA2 source stability identity changed")

    expected_inputs = {
        "failed_source_qualification_manifest",
        "failed_source_qualification_report",
        "failed_source_run_config",
        "failed_source_collection_summary",
        "source_v1_manifest",
        "source_v1_summary",
        "supplement_manifest",
        "supplement_qualification_manifest",
        "stability_runtime",
        "source_v4_qualification_manifest",
        "source_v4_official_adaptive_manifest",
        "source_v4_realized_dynamic_manifest",
        "source_v4_run_config",
        "source_v4_collection_summary",
    }
    if set(dict(config.get("inputs") or {})) != expected_inputs:
        raise ValueError("DA2 source stability input registry changed")
    paths = _input_paths(config, project_root)

    observed = dict(config.get("observed_failure") or {})
    if (
        int(observed.get("expected_reset_count", -1)) != 32
        or int(observed.get("valid_reset_count", -1)) != 29
        or int(observed.get("error_reset_count", -1)) != 3
        or int(observed.get("timeout_reset_count", -1)) != 0
        or int(observed.get("policy_episode_count", -1)) != 0
        or observed.get("error")
        != "ValueError: state contains an empty agent path"
        or {
            (str(value[0]), int(value[1]))
            for value in observed.get("failed_job_keys") or ()
        }
        != EXPECTED_FAILED_KEYS
        or observed.get("mechanism")
        != "initial_pp_exceeded_native_600_second_budget_before_all_agent_paths_were_built"
        or bool(observed.get("max_repair_iterations_affects_initialization"))
    ):
        raise ValueError("DA2 source stability failure registration changed")

    failure_rows = _read_jsonl(paths["failed_source_qualification_manifest"])
    failure_report = _read_json(paths["failed_source_qualification_report"])
    failed = {
        _job_key(row)
        for row in failure_rows
        if str(row.get("status")) != "ok"
    }
    if (
        len(failure_rows) != 32
        or sum(str(row.get("status")) == "ok" for row in failure_rows) != 29
        or failed != EXPECTED_FAILED_KEYS
        or any(
            str(row.get("error"))
            != "ValueError: state contains an empty agent path"
            for row in failure_rows
            if str(row.get("status")) != "ok"
        )
        or failure_report.get("passed") is not False
        or int(failure_report.get("valid_count", -1)) != 29
        or int(failure_report.get("expected_reset_count", -1)) != 32
        or int(failure_report.get("nonzero_state_count", -1)) != 29
    ):
        raise ValueError("DA2 source stability failure evidence changed")

    source_summary = _read_json(paths["source_v1_summary"])
    if (
        int(source_summary.get("map_count", -1)) != 8
        or int(source_summary.get("task_count", -1)) != 16
        or dict(source_summary.get("layout_counts") or {})
        != EXPECTED_LAYOUT_COUNTS
    ):
        raise ValueError("DA2 source-v1 cohort changed")

    replacement_rule = dict(config.get("replacement_rule") or {})
    if replacement_rule != {
        "mode": "same_map_same_task_variant_next_lower_complete_registered_load",
        "reason": "remove_initial_pp_time_boundary_tasks_without_changing_map_or_topology_membership",
        "use_existing_complete_paired_reset_evidence": True,
        "require_both_solver_seeds_complete": True,
        "require_all_replacement_rows_fingerprinted": True,
        "require_distinct_task_ids": True,
        "allow_one_registered_zero_conflict_state": True,
        "failed_source_product_remains_immutable": True,
    }:
        raise ValueError("DA2 source stability replacement rule changed")

    replacement_rows = list(config.get("replacements") or ())
    configured_replacements = {
        str(row.get("removed_task_id")): str(row.get("replacement_task_id"))
        for row in replacement_rows
    }
    if configured_replacements != REPLACEMENTS or len(replacement_rows) != 2:
        raise ValueError("DA2 source stability replacements changed")

    supplement_rows = {
        str(row["task_id"]): row
        for row in _read_jsonl(paths["supplement_manifest"])
    }
    supplement_results = {
        _job_key(row): row
        for row in _read_jsonl(paths["supplement_qualification_manifest"])
    }
    for replacement in replacement_rows:
        task_id = str(replacement["replacement_task_id"])
        source = supplement_rows.get(task_id)
        registered_conflicts = dict(
            replacement["registered_initial_conflicts_by_seed"]
        )
        conflicts = {
            str(seed): int(registered_conflicts[str(seed)]) for seed in (1, 2)
        }
        if (
            source is None
            or str(source.get("map_id")) != str(replacement["map_id"])
            or str(source.get("layout_mode")) != str(replacement["layout_mode"])
            or int(source.get("agent_count", -1))
            != int(replacement["replacement_agent_count"])
        ):
            raise ValueError(f"replacement dataset row changed: {task_id}")
        for seed in (1, 2):
            result = supplement_results.get((task_id, seed))
            if (
                result is None
                or str(result.get("status")) != "ok"
                or not bool(result.get("initial_complete"))
                or not str(result.get("state_fingerprint", ""))
                or int(result.get("initial_conflicts", -1))
                != conflicts[str(seed)]
            ):
                raise ValueError(f"replacement reset evidence changed: {task_id}:{seed}")

    boundary = dict(config.get("outcome_boundary") or {})
    if (
        not bool(boundary.get("outcome_informed_initial_pp_recovery"))
        or bool(boundary.get("candidate_repair_outcomes_read"))
        or bool(boundary.get("controller_outcomes_read"))
        or bool(boundary.get("ttf_outcomes_read"))
        or bool(boundary.get("initial_pp_runtime_used_to_rank_replacements"))
        or boundary.get("claim_boundary")
        != "source_training_stability_only_not_clean_evaluation_or_ttf"
    ):
        raise ValueError("DA2 source stability outcome boundary changed")

    contract = dict(config.get("source_contract") or {})
    if contract != {
        "map_count": 8,
        "task_count": 16,
        "layout_counts": EXPECTED_LAYOUT_COUNTS,
        "solver_seeds": [1, 2],
        "source_policies": ["official_adaptive", "realized_dynamic"],
        "expected_qualification_rows": 32,
        "expected_nonzero_qualification_rows": 31,
        "expected_nonzero_by_solver_seed": {"1": 16, "2": 15},
        "expected_episode_rows_per_policy": 32,
        "expected_total_episode_rows": 64,
        "stopping_rule": "historical",
        "max_decisions": 12,
        "max_repair_iterations": 12,
        "metric_iteration_budget": 12,
        "workers": 1,
        "deterministic_pp_replay": True,
        "require_zero_qualification_errors": True,
        "require_zero_episode_errors": True,
        "require_zero_timeouts": True,
        "require_zero_invalid_actions": True,
        "failure_action": "preserve_complete_product_and_stop_without_success_filtering",
    }:
        raise ValueError("DA2 source stability source contract changed")

    runtime = _read_json(paths["stability_runtime"])
    qualification = dict(runtime.get("qualification") or {})
    environment = dict(runtime.get("environment") or {})
    if (
        int(runtime.get("workers", -1)) != 1
        or list(runtime.get("solver_seeds") or ()) != [1, 2]
        or list(runtime.get("policies") or ())
        != ["official_adaptive", "realized_dynamic"]
        or int(runtime.get("max_decisions", -1)) != 12
        or int(runtime.get("metric_iteration_budget", -1)) != 12
        or float(environment.get("time_limit", -1.0)) != 600.0
        or int(environment.get("max_repair_iterations", -1)) != 12
        or int(qualification.get("minimum_nonzero_states", -1)) != 31
        or int(qualification.get("minimum_nonzero_states_per_solver_seed", -1))
        != 15
        or dict(qualification.get("minimum_nonzero_states_per_agent_band") or {})
        != {"high": 31}
    ):
        raise ValueError("DA2 source stability runtime changed")

    frozen = dict(config.get("source_v4_frozen_capacity") or {})
    if frozen != {
        "raw_positive_pre_action_state_count": 1054,
        "maximum_obtainable_unique_state_ids": 297,
        "maximum_obtainable_unique_episode_ids": 154,
        "maximum_states_per_episode": 2,
        "must_remain_unfiltered": True,
    }:
        raise ValueError("DA2 source stability frozen capacity changed")


def _expected_task_ids(
    source_rows: list[dict[str, Any]], replacements: dict[str, str]
) -> set[str]:
    result = {str(row["task_id"]) for row in source_rows}
    for removed, replacement in replacements.items():
        if removed not in result or replacement in result:
            raise ValueError("DA2 source stability replacement identity is invalid")
        result.remove(removed)
        result.add(replacement)
    return result


def prepare_da2_source_stability_dataset(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_da2_source_stability_design(config, project_root=project_root)
    paths = _input_paths(config, project_root)
    source_manifest = paths["source_v1_manifest"]
    supplement_manifest = paths["supplement_manifest"]
    source_root = source_manifest.parents[1]
    supplement_root = supplement_manifest.parents[1]
    source_rows = _read_jsonl(source_manifest)
    supplement_rows = {
        str(row["task_id"]): row for row in _read_jsonl(supplement_manifest)
    }
    expected_ids = _expected_task_ids(source_rows, REPLACEMENTS)

    existing_summary = output / "dataset_summary.json"
    if existing_summary.is_file():
        summary = _read_json(existing_summary)
        if summary.get("design_sha256") != sha256_file(config_path):
            raise ValueError("DA2 stability dataset output belongs to another design")
        return summary
    if output.is_dir() and any(output.iterdir()):
        raise ValueError("DA2 stability dataset output is non-empty but incomplete")

    selected: list[tuple[Path, dict[str, Any]]] = []
    removed = set(REPLACEMENTS)
    for row in source_rows:
        if str(row["task_id"]) not in removed:
            selected.append((source_root, row))
    for replacement in REPLACEMENTS.values():
        if replacement not in supplement_rows:
            raise ValueError(f"missing replacement task: {replacement}")
        selected.append((supplement_root, supplement_rows[replacement]))

    final_rows: list[dict[str, Any]] = []
    for root, row in selected:
        for field in ("map_file", "scenario_file", "map_metadata_file", "task_file"):
            relative = Path(str(row[field]))
            _copy(root / SPLIT / relative, output / SPLIT / relative)
        final_rows.append(dict(row))
    final_rows.sort(key=lambda row: str(row["task_id"]))
    layouts = Counter(str(row["layout_mode"]) for row in final_rows)
    maps = {str(row["map_id"]) for row in final_rows}
    task_ids = {str(row["task_id"]) for row in final_rows}
    if (
        len(final_rows) != 16
        or len(maps) != 8
        or task_ids != expected_ids
        or dict(layouts) != EXPECTED_LAYOUT_COUNTS
    ):
        raise ValueError("DA2 source stability dataset dimensions changed")

    manifest = output / SPLIT / "manifest.jsonl"
    _write_jsonl(manifest, final_rows)
    summary = {
        "schema": "lns2.stride.robustaction_da2_source_stability_dataset.v1",
        "scientific_status": "outcome_informed_initial_pp_stability_source_cohort",
        "data_line_id": str(config["data_line_id"]),
        "design_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        "map_count": len(maps),
        "task_count": len(final_rows),
        "layout_counts": dict(layouts),
        "replacement_task_ids": sorted(REPLACEMENTS.values()),
        "removed_task_ids": sorted(REPLACEMENTS),
        "projected_nonzero_qualification_rows": 31,
        "projected_independent_episode_count": 64,
        "projected_state_capacity": 124,
        "candidate_repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "ttf_outcomes_read": False,
        "formal_speed_claim": False,
    }
    _write_json(existing_summary, summary)
    return summary


def _qualification_evidence(
    paths: dict[str, Path], expected_task_ids: set[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    removed = set(REPLACEMENTS)
    evidence = {
        _job_key(row): row
        for row in _read_jsonl(paths["failed_source_qualification_manifest"])
        if str(row.get("task_id")) not in removed
    }
    replacements = set(REPLACEMENTS.values())
    evidence.update(
        {
            _job_key(row): row
            for row in _read_jsonl(paths["supplement_qualification_manifest"])
            if str(row.get("task_id")) in replacements
        }
    )
    expected_keys = {
        (task_id, seed) for task_id in expected_task_ids for seed in (1, 2)
    }
    if set(evidence) != expected_keys:
        raise ValueError("DA2 stability reset evidence coverage changed")
    return evidence


def analyze_da2_source_stability_qualification(
    *,
    config_path: str | Path,
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    dataset = Path(dataset).resolve()
    qualification = Path(qualification).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_da2_source_stability_design(config, project_root=project_root)
    paths = _input_paths(config, project_root)
    source_rows = _read_jsonl(paths["source_v1_manifest"])
    expected_ids = _expected_task_ids(source_rows, REPLACEMENTS)
    expected_keys = {(task_id, seed) for task_id in expected_ids for seed in (1, 2)}
    dataset_rows = _read_jsonl(dataset / SPLIT / "manifest.jsonl")
    dataset_summary = _read_json(dataset / "dataset_summary.json")
    results_path = qualification / "qualification_manifest.jsonl"
    report_path = qualification / "qualification_report.json"
    run_path = qualification / "run_config.json"
    results = _read_jsonl(results_path)
    collector_report = _read_json(report_path)
    run_config = _read_json(run_path)
    evidence = _qualification_evidence(paths, expected_ids)
    indexed = {_job_key(row): row for row in results}
    row_errors: list[dict[str, Any]] = []
    forbidden: set[str] = set()
    for key, row in indexed.items():
        forbidden.update(FORBIDDEN_FIELDS & set(row))
        expected = evidence.get(key)
        if (
            key not in expected_keys
            or str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not str(row.get("state_fingerprint", ""))
            or expected is None
            or str(expected.get("status")) != "ok"
            or int(row.get("initial_conflicts", -1))
            != int(expected.get("initial_conflicts", -2))
            or str(row.get("state_fingerprint"))
            != str(expected.get("state_fingerprint"))
        ):
            row_errors.append(
                {
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "status": str(row.get("status")),
                    "error": str(row.get("error") or "evidence mismatch"),
                }
            )
    if len(indexed) != len(results) or set(indexed) != expected_keys:
        row_errors.append({"error": "incomplete or duplicate qualification keys"})

    nonzero = [row for row in results if int(row.get("initial_conflicts", 0)) > 0]
    nonzero_by_seed = Counter(str(int(row["solver_seed"])) for row in nonzero)
    runtime = _read_json(paths["stability_runtime"])
    expected_runtime = dict(runtime)
    expected_runtime["stopping_rule"] = "historical"
    layouts = Counter(str(row["layout_mode"]) for row in dataset_rows)
    maps = {str(row["map_id"]) for row in dataset_rows}
    gates = {
        "dataset_dimensions": (
            len(dataset_rows) == 16
            and len(maps) == 8
            and {str(row["task_id"]) for row in dataset_rows} == expected_ids
            and dict(layouts) == EXPECTED_LAYOUT_COUNTS
            and int(dataset_summary.get("projected_nonzero_qualification_rows", -1))
            == 31
        ),
        "qualification_dimensions": len(results) == 32 and set(indexed) == expected_keys,
        "all_resets_valid_and_reproduced": not row_errors,
        "forbidden_outcomes_absent": not forbidden,
        "nonzero_count_exact": len(nonzero) == 31,
        "nonzero_by_solver_seed_exact": dict(nonzero_by_seed) == {"1": 16, "2": 15},
        "collector_qualification_passed": bool(collector_report.get("passed")),
        "registered_runtime_exact": _contains(
            dict(run_config.get("configuration") or {}), expected_runtime
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "scientific_status": "outcome_informed_initial_pp_stability_qualification",
        "data_line_id": str(config["data_line_id"]),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(dataset / SPLIT / "manifest.jsonl"),
        "dataset_summary_sha256": sha256_file(dataset / "dataset_summary.json"),
        "qualification_manifest_sha256": sha256_file(results_path),
        "qualification_report_sha256": sha256_file(report_path),
        "run_config_sha256": sha256_file(run_path),
        "counts": {
            "map_count": len(maps),
            "task_count": len(dataset_rows),
            "qualification_row_count": len(results),
            "nonzero_state_count": len(nonzero),
            "row_error_count": len(row_errors),
        },
        "nonzero_by_solver_seed": dict(nonzero_by_seed),
        "row_errors": row_errors,
        "forbidden_fields_found": sorted(forbidden),
        "candidate_repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "ttf_outcomes_read": False,
        "formal_speed_claim": False,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_qualification_pass"
            if passed
            else "next_decision_on_qualification_failure"
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "stability_qualification_report.json", report)
    return report


def analyze_da2_source_stability_product(
    *,
    config_path: str | Path,
    dataset: str | Path,
    source: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    dataset = Path(dataset).resolve()
    source = Path(source).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_da2_source_stability_design(config, project_root=project_root)
    paths = _input_paths(config, project_root)
    qualification_audit = analyze_da2_source_stability_qualification(
        config_path=config_path,
        dataset=dataset,
        qualification=source,
        output=output,
    )
    contract = dict(config["source_contract"])
    dataset_rows = _read_jsonl(dataset / SPLIT / "manifest.jsonl")
    task_ids = {str(row["task_id"]) for row in dataset_rows}
    expected_keys = {
        (task_id, seed)
        for task_id in task_ids
        for seed in map(int, contract["solver_seeds"])
    }
    qualification = _read_jsonl(source / "qualification_manifest.jsonl")
    qualification_by_key = {_job_key(row): row for row in qualification}
    manifests = {
        policy: _read_jsonl(source / filename)
        for policy, filename in POLICY_MANIFESTS.items()
    }
    row_errors: list[dict[str, Any]] = []
    timeout_count = 0
    invalid_action_count = 0
    iteration_excess_count = 0
    fingerprint_mismatches: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    normalized_manifests: dict[str, list[dict[str, Any]]] = {}
    for policy, rows in manifests.items():
        by_key = {_job_key(row): row for row in rows}
        if len(by_key) != len(rows) or set(by_key) != expected_keys:
            row_errors.append({"source": policy, "error": "incomplete or duplicate job keys"})
        normalized_rows: list[dict[str, Any]] = []
        for key, row in by_key.items():
            status = str(row.get("status"))
            summary = dict(row.get("summary") or {})
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in episode_ids:
                row_errors.append({"source": policy, "job_key": list(key), "error": "duplicate episode id"})
            episode_ids.add(episode_id)
            if (
                key not in expected_keys
                or str(row.get("policy")) != policy
                or status not in {"ok", "resumed"}
                or row.get("error") is not None
            ):
                row_errors.append({"source": policy, "job_key": list(key), "error": row.get("error") or "invalid row"})
            if bool(summary.get("external_timeout")):
                timeout_count += 1
            invalid_action_count += int(summary.get("invalid_action_count", 0))
            if int(summary.get("repair_iterations", 0)) > int(contract["max_decisions"]):
                iteration_excess_count += 1
            expected_fingerprint = str(
                qualification_by_key.get(key, {}).get("state_fingerprint", "")
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
            normalized = dict(row)
            if status == "resumed":
                normalized["status"] = "ok"
            normalized_rows.append(normalized)
        normalized_manifests[policy] = normalized_rows

    maximum = int(config["capacity_contract"]["maximum_states_per_episode"])
    base_root = paths["source_v4_official_adaptive_manifest"].parent
    base_manifests = {
        "official_adaptive": _read_jsonl(paths["source_v4_official_adaptive_manifest"]),
        "realized_dynamic": _read_jsonl(paths["source_v4_realized_dynamic_manifest"]),
    }
    base_capacity, base_state_ids, base_trace_errors = _trace_capacity(
        base_root,
        base_manifests,
        namespace="source-v4",
        maximum_per_episode=maximum,
        max_decisions=int(contract["max_decisions"]),
    )
    recovery_capacity, recovery_state_ids, recovery_trace_errors = _trace_capacity(
        source,
        normalized_manifests,
        namespace="da2-stability-v2",
        maximum_per_episode=maximum,
        max_decisions=int(contract["max_decisions"]),
    )
    combined_state_ids = base_state_ids | recovery_state_ids
    combined = {
        "raw_positive_pre_action_state_count": int(base_capacity["raw_positive_pre_action_state_count"])
        + int(recovery_capacity["raw_positive_pre_action_state_count"]),
        "maximum_obtainable_unique_state_ids": int(base_capacity["maximum_obtainable_unique_state_ids"])
        + int(recovery_capacity["maximum_obtainable_unique_state_ids"]),
        "maximum_obtainable_unique_episode_ids": int(base_capacity["maximum_obtainable_unique_episode_ids"])
        + int(recovery_capacity["maximum_obtainable_unique_episode_ids"]),
        "materialized_result_blind_state_identity_count": len(combined_state_ids),
    }
    frozen = dict(config["source_v4_frozen_capacity"])
    capacity_contract = dict(config["capacity_contract"])
    gates = {
        "qualification_passed": bool(qualification_audit.get("passed")),
        "episode_dimensions": (
            all(len(rows) == 32 for rows in manifests.values())
            and sum(map(len, manifests.values())) == 64
        ),
        "all_episode_rows_valid": not row_errors,
        "zero_timeouts": timeout_count == 0,
        "zero_invalid_actions": invalid_action_count == 0,
        "historical_iteration_limit": iteration_excess_count == 0,
        "initial_fingerprints_match": not fingerprint_mismatches,
        "source_v4_traces_valid": not base_trace_errors,
        "source_v4_capacity_reproduced": (
            int(base_capacity["raw_positive_pre_action_state_count"])
            == int(frozen["raw_positive_pre_action_state_count"])
            and int(base_capacity["maximum_obtainable_unique_state_ids"])
            == int(frozen["maximum_obtainable_unique_state_ids"])
            and int(base_capacity["maximum_obtainable_unique_episode_ids"])
            == int(frozen["maximum_obtainable_unique_episode_ids"])
        ),
        "stability_v2_traces_valid": not recovery_trace_errors,
        "combined_state_capacity": int(combined["maximum_obtainable_unique_state_ids"])
        >= int(capacity_contract["minimum_combined_unique_state_ids"]),
        "combined_episode_capacity": int(combined["maximum_obtainable_unique_episode_ids"])
        >= int(capacity_contract["minimum_combined_unique_episode_ids"]),
    }
    passed = all(gates.values())
    report = {
        "schema": SOURCE_REPORT_SCHEMA,
        "scientific_status": "outcome_informed_source_stability_capacity_audit",
        "data_line_id": str(config["data_line_id"]),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": {
            policy: sha256_file(source / filename)
            for policy, filename in POLICY_MANIFESTS.items()
        },
        "counts": {
            "qualification_row_count": len(qualification),
            "episode_row_count": sum(map(len, manifests.values())),
            "row_error_count": len(row_errors),
            "timeout_count": timeout_count,
            "invalid_action_count": invalid_action_count,
            "iteration_excess_count": iteration_excess_count,
            "fingerprint_mismatch_count": len(fingerprint_mismatches),
        },
        "source_v4_capacity": base_capacity,
        "stability_v2_capacity": recovery_capacity,
        "combined_capacity": combined,
        "row_errors": row_errors,
        "fingerprint_mismatches": fingerprint_mismatches,
        "source_v4_trace_errors": base_trace_errors,
        "stability_v2_trace_errors": recovery_trace_errors,
        "candidate_repair_outcomes_read": False,
        "ttf_read_for_capacity": False,
        "source_episode_outcomes_used_to_filter_tasks_or_episodes": False,
        "formal_speed_claim": False,
        "gates": gates,
        "passed": passed,
        "next_decision": (
            config["next_decision_on_source_pass"]
            if passed
            else "stop_before_candidate_labels_and_report_exact_capacity"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "source_capacity_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "QUALIFICATION_REPORT_SCHEMA",
    "SOURCE_REPORT_SCHEMA",
    "analyze_da2_source_stability_product",
    "analyze_da2_source_stability_qualification",
    "prepare_da2_source_stability_dataset",
    "validate_da2_source_stability_design",
]
