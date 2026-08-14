from __future__ import annotations

import concurrent.futures
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from lns2_selector.runtime.hybridstructpool import merge_hybridstructpool_candidates


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_registration.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_audit_report.v1"
STATE_SCHEMA = "lns2.stride.hybridstructpool_state.v1"
EXPERIMENT_ID = "stride-hybridstructpool-v1"
FORBIDDEN_FIELDS = {
    "seed_mean",
    "first_fixed_half_mean",
    "second_fixed_half_mean",
    "no_progress_rate",
    "feasible_rate",
    "replan_success_rate",
    "future_trajectory",
    "ttf",
    "runtime",
}


def _load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_zero_solver_candidate_contract_audit"
    ):
        raise ValueError("HybridStructPool registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "causalclosure_cohort",
        "legacy_candidate_aggregates",
        "structural_coverage_report",
        "causalclosure_opportunity_report",
        "structshell_report",
    }:
        raise ValueError("HybridStructPool input registry changed")
    contract = dict(config.get("candidate_contract") or {})
    if (
        contract.get("full_v2_pool_preserved") is not True
        or contract.get("v2_anchor_preserved_separately") is not True
        or list(contract.get("structural_sizes_exposed_symmetrically") or ())
        != [8, 16, 24, 32]
        or contract.get("causalclosure_id") != "stride-causalclosurepool-v2"
        or int(contract.get("maximum_causal_candidates_per_state", -1)) != 12
        or contract.get("runtime_candidate_budget") is not None
        or contract.get("current_ranker_used") is not False
        or contract.get("candidate_outcomes_used_to_construct_pool") is not False
    ):
        raise ValueError("HybridStructPool candidate contract changed")
    execution = dict(config.get("execution") or {})
    if execution != {
        "workers": 16,
        "solver_calls_allowed": False,
        "native_pp_allowed": False,
        "future_rollout_allowed": False,
    }:
        raise ValueError("HybridStructPool execution boundary changed")
    return path, root, config, inputs


def _minimal(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "agents": sorted(set(map(int, row["agents"]))),
        "actual_size": int(row["actual_size"]),
        "candidate_kind": source,
        "selection_families": sorted(map(str, row.get("selection_families") or ())),
        "structpool_family_groups": sorted(
            map(str, row.get("structpool_family_groups") or ())
        ),
    }


def _state_job(job: dict[str, Any]) -> dict[str, Any]:
    state_key = str(job["state_fingerprint"])
    base = [_minimal(row, "base") for row in job["base"]]
    structural = [_minimal(row, "structural") for row in job["structural"]]
    causal = [_minimal(row, "causalclosure") for row in job["causal"]]
    result = merge_hybridstructpool_candidates(base, structural, causal)
    union_ids = {str(row["candidate_id"]) for row in result.candidates}
    robust_ids = set(map(str, job["legacy_robust_action_ids"]))
    causal_only_best_id = job.get("causalclosure_only_best_candidate_id")
    sizes = Counter(int(row["actual_size"]) for row in result.candidates)
    source_counts = Counter(
        source
        for values in result.provenance_by_candidate_id.values()
        for source in values
    )
    return {
        "schema": STATE_SCHEMA,
        "state_fingerprint": state_key,
        "map_id": str(job["map_id"]),
        "task_id": str(job["task_id"]),
        "solver_seed": int(job["solver_seed"]),
        "source_candidate_counts": {
            "v2_base": len(base),
            "structshell_equal_four_size": len(structural),
            "causalclosure_v2": len(causal),
        },
        "union_candidate_count": len(result.candidates),
        "challenger_count": len(result.challengers),
        "exact_duplicate_count": result.exact_duplicate_count,
        "source_membership_counts_after_merge": dict(sorted(source_counts.items())),
        "size_counts": {str(size): count for size, count in sorted(sizes.items())},
        "candidate_ids": sorted(union_ids),
        "provenance_by_candidate_id": {
            identity: list(values)
            for identity, values in sorted(result.provenance_by_candidate_id.items())
        },
        "base_candidate_recall_count": sum(
            str(row["candidate_id"]) in union_ids for row in base
        ),
        "structural_candidate_recall_count": sum(
            str(row["candidate_id"]) in union_ids for row in structural
        ),
        "causal_candidate_recall_count": sum(
            str(row["candidate_id"]) in union_ids for row in causal
        ),
        "legacy_robust_action_count": len(robust_ids),
        "legacy_robust_action_recall_count": len(robust_ids & union_ids),
        "causalclosure_only_best_candidate_id": causal_only_best_id,
        "causalclosure_only_best_preserved": (
            causal_only_best_id in union_ids if causal_only_best_id is not None else None
        ),
        "fixed_family_preference_used": False,
        "candidate_outcomes_used_to_construct_pool": False,
        "runtime_budget_applied": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def analyze_hybridstructpool(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, _root, config, inputs = _load_config(config_path)
    aggregates = _read_jsonl(inputs["legacy_candidate_aggregates"])
    causal_rows = _read_jsonl(inputs["causalclosure_cohort"])
    coverage = _read_json(inputs["structural_coverage_report"])
    opportunity = _read_json(inputs["causalclosure_opportunity_report"])
    structshell = _read_json(inputs["structshell_report"])
    if coverage.get("integrity_passed") is not True:
        raise ValueError("HybridStructPool structural coverage prerequisite failed")
    if opportunity.get("integrity_passed") is not True:
        raise ValueError("HybridStructPool CausalClosure opportunity prerequisite failed")
    if structshell.get("integrity_passed") is not True:
        raise ValueError("HybridStructPool StructShell prerequisite failed")

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_state[str(row["state_fingerprint"])].append(row)
    causal_by_state = {str(row["state_fingerprint"]): row for row in causal_rows}
    robust_by_state: dict[str, list[str]] = defaultdict(list)
    for row in coverage.get("legacy_robust_action_rows") or ():
        robust_by_state[str(row["state_fingerprint"])].append(str(row["candidate_id"]))
    legacy_opportunity = {
        str(row["state_fingerprint"]): bool(row["legacy_best_robustly_beats_v2_best"])
        for row in coverage.get("state_rows") or ()
    }
    opportunity_by_state = {
        str(row["state_fingerprint"]): row for row in opportunity.get("states") or ()
    }
    state_sets = [set(by_state), set(causal_by_state), set(legacy_opportunity), set(opportunity_by_state)]
    identity_errors: list[str] = []
    if any(state_set != state_sets[0] for state_set in state_sets[1:]):
        identity_errors.append("state_set_mismatch")

    jobs: list[dict[str, Any]] = []
    for state_key in sorted(set.intersection(*state_sets)):
        aggregate_rows = by_state[state_key]
        base = [row for row in aggregate_rows if row.get("candidate_kind") == "base"]
        structural = [
            row for row in aggregate_rows if row.get("candidate_kind") == "structural"
        ]
        causal = list(causal_by_state[state_key]["causalclosure_candidates"])
        outcome = opportunity_by_state[state_key]
        causal_only = bool(outcome["new_stably_dominates_base_best"]) and not bool(
            legacy_opportunity[state_key]
        )
        jobs.append(
            {
                "state_fingerprint": state_key,
                "map_id": str(causal_by_state[state_key]["map_id"]),
                "task_id": str(causal_by_state[state_key]["task_id"]),
                "solver_seed": int(causal_by_state[state_key]["solver_seed"]),
                "base": base,
                "structural": structural,
                "causal": causal,
                "legacy_robust_action_ids": robust_by_state.get(state_key, []),
                "causalclosure_only_best_candidate_id": (
                    str(outcome["new_best_candidate_id"]) if causal_only else None
                ),
            }
        )
    workers = int(config["execution"]["workers"])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        states = list(executor.map(_state_job, jobs))
    states.sort(key=lambda row: str(row["state_fingerprint"]))

    totals = dict(config["cohort"])
    base_total = sum(row["source_candidate_counts"]["v2_base"] for row in states)
    structural_total = sum(
        row["source_candidate_counts"]["structshell_equal_four_size"] for row in states
    )
    causal_total = sum(
        row["source_candidate_counts"]["causalclosure_v2"] for row in states
    )
    robust_total = sum(int(row["legacy_robust_action_count"]) for row in states)
    causal_only_rows = [
        row for row in states if row["causalclosure_only_best_candidate_id"] is not None
    ]
    recalls = {
        "base_candidate_recall": _ratio(
            sum(int(row["base_candidate_recall_count"]) for row in states), base_total
        ),
        "legacy_structural_candidate_recall": _ratio(
            sum(int(row["structural_candidate_recall_count"]) for row in states),
            structural_total,
        ),
        "legacy_robust_action_recall": _ratio(
            sum(int(row["legacy_robust_action_recall_count"]) for row in states),
            robust_total,
        ),
        "causalclosure_candidate_recall": _ratio(
            sum(int(row["causal_candidate_recall_count"]) for row in states),
            causal_total,
        ),
        "causalclosure_only_best_action_recall": _ratio(
            sum(bool(row["causalclosure_only_best_preserved"]) for row in causal_only_rows),
            len(causal_only_rows),
        ),
    }
    maps = sorted({str(row["map_id"]) for row in states})
    maps_all_sources = all(
        all(
            int(row["source_candidate_counts"][source]) > 0
            for source in ("v2_base", "structshell_equal_four_size", "causalclosure_v2")
        )
        for row in states
    )
    integrity = {
        "state_count": len(states) == int(totals["state_count"]),
        "map_count": len(maps) == int(totals["map_count"]),
        "base_candidate_count": base_total == int(totals["base_candidate_count"]),
        "legacy_structural_candidate_count": structural_total
        == int(totals["legacy_structural_candidate_count"]),
        "causalclosure_candidate_count": causal_total
        == int(totals["causalclosure_candidate_count"]),
        "legacy_robust_action_count": robust_total
        == int(totals["legacy_robust_action_count"]),
        "causalclosure_only_opportunity_state_count": len(causal_only_rows)
        == int(totals["causalclosure_only_opportunity_state_count"]),
        "zero_identity_errors": not identity_errors,
        "zero_forbidden_output_fields": all(
            not (FORBIDDEN_FIELDS & set(row)) for row in states
        ),
    }
    gate_config = dict(config["readiness_gates"])
    gates = {
        name: recalls[name] >= float(gate_config[name])
        for name in recalls
    }
    gates.update(
        {
            "all_maps_have_all_three_sources": maps_all_sources,
            "no_fixed_family_preference": all(
                row["fixed_family_preference_used"] is False for row in states
            ),
            "zero_identity_or_integrity_errors": all(integrity.values()),
        }
    )
    size_values = [
        int(size)
        for row in states
        for size, count in row["size_counts"].items()
        for _ in range(int(count))
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_zero_solver_candidate_contract_audit",
        "experiment_id": EXPERIMENT_ID,
        "registration_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "integrity": integrity,
        "identity_errors": sorted(identity_errors),
        "integrity_passed": all(integrity.values()),
        "state_count": len(states),
        "maps": maps,
        "source_candidate_counts": {
            "v2_base": base_total,
            "structshell_equal_four_size": structural_total,
            "causalclosure_v2": causal_total,
        },
        "union_candidate_count": sum(int(row["union_candidate_count"]) for row in states),
        "exact_duplicate_count": sum(int(row["exact_duplicate_count"]) for row in states),
        "mean_candidates_per_state": statistics.fmean(
            int(row["union_candidate_count"]) for row in states
        ),
        "candidate_size": {
            "minimum": min(size_values),
            "median": statistics.median(size_values),
            "mean": statistics.fmean(size_values),
            "maximum": max(size_values),
        },
        "membership_recall": recalls,
        "readiness_gates": gates,
        "candidate_contract_ready": all(gates.values()),
        "runtime_replacement_ready": False,
        "next_step": (
            "freeze_full_union_then_preregister_outcome_blind_budget_6_8_12_audit"
            if all(gates.values())
            else "stop_and_repair_candidate_contract_without_running_pp"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "hybridstructpool_states.jsonl", states)
    _write_json(output / "hybridstructpool_audit_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "STATE_SCHEMA",
    "analyze_hybridstructpool",
]
