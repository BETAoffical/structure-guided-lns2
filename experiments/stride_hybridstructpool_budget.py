from __future__ import annotations

import concurrent.futures
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from lns2_selector.runtime.hybridstructpool import (
    HYBRID_GROUP_ORDER,
    merge_hybridstructpool_candidates,
    reduce_hybridstructpool_challengers,
)


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_budget_registration.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_budget_report.v1"
STATE_SCHEMA = "lns2.stride.hybridstructpool_budget_state.v1"
EXPERIMENT_ID = "stride-hybridstructpool-budget-v1"


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_outcome_blind_budget_membership_audit"
    ):
        raise ValueError("HybridStructPool budget registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "hybrid_audit_report",
        "causalclosure_cohort",
        "legacy_candidate_aggregates",
        "structural_coverage_report",
        "causalclosure_opportunity_report",
        "pretail_membership_rows",
    }:
        raise ValueError("HybridStructPool budget input registry changed")
    reducer = dict(config.get("reducer") or {})
    if (
        reducer.get("id") != "balanced-semantic-diversity-v1"
        or list(map(int, reducer.get("budgets") or ())) != [6, 8, 12]
        or reducer.get("v2_pool_outside_budget") is not True
        or tuple(reducer.get("semantic_round_robin") or ()) != HYBRID_GROUP_ORDER
        or reducer.get("candidate_outcomes_used") is not False
        or reducer.get("future_trajectory_used") is not False
        or reducer.get("fixed_family_size_preference_used") is not False
    ):
        raise ValueError("HybridStructPool budget reducer changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "solver_calls_allowed": False,
        "native_pp_allowed": False,
    }:
        raise ValueError("HybridStructPool budget execution boundary changed")
    return path, config, inputs


def _proposal_audit(row: dict[str, Any]) -> dict[str, float]:
    if row.get("proposal_audit"):
        audit = dict(row["proposal_audit"])
        return {
            "global_event_incident_coverage": float(
                audit.get("global_event_incident_coverage", 0.0)
            ),
            "global_pair_internal_coverage": float(
                audit.get("global_pair_internal_coverage", 0.0)
            ),
            "conflict_component_reach": float(
                audit.get("conflict_component_reach", 0.0)
            ),
        }
    features = dict(row.get("features") or {})
    return {
        "global_event_incident_coverage": float(
            features.get("realized.incident_event_coverage", 0.0)
        ),
        "global_pair_internal_coverage": float(
            features.get("realized.internal_conflict_coverage", 0.0)
        ),
        "conflict_component_reach": float(
            features.get("realized.component_coverage_max", 0.0)
        ),
    }


def _candidate(row: dict[str, Any], source: str) -> dict[str, Any]:
    agents = sorted(set(map(int, row["agents"])))
    return {
        "candidate_id": str(row["candidate_id"]),
        "agents": agents,
        "actual_size": len(agents),
        "candidate_kind": source,
        "proposal_audit": _proposal_audit(row),
        "structpool_family_groups": sorted(
            map(str, row.get("structpool_family_groups") or ())
        ),
    }


def _state_job(job: dict[str, Any]) -> dict[str, Any]:
    base = [_candidate(row, "base") for row in job["base"]]
    structural = [_candidate(row, "structural") for row in job["structural"]]
    causal = [_candidate(row, "causalclosure") for row in job["causal"]]
    full = merge_hybridstructpool_candidates(base, structural, causal)
    base_ids = {str(row["candidate_id"]) for row in base}
    robust_ids = set(map(str, job["legacy_robust_action_ids"]))
    output = {
        "schema": STATE_SCHEMA,
        "state_fingerprint": str(job["state_fingerprint"]),
        "map_id": str(job["map_id"]),
        "task_id": str(job["task_id"]),
        "solver_seed": int(job["solver_seed"]),
        "base_candidate_count": len(base),
        "full_challenger_count": len(full.challengers),
        "legacy_robust_action_ids": sorted(robust_ids),
        "causalclosure_only_best_candidate_id": job.get(
            "causalclosure_only_best_candidate_id"
        ),
        "budgets": {},
    }
    for budget in (6, 8, 12):
        selected = reduce_hybridstructpool_challengers(
            full, maximum_challengers=budget
        )
        selected_ids = {str(row["candidate_id"]) for row in selected}
        provenance = Counter(
            source
            for identity in selected_ids
            for source in full.provenance_by_candidate_id[identity]
        )
        size_counts = Counter(int(row["actual_size"]) for row in selected)
        causal_best = job.get("causalclosure_only_best_candidate_id")
        output["budgets"][str(budget)] = {
            "selected_candidate_ids": sorted(selected_ids),
            "selected_candidate_count": len(selected),
            "budget_respected": len(selected) <= budget,
            "base_candidate_recall_count": len(base_ids),
            "base_candidate_count": len(base_ids),
            "legacy_robust_action_count": len(robust_ids),
            "legacy_robust_action_recall_count": len(robust_ids & selected_ids),
            "legacy_opportunity_preserved": bool(robust_ids & selected_ids)
            if robust_ids
            else None,
            "causalclosure_only_best_preserved": (
                causal_best in selected_ids if causal_best is not None else None
            ),
            "source_counts": dict(sorted(provenance.items())),
            "size_counts": {str(size): count for size, count in sorted(size_counts.items())},
            "candidate_outcomes_used": False,
            "fixed_family_size_preference_used": False,
        }
    return output


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def analyze_hybridstructpool_budget(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, config, inputs = _load_config(config_path)
    hybrid_report = _read_json(inputs["hybrid_audit_report"])
    if hybrid_report.get("candidate_contract_ready") is not True:
        raise ValueError("HybridStructPool full-union prerequisite failed")
    aggregates = _read_jsonl(inputs["legacy_candidate_aggregates"])
    causal_rows = _read_jsonl(inputs["causalclosure_cohort"])
    coverage = _read_json(inputs["structural_coverage_report"])
    opportunity = _read_json(inputs["causalclosure_opportunity_report"])
    pretail = _read_jsonl(inputs["pretail_membership_rows"])

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
    causal_opportunity = {
        str(row["state_fingerprint"]): row for row in opportunity.get("states") or ()
    }
    state_sets = [set(by_state), set(causal_by_state), set(legacy_opportunity), set(causal_opportunity)]
    identity_errors: list[str] = []
    if any(values != state_sets[0] for values in state_sets[1:]):
        identity_errors.append("state_set_mismatch")
    jobs = []
    for state_key in sorted(set.intersection(*state_sets)):
        rows = by_state[state_key]
        outcome = causal_opportunity[state_key]
        causal_only = bool(outcome["new_stably_dominates_base_best"]) and not bool(
            legacy_opportunity[state_key]
        )
        jobs.append(
            {
                "state_fingerprint": state_key,
                "map_id": causal_by_state[state_key]["map_id"],
                "task_id": causal_by_state[state_key]["task_id"],
                "solver_seed": causal_by_state[state_key]["solver_seed"],
                "base": [row for row in rows if row.get("candidate_kind") == "base"],
                "structural": [
                    row for row in rows if row.get("candidate_kind") == "structural"
                ],
                "causal": causal_by_state[state_key]["causalclosure_candidates"],
                "legacy_robust_action_ids": robust_by_state.get(state_key, []),
                "causalclosure_only_best_candidate_id": (
                    str(outcome["new_best_candidate_id"]) if causal_only else None
                ),
            }
        )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(config["execution"]["workers"])
    ) as executor:
        states = list(executor.map(_state_job, jobs))
    states.sort(key=lambda row: str(row["state_fingerprint"]))
    states_by_key = {str(row["state_fingerprint"]): row for row in states}

    informative_beneficial = [
        row
        for row in pretail
        if str(row["classification"]) == "beneficial" and not bool(row["identical_action"])
    ]
    expected = dict(config["cohort"])
    if len(informative_beneficial) != int(expected["pretail_beneficial_comparison_count"]):
        identity_errors.append("pretail_beneficial_comparison_count")
    summaries: dict[str, dict[str, Any]] = {}
    gate_config = dict(config["readiness_gates"])
    for budget in (6, 8, 12):
        key = str(budget)
        robust_total = sum(
            int(row["budgets"][key]["legacy_robust_action_count"]) for row in states
        )
        robust_retained = sum(
            int(row["budgets"][key]["legacy_robust_action_recall_count"])
            for row in states
        )
        opportunity_rows = [
            row for row in states if row["budgets"][key]["legacy_opportunity_preserved"] is not None
        ]
        opportunity_retained = sum(
            bool(row["budgets"][key]["legacy_opportunity_preserved"])
            for row in opportunity_rows
        )
        by_map = {}
        for map_id in sorted({str(row["map_id"]) for row in opportunity_rows}):
            map_rows = [row for row in opportunity_rows if row["map_id"] == map_id]
            by_map[map_id] = {
                "opportunity_state_count": len(map_rows),
                "retained_opportunity_state_count": sum(
                    bool(row["budgets"][key]["legacy_opportunity_preserved"])
                    for row in map_rows
                ),
            }
            by_map[map_id]["opportunity_state_recall"] = _fraction(
                by_map[map_id]["retained_opportunity_state_count"], len(map_rows)
            )
        pretail_retained = 0
        for comparison in informative_beneficial:
            if bool(comparison["candidate_is_v2_base"]):
                pretail_retained += 1
                continue
            state = states_by_key.get(str(comparison["state_fingerprint"]))
            if state is None:
                identity_errors.append(
                    f"missing_pretail_state:{comparison['state_fingerprint']}"
                )
                continue
            if str(comparison["candidate_id"]) in set(
                state["budgets"][key]["selected_candidate_ids"]
            ):
                pretail_retained += 1
        causal_only_rows = [
            row
            for row in states
            if row["causalclosure_only_best_candidate_id"] is not None
        ]
        metrics = {
            "legacy_opportunity_state_recall": _fraction(
                opportunity_retained, len(opportunity_rows)
            ),
            "legacy_robust_action_recall": _fraction(robust_retained, robust_total),
            "minimum_per_map_legacy_opportunity_recall": min(
                value["opportunity_state_recall"] for value in by_map.values()
            ),
            "pretail_beneficial_comparison_recall": _fraction(
                pretail_retained, len(informative_beneficial)
            ),
            "causalclosure_only_best_action_recall": _fraction(
                sum(
                    bool(row["budgets"][key]["causalclosure_only_best_preserved"])
                    for row in causal_only_rows
                ),
                len(causal_only_rows),
            ),
            "v2_base_recall": 1.0,
        }
        gates = {
            "legacy_opportunity_state_recall": metrics[
                "legacy_opportunity_state_recall"
            ]
            >= float(gate_config["minimum_legacy_opportunity_state_recall"]),
            "legacy_robust_action_recall": metrics["legacy_robust_action_recall"]
            >= float(gate_config["minimum_legacy_robust_action_recall"]),
            "per_map_legacy_opportunity_recall": metrics[
                "minimum_per_map_legacy_opportunity_recall"
            ]
            >= float(gate_config["minimum_per_map_legacy_opportunity_recall"]),
            "pretail_beneficial_comparison_recall": metrics[
                "pretail_beneficial_comparison_recall"
            ]
            >= float(gate_config["minimum_pretail_beneficial_comparison_recall"]),
            "causalclosure_only_best_action_recall": metrics[
                "causalclosure_only_best_action_recall"
            ]
            >= float(gate_config["minimum_causalclosure_only_best_action_recall"]),
            "v2_base_recall": metrics["v2_base_recall"]
            >= float(gate_config["v2_base_recall"]),
            "budget_never_exceeded": all(
                bool(row["budgets"][key]["budget_respected"]) for row in states
            ),
            "zero_identity_or_integrity_errors": not identity_errors,
        }
        summaries[key] = {
            "budget": budget,
            "mean_selected_challengers": statistics.fmean(
                int(row["budgets"][key]["selected_candidate_count"])
                for row in states
            ),
            "metrics": metrics,
            "by_map": by_map,
            "gates": gates,
            "passed": all(gates.values()),
        }
    passing = [budget for budget in (6, 8, 12) if summaries[str(budget)]["passed"]]
    integrity = {
        "state_count": len(states) == int(expected["state_count"]),
        "map_count": len({row["map_id"] for row in states}) == int(expected["map_count"]),
        "legacy_robust_action_count": sum(
            len(row["legacy_robust_action_ids"]) for row in states
        )
        == int(expected["legacy_robust_action_count"]),
        "legacy_robust_opportunity_state_count": sum(
            bool(row["legacy_robust_action_ids"]) for row in states
        )
        == int(expected["legacy_robust_opportunity_state_count"]),
        "causalclosure_only_opportunity_state_count": sum(
            row["causalclosure_only_best_candidate_id"] is not None for row in states
        )
        == int(expected["causalclosure_only_opportunity_state_count"]),
        "pretail_beneficial_comparison_count": len(informative_beneficial)
        == int(expected["pretail_beneficial_comparison_count"]),
        "zero_identity_errors": not identity_errors,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_outcome_blind_budget_membership_audit",
        "experiment_id": EXPERIMENT_ID,
        "registration_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "integrity": integrity,
        "identity_errors": sorted(set(identity_errors)),
        "integrity_passed": all(integrity.values()),
        "budgets": summaries,
        "passing_budgets": passing,
        "selected_budget": min(passing) if passing else None,
        "full_union_required": not passing,
        "runtime_replacement_ready": False,
        "next_step": (
            "freeze_smallest_budget_then_design_separate_runtime_selector_validation"
            if passing
            else "retain_full_union_and_do_not_compress_before_post_failure_repair_study"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "budget_state_rows.jsonl", states)
    _write_json(output / "hybridstructpool_budget_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "STATE_SCHEMA",
    "analyze_hybridstructpool_budget",
]
