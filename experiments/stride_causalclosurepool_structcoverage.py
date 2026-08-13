from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_marginalpool_action_replay import stable_dominates


CONFIG_SCHEMA = "lns2.stride.causalclosurepool_structcoverage_registration.v1"
REPORT_SCHEMA = "lns2.stride.causalclosurepool_structcoverage_report.v1"
EXPERIMENT_ID = "stride-causalclosurepool-structcoverage-v1"


def _load_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_posthoc_structural_coverage_diagnostic"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("structural-coverage registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "causalclosure_cohort",
        "legacy_candidate_aggregates",
        "causalclosure_opportunity_report",
    }:
        raise ValueError("structural-coverage input registry changed")
    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "state_count": 78,
        "base_candidate_count": 1367,
        "legacy_structural_candidate_count": 1135,
        "causalclosure_candidate_count": 930,
        "all_states_required": True,
        "development_outcome_enriched": True,
        "no_result_based_state_exclusion": True,
    }:
        raise ValueError("structural-coverage cohort contract changed")
    analysis = dict(config.get("analysis") or {})
    expected_analysis = {
        "exact_agent_set_threshold": 1.0,
        "near_jaccard_threshold": 0.8,
        "descriptive_jaccard_threshold": 0.6,
        "medium_size_minimum": 8,
        "medium_size_maximum": 17,
        "stable_dominance_minimum_seed_mean_advantage": 0.02,
        "no_progress_rate_must_not_increase": True,
        "both_fixed_seed_halves_must_improve": True,
        "family_membership_is_multi_label": True,
    }
    if analysis != expected_analysis:
        raise ValueError("structural-coverage analysis contract changed")
    gates = dict(config.get("decision_gates") or {})
    if gates != {
        "minimum_old_opportunity_state_recall": 0.9,
        "minimum_old_robust_action_jaccard_0_8_coverage": 0.9,
        "minimum_per_map_old_opportunity_state_recall": 0.8,
        "zero_identity_or_coverage_errors": True,
    }:
        raise ValueError("structural-coverage decision gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "development_posthoc_coverage_diagnostic_only": True,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
        "current_generator_promotion_allowed": False,
    }:
        raise ValueError("structural-coverage claim boundary changed")
    return path, root, config, inputs


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(map(int, left)), set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def closest_action(
    legacy: dict[str, Any], causal_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not causal_actions:
        raise ValueError("state has no CausalClosure actions")
    ranked = sorted(
        causal_actions,
        key=lambda row: (
            -jaccard(legacy["agents"], row["agents"]),
            abs(int(legacy["actual_size"]) - int(row["actual_size"])),
            str(row["candidate_id"]),
        ),
    )
    best = ranked[0]
    return {
        "candidate_id": str(best["candidate_id"]),
        "actual_size": int(best["actual_size"]),
        "jaccard": jaccard(legacy["agents"], best["agents"]),
        "exact_agent_set": set(map(int, legacy["agents"]))
        == set(map(int, best["agents"])),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _opportunity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_count = sum(bool(row["legacy_best_robustly_beats_v2_best"]) for row in rows)
    causal_count = sum(bool(row["causalclosure_robustly_beats_v2_best"]) for row in rows)
    both = sum(
        bool(row["legacy_best_robustly_beats_v2_best"])
        and bool(row["causalclosure_robustly_beats_v2_best"])
        for row in rows
    )
    return {
        "state_count": len(rows),
        "legacy_opportunity_state_count": old_count,
        "causalclosure_opportunity_state_count": causal_count,
        "shared_opportunity_state_count": both,
        "lost_legacy_opportunity_state_count": sum(
            bool(row["legacy_best_robustly_beats_v2_best"])
            and not bool(row["causalclosure_robustly_beats_v2_best"])
            for row in rows
        ),
        "new_causalclosure_opportunity_state_count": sum(
            not bool(row["legacy_best_robustly_beats_v2_best"])
            and bool(row["causalclosure_robustly_beats_v2_best"])
            for row in rows
        ),
        "old_opportunity_state_recall": _fraction(both, old_count),
        "causalclosure_opportunity_precision_against_legacy": _fraction(
            both, causal_count
        ),
    }


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_count": len(rows),
        "exact_agent_set_count": sum(bool(row["exact_agent_set"]) for row in rows),
        "exact_agent_set_fraction": _fraction(
            sum(bool(row["exact_agent_set"]) for row in rows), len(rows)
        ),
        "jaccard_at_least_0_8_count": sum(
            float(row["maximum_causalclosure_jaccard"]) >= 0.8 for row in rows
        ),
        "jaccard_at_least_0_8_fraction": _fraction(
            sum(float(row["maximum_causalclosure_jaccard"]) >= 0.8 for row in rows),
            len(rows),
        ),
        "jaccard_at_least_0_6_count": sum(
            float(row["maximum_causalclosure_jaccard"]) >= 0.6 for row in rows
        ),
        "jaccard_at_least_0_6_fraction": _fraction(
            sum(float(row["maximum_causalclosure_jaccard"]) >= 0.6 for row in rows),
            len(rows),
        ),
        "mean_maximum_jaccard": (
            sum(float(row["maximum_causalclosure_jaccard"]) for row in rows)
            / len(rows)
            if rows
            else None
        ),
    }


def analyze_structural_coverage(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, _root, config, inputs = _load_config(config_path)
    cohort_rows = _read_jsonl(inputs["causalclosure_cohort"])
    aggregate_rows = _read_jsonl(inputs["legacy_candidate_aggregates"])
    opportunity = _read_json(inputs["causalclosure_opportunity_report"])
    cohort_by_state = {
        str(row["state_fingerprint"]): dict(row) for row in cohort_rows
    }
    opportunity_by_state = {
        str(row["state_fingerprint"]): dict(row)
        for row in opportunity.get("states") or ()
    }
    aggregates_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        aggregates_by_state[str(row["state_fingerprint"])].append(dict(row))
    state_keys = set(cohort_by_state)
    identity_errors: list[str] = []
    if state_keys != set(opportunity_by_state):
        identity_errors.append("opportunity_state_set")
    if state_keys != set(aggregates_by_state):
        identity_errors.append("aggregate_state_set")
    state_rows: list[dict[str, Any]] = []
    legacy_action_rows: list[dict[str, Any]] = []
    for state_key in sorted(state_keys):
        cohort = cohort_by_state[state_key]
        rows = aggregates_by_state.get(state_key, [])
        base = [row for row in rows if row.get("candidate_kind") == "base"]
        structural = [
            row for row in rows if row.get("candidate_kind") == "structural"
        ]
        causal = [dict(row) for row in cohort["causalclosure_candidates"]]
        registered_base_ids = set(map(str, cohort["base_candidate_ids"]))
        observed_base_ids = {str(row["candidate_id"]) for row in base}
        if registered_base_ids != observed_base_ids:
            identity_errors.append(f"base_candidate_set:{state_key}")
        if not base or not structural or not causal:
            identity_errors.append(f"empty_pool:{state_key}")
            continue
        base_best = max(
            base, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"]))
        )
        legacy_best = max(
            structural,
            key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])),
        )
        robust_actions = [
            row for row in structural if stable_dominates(row, base_best)
        ]
        legacy_best_overlap = closest_action(legacy_best, causal)
        for action in structural:
            overlap = closest_action(action, causal)
            legacy_action_rows.append(
                {
                    "state_fingerprint": state_key,
                    "map_id": str(cohort["map_id"]),
                    "candidate_id": str(action["candidate_id"]),
                    "actual_size": int(action["actual_size"]),
                    "structpool_family_groups": sorted(
                        map(str, action.get("structpool_family_groups") or ())
                    ),
                    "robustly_beats_v2_best": stable_dominates(action, base_best),
                    "closest_causalclosure_candidate_id": overlap["candidate_id"],
                    "closest_causalclosure_size": overlap["actual_size"],
                    "maximum_causalclosure_jaccard": overlap["jaccard"],
                    "exact_agent_set": overlap["exact_agent_set"],
                }
            )
        causal_result = opportunity_by_state.get(state_key)
        if causal_result is None:
            identity_errors.append(f"missing_opportunity:{state_key}")
            continue
        state_rows.append(
            {
                "state_fingerprint": state_key,
                "map_id": str(cohort["map_id"]),
                "base_candidate_count": len(base),
                "legacy_structural_candidate_count": len(structural),
                "causalclosure_candidate_count": len(causal),
                "v2_best_candidate_id": str(base_best["candidate_id"]),
                "legacy_best_candidate_id": str(legacy_best["candidate_id"]),
                "legacy_best_size": int(legacy_best["actual_size"]),
                "legacy_best_family_groups": sorted(
                    map(str, legacy_best.get("structpool_family_groups") or ())
                ),
                "legacy_best_seed_mean_advantage": float(legacy_best["seed_mean"])
                - float(base_best["seed_mean"]),
                "legacy_best_robustly_beats_v2_best": stable_dominates(
                    legacy_best, base_best
                ),
                "legacy_robust_action_count": len(robust_actions),
                "legacy_best_closest_causalclosure_candidate_id": legacy_best_overlap[
                    "candidate_id"
                ],
                "legacy_best_maximum_causalclosure_jaccard": legacy_best_overlap[
                    "jaccard"
                ],
                "legacy_best_exactly_preserved": legacy_best_overlap[
                    "exact_agent_set"
                ],
                "causalclosure_robustly_beats_v2_best": bool(
                    causal_result["new_stably_dominates_base_best"]
                ),
            }
        )
    expected = dict(config["cohort"])
    integrity = {
        "state_count": len(state_rows) == int(expected["state_count"]),
        "base_candidate_count": sum(
            int(row["base_candidate_count"]) for row in state_rows
        )
        == int(expected["base_candidate_count"]),
        "legacy_structural_candidate_count": len(legacy_action_rows)
        == int(expected["legacy_structural_candidate_count"]),
        "causalclosure_candidate_count": sum(
            int(row["causalclosure_candidate_count"]) for row in state_rows
        )
        == int(expected["causalclosure_candidate_count"]),
        "opportunity_report_integrity": opportunity.get("integrity_passed") is True,
        "zero_identity_errors": not identity_errors,
    }
    overall = _opportunity_summary(state_rows)
    robust_action_rows = [
        row for row in legacy_action_rows if row["robustly_beats_v2_best"]
    ]
    robust_coverage = _coverage_summary(robust_action_rows)
    all_coverage = _coverage_summary(legacy_action_rows)
    by_map = {
        map_id: {
            "opportunity": _opportunity_summary(
                [row for row in state_rows if row["map_id"] == map_id]
            ),
            "legacy_robust_action_coverage": _coverage_summary(
                [row for row in robust_action_rows if row["map_id"] == map_id]
            ),
        }
        for map_id in sorted({row["map_id"] for row in state_rows})
    }
    family_names = sorted(
        {
            family
            for row in legacy_action_rows
            for family in row["structpool_family_groups"]
        }
    )
    by_family = {
        family: _coverage_summary(
            [
                row
                for row in robust_action_rows
                if family in row["structpool_family_groups"]
            ]
        )
        for family in family_names
    }
    by_size = {
        str(size): _coverage_summary(
            [row for row in robust_action_rows if int(row["actual_size"]) == size]
        )
        for size in (8, 16, 24, 32)
    }
    gates_config = dict(config["decision_gates"])
    per_map_recalls = [
        summary["opportunity"]["old_opportunity_state_recall"]
        for summary in by_map.values()
        if summary["opportunity"]["legacy_opportunity_state_count"] > 0
    ]
    gates = {
        "old_opportunity_state_recall": (
            overall["old_opportunity_state_recall"] is not None
            and overall["old_opportunity_state_recall"]
            >= float(gates_config["minimum_old_opportunity_state_recall"])
        ),
        "old_robust_action_jaccard_0_8_coverage": (
            robust_coverage["jaccard_at_least_0_8_fraction"] is not None
            and robust_coverage["jaccard_at_least_0_8_fraction"]
            >= float(
                gates_config[
                    "minimum_old_robust_action_jaccard_0_8_coverage"
                ]
            )
        ),
        "per_map_old_opportunity_state_recall": bool(per_map_recalls)
        and min(per_map_recalls)
        >= float(gates_config["minimum_per_map_old_opportunity_state_recall"]),
        "zero_identity_or_coverage_errors": all(integrity.values()),
    }
    preserves = all(gates.values())
    medium_min = int(config["analysis"]["medium_size_minimum"])
    medium_max = int(config["analysis"]["medium_size_maximum"])
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_development_posthoc_structural_coverage_diagnostic",
        "experiment_id": EXPERIMENT_ID,
        "registration_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "integrity": integrity,
        "integrity_errors": sorted(identity_errors),
        "integrity_passed": all(integrity.values()),
        "overall_opportunity": overall,
        "all_legacy_structural_action_coverage": all_coverage,
        "legacy_robust_action_coverage": robust_coverage,
        "legacy_robust_action_medium_size_fraction": _fraction(
            sum(
                medium_min <= int(row["actual_size"]) <= medium_max
                for row in robust_action_rows
            ),
            len(robust_action_rows),
        ),
        "by_map": by_map,
        "by_family": by_family,
        "by_size": by_size,
        "state_rows": state_rows,
        "legacy_robust_action_rows": robust_action_rows,
        "decision_gates": gates,
        "current_pool_preserves_old_structural_opportunity": preserves,
        "hybrid_pool_required": not preserves,
        "next_step": (
            "independent_result_blind_forced_continuation_confirmation"
            if preserves
            else "preregister_outcome_blind_hybrid_topology_causal_pool_design"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "structural_coverage_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "analyze_structural_coverage",
    "closest_action",
    "jaccard",
]
