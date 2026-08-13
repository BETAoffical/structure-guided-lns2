from __future__ import annotations

import concurrent.futures
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_structshell_audit import ALLOWED_SIZES, parse_structpool_family


CONFIG_SCHEMA = "lns2.stride.shellbudget_audit_registration.v1"
REPORT_SCHEMA = "lns2.stride.shellbudget_audit_report.v1"
EXPERIMENT_ID = "stride-shellbudget-audit-v1"
BUDGETS = (6, 8, 12)
STRUCTURAL_COVERAGE_FIELDS = (
    "realized.incident_event_coverage",
    "realized.incident_conflict_coverage",
    "realized.internal_conflict_coverage",
    "realized.component_coverage_max",
)
FORBIDDEN_SELECTION_FIELDS = (
    "seed_mean",
    "first_fixed_half_mean",
    "second_fixed_half_mean",
    "no_progress_rate",
    "classification",
    "reason",
    "ttf",
    "runtime",
    "future_trajectory",
)


def _load_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_ranker_free_equal_four_size_budget_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("ShellBudget registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label="ShellBudget audit")
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "structshell_report",
        "non_tail_size_rows",
        "maze_candidate_aggregates",
        "maze_state_rule_rows",
        "pretail_membership_rows",
    }:
        raise ValueError("ShellBudget registered inputs changed")
    contract = dict(config.get("candidate_contract") or {})
    if (
        tuple(map(int, contract.get("allowed_sizes") or ())) != ALLOWED_SIZES
        or tuple(map(int, contract.get("candidate_budgets") or ())) != BUDGETS
        or contract.get("v2_pool_and_anchor_retained_separately") is not True
        or contract.get("fixed_family_preferred_sizes_allowed") is not False
        or contract.get("posthoc_budget_or_rule_selection_allowed") is not False
    ):
        raise ValueError("ShellBudget candidate contract changed")
    reducer = dict(config.get("primary_reducer") or {})
    if (
        reducer.get("id") != "size_family_balanced_maximin_v1"
        or tuple(reducer.get("structural_coverage_fields") or ())
        != STRUCTURAL_COVERAGE_FIELDS
        or tuple(reducer.get("forbidden_selection_fields") or ())
        != FORBIDDEN_SELECTION_FIELDS
    ):
        raise ValueError("ShellBudget primary reducer changed")
    boundary = dict(config.get("claim_boundary") or {})
    if not all(
        boundary.get(name) is expected
        for name, expected in {
            "retrospective_diagnostic_only": True,
            "outcome_enriched_development_data": True,
            "candidate_generation_changed": False,
            "ranker_used": False,
            "model_training_allowed": False,
            "runtime_integration_allowed": False,
            "ttf_experiment_allowed": False,
            "long_tail_avoidance_claim_allowed": False,
            "default_controller_replacement_allowed": False,
        }.items()
    ):
        raise ValueError("ShellBudget claim boundary changed")
    return path, root, config, inputs


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    labels = sorted(
        {
            str(label)
            for label in row.get("selection_families") or ()
            if str(label).startswith("structpool-")
        }
    )
    if not labels:
        raise ValueError("ShellBudget candidate has no StructPool provenance")
    parsed = [parse_structpool_family(label) for label in labels]
    sizes = {int(size) for _family, size in parsed}
    if len(sizes) != 1:
        raise ValueError("ShellBudget exact-deduplicated candidate spans nominal sizes")
    features = dict(row["features"])
    coverage = statistics.fmean(
        float(features[field]) for field in STRUCTURAL_COVERAGE_FIELDS
    )
    agents = tuple(sorted(set(map(int, row["agents"]))))
    if not agents or len(agents) != int(row["actual_size"]):
        raise ValueError("ShellBudget candidate agent identity changed")
    return {
        "candidate_id": str(row["candidate_id"]),
        "agents": agents,
        "nominal_size": next(iter(sizes)),
        "families": tuple(sorted({family for family, _size in parsed})),
        "family_size_slots": tuple(sorted(set(parsed))),
        "structural_coverage": coverage,
    }


def _merge_candidate_views(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        view = _candidate_view(row)
        candidate_id = str(view["candidate_id"])
        previous = by_id.get(candidate_id)
        if previous is not None and previous != view:
            raise ValueError(f"ShellBudget duplicate candidate changed: {candidate_id}")
        by_id[candidate_id] = view
    if not by_id:
        raise ValueError("ShellBudget state has no structural candidates")
    return [by_id[candidate_id] for candidate_id in sorted(by_id)]


def _jaccard_distance(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def reduce_size_family_balanced_maximin(
    candidates: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("ShellBudget budget must be positive")
    candidates = sorted(candidates, key=lambda row: str(row["candidate_id"]))
    if len(candidates) <= budget:
        return candidates
    active_sizes = sorted({int(row["nominal_size"]) for row in candidates})
    if tuple(active_sizes) != ALLOWED_SIZES:
        raise ValueError("ShellBudget state lost an equal-size alternative")
    active_families = sorted(
        {str(family) for row in candidates for family in row["families"]}
    )
    size_counts = {size: 0 for size in active_sizes}
    family_counts = {family: 0 for family in active_families}
    covered_slots: set[tuple[str, int]] = set()
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < budget:
        balance_feasible = []
        for candidate in remaining:
            prospective = dict(size_counts)
            prospective[int(candidate["nominal_size"])] += 1
            if max(prospective.values()) - min(prospective.values()) <= 1:
                balance_feasible.append(candidate)
        if not balance_feasible:
            break

        def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
            families = tuple(map(str, candidate["families"]))
            slots = set(candidate["family_size_slots"])
            minimum_distance = (
                min(
                    _jaccard_distance(candidate["agents"], chosen["agents"])
                    for chosen in selected
                )
                if selected
                else 1.0
            )
            return (
                size_counts[int(candidate["nominal_size"])],
                min(family_counts[family] for family in families),
                -sum(family_counts[family] == 0 for family in families),
                -len(slots - covered_slots),
                -minimum_distance,
                -float(candidate["structural_coverage"]),
                str(candidate["candidate_id"]),
            )

        chosen = min(balance_feasible, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        size_counts[int(chosen["nominal_size"])] += 1
        for family in chosen["families"]:
            family_counts[str(family)] += 1
        covered_slots.update(chosen["family_size_slots"])
    return selected


def _outcome_by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = ("seed_mean", "first_fixed_half_mean", "second_fixed_half_mean")
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        values = {field: float(row[field]) for field in fields}
        previous = output.get(candidate_id)
        if previous is not None and previous != values:
            raise ValueError(f"ShellBudget duplicated outcome changed: {candidate_id}")
        output[candidate_id] = values
    return output


def _best_ids(values: dict[str, dict[str, float]], field: str) -> tuple[set[str], float]:
    maximum = max(float(row[field]) for row in values.values())
    return (
        {
            candidate_id
            for candidate_id, row in values.items()
            if float(row[field]) == maximum
        },
        maximum,
    )


def _size_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(size): sum(int(row["nominal_size"]) == size for row in selected)
        for size in ALLOWED_SIZES
    }


def _non_tail_state_job(
    item: tuple[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    state_id, rows = item
    candidates = _merge_candidate_views(rows)
    outcomes = _outcome_by_candidate(rows)
    if set(outcomes) != {str(row["candidate_id"]) for row in candidates}:
        raise ValueError(f"ShellBudget non-tail candidate/outcome mismatch: {state_id}")
    full_best, full_mean = _best_ids(outcomes, "seed_mean")
    first_best, _first_mean = _best_ids(outcomes, "first_fixed_half_mean")
    second_best, _second_mean = _best_ids(outcomes, "second_fixed_half_mean")
    output = []
    for budget in BUDGETS:
        selected = reduce_size_family_balanced_maximin(candidates, budget)
        selected_ids = {str(row["candidate_id"]) for row in selected}
        selected_best = max(float(outcomes[candidate_id]["seed_mean"]) for candidate_id in selected_ids)
        counts = _size_counts(selected)
        output.append(
            {
                "state_id": state_id,
                "map_id": str(rows[0]["map_id"]),
                "budget": budget,
                "candidate_count": len(candidates),
                "selected_candidate_count": len(selected),
                "selected_candidate_ids": [str(row["candidate_id"]) for row in selected],
                "selected_size_counts": counts,
                "size_count_spread": max(counts.values()) - min(counts.values()),
                "size_balance_required": len(candidates) > budget,
                "global_best_retained": bool(full_best & selected_ids),
                "first_half_best_retained": bool(first_best & selected_ids),
                "second_half_best_retained": bool(second_best & selected_ids),
                "normalized_regret": full_mean - selected_best,
            }
        )
    return output


def _maze_state_job(
    item: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> list[dict[str, Any]]:
    state_fingerprint, rows, state = item
    structural = [
        row for row in rows if str(row.get("candidate_kind")) == "structural"
    ]
    candidates = _merge_candidate_views(structural)
    robust_ids = set(map(str, state["robust_action_ids"]))
    candidate_ids = {str(row["candidate_id"]) for row in candidates}
    if not robust_ids <= candidate_ids:
        raise ValueError(f"ShellBudget Maze robust action left pool: {state_fingerprint}")
    output = []
    for budget in BUDGETS:
        selected = reduce_size_family_balanced_maximin(candidates, budget)
        selected_ids = {str(row["candidate_id"]) for row in selected}
        counts = _size_counts(selected)
        retained = robust_ids & selected_ids
        output.append(
            {
                "state_fingerprint": state_fingerprint,
                "map_id": str(state["map_id"]),
                "budget": budget,
                "candidate_count": len(candidates),
                "selected_candidate_count": len(selected),
                "selected_candidate_ids": [str(row["candidate_id"]) for row in selected],
                "selected_size_counts": counts,
                "size_count_spread": max(counts.values()) - min(counts.values()),
                "size_balance_required": len(candidates) > budget,
                "robust_action_count": len(robust_ids),
                "retained_robust_action_count": len(retained),
                "has_structural_opportunity": bool(robust_ids),
                "preserves_opportunity": bool(retained),
            }
        )
    return output


def _summarize_non_tail(rows: list[dict[str, Any]]) -> dict[str, Any]:
    map_regrets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        map_regrets[str(row["map_id"])].append(float(row["normalized_regret"]))
    map_means = {
        map_id: statistics.fmean(values)
        for map_id, values in sorted(map_regrets.items())
    }
    return {
        "state_count": len(rows),
        "full_candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "selected_candidate_count": sum(
            int(row["selected_candidate_count"]) for row in rows
        ),
        "maximum_selected_candidate_count": max(
            int(row["selected_candidate_count"]) for row in rows
        ),
        "maximum_size_count_spread": max(int(row["size_count_spread"]) for row in rows),
        "maximum_required_size_count_spread": max(
            (
                int(row["size_count_spread"])
                for row in rows
                if row["size_balance_required"]
            ),
            default=0,
        ),
        "global_best_retention": statistics.fmean(
            bool(row["global_best_retained"]) for row in rows
        ),
        "first_half_best_retention": statistics.fmean(
            bool(row["first_half_best_retained"]) for row in rows
        ),
        "second_half_best_retention": statistics.fmean(
            bool(row["second_half_best_retained"]) for row in rows
        ),
        "mean_normalized_regret": statistics.fmean(
            float(row["normalized_regret"]) for row in rows
        ),
        "maximum_map_mean_regret": max(map_means.values()),
        "map_mean_regret": map_means,
    }


def _summarize_maze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opportunity = [row for row in rows if row["has_structural_opportunity"]]
    robust_count = sum(int(row["robust_action_count"]) for row in rows)
    retained_count = sum(int(row["retained_robust_action_count"]) for row in rows)
    by_map: dict[str, dict[str, Any]] = {}
    for map_id in sorted({str(row["map_id"]) for row in rows}):
        map_rows = [row for row in rows if str(row["map_id"]) == map_id]
        map_opportunity = [row for row in map_rows if row["has_structural_opportunity"]]
        preserved = sum(bool(row["preserves_opportunity"]) for row in map_opportunity)
        by_map[map_id] = {
            "state_count": len(map_rows),
            "opportunity_state_count": len(map_opportunity),
            "preserved_opportunity_state_count": preserved,
            "opportunity_state_recall": preserved / len(map_opportunity),
        }
    preserved = sum(bool(row["preserves_opportunity"]) for row in opportunity)
    return {
        "state_count": len(rows),
        "full_candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "selected_candidate_count": sum(
            int(row["selected_candidate_count"]) for row in rows
        ),
        "maximum_selected_candidate_count": max(
            int(row["selected_candidate_count"]) for row in rows
        ),
        "maximum_size_count_spread": max(int(row["size_count_spread"]) for row in rows),
        "maximum_required_size_count_spread": max(
            (
                int(row["size_count_spread"])
                for row in rows
                if row["size_balance_required"]
            ),
            default=0,
        ),
        "opportunity_state_count": len(opportunity),
        "preserved_opportunity_state_count": preserved,
        "opportunity_state_recall": preserved / len(opportunity),
        "robust_action_count": robust_count,
        "retained_robust_action_count": retained_count,
        "robust_action_recall": retained_count / robust_count,
        "by_map": by_map,
    }


def _tail_rows(
    source_rows: list[dict[str, Any]], maze_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = {
        (str(row["state_fingerprint"]), int(row["budget"])): set(
            map(str, row["selected_candidate_ids"])
        )
        for row in maze_rows
    }
    output = []
    for row in source_rows:
        retained = {}
        for budget in BUDGETS:
            retained[str(budget)] = bool(row["candidate_is_v2_base"]) or str(
                row["candidate_id"]
            ) in selected[(str(row["state_fingerprint"]), budget)]
        output.append(
            {
                "case_id": str(row["case_id"]),
                "trial_index": int(row["trial_index"]),
                "arm": str(row["arm"]),
                "state_fingerprint": str(row["state_fingerprint"]),
                "map_id": str(row["map_id"]),
                "candidate_id": str(row["candidate_id"]),
                "candidate_is_v2_base": bool(row["candidate_is_v2_base"]),
                "classification": str(row["classification"]),
                "identical_action": bool(row["identical_action"]),
                "retained_by_budget": retained,
            }
        )
    return output


def _summarize_tail(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    informative = [row for row in rows if not row["identical_action"]]
    by_classification = {}
    for classification in ("beneficial", "neutral", "adverse"):
        selected = [
            row for row in informative if row["classification"] == classification
        ]
        retained = sum(bool(row["retained_by_budget"][str(budget)]) for row in selected)
        by_classification[classification] = {
            "comparison_count": len(selected),
            "retained_comparison_count": retained,
            "retained_comparison_fraction": retained / len(selected),
        }
    return {
        "informative_comparison_count": len(informative),
        "by_classification": by_classification,
        "claim_boundary": "membership_only_not_online_selection_or_loop_avoidance",
    }


def analyze_shellbudget(
    config_path: str | Path, output: str | Path, *, workers: int = 16
) -> dict[str, Any]:
    config_path, _root, config, inputs = _load_config(config_path)
    if workers != int(config["analysis"]["workers"]):
        raise ValueError("ShellBudget audit must use the registered 16 workers")
    structshell = _read_json(inputs["structshell_report"])
    non_tail_source = _read_jsonl(inputs["non_tail_size_rows"])
    maze_source = _read_jsonl(inputs["maze_candidate_aggregates"])
    maze_state_source = _read_jsonl(inputs["maze_state_rule_rows"])
    pretail_source = _read_jsonl(inputs["pretail_membership_rows"])

    non_tail_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in non_tail_source:
        non_tail_groups[str(row["state_id"])].append(dict(row))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        non_tail_nested = list(executor.map(_non_tail_state_job, sorted(non_tail_groups.items())))
    non_tail_rows = [row for group in non_tail_nested for row in group]

    maze_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in maze_source:
        maze_groups[str(row["state_fingerprint"])].append(dict(row))
    state_by_fingerprint = {
        str(row["state_fingerprint"]): dict(row) for row in maze_state_source
    }
    maze_jobs = [
        (fingerprint, rows, state_by_fingerprint[fingerprint])
        for fingerprint, rows in sorted(maze_groups.items())
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        maze_nested = list(executor.map(_maze_state_job, maze_jobs))
    maze_rows = [row for group in maze_nested for row in group]
    tail_rows = _tail_rows(pretail_source, maze_rows)

    expected_non_tail = dict(config["cohorts"]["non_tail"])
    expected_maze = dict(config["cohorts"]["maze_difficult"])
    expected_tail = dict(config["cohorts"]["bounded_continuation"])
    non_tail_unique_candidates = {
        (str(row["state_id"]), str(row["candidate_id"])) for row in non_tail_source
    }
    integrity = {
        "structshell_integrity": structshell.get("integrity_passed") is True,
        "structshell_equal_grid_fallback": structshell.get("recommended_size_policy")
        == "retain_equal_four_size_grid_8_16_24_32",
        "non_tail_state_count": len(non_tail_groups)
        == int(expected_non_tail["state_count"]),
        "non_tail_candidate_count": len(non_tail_unique_candidates)
        == int(expected_non_tail["candidate_count"]),
        "non_tail_map_count": len({str(row["map_id"]) for row in non_tail_source})
        == int(expected_non_tail["map_count"]),
        "non_tail_complete_budget_matrix": len(non_tail_rows)
        == int(expected_non_tail["state_count"]) * len(BUDGETS),
        "maze_state_count": len(maze_groups) == int(expected_maze["state_count"]),
        "maze_candidate_count": len(maze_source)
        == int(expected_maze["candidate_count"]),
        "maze_structural_candidate_count": sum(
            str(row.get("candidate_kind")) == "structural" for row in maze_source
        )
        == int(expected_maze["structural_candidate_count"]),
        "maze_robust_action_count": sum(
            len(row["robust_action_ids"]) for row in maze_state_source
        )
        == int(expected_maze["robust_action_count"]),
        "maze_opportunity_state_count": sum(
            bool(row["has_structural_opportunity"]) for row in maze_state_source
        )
        == int(expected_maze["opportunity_state_count"]),
        "maze_complete_budget_matrix": len(maze_rows)
        == int(expected_maze["state_count"]) * len(BUDGETS),
        "pretail_comparison_count": len(tail_rows)
        == int(expected_tail["paired_comparison_count"]),
        "pretail_informative_comparison_count": sum(
            not row["identical_action"] for row in tail_rows
        )
        == int(expected_tail["informative_comparison_count"]),
        "pretail_beneficial_comparison_count": sum(
            not row["identical_action"] and row["classification"] == "beneficial"
            for row in tail_rows
        )
        == int(expected_tail["beneficial_comparison_count"]),
        "budgets_respected": all(
            int(row["selected_candidate_count"]) <= int(row["budget"])
            for row in non_tail_rows + maze_rows
        ),
        "size_balance_respected": all(
            not row["size_balance_required"]
            or int(row["size_count_spread"])
            <= int(config["readiness_gates"]["maximum_size_count_spread"])
            for row in non_tail_rows + maze_rows
        ),
    }

    thresholds = dict(config["readiness_gates"])
    budgets: dict[str, dict[str, Any]] = {}
    for budget in BUDGETS:
        non_tail = _summarize_non_tail(
            [row for row in non_tail_rows if int(row["budget"]) == budget]
        )
        maze = _summarize_maze(
            [row for row in maze_rows if int(row["budget"]) == budget]
        )
        tail = _summarize_tail(tail_rows, budget)
        per_map_recalls = [
            float(row["opportunity_state_recall"])
            for row in maze["by_map"].values()
        ]
        beneficial_recall = float(
            tail["by_classification"]["beneficial"][
                "retained_comparison_fraction"
            ]
        )
        gates = {
            "non_tail_global_best_retention": float(non_tail["global_best_retention"])
            >= float(thresholds["minimum_non_tail_global_best_retention"]),
            "non_tail_mean_normalized_regret": float(non_tail["mean_normalized_regret"])
            <= float(thresholds["maximum_non_tail_mean_normalized_regret"]),
            "non_tail_map_mean_regret": float(non_tail["maximum_map_mean_regret"])
            <= float(thresholds["maximum_non_tail_map_mean_regret"]),
            "non_tail_first_half_best_retention": float(
                non_tail["first_half_best_retention"]
            )
            >= float(thresholds["minimum_non_tail_first_half_best_retention"]),
            "non_tail_second_half_best_retention": float(
                non_tail["second_half_best_retention"]
            )
            >= float(thresholds["minimum_non_tail_second_half_best_retention"]),
            "maze_opportunity_state_recall": float(maze["opportunity_state_recall"])
            >= float(thresholds["minimum_maze_opportunity_state_recall"]),
            "maze_robust_action_recall": float(maze["robust_action_recall"])
            >= float(thresholds["minimum_maze_robust_action_recall"]),
            "maze_per_map_opportunity_state_recall": min(per_map_recalls)
            >= float(thresholds["minimum_maze_per_map_opportunity_state_recall"]),
            "pretail_beneficial_membership_recall": beneficial_recall
            >= float(thresholds["minimum_pretail_beneficial_membership_recall"]),
            "size_count_spread": max(
                int(non_tail["maximum_required_size_count_spread"]),
                int(maze["maximum_required_size_count_spread"]),
            )
            <= int(thresholds["maximum_size_count_spread"]),
            "zero_identity_or_integrity_errors": all(integrity.values()),
        }
        budgets[str(budget)] = {
            "non_tail": non_tail,
            "maze_difficult": maze,
            "bounded_continuation_membership": tail,
            "readiness_gates": gates,
            "passed": all(gates.values()),
        }
    passing = [budget for budget in BUDGETS if budgets[str(budget)]["passed"]]
    selected_budget = passing[0] if passing else None
    decision = (
        f"retain_ranker_free_equal_four_size_budget_{selected_budget}"
        if selected_budget is not None
        else "retain_full_exact_deduplicated_equal_four_size_pool"
    )

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "non_tail_budget_rows.jsonl", non_tail_rows)
    _write_jsonl(output / "maze_budget_rows.jsonl", maze_rows)
    _write_jsonl(output / "pretail_budget_membership_rows.jsonl", tail_rows)
    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "non_tail_budget_rows.jsonl",
            "maze_budget_rows.jsonl",
            "pretail_budget_membership_rows.jsonl",
        )
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_ranker_free_equal_four_size_budget_audit",
        "experiment_id": EXPERIMENT_ID,
        "registration_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "workers": workers,
        "primary_reducer": dict(config["primary_reducer"]),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "budgets": budgets,
        "passing_budgets": passing,
        "selected_budget": selected_budget,
        "decision": decision,
        "next_step": (
            "preregister_independent_result_blind_bounded_forced_continuation"
            if selected_budget is not None
            else "retain_full_pool_and_design_preloop_selector_without_candidate_compression"
        ),
        "artifacts": artifacts,
        "claim_boundary": dict(config["claim_boundary"]),
    }
    _write_json(output / "shellbudget_audit_report.json", report)
    return report


__all__ = [
    "BUDGETS",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "STRUCTURAL_COVERAGE_FIELDS",
    "analyze_shellbudget",
    "reduce_size_family_balanced_maximin",
]
