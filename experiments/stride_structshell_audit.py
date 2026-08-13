from __future__ import annotations

import concurrent.futures
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_marginalpool_action_replay import stable_dominates


CONFIG_SCHEMA = "lns2.stride.structshell_audit_registration.v1"
REPORT_SCHEMA = "lns2.stride.structshell_audit_report.v1"
EXPERIMENT_ID = "stride-structshell-audit-v1"
ALLOWED_SIZES = (8, 16, 24, 32)
RULES = (
    "structural_knee",
    "support_nearest",
    "fixed_preferred",
    "equal_four_size_grid",
)
FAMILY_LABELS = {
    "structpool-bottleneck-crossing": "bottleneck_crossing",
    "structpool-conflict-component": "conflict_component",
    "structpool-boundary-articulation": "topology_boundary_articulation",
    "structpool-boundary-low_degree": "topology_boundary_low_degree",
    "structpool-spatiotemporal-hotspot": "spatiotemporal_hotspot",
    "structpool-path-overlap": "path_overlap",
}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def parse_structpool_family(label: str) -> tuple[str, int]:
    prefix, separator, raw_size = str(label).rpartition(":")
    if not separator or prefix not in FAMILY_LABELS:
        raise ValueError(f"unsupported StructPool family label: {label}")
    size = int(raw_size)
    if size not in ALLOWED_SIZES:
        raise ValueError(f"unsupported StructPool nominal size: {label}")
    return FAMILY_LABELS[prefix], size


def _family_signal(row: dict[str, Any], family: str) -> float:
    features = dict(row["features"])
    if family == "bottleneck_crossing":
        return float(features["realized.internal_conflict_coverage"])
    if family == "conflict_component":
        return float(features["realized.component_coverage_max"])
    if family in {
        "topology_boundary_articulation",
        "topology_boundary_low_degree",
    }:
        return float(features["realized.boundary_conflict_edges"]) / max(
            1.0, float(features["state.colliding_pairs"])
        )
    if family == "spatiotemporal_hotspot":
        return float(features["realized.incident_event_coverage"])
    if family == "path_overlap":
        return float(features["realized.incident_conflict_coverage"])
    raise ValueError(f"unsupported StructPool family: {family}")


def structural_vector(row: dict[str, Any], family: str) -> tuple[float, ...]:
    features = dict(row["features"])
    return (
        float(features["realized.incident_event_coverage"]),
        float(features["realized.internal_conflict_coverage"]),
        float(features["realized.component_coverage_max"]),
        _family_signal(row, family),
    )


def select_structural_knee(
    rows: Iterable[dict[str, Any]], family: str
) -> dict[str, Any]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["nominal_size"]), str(row["candidate_id"])),
    )
    if not ordered:
        raise ValueError("structural knee requires candidates")
    if len(ordered) == 1:
        return ordered[0]
    vectors = [structural_vector(row, family) for row in ordered]
    minima = [min(vector[index] for vector in vectors) for index in range(4)]
    maxima = [max(vector[index] for vector in vectors) for index in range(4)]
    sizes = [int(row["nominal_size"]) for row in ordered]
    minimum_size, maximum_size = min(sizes), max(sizes)
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for row, vector in zip(ordered, vectors):
        normalized = [
            (value - minima[index]) / (maxima[index] - minima[index])
            if maxima[index] > minima[index]
            else 0.0
            for index, value in enumerate(vector)
        ]
        size = int(row["nominal_size"])
        normalized_size = (
            (size - minimum_size) / (maximum_size - minimum_size)
            if maximum_size > minimum_size
            else 0.0
        )
        score = statistics.fmean(normalized) - normalized_size
        scored.append((score, size, str(row["candidate_id"]), row))
    return min(scored, key=lambda item: (-item[0], item[1], item[2]))[3]


def _support_count(row: dict[str, Any], family: str) -> int:
    if "support_count" in row:
        return int(row["support_count"])
    for label, value in dict(row.get("structpool_support_count_by_family") or {}).items():
        parsed_family, _size = parse_structpool_family(str(label))
        if parsed_family == family:
            return int(value)
    raise ValueError(f"missing support count for {family}")


def select_support_nearest(
    rows: Iterable[dict[str, Any]], family: str
) -> dict[str, Any]:
    ordered = [dict(row) for row in rows]
    if not ordered:
        raise ValueError("support-nearest requires candidates")
    support = _support_count(ordered[0], family)
    return min(
        ordered,
        key=lambda row: (
            abs(int(row["nominal_size"]) - support),
            int(row["nominal_size"]),
            str(row["candidate_id"]),
        ),
    )


def select_fixed_preferred(
    rows: Iterable[dict[str, Any]], preferred_size: int
) -> dict[str, Any] | None:
    choices = [dict(row) for row in rows if int(row["nominal_size"]) == preferred_size]
    return min(choices, key=lambda row: str(row["candidate_id"])) if choices else None


def _rule_candidates(
    rows_by_family: dict[str, list[dict[str, Any]]],
    fixed: dict[str, int],
) -> dict[str, set[str]]:
    selected = {rule: set() for rule in RULES}
    for family, rows in sorted(rows_by_family.items()):
        selected["structural_knee"].add(
            str(select_structural_knee(rows, family)["candidate_id"])
        )
        selected["support_nearest"].add(
            str(select_support_nearest(rows, family)["candidate_id"])
        )
        fixed_row = select_fixed_preferred(rows, int(fixed[family]))
        if fixed_row is not None:
            selected["fixed_preferred"].add(str(fixed_row["candidate_id"]))
        selected["equal_four_size_grid"].update(
            str(row["candidate_id"]) for row in rows
        )
    return selected


def _load_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_retrospective_structpool_shell_and_cutpoint_diagnostic"
        or config.get("pre_registration_head")
        != "b6ff0cec8d7ebf2d1033800bcad8e0a3dcf64884"
    ):
        raise ValueError("StructShell audit registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "size_ablation_rows",
        "size_ablation_report",
        "maze_candidate_aggregates",
        "maze_checkpoint_results",
        "maze_action_replay_report",
        "legacy_structural_coverage_report",
        "pretail_report",
        "pretail_schedule",
    }:
        raise ValueError("StructShell audit input registry changed")
    contract = dict(config.get("candidate_contract") or {})
    if (
        tuple(map(int, contract.get("allowed_sizes") or ())) != ALLOWED_SIZES
        or contract.get("primary_outcome_blind_rule") != "structural_knee"
        or list(contract.get("comparators") or ())
        != ["support_nearest", "fixed_preferred", "equal_four_size_grid"]
        or bool(contract.get("posthoc_best_rule_selection_allowed"))
        or contract.get("missing_family_or_size_is_not_imputed") is not True
    ):
        raise ValueError("StructShell candidate contract changed")
    if set(contract.get("fixed_preferred_size_comparator") or {}) != set(
        FAMILY_LABELS.values()
    ):
        raise ValueError("StructShell fixed-size comparator changed")
    analysis = dict(config.get("analysis") or {})
    if (
        int(analysis.get("workers", -1)) != 16
        or analysis.get("one_step_quality_is_not_loop_avoidance") is not True
        or bool(analysis.get("runtime_or_ttf_fields_used"))
    ):
        raise ValueError("StructShell analysis boundary changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "retrospective_diagnostic_only": True,
        "outcome_enriched_development_data": True,
        "candidate_generation_changed": False,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
        "default_controller_replacement_allowed": False,
    }:
        raise ValueError("StructShell claim boundary changed")
    return path, root, config, inputs


def _size_group_job(
    item: tuple[tuple[str, str], list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (state_id, family), rows = item
    rows = sorted(rows, key=lambda row: int(row["nominal_size"]))
    if tuple(int(row["nominal_size"]) for row in rows) != ALLOWED_SIZES:
        raise ValueError(f"incomplete size grid: {state_id} {family}")
    selected = {
        "structural_knee": select_structural_knee(rows, family),
        "support_nearest": select_support_nearest(rows, family),
    }
    preferred = {
        "bottleneck_crossing": 8,
        "conflict_component": 24,
        "topology_boundary_articulation": 16,
        "topology_boundary_low_degree": 16,
        "spatiotemporal_hotspot": 16,
        "path_overlap": 32,
    }
    selected["fixed_preferred"] = select_fixed_preferred(rows, preferred[family])
    best = max(rows, key=lambda row: (float(row["seed_mean"]), -int(row["nominal_size"])))
    first_best = max(
        rows,
        key=lambda row: (float(row["first_fixed_half_mean"]), -int(row["nominal_size"])),
    )
    second_best = max(
        rows,
        key=lambda row: (float(row["second_fixed_half_mean"]), -int(row["nominal_size"])),
    )
    group = {
        "state_id": state_id,
        "map_id": str(rows[0]["map_id"]),
        "family": family,
        "best_size": int(best["nominal_size"]),
        "first_half_best_size": int(first_best["nominal_size"]),
        "second_half_best_size": int(second_best["nominal_size"]),
        "seed_half_best_size_agreement": int(first_best["nominal_size"])
        == int(second_best["nominal_size"]),
        "rules": {
            rule: {
                "selected_size": int(row["nominal_size"]),
                "candidate_id": str(row["candidate_id"]),
                "current_step_regret": float(best["seed_mean"])
                - float(row["seed_mean"]),
                "exact_best_size": int(row["nominal_size"])
                == int(best["nominal_size"]),
            }
            for rule, row in selected.items()
            if row is not None
        },
    }
    shells: list[dict[str, Any]] = []
    by_size = {int(row["nominal_size"]): row for row in rows}
    for lower_size, upper_size in zip(ALLOWED_SIZES, ALLOWED_SIZES[1:]):
        lower, upper = by_size[lower_size], by_size[upper_size]
        lower_agents = set(map(int, lower["agents"]))
        upper_agents = set(map(int, upper["agents"]))
        union = lower_agents | upper_agents
        lower_vector = structural_vector(lower, family)
        upper_vector = structural_vector(upper, family)
        shells.append(
            {
                "state_id": state_id,
                "map_id": str(lower["map_id"]),
                "family": family,
                "lower_size": lower_size,
                "upper_size": upper_size,
                "nested": lower_agents <= upper_agents,
                "lost_agent_count": len(lower_agents - upper_agents),
                "added_shell_agent_count": len(upper_agents - lower_agents),
                "jaccard": len(lower_agents & upper_agents) / len(union),
                "structural_vector_delta": [
                    upper_value - lower_value
                    for lower_value, upper_value in zip(lower_vector, upper_vector)
                ],
                "seed_mean_delta": float(upper["seed_mean"])
                - float(lower["seed_mean"]),
                "no_progress_rate_delta": float(upper["no_progress_rate"])
                - float(lower["no_progress_rate"]),
                "repair_success_rate_delta": float(upper["repair_success_rate"])
                - float(lower["repair_success_rate"]),
            }
        )
    return group, shells


def _maze_state_job(
    item: tuple[str, list[dict[str, Any]], dict[str, int]]
) -> dict[str, Any]:
    state_key, rows, fixed = item
    base = [row for row in rows if str(row.get("candidate_kind")) == "base"]
    structural = [
        row for row in rows if str(row.get("candidate_kind")) == "structural"
    ]
    if not base or not structural:
        raise ValueError(f"missing Maze pool: {state_key}")
    base_best = max(base, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])))
    robust_ids = {
        str(row["candidate_id"])
        for row in structural
        if stable_dominates(row, base_best)
    }
    by_family: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in structural:
        for label in row.get("selection_families") or ():
            if not str(label).startswith("structpool-"):
                continue
            family, size = parse_structpool_family(str(label))
            expanded = {**row, "nominal_size": size}
            previous = by_family[family].get(size)
            if previous is not None and str(previous["candidate_id"]) != str(
                row["candidate_id"]
            ):
                raise ValueError(f"duplicate Maze family-size: {state_key} {family} {size}")
            by_family[family][size] = expanded
    family_rows = {
        family: list(rows_by_size.values())
        for family, rows_by_size in by_family.items()
    }
    selected = _rule_candidates(family_rows, fixed)
    return {
        "state_fingerprint": state_key,
        "map_id": str(rows[0]["map_id"]),
        "base_candidate_ids": sorted(str(row["candidate_id"]) for row in base),
        "structural_candidate_count": len(structural),
        "robust_action_ids": sorted(robust_ids),
        "has_structural_opportunity": bool(robust_ids),
        "rule_selected_candidate_ids": {
            rule: sorted(candidate_ids) for rule, candidate_ids in selected.items()
        },
        "rules": {
            rule: {
                "selected_candidate_count": len(candidate_ids),
                "selected_robust_action_count": len(candidate_ids & robust_ids),
                "preserves_opportunity": bool(candidate_ids & robust_ids),
            }
            for rule, candidate_ids in selected.items()
        },
    }


def _aggregate_size_rule(
    groups: list[dict[str, Any]], rule: str
) -> dict[str, Any]:
    rows = [row["rules"][rule] for row in groups]
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["selected_size"])] += 1
    return {
        "group_count": len(rows),
        "selected_size_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "mean_current_step_regret": statistics.fmean(
            float(row["current_step_regret"]) for row in rows
        ),
        "exact_best_size_fraction": statistics.fmean(
            bool(row["exact_best_size"]) for row in rows
        ),
    }


def _aggregate_maze_rule(
    states: list[dict[str, Any]], rule: str
) -> dict[str, Any]:
    opportunity = [row for row in states if row["has_structural_opportunity"]]
    robust_count = sum(len(row["robust_action_ids"]) for row in states)
    selected_robust = sum(
        int(row["rules"][rule]["selected_robust_action_count"]) for row in states
    )
    return {
        "state_count": len(states),
        "opportunity_state_count": len(opportunity),
        "preserved_opportunity_state_count": sum(
            bool(row["rules"][rule]["preserves_opportunity"]) for row in opportunity
        ),
        "opportunity_state_recall": _fraction(
            sum(bool(row["rules"][rule]["preserves_opportunity"]) for row in opportunity),
            len(opportunity),
        ),
        "robust_action_count": robust_count,
        "retained_robust_action_count": selected_robust,
        "robust_action_recall": _fraction(selected_robust, robust_count),
        "mean_selected_structural_candidate_count": statistics.fmean(
            int(row["rules"][rule]["selected_candidate_count"]) for row in states
        ),
    }


def _tail_rows(
    comparisons: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    maze_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    case_state = {
        str(row["case_id"]): str(row["state_fingerprint"]) for row in checkpoints
    }
    schedule_by_key = {
        (str(row["case_id"]), int(row["trial_index"]), str(row["arm"])): row
        for row in schedule
    }
    state_by_key = {str(row["state_fingerprint"]): row for row in maze_states}
    output: list[dict[str, Any]] = []
    for comparison in comparisons:
        key = (
            str(comparison["case_id"]),
            int(comparison["trial_index"]),
            str(comparison["arm"]),
        )
        item = schedule_by_key.get(key)
        state_key = case_state.get(key[0])
        state = state_by_key.get(str(state_key)) if state_key is not None else None
        if item is None or state is None:
            errors.append(f"missing_pretail_identity:{key}")
            continue
        candidate_id = str(item["candidate_id"])
        base = candidate_id in set(state["base_candidate_ids"])
        output.append(
            {
                "case_id": key[0],
                "trial_index": key[1],
                "arm": key[2],
                "state_fingerprint": state_key,
                "map_id": str(state["map_id"]),
                "candidate_id": candidate_id,
                "classification": str(comparison["classification"]),
                "reason": str(comparison["reason"]),
                "identical_action": bool(comparison["identical_action"]),
                "candidate_is_v2_base": base,
                "retained_by_rule": {
                    rule: base
                    or candidate_id
                    in set(state["rule_selected_candidate_ids"][rule])
                    for rule in RULES
                },
            }
        )
    return output, errors


def _aggregate_tail_rule(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    informative = [row for row in rows if not row["identical_action"]]
    by_class = {}
    for classification in ("beneficial", "neutral", "adverse"):
        selected = [row for row in informative if row["classification"] == classification]
        retained = sum(bool(row["retained_by_rule"][rule]) for row in selected)
        by_class[classification] = {
            "comparison_count": len(selected),
            "retained_comparison_count": retained,
            "retained_comparison_fraction": _fraction(retained, len(selected)),
        }
    return {
        "informative_comparison_count": len(informative),
        "by_classification": by_class,
        "claim_boundary": "membership_coverage_only_not_online_selection_or_loop_avoidance",
    }


def analyze_structshell(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int = 16,
) -> dict[str, Any]:
    config_path, _root, config, inputs = _load_config(config_path)
    if workers != int(config["analysis"]["workers"]):
        raise ValueError("StructShell audit must use the registered 16 workers")
    size_report = _read_json(inputs["size_ablation_report"])
    size_rows = _read_jsonl(inputs["size_ablation_rows"])
    maze_rows = _read_jsonl(inputs["maze_candidate_aggregates"])
    checkpoint_rows = _read_jsonl(inputs["maze_checkpoint_results"])
    action_report = _read_json(inputs["maze_action_replay_report"])
    coverage_report = _read_json(inputs["legacy_structural_coverage_report"])
    pretail_report = _read_json(inputs["pretail_report"])
    pretail_schedule = _read_jsonl(inputs["pretail_schedule"])

    size_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in size_rows:
        size_groups[(str(row["state_id"]), str(row["family"]))].append(dict(row))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        size_results = list(executor.map(_size_group_job, sorted(size_groups.items())))
    size_group_rows = [row for row, _shells in size_results]
    shell_rows = [shell for _row, shells in size_results for shell in shells]

    maze_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in maze_rows:
        maze_by_state[str(row["state_fingerprint"])].append(dict(row))
    fixed = {
        str(key): int(value)
        for key, value in dict(
            config["candidate_contract"]["fixed_preferred_size_comparator"]
        ).items()
    }
    maze_jobs = [
        (state_key, rows, fixed) for state_key, rows in sorted(maze_by_state.items())
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        maze_state_rows = list(executor.map(_maze_state_job, maze_jobs))

    tail_rows, identity_errors = _tail_rows(
        list(pretail_report.get("paired_comparisons") or ()),
        pretail_schedule,
        checkpoint_rows,
        maze_state_rows,
    )
    size_expected = dict(config["cohorts"]["non_tail_size_grid"])
    maze_expected = dict(config["cohorts"]["maze_difficult_states"])
    tail_expected = dict(config["cohorts"]["bounded_forced_continuation"])
    integrity = {
        "size_ablation_report_complete": size_report.get("scientific_status")
        == "completed_current_step_family_size_ablation",
        "size_state_count": len({str(row["state_id"]) for row in size_rows})
        == int(size_expected["state_count"]),
        "size_candidate_count": int(size_report.get("candidate_count", -1))
        == int(size_expected["candidate_count"]),
        "expanded_size_row_count": len(size_rows)
        == int(size_expected["expanded_family_size_row_count"]),
        "maze_state_count": len(maze_state_rows) == int(maze_expected["state_count"]),
        "maze_candidate_count": len(maze_rows) == int(maze_expected["candidate_count"]),
        "maze_base_candidate_count": sum(
            str(row.get("candidate_kind")) == "base" for row in maze_rows
        )
        == int(maze_expected["base_candidate_count"]),
        "maze_structural_candidate_count": sum(
            str(row.get("candidate_kind")) == "structural" for row in maze_rows
        )
        == int(maze_expected["structural_candidate_count"]),
        "maze_action_replay_integrity": action_report.get("integrity_passed") is True,
        "legacy_coverage_integrity": coverage_report.get("integrity_passed") is True,
        "legacy_robust_action_count": len(
            coverage_report.get("legacy_robust_action_rows") or ()
        )
        == int(maze_expected["legacy_robust_action_count"]),
        "legacy_opportunity_state_count": sum(
            bool(row["has_structural_opportunity"]) for row in maze_state_rows
        )
        == int(maze_expected["legacy_opportunity_state_count"]),
        "pretail_integrity": pretail_report.get("integrity_passed") is True,
        "pretail_case_count": int(pretail_report.get("case_count", -1))
        == int(tail_expected["case_count"]),
        "pretail_schedule_count": len(pretail_schedule)
        == int(tail_expected["schedule_entry_count"]),
        "pretail_comparison_count": len(tail_rows)
        == int(tail_expected["paired_comparison_count"]),
        "zero_identity_errors": not identity_errors,
    }

    shell_by_family_transition: dict[str, Any] = {}
    for family in sorted({row["family"] for row in shell_rows}):
        for lower, upper in zip(ALLOWED_SIZES, ALLOWED_SIZES[1:]):
            rows = [
                row
                for row in shell_rows
                if row["family"] == family
                and int(row["lower_size"]) == lower
                and int(row["upper_size"]) == upper
            ]
            shell_by_family_transition[f"{family}:{lower}_to_{upper}"] = {
                "row_count": len(rows),
                "nested_fraction": _mean(bool(row["nested"]) for row in rows),
                "mean_lost_agent_count": _mean(
                    float(row["lost_agent_count"]) for row in rows
                ),
                "mean_added_shell_agent_count": _mean(
                    float(row["added_shell_agent_count"]) for row in rows
                ),
                "mean_jaccard": _mean(float(row["jaccard"]) for row in rows),
                "mean_seed_mean_delta": _mean(
                    float(row["seed_mean_delta"]) for row in rows
                ),
                "mean_no_progress_rate_delta": _mean(
                    float(row["no_progress_rate_delta"]) for row in rows
                ),
            }

    size_rule_summary = {
        rule: _aggregate_size_rule(size_group_rows, rule)
        for rule in RULES
        if rule != "equal_four_size_grid"
    }
    seed_half_agreement = statistics.fmean(
        bool(row["seed_half_best_size_agreement"]) for row in size_group_rows
    )
    maze_rule_summary = {
        rule: _aggregate_maze_rule(maze_state_rows, rule) for rule in RULES
    }
    by_map = {
        map_id: {
            rule: _aggregate_maze_rule(
                [row for row in maze_state_rows if row["map_id"] == map_id], rule
            )
            for rule in RULES
        }
        for map_id in sorted({row["map_id"] for row in maze_state_rows})
    }
    tail_rule_summary = {
        rule: _aggregate_tail_rule(tail_rows, rule) for rule in RULES
    }
    primary = "structural_knee"
    thresholds = dict(config["readiness_gates"])
    per_map_recalls = [
        summary[primary]["opportunity_state_recall"]
        for summary in by_map.values()
        if summary[primary]["opportunity_state_count"] > 0
    ]
    beneficial_recall = tail_rule_summary[primary]["by_classification"]["beneficial"][
        "retained_comparison_fraction"
    ]
    gates = {
        "primary_maze_opportunity_state_recall": float(
            maze_rule_summary[primary]["opportunity_state_recall"] or 0.0
        )
        >= float(thresholds["minimum_primary_maze_opportunity_state_recall"]),
        "primary_maze_robust_action_recall": float(
            maze_rule_summary[primary]["robust_action_recall"] or 0.0
        )
        >= float(thresholds["minimum_primary_maze_robust_action_recall"]),
        "primary_per_map_opportunity_state_recall": bool(per_map_recalls)
        and min(map(float, per_map_recalls))
        >= float(thresholds["minimum_primary_per_map_opportunity_state_recall"]),
        "primary_pretail_beneficial_comparison_recall": float(
            beneficial_recall or 0.0
        )
        >= float(
            thresholds["minimum_primary_pretail_beneficial_comparison_recall"]
        ),
        "primary_non_tail_mean_current_step_regret": float(
            size_rule_summary[primary]["mean_current_step_regret"]
        )
        <= float(thresholds["maximum_primary_non_tail_mean_current_step_regret"]),
        "non_tail_seed_half_best_size_agreement": seed_half_agreement
        >= float(thresholds["minimum_non_tail_seed_half_best_size_agreement"]),
        "zero_identity_or_integrity_errors": all(integrity.values()),
    }
    ready = all(gates.values())
    recommendation = (
        "preregister_result_blind_structural_knee_forced_continuation"
        if ready
        else "retain_equal_four_size_grid_8_16_24_32"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_retrospective_structpool_shell_and_cutpoint_diagnostic",
        "experiment_id": EXPERIMENT_ID,
        "registration_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "workers": workers,
        "integrity": integrity,
        "integrity_errors": sorted(identity_errors),
        "integrity_passed": all(integrity.values()),
        "shell_analysis": {
            "shell_row_count": len(shell_rows),
            "by_family_transition": shell_by_family_transition,
        },
        "non_tail_size_diagnostics": {
            "state_family_group_count": len(size_group_rows),
            "seed_half_best_size_agreement": seed_half_agreement,
            "rules": size_rule_summary,
        },
        "maze_difficult_state_diagnostics": {
            "rules": maze_rule_summary,
            "by_map": by_map,
        },
        "bounded_continuation_membership_diagnostics": {
            "rules": tail_rule_summary,
            "warning": "retention is not online selection and does not prove loop avoidance",
        },
        "readiness_gates": gates,
        "natural_structural_cutpoint_design_ready": ready,
        "recommended_size_policy": recommendation,
        "next_step": (
            "independent_result_blind_bounded_forced_continuation_registration"
            if ready
            else "preregister_ranker_free_equal_four_size_budget_reducer_audit"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "shell_rows.jsonl", shell_rows)
    _write_jsonl(output / "size_group_rows.jsonl", size_group_rows)
    _write_jsonl(output / "maze_state_rule_rows.jsonl", maze_state_rows)
    _write_jsonl(output / "pretail_membership_rows.jsonl", tail_rows)
    report["artifacts"] = {
        name: sha256_file(output / name)
        for name in (
            "shell_rows.jsonl",
            "size_group_rows.jsonl",
            "maze_state_rule_rows.jsonl",
            "pretail_membership_rows.jsonl",
        )
    }
    _write_json(output / "structshell_audit_report.json", report)
    return report


__all__ = [
    "ALLOWED_SIZES",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "analyze_structshell",
    "parse_structpool_family",
    "select_fixed_preferred",
    "select_structural_knee",
    "select_support_nearest",
    "structural_vector",
]
