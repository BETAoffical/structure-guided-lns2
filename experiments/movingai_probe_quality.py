from __future__ import annotations

import collections
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator

from experiments.movingai_mechanism_probe import (
    MODEL_SEED,
    _actual_neighborhood,
    _candidate_key,
    _family,
    _group_statistic,
    _horizon_one,
    _mean,
    _read_json,
    _read_jsonl,
    _ratio,
    _write_json,
    summarize_probe_records,
)


QUALITY_SCHEMA_VERSION = 1


def _median(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return statistics.median(numbers) if numbers else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return _ratio(len(left_set & right_set), len(union)) if union else 1.0


def _mean_pairwise_jaccard(values: list[Iterable[Any]]) -> float:
    pairs = [
        _jaccard(left, right)
        for left, right in itertools.combinations(values, 2)
    ]
    return _mean(pairs) if pairs else 1.0


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return _ratio(numerator, left_scale * right_scale)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _unique_multiset_permutations(values: list[Any]) -> Iterator[tuple[Any, ...]]:
    counts = collections.Counter(values)
    ordered = sorted(counts, key=str)
    result: list[Any] = []

    def visit() -> Iterator[tuple[Any, ...]]:
        if len(result) == len(values):
            yield tuple(result)
            return
        for value in ordered:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            result.append(value)
            yield from visit()
            result.pop()
            counts[value] += 1

    yield from visit()


def _multiset_permutation_count(values: list[Any]) -> int:
    result = math.factorial(len(values))
    for count in collections.Counter(values).values():
        result //= math.factorial(count)
    return result


def _dominates(
    left: dict[str, float], right: dict[str, float], *, compute: bool, runtime: bool
) -> bool:
    left_values = [-left["solved_rate"], left["conflicts_after"]]
    right_values = [-right["solved_rate"], right["conflicts_after"]]
    if compute:
        left_values.append(left["generated"])
        right_values.append(right["generated"])
    if runtime:
        left_values.append(left["runtime"])
        right_values.append(right["runtime"])
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _aggregate_candidate(candidate: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_key": candidate["candidate_key"],
        "family": candidate["family"],
        "heuristic": candidate["heuristic"],
        "size": candidate["size"],
        "seed_agent": candidate["seed_agent"],
        "solved_rate": _mean(trial["solved"] for trial in trials),
        "conflicts_after": _mean(trial["conflicts_after"] for trial in trials),
        "generated": _mean(trial["generated"] for trial in trials),
        "runtime": _mean(trial["runtime"] for trial in trials),
    }


def _pareto_keys(
    candidates: list[dict[str, Any]], *, compute: bool = False, runtime: bool = False
) -> set[str]:
    return {
        candidate["candidate_key"]
        for candidate in candidates
        if not any(
            other is not candidate
            and _dominates(other, candidate, compute=compute, runtime=runtime)
            for other in candidates
        )
    }


def _best_effectiveness_keys(candidates: list[dict[str, Any]]) -> set[str]:
    maximum_solved = max(
        (candidate["solved_rate"] for candidate in candidates), default=0.0
    )
    feasible = [
        candidate
        for candidate in candidates
        if candidate["solved_rate"] == maximum_solved
    ]
    minimum_conflicts = min(
        (candidate["conflicts_after"] for candidate in feasible), default=0.0
    )
    return {
        candidate["candidate_key"]
        for candidate in feasible
        if candidate["conflicts_after"] == minimum_conflicts
    }


def _statewise_eta_squared(candidates: list[dict[str, Any]], before: float) -> float:
    denominator = max(1.0, before)
    values = [
        value / denominator
        for candidate in candidates
        for value in candidate["trial_conflicts"]
    ]
    if not values:
        return 0.0
    grand_mean = _mean(values)
    between = 0.0
    within = 0.0
    for candidate in candidates:
        selected = [value / denominator for value in candidate["trial_conflicts"]]
        candidate_mean = _mean(selected)
        between += len(selected) * (candidate_mean - grand_mean) ** 2
        within += sum((value - candidate_mean) ** 2 for value in selected)
    return _ratio(between, between + within)


def _bootstrap_task_mean(
    state_rows: list[dict[str, Any]], samples: int
) -> dict[str, Any]:
    by_task: dict[str, list[float]] = collections.defaultdict(list)
    for row in state_rows:
        by_task[row["task_id"]].append(float(row["action_effect_eta_squared"]))
    units = sorted(by_task)
    unit_values = {key: _mean(by_task[key]) for key in units}
    rng = random.Random(MODEL_SEED + 1709)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(units) for _ in units]
        estimates.append(_mean(unit_values[key] for key in selected))
    return {
        "samples": samples,
        "unit": "task_instance",
        "mean": _mean(unit_values.values()),
        "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
    }


def _density_alignment_statistic(
    units: list[dict[str, Any]], counts: list[int], families: list[str]
) -> float:
    differences: list[dict[str, float]] = []
    by_map: dict[str, list[int]] = collections.defaultdict(list)
    for index, unit in enumerate(units):
        by_map[str(unit["map_id"])].append(index)
    for indices in by_map.values():
        distinct = sorted({counts[index] for index in indices})
        if len(distinct) < 2:
            continue
        low = [index for index in indices if counts[index] == distinct[0]]
        high = [index for index in indices if counts[index] == distinct[-1]]
        differences.append(
            {
                family: _mean(units[index]["support"][family] for index in high)
                - _mean(units[index]["support"][family] for index in low)
                for family in families
            }
        )
    if not differences:
        return 0.0
    return _mean(
        _mean(row[family] for row in differences) ** 2 for family in families
    )


def _exact_context_signals(
    units: list[dict[str, Any]],
    families: list[str],
    *,
    maximum_exact_assignments: int = 100_000,
    monte_carlo_permutations: int = 10_000,
) -> dict[str, Any]:
    map_labels = [str(unit["map_id"]) for unit in units]
    counts = [int(unit["agent_count"]) for unit in units]
    observed_map = _group_statistic(units, map_labels, families)
    map_space = _multiset_permutation_count(map_labels)
    if map_space <= maximum_exact_assignments:
        map_method = "exact"
        map_null = [
            _group_statistic(units, list(labels), families)
            for labels in _unique_multiset_permutations(map_labels)
        ]
    else:
        map_method = "monte_carlo"
        rng = random.Random(MODEL_SEED + 4513)
        map_null = []
        for _ in range(monte_carlo_permutations):
            labels = list(map_labels)
            rng.shuffle(labels)
            map_null.append(_group_statistic(units, labels, families))

    by_map: dict[str, list[int]] = collections.defaultdict(list)
    for index, label in enumerate(map_labels):
        by_map[label].append(index)
    count_assignments = []
    for indices in by_map.values():
        original = [counts[index] for index in indices]
        count_assignments.append(
            (indices, list(_unique_multiset_permutations(original)))
        )
    density_space = math.prod(len(values) for _, values in count_assignments)
    if density_space <= maximum_exact_assignments:
        density_method = "exact"
        density_null = []
        for selections in itertools.product(
            *(values for _, values in count_assignments)
        ):
            shuffled = list(counts)
            for (indices, _), values in zip(count_assignments, selections):
                for index, value in zip(indices, values):
                    shuffled[index] = int(value)
            density_null.append(
                _density_alignment_statistic(units, shuffled, families)
            )
    else:
        density_method = "monte_carlo"
        rng = random.Random(MODEL_SEED + 6427)
        density_null = []
        for _ in range(monte_carlo_permutations):
            shuffled = list(counts)
            for indices in by_map.values():
                selected = [shuffled[index] for index in indices]
                rng.shuffle(selected)
                for index, value in zip(indices, selected):
                    shuffled[index] = value
            density_null.append(
                _density_alignment_statistic(units, shuffled, families)
            )
    observed_density = _density_alignment_statistic(units, counts, families)

    def permutation_summary(
        observed: float, values: list[float], method: str, space: int
    ) -> dict[str, Any]:
        extreme = sum(value >= observed for value in values)
        return {
            "observed": observed,
            "method": method,
            "assignment_space_size": space,
            "assignment_count": len(values),
            "percentile_strict": _ratio(sum(observed > value for value in values), len(values)),
            "upper_tail_p_value": (
                _ratio(extreme, len(values))
                if method == "exact"
                else _ratio(extreme + 1, len(values) + 1)
            ),
            "null_range": [min(values, default=0.0), max(values, default=0.0)],
        }

    return {
        "unit": "repairable_task_instance",
        "map": permutation_summary(observed_map, map_null, map_method, map_space),
        "density_directional_alignment": permutation_summary(
            observed_density, density_null, density_method, density_space
        ),
        "note": (
            "Density uses signed high-minus-low family support aligned across maps; "
            "the earlier within-map squared statistic was invariant to two-level label swaps."
        ),
    }


def _candidate_trial_stability(
    state_rows: list[dict[str, Any]], bootstrap_samples: int
) -> dict[str, Any]:
    spearman = []
    best_jaccard = []
    pareto_family_jaccard = []
    winner_confidence = []
    rng = random.Random(MODEL_SEED + 3253)
    split_methods: collections.Counter[str] = collections.Counter()
    for state in state_rows:
        candidates = state["candidates"]
        episodes = sorted(
            {
                trial["episode_id"]
                for candidate in candidates
                for trial in candidate["trials"]
            }
        )
        split_selectors = []
        split_method = None
        if len(episodes) >= 2:
            split_method = "duplicate_solver_episode"
            split_selectors = [
                lambda trial, episode=episode: trial["episode_id"] == episode
                for episode in episodes[:2]
            ]
        elif min(candidate["trial_count"] for candidate in candidates) >= 4:
            split_method = "trial_index_parity"
            split_selectors = [
                lambda trial, parity=parity: trial["trial_index"] % 2 == parity
                for parity in (0, 1)
            ]
        if split_selectors:
            split_methods[split_method] += 1
            halves = []
            for selector in split_selectors:
                halves.append(
                    [
                        _aggregate_candidate(
                            candidate,
                            [
                                trial
                                for trial in candidate["trials"]
                                if selector(trial)
                            ],
                        )
                        for candidate in candidates
                    ]
                )
            spearman.append(
                _spearman(
                    [row["conflicts_after"] for row in halves[0]],
                    [row["conflicts_after"] for row in halves[1]],
                )
            )
            best_sets = []
            pareto_families = []
            for half in halves:
                best_sets.append(_best_effectiveness_keys(half))
                keys = _pareto_keys(half)
                pareto_families.append(
                    {row["family"] for row in half if row["candidate_key"] in keys}
                )
            best_jaccard.append(_jaccard(*best_sets))
            pareto_family_jaccard.append(_jaccard(*pareto_families))

        support: collections.Counter[str] = collections.Counter()
        for _ in range(bootstrap_samples):
            aggregated = []
            for candidate in candidates:
                trials = candidate["trials"]
                sampled = [rng.choice(trials) for _ in trials]
                aggregated.append(_aggregate_candidate(candidate, sampled))
            best_keys = _best_effectiveness_keys(aggregated)
            families = {
                row["family"]
                for row in aggregated
                if row["candidate_key"] in best_keys
            }
            for family in families:
                support[family] += 1.0 / len(families)
        winner_confidence.append(_ratio(max(support.values(), default=0.0), bootstrap_samples))
    return {
        "split_methods": dict(sorted(split_methods.items())),
        "split_state_count": len(spearman),
        "mean_rank_spearman": _mean(spearman),
        "median_rank_spearman": _median(spearman),
        "mean_best_candidate_jaccard": _mean(best_jaccard),
        "mean_pareto_family_jaccard": _mean(pareto_family_jaccard),
        "winner_bootstrap_samples": bootstrap_samples,
        "mean_modal_family_confidence": _mean(winner_confidence),
        "states_below_80pct_winner_confidence_rate": _mean(
            value < 0.8 for value in winner_confidence
        ),
    }


def _neighborhood_quality(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_values: dict[str, list[float]] = collections.defaultdict(list)
    group_stability: dict[str, list[float]] = collections.defaultdict(list)
    jaccards = []
    conflict_variances = []
    size_matches = []
    seed_membership = []
    for state in state_rows:
        for candidate in state["candidates"]:
            neighborhoods = [trial["neighborhood"] for trial in candidate["trials"]]
            similarity = _mean_pairwise_jaccard(neighborhoods)
            stable = float(len({tuple(value) for value in neighborhoods}) == 1)
            family = candidate["family"]
            group_values[family].append(similarity)
            group_stability[family].append(stable)
            jaccards.append(similarity)
            conflict_variances.append(
                statistics.pvariance(candidate["trial_conflicts"])
                if len(candidate["trial_conflicts"]) > 1
                else 0.0
            )
            size_matches.extend(
                len(value) == candidate["size"] for value in neighborhoods
            )
            seed_membership.extend(
                candidate["seed_agent"] in value for value in neighborhoods
            )
    return {
        "mean_pairwise_jaccard": _mean(jaccards),
        "exact_neighborhood_stability_rate": _mean(
            value for values in group_stability.values() for value in values
        ),
        "requested_size_match_rate": _mean(size_matches),
        "seed_membership_rate": _mean(seed_membership),
        "jaccard_vs_conflict_variance_pearson": _pearson(jaccards, conflict_variances),
        "by_family": {
            family: {
                "candidate_count": len(values),
                "mean_pairwise_jaccard": _mean(values),
                "exact_stability_rate": _mean(group_stability[family]),
            }
            for family, values in sorted(group_values.items())
        },
    }


def _pareto_sensitivity(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compute_jaccards = []
    runtime_jaccards = []
    support: dict[str, collections.Counter[str]] = {
        "effectiveness": collections.Counter(),
        "compute_aware": collections.Counter(),
        "runtime_sensitive": collections.Counter(),
    }
    for state in state_rows:
        aggregated = [
            _aggregate_candidate(candidate, candidate["trials"])
            for candidate in state["candidates"]
        ]
        effectiveness = _pareto_keys(aggregated)
        compute = _pareto_keys(aggregated, compute=True)
        runtime = _pareto_keys(aggregated, compute=True, runtime=True)
        compute_jaccards.append(_jaccard(effectiveness, compute))
        runtime_jaccards.append(_jaccard(effectiveness, runtime))
        for name, keys in (
            ("effectiveness", effectiveness),
            ("compute_aware", compute),
            ("runtime_sensitive", runtime),
        ):
            for row in aggregated:
                if row["candidate_key"] in keys:
                    support[name][row["family"]] += 1
    return {
        "mean_effectiveness_compute_jaccard": _mean(compute_jaccards),
        "mean_effectiveness_runtime_jaccard": _mean(runtime_jaccards),
        "family_support_counts": {
            name: dict(sorted(values.items())) for name, values in support.items()
        },
    }


def _build_quality_rows(
    states: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_index = {str(row["state_id"]): row for row in states}
    canonical = {
        state_id: str(row.get("state_fingerprint", state_id))
        for state_id, row in state_index.items()
    }
    sources: dict[str, dict[str, Any]] = {}
    source_ids: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in state_index.items():
        fingerprint = canonical[state_id]
        sources.setdefault(fingerprint, row)
        source_ids[fingerprint].append(state_id)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    auc_identity_errors = []
    for outcome in outcomes:
        state_id = str(outcome["state_id"])
        if state_id not in state_index:
            continue
        horizon = _horizon_one(outcome)
        trajectory = outcome.get("conflict_trajectory", [])
        before = float(trajectory[0]) if trajectory else float(
            outcome["steps"][0]["conflicts"]
        )
        expected_auc = (before + float(horizon["conflicts_after"])) / 2.0
        auc_identity_errors.append(abs(float(horizon["conflict_auc"]) - expected_auc))
        grouped[(canonical[state_id], _candidate_key(outcome["candidate_action"]))].append(
            {
                "episode_id": str(outcome["episode_id"]),
                "trial_index": int(outcome["trial_index"]),
                "trial_seed": int(outcome.get("trial_seed", -1)),
                "solved": bool(horizon["solved"]),
                "conflicts_before": before,
                "conflicts_after": float(horizon["conflicts_after"]),
                "conflict_auc": float(horizon["conflict_auc"]),
                "generated": float(horizon["low_level_delta"]["generated"]),
                "runtime": float(horizon.get("branch_runtime", 0.0)),
                "neighborhood": _actual_neighborhood(outcome),
                "action": outcome["candidate_action"],
            }
        )

    candidates_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for (fingerprint, candidate_key), trials in sorted(grouped.items()):
        action = trials[0]["action"]
        candidates_by_state[fingerprint].append(
            {
                "candidate_key": candidate_key,
                "family": _family(action),
                "heuristic": str(action["heuristic"]),
                "size": int(action["neighborhood_size"]),
                "seed_agent": int(action["seed_agent"]),
                "trial_count": len(trials),
                "trial_conflicts": [row["conflicts_after"] for row in trials],
                "trials": trials,
            }
        )

    rows = []
    for fingerprint, source in sorted(sources.items()):
        candidates = candidates_by_state.get(fingerprint, [])
        if not candidates:
            continue
        context = source["state"].get("context", {})
        before = float(candidates[0]["trials"][0]["conflicts_before"])
        aggregated = [
            _aggregate_candidate(candidate, candidate["trials"])
            for candidate in candidates
        ]
        pareto = _pareto_keys(aggregated)
        rows.append(
            {
                "state_fingerprint": fingerprint,
                "source_state_ids": sorted(source_ids[fingerprint]),
                "map_id": str(context.get("map_id", "unknown")),
                "task_id": str(context.get("task_id", "unknown")),
                "agent_count": int(context.get("agent_count", 0)),
                "scenario_type": str(context.get("scenario_type", "unknown")),
                "decision_index": int(source.get("decision_index", -1)),
                "conflicts_before": before,
                "candidate_count": len(candidates),
                "trial_count_min": min(candidate["trial_count"] for candidate in candidates),
                "trial_count_max": max(candidate["trial_count"] for candidate in candidates),
                "action_effect_eta_squared": _statewise_eta_squared(candidates, before),
                "pareto_families": sorted(
                    row["family"] for row in aggregated if row["candidate_key"] in pareto
                ),
                "candidates": candidates,
            }
        )
    return rows, {
        "horizon1_auc_affine_identity_max_error": max(auc_identity_errors, default=0.0),
        "horizon1_auc_is_independent_objective": False,
    }


def audit_probe_records(
    qualification: list[dict[str, Any]],
    states: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    state_rows, label_quality = _build_quality_rows(states, outcomes)
    qualified_repair_tasks = {
        str(row["task_id"])
        for row in qualification
        if str(row.get("status")) == "ok" and bool(row.get("repairable"))
    }
    task_context = {}
    for row in state_rows:
        task_context.setdefault(
            row["task_id"],
            {
                "map_id": row["map_id"],
                "agent_count": row["agent_count"],
                "scenario_type": row["scenario_type"],
            },
        )
    labeled_tasks = set(task_context)
    labeled_maps = {str(value["map_id"]) for value in task_context.values()}
    counts_by_map: dict[str, set[int]] = collections.defaultdict(set)
    scenarios_by_map: dict[str, set[str]] = collections.defaultdict(set)
    for value in task_context.values():
        counts_by_map[value["map_id"]].add(value["agent_count"])
        scenarios_by_map[value["map_id"]].add(value["scenario_type"])
    density_pair_maps = sum(len(values) >= 2 for values in counts_by_map.values())

    all_families = sorted(
        {
            candidate["family"]
            for row in state_rows
            for candidate in row["candidates"]
        }
    )
    task_states: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in state_rows:
        task_states[row["task_id"]].append(row)
    units = []
    for task_id, selected in sorted(task_states.items()):
        units.append(
            {
                "task_id": task_id,
                "map_id": selected[0]["map_id"],
                "agent_count": selected[0]["agent_count"],
                "support": {
                    family: _mean(
                        family in row["pareto_families"] for row in selected
                    )
                    for family in all_families
                },
            }
        )
    context = _exact_context_signals(
        units,
        all_families,
        maximum_exact_assignments=int(settings["maximum_exact_assignments"]),
        monte_carlo_permutations=int(settings["monte_carlo_permutations"]),
    )
    etas = [float(row["action_effect_eta_squared"]) for row in state_rows]
    action_effect = {
        "definition": "statewise eta-squared on normalized Horizon-1 remaining conflicts",
        "mean": _mean(etas),
        "median": _median(etas),
        "minimum": min(etas, default=0.0),
        "maximum": max(etas, default=0.0),
        "states_at_or_above_0_5_rate": _mean(value >= 0.5 for value in etas),
        "task_bootstrap": _bootstrap_task_mean(
            state_rows, int(settings["bootstrap_samples"])
        ),
    }
    trial_stability = _candidate_trial_stability(
        state_rows, int(settings["winner_bootstrap_samples"])
    )
    neighborhood = _neighborhood_quality(state_rows)
    pareto = _pareto_sensitivity(state_rows)
    pooled_trials = min(
        (row["trial_count_min"] for row in state_rows), default=0
    )
    raw_state_count = len(states)
    unique_states = len(state_rows)
    scenario_minimum = min(
        (len(values) for values in scenarios_by_map.values()), default=0
    )

    gates = {
        "independent_state_coverage": {
            "actual": unique_states,
            "requirement": int(settings["minimum_unique_states"]),
        },
        "repairable_task_coverage": {
            "actual": len(labeled_tasks),
            "requirement": int(settings["minimum_repairable_tasks"]),
        },
        "repair_label_map_coverage": {
            "actual": len(labeled_maps),
            "requirement": int(settings["minimum_repair_label_maps"]),
        },
        "within_map_density_coverage": {
            "actual": density_pair_maps,
            "requirement": int(settings["minimum_density_pair_maps"]),
        },
        "task_scenario_replication": {
            "actual": scenario_minimum,
            "requirement": int(settings["minimum_scenarios_per_map"]),
        },
        "layout_family_replication": {
            "actual": 1,
            "requirement": int(settings["minimum_layout_replicates_per_family"]),
            "note": "Each benchmark map is currently its only layout-family representative.",
        },
        "action_trial_coverage": {
            "actual": pooled_trials,
            "requirement": int(settings["minimum_trials_per_candidate"]),
        },
        "state_normalized_action_signal": {
            "actual": action_effect["mean"],
            "requirement": float(settings["minimum_mean_statewise_action_effect"]),
        },
        "trial_split_rank_stability": {
            "actual": trial_stability["mean_rank_spearman"],
            "requirement": float(settings["minimum_trial_split_spearman"]),
        },
        "realized_neighborhood_stability": {
            "actual": neighborhood["mean_pairwise_jaccard"],
            "requirement": float(settings["minimum_neighborhood_jaccard"]),
        },
        "oracle_context_heterogeneity": {
            "actual": min(
                context["map"]["upper_tail_p_value"],
                context["density_directional_alignment"]["upper_tail_p_value"],
            ),
            "requirement": float(settings["maximum_exact_context_p_value"]),
            "comparison": "at_most",
        },
    }
    for gate in gates.values():
        gate["passed"] = (
            gate["actual"] <= gate["requirement"]
            if gate.get("comparison") == "at_most"
            else gate["actual"] >= gate["requirement"]
        )

    coverage_names = {
        "independent_state_coverage",
        "repairable_task_coverage",
        "repair_label_map_coverage",
        "within_map_density_coverage",
        "task_scenario_replication",
        "layout_family_replication",
    }
    coverage_passed = all(gates[name]["passed"] for name in coverage_names)
    trials_passed = gates["action_trial_coverage"]["passed"]
    if not coverage_passed and not trials_passed:
        decision = "expand_independent_context_coverage_and_action_trials"
    elif not coverage_passed:
        decision = "expand_independent_maps_and_scenarios_before_modeling"
    elif not trials_passed or not gates["trial_split_rank_stability"]["passed"]:
        decision = "increase_action_trials_before_modeling"
    elif not gates["realized_neighborhood_stability"]["passed"]:
        decision = "rank_realized_neighborhood_candidates_before_generation_policy"
    elif gates["oracle_context_heterogeneity"]["passed"]:
        decision = "confirm_contextual_ranking_on_independent_maps"
    else:
        decision = "retain_dynamic_policy_and_narrow_static_transfer_claim"

    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "threshold_status": (
            "post-probe diagnostic adequacy thresholds; not preregistered success criteria"
        ),
        "decision": decision,
        "passed": all(gate["passed"] for gate in gates.values()),
        "coverage": {
            "raw_state_rows": raw_state_count,
            "unique_state_fingerprints": unique_states,
            "solver_state_uniqueness_ratio": _ratio(unique_states, raw_state_count),
            "outcome_rows": len(outcomes),
            "qualified_repairable_task_instances": len(qualified_repair_tasks),
            "labeled_task_instances": len(labeled_tasks),
            "repair_label_maps": len(labeled_maps),
            "maps_with_two_repair_densities": density_pair_maps,
            "scenario_types": sorted(
                {row["scenario_type"] for row in state_rows}
            ),
            "minimum_scenarios_per_map": scenario_minimum,
            "source_policy": "official_adaptive_only",
            "decision_indices": dict(
                sorted(collections.Counter(row["decision_index"] for row in state_rows).items())
            ),
            "minimum_pooled_trials_per_candidate": pooled_trials,
        },
        "label_quality": label_quality,
        "action_effect": action_effect,
        "trial_stability": trial_stability,
        "realized_neighborhood": neighborhood,
        "pareto_sensitivity": pareto,
        "exact_context": context,
        "gates": gates,
        "limitations": [
            "Solver-seed state rows are deduplicated by fingerprint, not assumed independent.",
            "Task-scenario replication cannot replace independent layout-family replicas.",
            "Each layout family has one map, so this dataset cannot validate map transfer.",
            "Lower-density tasks are prefixes of higher-density tasks and are not independent OD samples.",
            "Horizon-1 AUC is an affine transform of remaining conflicts within a state.",
            "Exact context p-values are diagnostic only while action labels fail the trial-stability gate.",
            "All counterfactual states come from official Adaptive trajectories.",
            "Horizon 1 diagnoses immediate repair only, not time-to-feasible policy quality.",
        ],
        "recommended_next_collection": {
            "solver_seeds_for_state_acquisition": 1,
            "independent_action_trials_per_candidate": 8,
            "minimum_scenario_indices_per_map": 3,
            "minimum_maps_per_layout_family": 2,
            "minimum_unique_states": 24,
            "minimum_repairable_task_instances": 12,
            "minimum_maps_with_two_repairable_densities": 6,
            "qualification_first": True,
            "density_adjustment": (
                "Raise warehouse/den densities only after qualification; do not collect labels "
                "from zero-conflict tasks."
            ),
            "label_definition": (
                "Use Horizon-1 remaining conflicts as effectiveness; report generated nodes "
                "as a separate compute-aware sensitivity objective."
            ),
            "policy_scope": (
                "Do not train contextual ranking or RL until coverage and trial-stability gates pass."
            ),
        },
    }


def render_quality_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    action = report["action_effect"]
    stability = report["trial_stability"]
    neighborhood = report["realized_neighborhood"]
    context = report["exact_context"]
    completeness = report.get("collection_completeness", {})
    lines = [
        "# MovingAI probe quality audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Effective sample size",
        "",
        f"- Raw states: {coverage['raw_state_rows']}",
        f"- Unique state fingerprints: {coverage['unique_state_fingerprints']}",
        f"- Qualified repairable task instances: "
        f"{coverage['qualified_repairable_task_instances']}",
        f"- Task instances with labels: {coverage['labeled_task_instances']}",
        f"- Maps contributing repair labels: {coverage['repair_label_maps']}",
        f"- Maps with two repairable densities: {coverage['maps_with_two_repair_densities']}",
        f"- Pooled trials per candidate: {coverage['minimum_pooled_trials_per_candidate']}",
        "",
        "## Label and action quality",
        "",
        f"- Mean statewise action eta-squared: {action['mean']:.3f}",
        f"- Candidate split rank Spearman: {stability['mean_rank_spearman']:.3f} "
        f"({', '.join(stability['split_methods'])})",
        f"- Best-candidate split Jaccard: {stability['mean_best_candidate_jaccard']:.3f}",
        f"- Mean realized-neighborhood Jaccard: {neighborhood['mean_pairwise_jaccard']:.3f}",
        f"- Exact neighborhood stability: {neighborhood['exact_neighborhood_stability_rate']:.1%}",
        f"- Effectiveness/compute-aware Pareto Jaccard: "
        f"{report['pareto_sensitivity']['mean_effectiveness_compute_jaccard']:.3f}",
        f"- Horizon-1 AUC affine-identity error: "
        f"{report['label_quality']['horizon1_auc_affine_identity_max_error']:.3g}",
        "",
        "## Context identifiability",
        "",
        f"- Map permutation ({context['map']['method']}): "
        f"{context['map']['assignment_count']} evaluated, "
        f"p={context['map']['upper_tail_p_value']:.3f}",
        f"- Directional density permutation "
        f"({context['density_directional_alignment']['method']}): "
        f"{context['density_directional_alignment']['assignment_count']} evaluated, "
        f"p={context['density_directional_alignment']['upper_tail_p_value']:.3f}",
        f"- Scenario types represented: {len(coverage['scenario_types'])}; minimum "
        f"labeled scenarios per map: {coverage['minimum_scenarios_per_map']}",
        "- Only one map per layout family is present; this is not a transfer test.",
        "",
        "## Gates",
        "",
    ]
    if completeness:
        position = lines.index("## Label and action quality") - 1
        lines[position:position] = [
            f"- Recovered source episodes: "
            f"{completeness['recovered_source_episodes']}/"
            f"{completeness['eligible_source_episodes']}",
            f"- Partial analysis: {completeness['partial_allowed']}",
        ]
    if not report["pareto_sensitivity"].get("runtime_sensitive_valid", True):
        position = lines.index("## Context identifiability") - 1
        lines[position:position] = [
            "- Runtime-sensitive Pareto: INVALID (collector CPU contention); "
            "generated-node sensitivity remains valid.",
        ]
    for name, gate in report["gates"].items():
        lines.append(
            f"- {name}: **{'PASS' if gate['passed'] else 'FAIL'}** "
            f"(actual={gate['actual']:.3g}, requirement={gate['requirement']:.3g})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The {coverage['outcome_rows']:,} outcome rows are useful for finding data-design faults, but they are not "
            "enough to support a contextual-transfer claim. The next collection should target "
            "the failed coverage and stability gates rather than simply adding outcome rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_probe_quality(
    collection: str | Path,
    probe_config: str | Path,
    quality_config: str | Path,
    output: str | Path,
    *,
    require_complete: bool = True,
    runtime_metrics_valid: bool = True,
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    run_config = _read_json(collection_root / "run_config.json")
    probe_settings = _read_json(Path(probe_config).resolve())
    quality_settings = _read_json(Path(quality_config).resolve())
    qualification = _read_jsonl(collection_root / "qualification_manifest.jsonl")
    baseline = _read_jsonl(collection_root / "collection_manifest.jsonl")
    manifest = _read_jsonl(collection_root / "counterfactual_manifest.jsonl")
    states = []
    outcomes = []
    errors = 0
    for row in manifest:
        errors += int(row.get("error_count", 0))
        if str(row.get("status")) not in {"ok", "resumed"}:
            continue
        states.extend(_read_jsonl(collection_root / str(row["states_file"])))
        outcomes.extend(_read_jsonl(collection_root / str(row["outcomes_file"])))
    original = summarize_probe_records(
        qualification,
        baseline,
        states,
        outcomes,
        probe_settings,
        run_config["configuration"],
        require_complete=require_complete,
        counterfactual_error_count=errors,
    )
    report = audit_probe_records(qualification, states, outcomes, quality_settings)
    report["pareto_sensitivity"]["runtime_sensitive_valid"] = bool(
        runtime_metrics_valid
    )
    if not runtime_metrics_valid:
        report["pareto_sensitivity"]["runtime_invalid_reason"] = (
            "overlapping WSL collectors after host command timeouts caused CPU contention"
        )
    report["source_probe"] = {
        "integrity_passed": original["integrity"]["passed"],
        "decision": original["decision"],
        "action_effect_ratio_pooled": original["mechanism"]["action_effect_ratio"],
        "configuration_fingerprint": run_config["configuration_fingerprint"],
        "dataset_fingerprint": run_config["dataset_fingerprint"],
        "require_complete": require_complete,
        "volume_checks": original["integrity"]["volume_checks"],
    }
    source_manifest = _read_jsonl(
        collection_root / "counterfactual_source_manifest.jsonl"
    )
    recovered_ids = {str(row["episode_id"]) for row in manifest}
    eligible_ids = (
        {
            str(row["episode_id"])
            for row in source_manifest
            if bool(row.get("eligible"))
        }
        if source_manifest
        else set(recovered_ids)
    )
    report["collection_completeness"] = {
        "partial_allowed": not require_complete,
        "eligible_source_episodes": len(eligible_ids),
        "recovered_source_episodes": len(recovered_ids),
        "missing_source_episode_ids": sorted(eligible_ids - recovered_ids),
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "movingai_probe_quality.json", report)
    (output_root / "movingai_probe_quality.md").write_text(
        render_quality_markdown(report), encoding="utf-8"
    )
    return report
