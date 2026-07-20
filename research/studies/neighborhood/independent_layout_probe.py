from __future__ import annotations

import collections
import itertools
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from research.studies.neighborhood.movingai_mechanism_probe import (
    MODEL_SEED,
    _candidate_key,
    _group_statistic,
    _mean,
    _read_json,
    _read_jsonl,
    _write_json,
)
from research.studies.neighborhood.movingai_probe_quality import (
    _aggregate_candidate,
    _build_quality_rows,
    _candidate_trial_stability,
    _neighborhood_quality,
    _pareto_keys,
    _pareto_sensitivity,
    _quantile,
    _unique_multiset_permutations,
)


PROBE_SCHEMA_VERSION = 1


def _variant_axes(task_variant: str) -> tuple[str, int]:
    pieces = str(task_variant).rsplit("_", 1)
    if len(pieces) != 2 or pieces[0] not in {"balanced", "bottleneck"}:
        raise ValueError(f"invalid independent-probe task variant: {task_variant}")
    density = int(pieces[1])
    if density not in {80, 100}:
        raise ValueError(f"invalid independent-probe agent count: {density}")
    return pieces[0], density


def _dataset_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_root.glob("*/manifest.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _task_signature(path: Path, limit: int | None = None) -> tuple[Any, ...]:
    payload = _read_json(path)
    starts = list(payload.get("starts", []))
    goals = list(payload.get("goals", []))
    if limit is not None:
        starts = starts[:limit]
        goals = goals[:limit]
    return tuple((tuple(start), tuple(goal)) for start, goal in zip(starts, goals))


def _seed_sets(rows: Iterable[dict[str, Any]]) -> tuple[set[int], set[int]]:
    return (
        {int(row["map_seed"]) for row in rows},
        {int(row["task_seed"]) for row in rows},
    )


def validate_probe_dataset(
    dataset: str | Path,
    settings: dict[str, Any],
    reference_datasets: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(dataset).resolve()
    rows = _dataset_rows(root)
    errors: list[str] = []
    expected_split = str(settings["expected_split"])
    expected_layouts = {
        str(name): int(count)
        for name, count in settings["expected_layouts"].items()
    }
    expected_variants = set(map(str, settings["expected_task_variants"]))

    if not rows:
        errors.append("dataset contains no split manifests")
    if any(str(row.get("split")) != expected_split for row in rows):
        errors.append("dataset contains a non-probe split")
    if len({str(row.get("task_id")) for row in rows}) != len(rows):
        errors.append("task IDs are not unique")
    if len({int(row.get("task_seed", -1)) for row in rows}) != len(rows):
        errors.append("task seeds are not unique")

    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row.get("map_id"))].append(row)
        try:
            od_mode, density = _variant_axes(str(row.get("task_variant")))
        except (TypeError, ValueError) as error:
            errors.append(str(error))
            continue
        if int(row.get("agent_count", -1)) != density:
            errors.append(f"{row.get('task_id')}: variant/agent-count mismatch")
        expected_scenario = (
            "balanced_bidirectional" if od_mode == "balanced" else "bottleneck_pressure"
        )
        if str(row.get("scenario_type")) != expected_scenario:
            errors.append(f"{row.get('task_id')}: variant/scenario mismatch")

    actual_layouts: collections.Counter[str] = collections.Counter()
    independent_task_pairs = 0
    for map_id, selected in sorted(by_map.items()):
        layout_values = {str(row.get("layout_mode")) for row in selected}
        if len(layout_values) != 1:
            errors.append(f"{map_id}: inconsistent layout mode")
            continue
        actual_layouts[next(iter(layout_values))] += 1
        variants = {str(row.get("task_variant")) for row in selected}
        if variants != expected_variants:
            errors.append(f"{map_id}: incomplete four-task factorial")
        by_variant = {str(row["task_variant"]): row for row in selected}
        for od_mode in ("balanced", "bottleneck"):
            low = by_variant.get(f"{od_mode}_80")
            high = by_variant.get(f"{od_mode}_100")
            if low is None or high is None:
                continue
            low_path = root / str(low["split"]) / str(low["task_file"])
            high_path = root / str(high["split"]) / str(high["task_file"])
            if low_path.is_file() and high_path.is_file():
                if _task_signature(low_path) == _task_signature(high_path, 80):
                    errors.append(f"{map_id}: {od_mode} 80-agent task is a 100-agent prefix")
                else:
                    independent_task_pairs += 1

    if dict(sorted(actual_layouts.items())) != dict(sorted(expected_layouts.items())):
        errors.append("layout replication does not match the preregistered 2/2/2 design")

    map_seeds, task_seeds = _seed_sets(rows) if rows else (set(), set())
    reference_reports = []
    for reference in reference_datasets:
        reference_root = Path(reference).resolve()
        if not reference_root.is_dir():
            reference_reports.append(
                {"path": str(reference_root), "available": False, "overlap": False}
            )
            continue
        reference_rows = _dataset_rows(reference_root)
        reference_map_seeds, reference_task_seeds = _seed_sets(reference_rows)
        shared_maps = sorted(map_seeds & reference_map_seeds)
        shared_tasks = sorted(task_seeds & reference_task_seeds)
        overlap = bool(shared_maps or shared_tasks)
        if overlap:
            errors.append(f"seed overlap with reference dataset {reference_root}")
        reference_reports.append(
            {
                "path": str(reference_root),
                "available": True,
                "overlap": overlap,
                "shared_map_seeds": shared_maps,
                "shared_task_seeds": shared_tasks,
            }
        )

    return {
        "passed": not errors,
        "errors": errors,
        "row_count": len(rows),
        "map_count": len(by_map),
        "layout_counts": dict(sorted(actual_layouts.items())),
        "task_variant_counts": dict(
            sorted(collections.Counter(str(row.get("task_variant")) for row in rows).items())
        ),
        "independent_density_task_pairs": independent_task_pairs,
        "reference_seed_checks": reference_reports,
    }


def _eligible_qualification(row: dict[str, Any]) -> bool:
    return (
        str(row.get("status")) == "ok"
        and bool(row.get("repairable"))
        and 1 <= int(row.get("initial_conflicts", 0)) <= 200
        and int(row.get("agent_count", 0)) <= 100
    )


def qualification_summary(
    manifest: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    settings: dict[str, Any],
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    expected_seed = int(settings["expected_solver_seed"])
    manifest_index = {str(row["task_id"]): row for row in manifest}
    errors = []
    if len(qualification) != len(manifest):
        errors.append("qualification row count does not match the 24 tasks")
    if {str(row.get("task_id")) for row in qualification} != set(manifest_index):
        errors.append("qualification task coverage does not match the dataset")
    if any(str(row.get("status")) != "ok" for row in qualification):
        errors.append("qualification contains a runtime error")
    if any(int(row.get("solver_seed", -1)) != expected_seed for row in qualification):
        errors.append("qualification used a non-preregistered solver seed")

    repairable = [row for row in qualification if _eligible_qualification(row)]
    by_layout: collections.Counter[str] = collections.Counter(
        str(row.get("layout_mode")) for row in repairable
    )
    context_by_map: dict[str, dict[str, set[Any]]] = collections.defaultdict(
        lambda: {"od": set(), "density": set()}
    )
    for row in repairable:
        source = manifest_index.get(str(row.get("task_id")))
        if source is None:
            continue
        od_mode, density = _variant_axes(str(source["task_variant"]))
        context_by_map[str(source["map_id"])]["od"].add(od_mode)
        context_by_map[str(source["map_id"])]["density"].add(density)

    minimum_layout = int(settings["minimum_repairable_tasks_per_layout"])
    context_ready_maps = [
        map_id
        for map_id, values in context_by_map.items()
        if len(values["od"]) >= int(settings["minimum_od_modes_per_map"])
        and len(values["density"]) >= int(settings["minimum_densities_per_map"])
    ]
    gates = {
        "dataset_integrity": bool(dataset_validation["passed"]),
        "qualification_integrity": not errors,
        "repairable_task_count": len(repairable)
        >= int(settings["minimum_repairable_tasks"]),
        "repairable_layout_coverage": all(
            by_layout.get(layout, 0) >= minimum_layout
            for layout in settings["expected_layouts"]
        ),
        "within_map_od_density_coverage": len(context_ready_maps)
        == sum(int(value) for value in settings["expected_layouts"].values()),
    }
    return {
        "passed": all(gates.values()),
        "errors": errors,
        "gates": gates,
        "valid_count": sum(str(row.get("status")) == "ok" for row in qualification),
        "repairable_count": len(repairable),
        "repairable_by_layout": dict(sorted(by_layout.items())),
        "context_ready_maps": sorted(context_ready_maps),
        "initial_conflict_range": [
            min((int(row["initial_conflicts"]) for row in repairable), default=0),
            max((int(row["initial_conflicts"]) for row in repairable), default=0),
        ],
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (total - rank)))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _permutation_summary(observed: float, values: list[float]) -> dict[str, Any]:
    return {
        "observed": observed,
        "method": "exact",
        "assignment_count": len(values),
        "upper_tail_p_value": (
            sum(value >= observed - 1e-15 for value in values) / len(values)
            if values
            else 1.0
        ),
        "null_range": [min(values, default=0.0), max(values, default=0.0)],
    }


def paired_difference_test(
    differences: list[dict[str, float]], families: list[str]
) -> dict[str, Any]:
    def statistic(signs: tuple[int, ...]) -> float:
        if not differences or not families:
            return 0.0
        return _mean(
            _mean(
                signs[index] * differences[index][family]
                for index in range(len(differences))
            )
            ** 2
            for family in families
        )

    observed_signs = tuple(1 for _ in differences)
    observed = statistic(observed_signs)
    null = [
        statistic(signs)
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    result = _permutation_summary(observed, null)
    result["pair_count"] = len(differences)
    result["unit"] = "complete within-map pair"
    return result


def _layout_test(units: list[dict[str, Any]], families: list[str]) -> dict[str, Any]:
    by_map: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for unit in units:
        by_map[str(unit["map_id"])].append(unit)
    map_units = []
    labels = []
    for map_id, selected in sorted(by_map.items()):
        map_units.append(
            {
                "map_id": map_id,
                "support": {
                    family: _mean(row["support"][family] for row in selected)
                    for family in families
                },
            }
        )
        labels.append(str(selected[0]["layout_mode"]))
    observed = _group_statistic(map_units, labels, families)
    null = [
        _group_statistic(map_units, list(permutation), families)
        for permutation in _unique_multiset_permutations(labels)
    ]
    result = _permutation_summary(observed, null)
    result["map_count"] = len(map_units)
    result["unit"] = "independent map after averaging its labeled tasks"
    return result


def _paired_context_tests(
    units: list[dict[str, Any]], families: list[str]
) -> dict[str, Any]:
    indexed = {
        (str(row["map_id"]), str(row["od_mode"]), int(row["density"])): row
        for row in units
    }
    od_differences = []
    density_differences = []
    maps = sorted({str(row["map_id"]) for row in units})
    for map_id in maps:
        for density in (80, 100):
            balanced = indexed.get((map_id, "balanced", density))
            bottleneck = indexed.get((map_id, "bottleneck", density))
            if balanced is not None and bottleneck is not None:
                od_differences.append(
                    {
                        family: bottleneck["support"][family]
                        - balanced["support"][family]
                        for family in families
                    }
                )
        for od_mode in ("balanced", "bottleneck"):
            low = indexed.get((map_id, od_mode, 80))
            high = indexed.get((map_id, od_mode, 100))
            if low is not None and high is not None:
                density_differences.append(
                    {
                        family: high["support"][family] - low["support"][family]
                        for family in families
                    }
                )
    return {
        "layout": _layout_test(units, families),
        "od": paired_difference_test(od_differences, families),
        "density": paired_difference_test(density_differences, families),
    }


def _map_bootstrap(
    rows: list[dict[str, Any]], field: str, samples: int
) -> dict[str, Any]:
    by_map: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_map[str(row["map_id"])].append(float(row[field]))
    maps = sorted(by_map)
    means = {map_id: _mean(by_map[map_id]) for map_id in maps}
    rng = random.Random(MODEL_SEED + 19081)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(maps) for _ in maps]
        estimates.append(_mean(means[map_id] for map_id in selected))
    return {
        "samples": samples,
        "unit": "map",
        "mean": _mean(means.values()),
        "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
    }


def _fixed_family_dominance(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique: collections.Counter[str] = collections.Counter()
    pareto_support: collections.Counter[str] = collections.Counter()
    for state in state_rows:
        aggregated = [
            _aggregate_candidate(candidate, candidate["trials"])
            for candidate in state["candidates"]
        ]
        keys = _pareto_keys(aggregated)
        families = {
            row["family"] for row in aggregated if row["candidate_key"] in keys
        }
        for family in families:
            pareto_support[family] += 1
        if len(families) == 1:
            unique[next(iter(families))] += 1
    state_count = len(state_rows)
    shares = {
        family: count / state_count if state_count else 0.0
        for family, count in sorted(unique.items())
    }
    return {
        "unique_pareto_counts": dict(sorted(unique.items())),
        "unique_pareto_shares": shares,
        "pareto_support_counts": dict(sorted(pareto_support.items())),
        "maximum_unique_pareto_share": max(shares.values(), default=0.0),
    }


def _enrich_state_rows(
    state_rows: list[dict[str, Any]], states: list[dict[str, Any]]
) -> None:
    contexts = {
        str(row.get("state_fingerprint", row["state_id"])): row["state"].get(
            "context", {}
        )
        for row in states
    }
    for row in state_rows:
        context = contexts.get(str(row["state_fingerprint"]), {})
        row["split"] = str(context.get("split", "unknown"))
        row["layout_mode"] = str(context.get("layout_mode", "unknown"))
        row["task_variant"] = str(context.get("task_variant", "unknown"))
        row["od_mode"], row["density"] = _variant_axes(row["task_variant"])


def analyze_probe_records(
    manifest: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    states: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    collection_config: dict[str, Any],
    settings: dict[str, Any],
    dataset_validation: dict[str, Any],
    *,
    counterfactual_errors: int = 0,
) -> dict[str, Any]:
    qualification_report = qualification_summary(
        manifest, qualification, settings, dataset_validation
    )
    state_rows, label_quality = _build_quality_rows(states, outcomes)
    _enrich_state_rows(state_rows, states)
    manifest_tasks = {str(row["task_id"]) for row in manifest}
    state_ids = {str(row["state_id"]) for row in states}
    expected_trials = int(collection_config["counterfactual"]["trials"])
    grouped_trials: collections.Counter[tuple[str, str]] = collections.Counter()
    trial_indices: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    orphan_outcomes = 0
    invalid_actions = 0
    for outcome in outcomes:
        state_id = str(outcome.get("state_id"))
        if state_id not in state_ids:
            orphan_outcomes += 1
            continue
        key = (state_id, _candidate_key(outcome["candidate_action"]))
        grouped_trials[key] += 1
        trial_indices[key].add(int(outcome.get("trial_index", -1)))
        invalid_actions += not bool(outcome.get("action_valid"))
    trial_mismatches = sum(
        count != expected_trials
        or trial_indices[key] != set(range(expected_trials))
        for key, count in grouped_trials.items()
    )
    observed_candidates: collections.Counter[str] = collections.Counter(
        state_id for state_id, _ in grouped_trials
    )
    candidate_count_mismatches = sum(
        observed_candidates.get(str(row["state_id"]), 0)
        != int(row.get("candidate_count", 0))
        for row in states
    )
    expected_families = {
        f"{heuristic}:{int(size)}"
        for heuristic in collection_config["counterfactual"]["heuristics"]
        for size in collection_config["counterfactual"]["neighborhood_sizes"]
    }
    candidate_family_mismatches = 0
    for row in state_rows:
        by_seed: dict[int, set[str]] = collections.defaultdict(set)
        for candidate in row["candidates"]:
            by_seed[int(candidate["seed_agent"])].add(str(candidate["family"]))
        candidate_family_mismatches += sum(
            families != expected_families for families in by_seed.values()
        )
    source_state_counts: collections.Counter[str] = collections.Counter(
        str(row.get("episode_id")) for row in states
    )
    eligible_tasks = {
        str(row["task_id"]) for row in qualification if _eligible_qualification(row)
    }
    baseline_ok = (
        len(baseline) == len(manifest)
        and all(str(row.get("status")) in {"ok", "resumed"} for row in baseline)
        and {str(row.get("task_id")) for row in baseline} == manifest_tasks
    )
    labeled_tasks = {str(row["task_id"]) for row in state_rows}
    integrity_errors = {
        "baseline_incomplete": not baseline_ok,
        "counterfactual_errors": int(counterfactual_errors),
        "orphan_outcomes": orphan_outcomes,
        "invalid_actions": invalid_actions,
        "trial_mismatches": trial_mismatches,
        "candidate_count_mismatches": candidate_count_mismatches,
        "candidate_family_mismatches": candidate_family_mismatches,
        "duplicate_state_ids": len(states)
        - len({str(row["state_id"]) for row in states}),
        "duplicate_state_fingerprints": len(states)
        - len({str(row.get("state_fingerprint")) for row in states}),
        "duplicate_or_multiple_source_states": sum(
            count != 1 for count in source_state_counts.values()
        ),
        "duplicate_task_states": sum(
            count != 1
            for count in collections.Counter(
                str(row["task_id"]) for row in state_rows
            ).values()
        ),
        "unexpected_label_tasks": len(labeled_tasks - eligible_tasks),
        "missing_label_tasks": len(eligible_tasks - labeled_tasks),
        "non_probe_labels": sum(
            row["split"] != str(settings["expected_split"]) for row in state_rows
        ),
    }
    integrity_passed = (
        qualification_report["passed"]
        and all(not value for value in integrity_errors.values())
        and len(outcomes) <= 6912
    )

    trial_stability = _candidate_trial_stability(
        state_rows, int(settings["winner_bootstrap_samples"])
    )
    neighborhood = _neighborhood_quality(state_rows)
    pareto_sensitivity = _pareto_sensitivity(state_rows)
    dominance = _fixed_family_dominance(state_rows)
    all_families = sorted(
        {
            candidate["family"]
            for row in state_rows
            for candidate in row["candidates"]
        }
    )
    units = [
        {
            "task_id": row["task_id"],
            "map_id": row["map_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row["task_variant"],
            "od_mode": row["od_mode"],
            "density": row["density"],
            "support": {
                family: float(family in set(row["pareto_families"]))
                for family in all_families
            },
        }
        for row in state_rows
    ]
    context_tests = _paired_context_tests(units, all_families)
    adjusted = holm_adjust(
        {
            name: float(result["upper_tail_p_value"])
            for name, result in context_tests.items()
        }
    )
    for name, value in adjusted.items():
        context_tests[name]["holm_adjusted_p_value"] = value

    label_context: dict[str, dict[str, set[Any]]] = collections.defaultdict(
        lambda: {"od": set(), "density": set(), "layout": set()}
    )
    for unit in units:
        values = label_context[str(unit["map_id"])]
        values["od"].add(unit["od_mode"])
        values["density"].add(unit["density"])
        values["layout"].add(unit["layout_mode"])
    labeled_layouts: collections.Counter[str] = collections.Counter(
        next(iter(values["layout"]))
        for values in label_context.values()
        if values["layout"]
    )
    label_coverage = {
        "map_count": len(label_context),
        "layout_counts": dict(sorted(labeled_layouts.items())),
        "all_maps_have_two_od_modes": all(
            len(values["od"]) >= int(settings["minimum_od_modes_per_map"])
            for values in label_context.values()
        ),
        "all_maps_have_two_densities": all(
            len(values["density"]) >= int(settings["minimum_densities_per_map"])
            for values in label_context.values()
        ),
        "od_pair_count": int(context_tests["od"]["pair_count"]),
        "density_pair_count": int(context_tests["density"]["pair_count"]),
    }
    label_coverage_passed = (
        label_coverage["map_count"]
        == sum(int(value) for value in settings["expected_layouts"].values())
        and all(
            labeled_layouts.get(layout, 0)
            >= int(settings["minimum_labeled_maps_per_layout"])
            for layout in settings["expected_layouts"]
        )
        and label_coverage["all_maps_have_two_od_modes"]
        and label_coverage["all_maps_have_two_densities"]
        and label_coverage["od_pair_count"] >= int(settings["minimum_paired_units"])
        and label_coverage["density_pair_count"]
        >= int(settings["minimum_paired_units"])
    )

    etas = [float(row["action_effect_eta_squared"]) for row in state_rows]
    minimum_trials = min(
        (int(row["trial_count_min"]) for row in state_rows), default=0
    )
    context_passed = min(adjusted.values(), default=1.0) <= float(
        settings["maximum_holm_p_value"]
    )
    gates = {
        "integrity": integrity_passed,
        "qualification": qualification_report["passed"],
        "label_context_coverage": label_coverage_passed,
        "action_trial_coverage": minimum_trials
        >= int(settings["minimum_trials_per_candidate"]),
        "action_eta_squared": _mean(etas)
        >= float(settings["minimum_mean_action_eta_squared"]),
        "trial_split_spearman": trial_stability["mean_rank_spearman"]
        >= float(settings["minimum_trial_split_spearman"]),
        "pareto_family_jaccard": trial_stability["mean_pareto_family_jaccard"]
        >= float(settings["minimum_pareto_family_jaccard"]),
        "no_fixed_action_dominance": dominance["maximum_unique_pareto_share"]
        <= float(settings["maximum_fixed_unique_pareto_share"]),
        "holm_corrected_context_heterogeneity": context_passed,
    }
    stability_names = {
        "action_eta_squared",
        "trial_split_spearman",
        "pareto_family_jaccard",
        "no_fixed_action_dominance",
    }
    if not gates["integrity"] or not gates["qualification"]:
        decision = "stop_invalid_or_incomplete_collection"
    elif not gates["label_context_coverage"]:
        decision = "stop_and_adjust_task_conflict_pressure"
    elif not all(gates[name] for name in stability_names):
        decision = "stop_and_redefine_action_space"
    elif not gates["holm_corrected_context_heterogeneity"]:
        decision = "narrow_to_dynamic_or_realized_policy"
    else:
        decision = "expand_to_independent_train_validation"
    realized_stable = neighborhood["mean_pairwise_jaccard"] >= float(
        settings["minimum_realized_neighborhood_jaccard"]
    )

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "decision": decision,
        "passed": all(gates.values()),
        "recommended_policy_surface": (
            "expected_action_distribution"
            if realized_stable
            else "realized_neighborhood_ranking"
        ),
        "pre_registration": {
            "map_count": 6,
            "task_count": 24,
            "maximum_states": 24,
            "maximum_seed_agents": 4,
            "heuristics": ["target", "collision", "random"],
            "neighborhood_sizes": [4, 8, 16],
            "trials": expected_trials,
            "horizon": 1,
            "maximum_outcomes": 6912,
        },
        "dataset_validation": dataset_validation,
        "qualification": qualification_report,
        "integrity": {
            "passed": integrity_passed,
            "errors": integrity_errors,
            "baseline_rows": len(baseline),
            "state_rows": len(states),
            "unique_state_rows": len(state_rows),
            "outcome_rows": len(outcomes),
            "minimum_trials_per_candidate": minimum_trials,
        },
        "label_coverage": label_coverage,
        "label_quality": label_quality,
        "action_effect": {
            "mean_eta_squared": _mean(etas),
            "median_eta_squared": statistics.median(etas) if etas else 0.0,
            "map_bootstrap": _map_bootstrap(
                state_rows,
                "action_effect_eta_squared",
                int(settings["map_bootstrap_samples"]),
            ),
        },
        "trial_stability": trial_stability,
        "fixed_action_dominance": dominance,
        "realized_neighborhood": {
            **neighborhood,
            "routing_threshold": float(
                settings["minimum_realized_neighborhood_jaccard"]
            ),
            "routing_only": True,
        },
        "pareto_sensitivity": pareto_sensitivity,
        "context_tests": context_tests,
        "gates": gates,
        "limitations": [
            "This is a mechanism and heterogeneity probe, not a learned-policy result.",
            "Only immediate Horizon-1 repair effectiveness is a scientific label.",
            "Generated nodes are compute-aware sensitivity; runtime is excluded.",
            "All source states come from official Adaptive trajectories.",
            "Low realized-neighborhood Jaccard routes the next method to "
            "candidate ranking; it is not a veto.",
        ],
    }


def render_qualification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# InitLNS independent-layout qualification",
        "",
        f"Gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"- Valid runs: {report['valid_count']}/24",
        f"- Eligible repairable tasks: {report['repairable_count']}/24",
        f"- Repairable by layout: `{report['repairable_by_layout']}`",
        f"- Maps with both OD modes and densities: {len(report['context_ready_maps'])}/6",
        f"- Eligible initial-conflict range: {report['initial_conflict_range']}",
        "",
        "Counterfactual collection may start only when every gate above passes.",
    ]
    return "\n".join(lines) + "\n"


def render_probe_markdown(report: dict[str, Any]) -> str:
    action = report["action_effect"]
    stability = report["trial_stability"]
    neighborhood = report["realized_neighborhood"]
    lines = [
        "# InitLNS independent-layout mechanism confirmation",
        "",
        f"Decision: `{report['decision']}`",
        f"Recommended policy surface: `{report['recommended_policy_surface']}`",
        "",
        "## Coverage",
        "",
        f"- Qualification: {report['qualification']['repairable_count']}/24 eligible repairable",
        f"- States: {report['integrity']['unique_state_rows']}",
        f"- Outcomes: {report['integrity']['outcome_rows']:,}/6,912 maximum",
        f"- OD pairs: {report['label_coverage']['od_pair_count']}",
        f"- Density pairs: {report['label_coverage']['density_pair_count']}",
        "",
        "## Stability",
        "",
        f"- Mean action eta-squared: {action['mean_eta_squared']:.3f}",
        f"- Trial-split Spearman: {stability['mean_rank_spearman']:.3f}",
        f"- Pareto-family Jaccard: {stability['mean_pareto_family_jaccard']:.3f}",
        "- Maximum fixed-family unique-Pareto share: "
        f"{report['fixed_action_dominance']['maximum_unique_pareto_share']:.1%}",
        "- Realized-neighborhood Jaccard: "
        f"{neighborhood['mean_pairwise_jaccard']:.3f} (routing only)",
        "",
        "## Context",
        "",
    ]
    for name, result in report["context_tests"].items():
        lines.append(
            f"- {name}: raw p={result['upper_tail_p_value']:.4f}, "
            f"Holm p={result['holm_adjusted_p_value']:.4f}"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in report["gates"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This report tests whether actions are repeatable and whether oracle "
            "preferences vary across independent layouts, OD modes, or densities. "
            "It does not train a supervised model or RL policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_counterfactual(
    collection_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    states: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    errors = 0
    for row in _read_jsonl(collection_root / "counterfactual_manifest.jsonl"):
        errors += int(row.get("error_count", 0))
        if str(row.get("status")) not in {"ok", "resumed"}:
            continue
        states.extend(_read_jsonl(collection_root / str(row["states_file"])))
        outcomes.extend(_read_jsonl(collection_root / str(row["outcomes_file"])))
    return states, outcomes, errors


def run_independent_probe_analysis(
    dataset: str | Path,
    collection: str | Path,
    quality_config: str | Path,
    output: str | Path,
    *,
    qualification_only: bool = False,
    reference_datasets: Iterable[str | Path] = (),
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    settings = _read_json(Path(quality_config).resolve())
    references = list(reference_datasets) or list(settings.get("reference_datasets", []))
    validation = validate_probe_dataset(dataset_root, settings, references)
    manifest = _dataset_rows(dataset_root)
    qualification = _read_jsonl(collection_root / "qualification_manifest.jsonl")
    qualification_report = qualification_summary(
        manifest, qualification, settings, validation
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if qualification_only:
        _write_json(output_root / "qualification_gate.json", qualification_report)
        (output_root / "qualification_gate.md").write_text(
            render_qualification_markdown(qualification_report), encoding="utf-8"
        )
        return qualification_report

    run_config = _read_json(collection_root / "run_config.json")
    baseline = _read_jsonl(collection_root / "collection_manifest.jsonl")
    states, outcomes, errors = _load_counterfactual(collection_root)
    report = analyze_probe_records(
        manifest,
        qualification,
        baseline,
        states,
        outcomes,
        run_config["configuration"],
        settings,
        validation,
        counterfactual_errors=errors,
    )
    _write_json(output_root / "independent_layout_probe.json", report)
    (output_root / "independent_layout_probe.md").write_text(
        render_probe_markdown(report), encoding="utf-8"
    )
    return report
