from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


QUALITY_SCHEMA_VERSION = 1
COMPLETE_STATUSES = {"ok", "resumed"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing collection file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"collection path escapes its root: {relative}") from error
    return path


def _number_summary(values: Iterable[float | int]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
    }


def _horizon_outcome(row: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    for value in row.get("horizon_outcomes", []):
        if int(value["horizon"]) == horizon:
            return value
    return None


def _candidate_key(row: dict[str, Any]) -> str:
    action = row["candidate_action"]
    return ":".join(
        (
            str(action["seed_agent"]),
            str(action["heuristic"]),
            str(action["neighborhood_size"]),
        )
    )


def _family_key(row: dict[str, Any]) -> str:
    action = row["candidate_action"]
    return f"{action['heuristic']}:{int(action['neighborhood_size'])}"


def _objective_values(value: dict[str, Any]) -> tuple[float, ...]:
    low_level = value.get("low_level_delta", {})
    return (
        -float(bool(value["solved"])),
        float(value["conflicts_after"]),
        float(value["conflict_auc"]),
        float(low_level.get("generated", 0)),
        float(value["branch_runtime"]),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = _objective_values(left)
    right_values = _objective_values(right)
    return all(
        first <= second for first, second in zip(left_values, right_values)
    ) and any(first < second for first, second in zip(left_values, right_values))


def pareto_indices(values: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, value in enumerate(values)
        if not any(
            other_index != index and _dominates(other, value)
            for other_index, other in enumerate(values)
        )
    ]


def _outcome_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(value["solved"]),
        int(value["conflicts_after"]),
        float(value["conflict_auc"]),
    )


def _stage_labels(states: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    by_episode: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for state in states:
        by_episode[str(state["episode_id"])].append(state)
    for values in by_episode.values():
        values.sort(key=lambda item: int(item["decision_index"]))
        if len(values) == 1:
            labels[str(values[0]["state_id"])] = "only"
            continue
        for index, value in enumerate(values):
            if index == 0:
                label = "early"
            elif index == len(values) - 1:
                label = "late"
            else:
                label = "middle"
            labels[str(value["state_id"])] = label
    return labels


def _state_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    informative = sum(bool(row["informative"]) for row in records)
    ties = sum(bool(row["complete_tie"]) for row in records)
    family_coverage = sum(bool(row["full_family_coverage"]) for row in records)
    seed_coverage = sum(bool(row["full_seed_coverage"]) for row in records)
    return {
        "episode_count": len({str(row["episode_id"]) for row in records}),
        "state_count": count,
        "seed_agent_selection_count": sum(
            int(row["selected_seed_count"]) for row in records
        ),
        "expected_seed_agent_selection_count": sum(
            int(row["expected_seed_count"]) for row in records
        ),
        "full_seed_coverage_count": seed_coverage,
        "full_seed_coverage_rate": seed_coverage / count if count else 0.0,
        "candidate_count": sum(int(row["candidate_count"]) for row in records),
        "informative_count": informative,
        "informative_rate": informative / count if count else 0.0,
        "complete_tie_count": ties,
        "complete_tie_rate": ties / count if count else 0.0,
        "full_family_coverage_count": family_coverage,
        "full_family_coverage_rate": family_coverage / count if count else 0.0,
        "valid_candidate_count": sum(
            int(row["valid_candidate_count"]) for row in records
        ),
        "distinct_outcomes": _number_summary(
            int(row["distinct_outcome_count"]) for row in records
        ),
        "pareto_count": _number_summary(int(row["pareto_count"]) for row in records),
        "pareto_fraction": _number_summary(
            float(row["pareto_fraction"]) for row in records
        ),
    }


def _group_state_records(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {name: _state_summary(values) for name, values in sorted(grouped.items())}


def _baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["policy"]))].append(row)
    result: dict[str, Any] = {}
    for (split, policy), values in sorted(grouped.items()):
        summaries = [dict(row.get("summary", {})) for row in values]
        key = f"{split}/{policy}"
        result[key] = {
            "episode_count": len(values),
            "error_count": sum(row.get("status") == "error" for row in values),
            "success_count": sum(bool(value.get("success")) for value in summaries),
            "success_rate": (
                sum(bool(value.get("success")) for value in summaries) / len(values)
                if values
                else 0.0
            ),
            "time_to_feasible": _number_summary(
                value["time_to_feasible"]
                for value in summaries
                if value.get("time_to_feasible") is not None
            ),
            "conflict_auc": _number_summary(
                value["conflict_auc"]
                for value in summaries
                if value.get("conflict_auc") is not None
            ),
        }
    return result


def _outcome_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(outcomes)
    return {
        "outcome_count": count,
        "solved_count": sum(bool(row["solved"]) for row in outcomes),
        "solved_rate": (
            sum(bool(row["solved"]) for row in outcomes) / count if count else 0.0
        ),
        "conflict_reduction": _number_summary(
            int(row["conflict_reduction"]) for row in outcomes
        ),
        "conflicts_after": _number_summary(
            int(row["conflicts_after"]) for row in outcomes
        ),
        "conflict_auc": _number_summary(
            float(row["conflict_auc"]) for row in outcomes
        ),
        "cost_improvement": _number_summary(
            int(row["cost_improvement"]) for row in outcomes
        ),
        "branch_runtime": _number_summary(
            float(row["branch_runtime"]) for row in outcomes
        ),
        "time_to_feasible": _number_summary(
            float(row["time_to_feasible"])
            for row in outcomes
            if row.get("time_to_feasible") is not None
        ),
        "expanded": _number_summary(
            int(row.get("low_level_delta", {}).get("expanded", 0))
            for row in outcomes
        ),
        "generated": _number_summary(
            int(row.get("low_level_delta", {}).get("generated", 0))
            for row in outcomes
        ),
    }


def _group_outcome_records(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record.get(key, "unknown"))].append(record["outcome"])
    return {
        name: _outcome_summary(values) for name, values in sorted(grouped.items())
    }


def _family_summary(
    records: list[dict[str, Any]], unique_wins: collections.Counter[str]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record["family"])].append(record["outcome"])
    result: dict[str, Any] = {}
    for family, outcomes in sorted(grouped.items()):
        result[family] = _outcome_summary(outcomes)
        result[family]["unique_pareto_family_wins"] = int(unique_wins[family])
    return result


def _stage_action_preferences(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record["repair_stage"])].append(record)
    result: dict[str, dict[str, Any]] = {}
    for stage, values in sorted(grouped.items()):
        pareto_occurrences: collections.Counter[str] = collections.Counter()
        unique_wins: collections.Counter[str] = collections.Counter()
        for value in values:
            pareto_occurrences.update(value["pareto_families"])
            if value["unique_pareto_family"] is not None:
                unique_wins[str(value["unique_pareto_family"])] += 1
        result[stage] = {
            "state_count": len(values),
            "pareto_state_count_by_family": dict(sorted(pareto_occurrences.items())),
            "pareto_state_rate_by_family": {
                family: count / len(values)
                for family, count in sorted(pareto_occurrences.items())
            },
            "unique_pareto_wins_by_family": dict(sorted(unique_wins.items())),
        }
    return result


def _gate(
    name: str, actual: Any, required: Any, passed: bool, comparison: str
) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "required": required,
        "comparison": comparison,
        "passed": bool(passed),
    }


def analyze_collection(
    collection: str | Path,
    *,
    expected_qualification_runs: int = 72,
    expected_baseline_episodes: int = 288,
    expected_source_episodes: int = 72,
    minimum_states: int = 72,
    minimum_outcomes: int = 1296,
    minimum_informative_rate: float = 0.6,
    maximum_family_dominance: float = 0.8,
) -> dict[str, Any]:
    root = Path(collection).resolve()
    run_config = _read_json(root / "run_config.json")
    qualification = _read_jsonl(root / "qualification_manifest.jsonl")
    baselines = _read_jsonl(root / "collection_manifest.jsonl")
    counterfactual = _read_jsonl(root / "counterfactual_manifest.jsonl")
    configuration = dict(run_config["configuration"])
    cf_config = dict(configuration["counterfactual"])
    allowed_splits = {str(value) for value in cf_config["eligible_splits"]}
    horizons = sorted(int(value) for value in cf_config["horizons"])
    expected_families = {
        f"{heuristic}:{int(size)}"
        for heuristic in cf_config["heuristics"]
        for size in cf_config["neighborhood_sizes"]
    }
    run_fingerprint = str(run_config["run_fingerprint"])

    states: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fingerprint_mismatches = 0
    counterfactual_error_count = 0
    cross_split_manifests = 0
    for manifest in counterfactual:
        counterfactual_error_count += int(manifest.get("error_count", 0))
        if str(manifest.get("split", "unknown")) not in allowed_splits:
            cross_split_manifests += 1
        episode_states = _read_jsonl(
            _resolve(root, str(manifest["states_file"]))
        )
        episode_outcomes = _read_jsonl(
            _resolve(root, str(manifest["outcomes_file"]))
        )
        episode_errors = _read_jsonl(
            _resolve(root, str(manifest["errors_file"]))
        )
        for row in episode_states + episode_outcomes:
            if str(row.get("run_fingerprint")) != run_fingerprint:
                fingerprint_mismatches += 1
        states.extend(episode_states)
        outcomes.extend(episode_outcomes)
        errors.extend(episode_errors)

    replay_mismatches = sum(
        "fingerprint mismatch" in str(row.get("error", "")).lower()
        for row in errors
    )
    invalid_actions = sum(not bool(row.get("action_valid")) for row in outcomes)
    stage_labels = _stage_labels(states)
    state_index = {str(row["state_id"]): row for row in states}
    outcomes_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in outcomes:
        outcomes_by_state[str(row["state_id"])].append(row)

    cross_split_states = 0
    family_coverage_failures = 0
    seed_coverage_failures = 0
    state_records: dict[int, list[dict[str, Any]]] = {
        horizon: [] for horizon in horizons
    }
    outcome_records: dict[int, list[dict[str, Any]]] = {
        horizon: [] for horizon in horizons
    }
    pareto_keys: dict[tuple[str, int], set[str]] = {}
    unique_family_wins: dict[int, collections.Counter[str]] = {
        horizon: collections.Counter() for horizon in horizons
    }

    for state_id, state_row in state_index.items():
        state = dict(state_row["state"])
        context = dict(state.get("context", {}))
        split = str(context.get("split", "unknown"))
        if split not in allowed_splits:
            cross_split_states += 1
        rows = outcomes_by_state.get(state_id, [])
        actual_families = {_family_key(row) for row in rows}
        full_family_coverage = expected_families.issubset(actual_families)
        if not full_family_coverage:
            family_coverage_failures += 1
        selected_seed_count = len(
            {
                int(row["candidate_action"]["seed_agent"])
                for row in rows
                if row.get("candidate_action", {}).get("seed_agent") is not None
            }
        )
        available_conflict_agents = sum(
            int(agent.get("conflict_degree", 0)) > 0
            for agent in state.get("agents", [])
        )
        expected_seed_count = min(
            int(cf_config["max_seed_agents"]), available_conflict_agents
        )
        full_seed_coverage = selected_seed_count == expected_seed_count
        if not full_seed_coverage:
            seed_coverage_failures += 1
        for horizon in horizons:
            candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row in rows:
                value = _horizon_outcome(row, horizon)
                if (
                    bool(row.get("action_valid"))
                    and value is not None
                    and bool(value.get("available"))
                    and value.get("conflict_auc") is not None
                ):
                    candidates.append((row, value))
                    action = dict(row["candidate_action"])
                    outcome_records[horizon].append(
                        {
                            "state_id": state_id,
                            "episode_id": str(state_row["episode_id"]),
                            "split": split,
                            "layout_mode": str(
                                context.get("layout_mode", "unknown")
                            ),
                            "scenario_type": str(
                                context.get("scenario_type", "unknown")
                            ),
                            "task_variant": str(
                                context.get("task_variant", "unknown")
                            ),
                            "agent_count": int(context.get("agent_count", 0)),
                            "repair_stage": stage_labels.get(state_id, "unknown"),
                            "heuristic": str(action["heuristic"]),
                            "neighborhood_size": int(action["neighborhood_size"]),
                            "seed_agent": int(action["seed_agent"]),
                            "family": _family_key(row),
                            "outcome": value,
                        }
                    )
            values = [value for _, value in candidates]
            indices = pareto_indices(values) if values else []
            keys = {_candidate_key(candidates[index][0]) for index in indices}
            families = {_family_key(candidates[index][0]) for index in indices}
            pareto_keys[(state_id, horizon)] = keys
            unique_family = next(iter(families)) if len(families) == 1 else None
            if unique_family is not None:
                unique_family_wins[horizon][unique_family] += 1
            signatures = {_outcome_signature(value) for value in values}
            record = {
                "state_id": state_id,
                "episode_id": str(state_row["episode_id"]),
                "horizon": horizon,
                "split": split,
                "layout_mode": str(context.get("layout_mode", "unknown")),
                "scenario_type": str(context.get("scenario_type", "unknown")),
                "task_variant": str(context.get("task_variant", "unknown")),
                "agent_count": int(context.get("agent_count", 0)),
                "repair_stage": stage_labels.get(state_id, "unknown"),
                "candidate_count": len(rows),
                "valid_candidate_count": len(values),
                "selected_seed_count": selected_seed_count,
                "expected_seed_count": expected_seed_count,
                "full_seed_coverage": full_seed_coverage,
                "full_family_coverage": full_family_coverage,
                "distinct_outcome_count": len(signatures),
                "informative": len(signatures) > 1,
                "complete_tie": len(signatures) == 1 and bool(values),
                "pareto_count": len(indices),
                "pareto_fraction": len(indices) / len(values) if values else 0.0,
                "pareto_families": sorted(families),
                "unique_pareto_family": unique_family,
            }
            state_records[horizon].append(record)

    overlaps = []
    if 1 in horizons and 4 in horizons:
        for state_id in state_index:
            first = pareto_keys.get((state_id, 1), set())
            fourth = pareto_keys.get((state_id, 4), set())
            union = first | fourth
            if union:
                overlaps.append(len(first & fourth) / len(union))

    horizon_summaries = {
        str(horizon): _state_summary(state_records[horizon])
        for horizon in horizons
    }
    fourth_records = state_records.get(4, [])
    fourth_count = len(fourth_records)
    informative_rate = (
        sum(bool(row["informative"]) for row in fourth_records) / fourth_count
        if fourth_count
        else 0.0
    )
    fourth_wins = unique_family_wins.get(4, collections.Counter())
    dominant_family, dominant_count = (
        fourth_wins.most_common(1)[0] if fourth_wins else (None, 0)
    )
    family_dominance = dominant_count / fourth_count if fourth_count else 0.0

    qualification_errors = sum(row.get("status") == "error" for row in qualification)
    qualification_valid = sum(row.get("status") == "ok" for row in qualification)
    baseline_errors = sum(row.get("status") == "error" for row in baselines)
    complete_baselines = sum(
        row.get("status") in COMPLETE_STATUSES for row in baselines
    )
    complete_counterfactual = sum(
        bool(row.get("complete")) and row.get("status") in COMPLETE_STATUSES
        for row in counterfactual
    )
    counterfactual_errors = max(counterfactual_error_count, len(errors))
    gates = [
        _gate(
            "qualification_runs",
            len(qualification),
            expected_qualification_runs,
            len(qualification) == expected_qualification_runs,
            "==",
        ),
        _gate(
            "qualification_valid",
            qualification_valid,
            expected_qualification_runs,
            qualification_valid == expected_qualification_runs
            and qualification_errors == 0,
            "==",
        ),
        _gate(
            "baseline_episodes",
            len(baselines),
            expected_baseline_episodes,
            len(baselines) == expected_baseline_episodes,
            "==",
        ),
        _gate(
            "complete_baseline_episodes",
            complete_baselines,
            expected_baseline_episodes,
            complete_baselines == expected_baseline_episodes,
            "==",
        ),
        _gate("baseline_errors", baseline_errors, 0, baseline_errors == 0, "=="),
        _gate(
            "counterfactual_source_episodes",
            complete_counterfactual,
            expected_source_episodes,
            complete_counterfactual == expected_source_episodes,
            "==",
        ),
        _gate(
            "state_count",
            len(states),
            minimum_states,
            len(states) >= minimum_states,
            ">=",
        ),
        _gate(
            "outcome_count",
            len(outcomes),
            minimum_outcomes,
            len(outcomes) >= minimum_outcomes,
            ">=",
        ),
        _gate(
            "counterfactual_errors",
            counterfactual_errors,
            0,
            counterfactual_errors == 0,
            "==",
        ),
        _gate("replay_mismatches", replay_mismatches, 0, replay_mismatches == 0, "=="),
        _gate(
            "run_fingerprint_mismatches",
            fingerprint_mismatches,
            0,
            fingerprint_mismatches == 0,
            "==",
        ),
        _gate("invalid_actions", invalid_actions, 0, invalid_actions == 0, "=="),
        _gate(
            "test_ood_labels",
            cross_split_manifests + cross_split_states,
            0,
            cross_split_manifests == 0 and cross_split_states == 0,
            "==",
        ),
        _gate(
            "family_coverage_failures",
            family_coverage_failures,
            0,
            family_coverage_failures == 0,
            "==",
        ),
        _gate(
            "seed_coverage_failures",
            seed_coverage_failures,
            0,
            seed_coverage_failures == 0,
            "==",
        ),
        _gate(
            "horizon_4_informative_rate",
            informative_rate,
            minimum_informative_rate,
            informative_rate >= minimum_informative_rate,
            ">=",
        ),
        _gate(
            "fixed_family_dominance",
            family_dominance,
            maximum_family_dominance,
            family_dominance <= maximum_family_dominance,
            "<=",
        ),
    ]

    report = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "collection_root": str(root),
        "run_fingerprint": run_fingerprint,
        "splits": sorted(allowed_splits),
        "configuration": {
            "solver_seeds": configuration["solver_seeds"],
            "policies": configuration["policies"],
            "counterfactual": cf_config,
        },
        "counts": {
            "qualification_runs": len(qualification),
            "qualification_valid": qualification_valid,
            "baseline_episodes": len(baselines),
            "complete_baseline_episodes": complete_baselines,
            "counterfactual_source_episodes": len(counterfactual),
            "complete_counterfactual_episodes": complete_counterfactual,
            "states": len(states),
            "outcomes": len(outcomes),
            "errors": len(errors),
        },
        "integrity": {
            "qualification_errors": qualification_errors,
            "baseline_errors": baseline_errors,
            "counterfactual_errors": counterfactual_error_count,
            "counterfactual_error_rows": len(errors),
            "replay_mismatches": replay_mismatches,
            "run_fingerprint_mismatches": fingerprint_mismatches,
            "invalid_actions": invalid_actions,
            "test_ood_manifests": cross_split_manifests,
            "test_ood_states": cross_split_states,
            "family_coverage_failures": family_coverage_failures,
            "seed_coverage_failures": seed_coverage_failures,
        },
        "baselines": _baseline_summary(baselines),
        "horizons": horizon_summaries,
        "horizon_pareto_overlap": _number_summary(overlaps),
        "action_families": {
            str(horizon): _family_summary(
                outcome_records[horizon], unique_family_wins[horizon]
            )
            for horizon in horizons
        },
        "performance_by_horizon": {
            str(horizon): {
                key: _group_outcome_records(outcome_records[horizon], key)
                for key in (
                    "split",
                    "layout_mode",
                    "scenario_type",
                    "task_variant",
                    "agent_count",
                    "repair_stage",
                    "heuristic",
                    "neighborhood_size",
                )
            }
            for horizon in horizons
        },
        "coverage": _state_summary(fourth_records),
        "contexts_horizon_4": {
            key: _group_state_records(fourth_records, key)
            for key in (
                "split",
                "layout_mode",
                "scenario_type",
                "task_variant",
                "agent_count",
                "repair_stage",
            )
        },
        "fixed_family_dominance": {
            "family": dominant_family,
            "unique_win_count": dominant_count,
            "state_count": fourth_count,
            "rate": family_dominance,
        },
        "stage_action_preferences_horizon_4": _stage_action_preferences(
            fourth_records
        ),
        "acceptance": {
            "passed": all(bool(gate["passed"]) for gate in gates),
            "gates": gates,
        },
    }
    return report


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    coverage = report["coverage"]
    acceptance = report["acceptance"]
    lines = [
        "# Repair Calibration Quality Report",
        "",
        f"Overall acceptance: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
        "",
        "## Coverage",
        "",
        f"- Qualification runs: {counts['qualification_runs']}",
        f"- Baseline episodes: {counts['baseline_episodes']}",
        f"- Counterfactual source episodes: {counts['counterfactual_source_episodes']}",
        f"- States: {counts['states']}",
        f"- Seed-agent selections: {coverage['seed_agent_selection_count']}",
        "- Full seed-agent coverage: "
        f"{coverage['full_seed_coverage_rate']:.4f}",
        f"- Outcomes: {counts['outcomes']}",
        "- Full action-family coverage: "
        f"{coverage['full_family_coverage_rate']:.4f}",
        "",
        "## Acceptance Gates",
        "",
        "| Gate | Actual | Requirement | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for gate in acceptance["gates"]:
        lines.append(
            "| "
            + str(gate["name"])
            + " | "
            + _format_number(gate["actual"])
            + " | "
            + str(gate["comparison"])
            + " "
            + _format_number(gate["required"])
            + " | "
            + ("PASS" if gate["passed"] else "FAIL")
            + " |"
        )
    lines.extend(
        [
            "",
            "## Horizon Quality",
            "",
            "| Horizon | States | Informative rate | Tie rate | "
            "Median Pareto fraction |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon, value in report["horizons"].items():
        lines.append(
            f"| {horizon} | {value['state_count']} | "
            f"{value['informative_rate']:.4f} | "
            f"{value['complete_tie_rate']:.4f} | "
            f"{_format_number(value['pareto_fraction']['median'])} |"
        )
    lines.extend(
        [
            "",
            "## Horizon 4 Context Performance",
            "",
            "| Dimension | Value | Outcomes | Solved rate | Mean conflict "
            "reduction | Mean AUC | Mean runtime | Mean generated |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    context_dimensions = (
        ("Split", "split"),
        ("Layout", "layout_mode"),
        ("Task", "task_variant"),
        ("Agents", "agent_count"),
        ("Stage", "repair_stage"),
        ("Heuristic", "heuristic"),
        ("Size", "neighborhood_size"),
    )
    performance = report["performance_by_horizon"].get("4", {})
    for label, key in context_dimensions:
        for name, value in performance.get(key, {}).items():
            lines.append(
                f"| {label} | {name} | {value['outcome_count']} | "
                f"{value['solved_rate']:.4f} | "
                f"{_format_number(value['conflict_reduction']['mean'])} | "
                f"{_format_number(value['conflict_auc']['mean'])} | "
                f"{_format_number(value['branch_runtime']['mean'])} | "
                f"{_format_number(value['generated']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Horizon 4 Action Families",
            "",
            "| Family | Outcomes | Solved rate | Mean conflicts | Mean AUC | "
            "Unique Pareto wins |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, value in report["action_families"].get("4", {}).items():
        lines.append(
            f"| {family} | {value['outcome_count']} | "
            f"{value['solved_rate']:.4f} | "
            f"{_format_number(value['conflicts_after']['mean'])} | "
            f"{_format_number(value['conflict_auc']['mean'])} | "
            f"{value['unique_pareto_family_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Horizon 4 Stage Preferences",
            "",
            "| Stage | States | Most frequent Pareto family | "
            "Most frequent unique winner |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for stage, value in report["stage_action_preferences_horizon_4"].items():
        pareto_counts = value["pareto_state_count_by_family"]
        unique_counts = value["unique_pareto_wins_by_family"]
        pareto = (
            max(pareto_counts.items(), key=lambda item: (item[1], item[0]))
            if pareto_counts
            else ("n/a", 0)
        )
        unique = (
            max(unique_counts.items(), key=lambda item: (item[1], item[0]))
            if unique_counts
            else ("n/a", 0)
        )
        lines.append(
            f"| {stage} | {value['state_count']} | "
            f"{pareto[0]} ({pareto[1]}) | {unique[0]} ({unique[1]}) |"
        )
    lines.append("")
    return "\n".join(lines)


def write_quality_report(
    collection: str | Path,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    **thresholds: Any,
) -> dict[str, Any]:
    root = Path(collection).resolve()
    report = analyze_collection(root, **thresholds)
    destination_json = Path(json_path) if json_path else root / "quality_report.json"
    destination_markdown = (
        Path(markdown_path) if markdown_path else root / "quality_report.md"
    )
    _write_json(destination_json, report)
    destination_markdown.parent.mkdir(parents=True, exist_ok=True)
    destination_markdown.write_text(render_markdown(report), encoding="utf-8")
    return report
