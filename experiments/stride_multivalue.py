from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


MULTIHORIZON_VALUE_SCHEMA = "lns2.stride.multihorizon_value.v1"
MULTIVALUE_MODEL_ID = "stride-multivalue-v1"
MULTIVALUE_HORIZONS = (1, 8, 32, 128)
MULTIVALUE_TEACHERS = ("v2-full", "official-adaptive")
ALLOWED_STOP_REASONS = ("success", "repair_limit", "wall_timeout")


def _pair_set(state: Mapping[str, Any]) -> set[tuple[int, int]]:
    result = set()
    for raw in state.get("conflict_edges", ()):
        edge = tuple(sorted(map(int, raw)))
        if len(edge) != 2 or edge[0] == edge[1]:
            raise ValueError("multi-value state contains an invalid conflict edge")
        result.add(edge)
    reported = int(state.get("num_of_colliding_pairs", len(result)))
    if reported != len(result):
        raise ValueError("multi-value conflict count and edge set disagree")
    return result


def _components(edges: set[tuple[int, int]]) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    result: list[set[int]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        members = {start}
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    members.add(neighbor)
                    stack.append(neighbor)
        result.append(members)
    return result


def _neighborhood(transition: Mapping[str, Any]) -> tuple[int, ...]:
    metrics = transition.get("metrics")
    raw = metrics.get("neighborhood") if isinstance(metrics, Mapping) else None
    if not isinstance(raw, list) or not raw:
        action = transition.get("action")
        raw = action.get("agents") if isinstance(action, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("multi-value transition has no realized neighborhood")
    agents = tuple(sorted(set(map(int, raw))))
    if len(agents) != len(raw):
        raise ValueError("multi-value transition neighborhood contains duplicates")
    return agents


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _repeat_metrics(transitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    neighborhoods = [_neighborhood(row) for row in transitions]
    if not neighborhoods:
        return {
            "neighborhood_count": 0,
            "exact_repeat_count": 0,
            "exact_repeat_rate": 0.0,
            "mean_max_prior_jaccard": 0.0,
            "maximum_exact_repeat_streak": 0,
        }
    exact = 0
    maximum_jaccard: list[float] = []
    streak = 0
    maximum_streak = 0
    for index, neighborhood in enumerate(neighborhoods):
        prior = neighborhoods[:index]
        repeated = neighborhood in prior
        exact += int(repeated)
        streak = streak + 1 if index and neighborhood == neighborhoods[index - 1] else 0
        maximum_streak = max(maximum_streak, streak)
        maximum_jaccard.append(
            max(
                (jaccard_similarity(neighborhood, row) for row in prior),
                default=0.0,
            )
        )
    return {
        "neighborhood_count": len(neighborhoods),
        "exact_repeat_count": exact,
        "exact_repeat_rate": exact / len(neighborhoods),
        "mean_max_prior_jaccard": sum(maximum_jaccard) / len(maximum_jaccard),
        "maximum_exact_repeat_streak": maximum_streak,
    }


def _structural_metrics(
    initial_edges: set[tuple[int, int]], current_edges: set[tuple[int, int]]
) -> dict[str, float]:
    denominator = max(1, len(initial_edges))
    initial_components = _components(initial_edges)
    current_components = _components(current_edges)
    initial_component_by_agent = {
        agent: index
        for index, component in enumerate(initial_components)
        for agent in component
    }
    remerged = sum(
        len(
            {
                initial_component_by_agent[agent]
                for agent in component
                if agent in initial_component_by_agent
            }
        )
        >= 2
        for component in current_components
    )
    persistent_components = sum(
        any(left in component or right in component for left, right in current_edges)
        for component in initial_components
    )
    return {
        "original_conflict_edge_residual_rate": len(
            initial_edges & current_edges
        )
        / denominator,
        "new_conflict_edge_rate": len(current_edges - initial_edges) / denominator,
        "conflict_edge_migration_rate": len(initial_edges ^ current_edges)
        / denominator,
        "initial_component_persistence_rate": (
            persistent_components / len(initial_components)
            if initial_components
            else 0.0
        ),
        "component_remerge_rate": (
            remerged / len(current_components) if current_components else 0.0
        ),
    }


def multihorizon_targets(
    states: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    *,
    stop_reason: str,
    horizons: Sequence[int] = MULTIVALUE_HORIZONS,
) -> dict[str, Any]:
    """Build distributional future labels without imputing censored outcomes."""

    if len(states) != len(transitions) + 1 or not states:
        raise ValueError("multi-value states and transitions are not aligned")
    if stop_reason not in ALLOWED_STOP_REASONS:
        raise ValueError("multi-value rollout has an unsupported stop reason")
    normalized_horizons = tuple(map(int, horizons))
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("multi-value horizons must be positive")
    if tuple(sorted(set(normalized_horizons))) != normalized_horizons:
        raise ValueError("multi-value horizons must be strictly increasing")
    conflicts = [len(_pair_set(state)) for state in states]
    initial_conflicts = conflicts[0]
    initial_edges = _pair_set(states[0])
    feasible_indices = [
        index for index, state in enumerate(states) if bool(state.get("feasible"))
    ]
    feasible_step = min(feasible_indices) if feasible_indices else None
    targets: dict[str, Any] = {}
    for horizon in normalized_horizons:
        cutoff = min(horizon, len(transitions))
        feasible_by_horizon = feasible_step is not None and feasible_step <= horizon
        horizon_observed = len(transitions) >= horizon or feasible_by_horizon
        right_censored = not horizon_observed and stop_reason in {
            "repair_limit",
            "wall_timeout",
        }
        observed_conflicts = conflicts[: cutoff + 1]
        denominator = max(1, initial_conflicts)
        observed_auc = sum(value / denominator for value in observed_conflicts) / len(
            observed_conflicts
        )
        if horizon_observed:
            padded = list(conflicts[: horizon + 1])
            if len(padded) < horizon + 1:
                if not feasible_by_horizon:
                    raise ValueError("non-feasible horizon cannot be padded")
                padded.extend([0] * (horizon + 1 - len(padded)))
            normalized_auc: float | None = sum(
                value / denominator for value in padded
            ) / len(padded)
            terminal_state = states[min(horizon, len(states) - 1)]
            terminal_edges = _pair_set(terminal_state)
            terminal_ratio: float | None = len(terminal_edges) / denominator
            structure: dict[str, float | None] = _structural_metrics(
                initial_edges, terminal_edges
            )
        else:
            normalized_auc = None
            terminal_ratio = None
            structure = {
                name: None
                for name in _structural_metrics(
                    initial_edges, _pair_set(states[cutoff])
                )
            }
        observed_structure = _structural_metrics(
            initial_edges, _pair_set(states[cutoff])
        )
        targets[str(horizon)] = {
            "horizon": horizon,
            "observed_repair_decisions": cutoff,
            "horizon_observed": horizon_observed,
            "right_censored": right_censored,
            "feasible_event_observed": feasible_by_horizon,
            "feasible_event_step": (
                int(feasible_step) if feasible_by_horizon else None
            ),
            "normalized_conflict_auc": normalized_auc,
            "observed_normalized_conflict_auc": observed_auc,
            "terminal_conflict_ratio": terminal_ratio,
            "observed_terminal_conflict_ratio": conflicts[cutoff] / denominator,
            **structure,
            "observed_structure": observed_structure,
            "neighborhood_history": _repeat_metrics(transitions[:cutoff]),
        }
    return {
        "schema": MULTIHORIZON_VALUE_SCHEMA,
        "initial_conflict_count": initial_conflicts,
        "repair_decision_count": len(transitions),
        "stop_reason": stop_reason,
        "feasible_step": feasible_step,
        "horizons": targets,
    }


def anchor_relative_targets(
    candidate: Mapping[str, Any], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate.get("schema") != MULTIHORIZON_VALUE_SCHEMA or anchor.get(
        "schema"
    ) != MULTIHORIZON_VALUE_SCHEMA:
        raise ValueError("anchor-relative targets require multi-value labels")
    if set(candidate["horizons"]) != set(anchor["horizons"]):
        raise ValueError("candidate and anchor horizons differ")
    result: dict[str, Any] = {}
    for horizon in candidate["horizons"]:
        left = dict(candidate["horizons"][horizon])
        right = dict(anchor["horizons"][horizon])
        complete = bool(left["horizon_observed"] and right["horizon_observed"])

        def delta(name: str) -> float | None:
            first, second = left.get(name), right.get(name)
            if not complete or first is None or second is None:
                return None
            return float(first) - float(second)

        result[horizon] = {
            "paired_horizon_observed": complete,
            "candidate_right_censored": bool(left["right_censored"]),
            "anchor_right_censored": bool(right["right_censored"]),
            "feasible_event_delta": int(left["feasible_event_observed"])
            - int(right["feasible_event_observed"]),
            "normalized_conflict_auc_delta": delta("normalized_conflict_auc"),
            "terminal_conflict_ratio_delta": delta("terminal_conflict_ratio"),
            "original_conflict_edge_residual_rate_delta": delta(
                "original_conflict_edge_residual_rate"
            ),
            "new_conflict_edge_rate_delta": delta("new_conflict_edge_rate"),
            "conflict_edge_migration_rate_delta": delta(
                "conflict_edge_migration_rate"
            ),
            "initial_component_persistence_rate_delta": delta(
                "initial_component_persistence_rate"
            ),
            "component_remerge_rate_delta": delta("component_remerge_rate"),
        }
    return {
        "schema": "lns2.stride.multihorizon_anchor_delta.v1",
        "horizons": result,
    }


@dataclass(frozen=True)
class WorkerPreflightMeasurement:
    workers: int
    throughput_per_second: float
    peak_rss_bytes: int
    error_count: int = 0


def choose_dynamic_worker_count(
    measurements: Sequence[WorkerPreflightMeasurement],
    *,
    logical_cpu_count: int,
    available_memory_bytes: int,
    maximum_workers: int = 16,
    reserved_logical_cpus: int = 2,
    memory_fraction: float = 0.75,
) -> int:
    if logical_cpu_count <= reserved_logical_cpus:
        raise ValueError("not enough logical CPUs for the reserved worker policy")
    if available_memory_bytes <= 0 or not 0.0 < memory_fraction <= 1.0:
        raise ValueError("invalid dynamic worker memory budget")
    worker_limit = min(maximum_workers, logical_cpu_count - reserved_logical_cpus)
    memory_limit = available_memory_bytes * memory_fraction
    eligible = [
        row
        for row in measurements
        if 0 < row.workers <= worker_limit
        and row.error_count == 0
        and row.peak_rss_bytes <= memory_limit
        and math.isfinite(row.throughput_per_second)
        and row.throughput_per_second > 0.0
    ]
    if not eligible:
        raise ValueError("no dynamic worker preflight configuration is eligible")
    return max(
        eligible,
        key=lambda row: (row.throughput_per_second, -row.workers),
    ).workers


__all__ = [
    "ALLOWED_STOP_REASONS",
    "MULTIHORIZON_VALUE_SCHEMA",
    "MULTIVALUE_HORIZONS",
    "MULTIVALUE_MODEL_ID",
    "MULTIVALUE_TEACHERS",
    "WorkerPreflightMeasurement",
    "anchor_relative_targets",
    "choose_dynamic_worker_count",
    "jaccard_similarity",
    "multihorizon_targets",
]
