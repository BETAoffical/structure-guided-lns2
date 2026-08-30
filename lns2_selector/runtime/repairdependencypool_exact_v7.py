from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime import repairdependencypool as _reference


REPAIRDEPENDENCYPOOL_EXACT_V7_ID = "stride-repairdependencypool-exact-v7"

_DIAGNOSTIC_KEYS = {
    "context_build_count",
    "core_count",
    "max_rolling_occupancy_entries",
    "max_rolling_transition_entries",
    "padded_path_cell_count",
}


@dataclass(frozen=True)
class _TemporalContext:
    paths: dict[int, tuple[int, ...]]
    agent_ids: tuple[int, ...]
    horizon: int
    corridor: frozenset[int]


@dataclass(frozen=True)
class _TimeBuckets:
    occupancy: dict[int, tuple[int, ...]]
    transitions: dict[tuple[int, int], tuple[int, ...]]
    occupancy_entries: int
    transition_entries: int


_EMPTY_BUCKETS = _TimeBuckets({}, {}, 0, 0)


def _clone_function_with_globals(
    function: Callable[..., Any], replacements: Mapping[str, Any]
) -> Callable[..., Any]:
    """Clone one function without mutating its defining module's globals."""

    namespace = dict(function.__globals__)
    namespace.update(replacements)
    cloned = types.FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    return cloned


def _build_temporal_context(
    state: dict[str, Any], analysis: StateAnalysis
) -> _TemporalContext:
    # Keep the reference implementation's duplicate-id, horizon, terminal-pad,
    # and corridor semantics exactly.  In particular, call its `_positions`
    # helper rather than restating the padding rule here.
    agents = {int(row["id"]): row for row in state["agents"]}
    horizon = max(len(row["path"]) for row in agents.values())
    paths = {
        agent: _reference._positions(row["path"], horizon)
        for agent, row in agents.items()
    }
    corridor = frozenset(
        int(cell)
        for cell in analysis.free_cells
        if int(cell) in analysis.articulation
        or int(analysis.degrees.get(cell, 4)) <= 2
    )
    return _TemporalContext(
        paths=paths,
        agent_ids=tuple(sorted(paths)),
        horizon=horizon,
        corridor=corridor,
    )


def _time_buckets(
    context: _TemporalContext,
    outside_agents: tuple[int, ...],
    time_index: int,
) -> _TimeBuckets:
    if time_index < 0 or time_index >= context.horizon:
        return _EMPTY_BUCKETS

    occupancy: dict[int, list[int]] = {}
    transitions: dict[tuple[int, int], list[int]] = {}
    for agent in outside_agents:
        path = context.paths[agent]
        current = path[time_index]
        occupancy.setdefault(current, []).append(agent)
        if time_index:
            previous = path[time_index - 1]
            if previous != current:
                transitions.setdefault((previous, current), []).append(agent)
    return _TimeBuckets(
        occupancy={cell: tuple(agents) for cell, agents in occupancy.items()},
        transitions={edge: tuple(agents) for edge, agents in transitions.items()},
        occupancy_entries=sum(map(len, occupancy.values())),
        transition_entries=sum(map(len, transitions.values())),
    )


def _rolling_temporal_evidence_for_cores(
    state: dict[str, Any],
    analysis: StateAnalysis,
    ordered_cores: tuple[tuple[int, ...], ...],
) -> tuple[tuple[dict[int, dict[str, Any]], ...], dict[str, int]]:
    context = _build_temporal_context(state, analysis)
    core_sets = tuple(frozenset(core) for core in ordered_cores)
    outside_by_core = tuple(
        tuple(agent for agent in context.agent_ids if agent not in core)
        for core in core_sets
    )

    # Access every member once before the rolling scan.  Valid generator cores
    # always come from this state; an inconsistent core must fail closed.
    for core in ordered_cores:
        for member in core:
            context.paths[member]

    overlaps: list[dict[int, int]] = [dict() for _ in ordered_cores]
    reverses: list[dict[int, int]] = [dict() for _ in ordered_cores]
    cells: list[dict[int, set[int]]] = [dict() for _ in ordered_cores]

    previous = [_EMPTY_BUCKETS for _ in ordered_cores]
    current = [
        _time_buckets(context, outside, 0) for outside in outside_by_core
    ]
    following = [
        _time_buckets(context, outside, 1) for outside in outside_by_core
    ]
    max_occupancy_entries = 0
    max_transition_entries = 0

    for time_index in range(context.horizon):
        max_occupancy_entries = max(
            max_occupancy_entries,
            sum(
                bucket.occupancy_entries
                for window in (previous, current, following)
                for bucket in window
            ),
        )
        max_transition_entries = max(
            max_transition_entries,
            sum(
                bucket.transition_entries
                for window in (previous, current, following)
                for bucket in window
            ),
        )

        for core_index, core in enumerate(ordered_cores):
            overlap_counts = overlaps[core_index]
            reverse_counts = reverses[core_index]
            corridor_cells = cells[core_index]
            occupancy_window = (
                previous[core_index].occupancy,
                current[core_index].occupancy,
                following[core_index].occupancy,
            )
            transition_window = (
                previous[core_index].transitions,
                current[core_index].transitions,
                following[core_index].transitions,
            )
            for member in core:
                member_path = context.paths[member]
                member_cell = member_path[time_index]
                if member_cell not in context.corridor:
                    continue
                for occupancy in occupancy_window:
                    for blocker in occupancy.get(member_cell, ()):
                        overlap_counts[blocker] = overlap_counts.get(blocker, 0) + 1
                        corridor_cells.setdefault(blocker, set()).add(member_cell)

                if time_index == 0:
                    continue
                member_previous = member_path[time_index - 1]
                if member_previous == member_cell:
                    continue
                reverse_edge = (member_cell, member_previous)
                for transitions in transition_window:
                    for blocker in transitions.get(reverse_edge, ()):
                        reverse_counts[blocker] = reverse_counts.get(blocker, 0) + 1
                        corridor_cells.setdefault(blocker, set()).update(
                            (member_previous, member_cell)
                        )

        previous, current = current, following
        next_time = time_index + 2
        following = [
            _time_buckets(context, outside, next_time)
            for outside in outside_by_core
        ]

    results: list[dict[int, dict[str, Any]]] = []
    for core_index, outside_agents in enumerate(outside_by_core):
        evidence: dict[int, dict[str, Any]] = {}
        for blocker in outside_agents:
            overlap_count = overlaps[core_index].get(blocker, 0)
            reverse_count = reverses[core_index].get(blocker, 0)
            if overlap_count and reverse_count:
                evidence[blocker] = {
                    "current_conflict_edges": set(),
                    "conflict_events": [],
                    "corridor_cells": set(cells[core_index].get(blocker, ())),
                    "temporal_overlap_count": overlap_count,
                    "reverse_queue_count": reverse_count,
                }
        results.append(evidence)

    return tuple(results), {
        "context_build_count": 1,
        "core_count": len(ordered_cores),
        "max_rolling_occupancy_entries": max_occupancy_entries,
        "max_rolling_transition_entries": max_transition_entries,
        "padded_path_cell_count": sum(len(path) for path in context.paths.values()),
    }


class _ExactInvocation:
    def __init__(self, state: dict[str, Any], analysis: StateAnalysis) -> None:
        self._state = state
        self._analysis = analysis
        self._core_drafts_calls = 0
        self._temporal_calls = 0
        self._ordered_cores: tuple[tuple[int, ...], ...] | None = None
        self._temporal_results: tuple[dict[int, dict[str, Any]], ...] | None = None
        self._diagnostics = {
            "context_build_count": 0,
            "core_count": 0,
            "max_rolling_occupancy_entries": 0,
            "max_rolling_transition_entries": 0,
            "padded_path_cell_count": 0,
        }

    def _assert_identity(
        self, state: dict[str, Any], analysis: StateAnalysis
    ) -> None:
        if state is not self._state or analysis is not self._analysis:
            raise RuntimeError(
                "RepairDependencyPool exact v7 state/analysis identity changed"
            )

    def core_drafts(
        self, state: dict[str, Any], analysis: StateAnalysis
    ) -> list[dict[str, Any]]:
        self._assert_identity(state, analysis)
        if self._core_drafts_calls:
            raise RuntimeError(
                "RepairDependencyPool exact v7 core draft call count changed"
            )
        self._core_drafts_calls += 1
        cores = _reference._core_drafts(state, analysis)
        self._ordered_cores = tuple(
            tuple(sorted(map(int, core["agents"]))) for core in cores
        )
        self._diagnostics["core_count"] = len(self._ordered_cores)
        return cores

    def temporal_evidence(
        self,
        state: dict[str, Any],
        analysis: StateAnalysis,
        core: set[int],
    ) -> dict[int, dict[str, Any]]:
        self._assert_identity(state, analysis)
        if self._ordered_cores is None:
            raise RuntimeError(
                "RepairDependencyPool exact v7 temporal call preceded core drafts"
            )
        if self._temporal_calls >= len(self._ordered_cores):
            raise RuntimeError(
                "RepairDependencyPool exact v7 temporal call count changed"
            )
        expected_core = self._ordered_cores[self._temporal_calls]
        if tuple(sorted(map(int, core))) != expected_core:
            raise RuntimeError(
                "RepairDependencyPool exact v7 ordered core identity changed"
            )
        if self._temporal_results is None:
            self._temporal_results, diagnostics = (
                _rolling_temporal_evidence_for_cores(
                    state, analysis, self._ordered_cores
                )
            )
            self._diagnostics.update(diagnostics)
        result = self._temporal_results[self._temporal_calls]
        self._temporal_calls += 1
        return result

    def finish(self) -> dict[str, int]:
        if self._core_drafts_calls not in {0, 1}:
            raise RuntimeError(
                "RepairDependencyPool exact v7 core draft call count changed"
            )
        expected_temporal_calls = len(self._ordered_cores or ())
        if self._temporal_calls != expected_temporal_calls:
            raise RuntimeError(
                "RepairDependencyPool exact v7 temporal call count changed"
            )
        expected_builds = int(expected_temporal_calls > 0)
        if self._diagnostics["context_build_count"] != expected_builds:
            raise RuntimeError(
                "RepairDependencyPool exact v7 context build count changed"
            )
        if set(self._diagnostics) != _DIAGNOSTIC_KEYS or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in self._diagnostics.values()
        ):
            raise RuntimeError(
                "RepairDependencyPool exact v7 diagnostics changed"
            )
        if (
            self._diagnostics["max_rolling_transition_entries"]
            > self._diagnostics["max_rolling_occupancy_entries"]
        ):
            raise RuntimeError(
                "RepairDependencyPool exact v7 rolling entry bounds changed"
            )
        return dict(self._diagnostics)


def generate_repairdependency_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    existing_candidates: Iterable[dict[str, Any]] = (),
    maximum_candidates: int = 6,
    maximum_added_agents: int = 8,
    maximum_neighborhood_size: int = 32,
    maximum_jaccard_similarity: float = 0.9,
    diagnostics: dict[str, int] | None = None,
) -> _reference.RepairDependencyPoolResult:
    """Return the exact legacy result using a bounded rolling temporal index.

    The reference public generator is cloned for this invocation.  Its module
    globals remain untouched; only its core-draft and temporal-evidence lookups
    are redirected to invocation-local callables.
    """

    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise TypeError("RepairDependencyPool exact v7 diagnostics must be a dict")
        if diagnostics:
            raise ValueError("RepairDependencyPool exact v7 diagnostics must be empty")

    invocation = _ExactInvocation(state, analysis)
    cloned = _clone_function_with_globals(
        _reference.generate_repairdependency_candidates,
        {
            "_core_drafts": invocation.core_drafts,
            "_temporal_corridor_evidence": invocation.temporal_evidence,
        },
    )
    result = cloned(
        state,
        analysis,
        existing_candidates=existing_candidates,
        maximum_candidates=maximum_candidates,
        maximum_added_agents=maximum_added_agents,
        maximum_neighborhood_size=maximum_neighborhood_size,
        maximum_jaccard_similarity=maximum_jaccard_similarity,
    )
    scalar_diagnostics = invocation.finish()
    if diagnostics is not None:
        diagnostics.update(scalar_diagnostics)
    return result


def generate_repairdependency_candidates_exact_v7(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    existing_candidates: Iterable[dict[str, Any]] = (),
    maximum_candidates: int = 6,
    maximum_added_agents: int = 8,
    maximum_neighborhood_size: int = 32,
    maximum_jaccard_similarity: float = 0.9,
    diagnostics: dict[str, int] | None = None,
) -> _reference.RepairDependencyPoolResult:
    """Expose the exact-v7 integration name without breaking the old entrypoint."""

    return generate_repairdependency_candidates(
        state,
        analysis,
        existing_candidates=existing_candidates,
        maximum_candidates=maximum_candidates,
        maximum_added_agents=maximum_added_agents,
        maximum_neighborhood_size=maximum_neighborhood_size,
        maximum_jaccard_similarity=maximum_jaccard_similarity,
        diagnostics=diagnostics,
    )


__all__ = [
    "REPAIRDEPENDENCYPOOL_EXACT_V7_ID",
    "generate_repairdependency_candidates",
    "generate_repairdependency_candidates_exact_v7",
]
