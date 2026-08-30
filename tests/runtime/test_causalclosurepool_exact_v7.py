from __future__ import annotations

import collections
import copy
import random
from dataclasses import asdict

import pytest

from experiments.state_analysis import (
    ConflictEvent,
    StateAnalysis,
    analyze_state,
    reconstruct_conflicts,
)
from lns2_selector.runtime import causalclosurepool as reference
from lns2_selector.runtime import causalclosurepool_exact_v7 as exact


_DIAGNOSTIC_KEYS = {
    "event_time_index_build_count",
    "event_time_index_entry_count",
    "event_slice_call_count",
    "event_slice_returned_count",
    "direct_completion_count",
    "oversized_closure_call_count",
    "closure_call_count",
    "event_slice_candidate_entries_skipped",
}


def _analysis(
    agents: list[dict],
    *,
    events: list[ConflictEvent],
    articulation: set[int] | None = None,
    cell_count: int = 128,
) -> StateAnalysis:
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for row in agents:
        visit_heat.update(map(int, row["path"]))
        agent_heat.update(set(map(int, row["path"])))

    pair_set = {
        tuple(sorted((int(event.left), int(event.right)))) for event in events
    }
    known_agents = sorted(int(row["id"]) for row in agents)
    adjacency = {agent: set() for agent in known_agents}
    for left, right in pair_set:
        adjacency[left].add(right)
        adjacency[right].add(left)
    component_id: dict[int, int] = {}
    component_members: dict[int, set[int]] = {}
    unvisited = set(known_agents)
    component = 0
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        members = {start}
        stack = [start]
        while stack:
            agent = stack.pop()
            for other in sorted(adjacency[agent]):
                if other in unvisited:
                    unvisited.remove(other)
                    members.add(other)
                    stack.append(other)
        component_members[component] = members
        component_id.update({agent: component for agent in members})
        component += 1

    articulation = set(articulation or ())
    return StateAnalysis(
        rows=8,
        cols=max(16, (cell_count + 7) // 8),
        free_cells=set(range(cell_count)),
        degrees={
            cell: (2 if cell in articulation else 3)
            for cell in range(cell_count)
        },
        articulation=articulation,
        obstacle_rate_2={},
        obstacle_rate_4={},
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=events,
        pair_set=pair_set,
        component_id=component_id,
        component_members=component_members,
    )


def _localized_state(
    extra_nearby_agents: int = 2,
) -> tuple[dict, StateAnalysis]:
    agents = [
        {"id": 0, "path": [0, 1, 2, 3, 4, 5], "conflict_degree": 1},
        {"id": 1, "path": [6, 1, 7, 8, 9, 10], "conflict_degree": 1},
        # Shares the conflict cell only outside the event window.
        {"id": 2, "path": [11, 12, 13, 14, 15, 1], "conflict_degree": 0},
    ]
    for offset in range(extra_nearby_agents):
        agents.append(
            {
                "id": 3 + offset,
                "path": [
                    20 + offset,
                    30 + offset,
                    1,
                    40 + offset,
                    50 + offset,
                    60 + offset,
                ],
                "conflict_degree": 0,
            }
        )
    events = [ConflictEvent(1, "vertex", 0, 1, (1,))]
    state = {"agents": agents, "conflict_edges": [[0, 1]]}
    return state, _analysis(
        agents,
        events=events,
        articulation={1},
    )


def _random_valid_state(seed: int) -> tuple[dict, StateAnalysis]:
    rng = random.Random(seed)
    rows = cols = 4
    agent_count = rng.randint(3, 7)
    horizon = rng.randint(3, 8)
    agents: list[dict] = []
    for agent in range(agent_count):
        # Agents 0 and 1 deliberately share their first cell so every case has
        # at least one valid event and therefore exercises the optimized path.
        cell = 5 if agent < 2 else rng.randrange(rows * cols)
        path = [cell]
        explicit_length = rng.randint(1, horizon)
        for _ in range(1, explicit_length):
            row, column = divmod(cell, cols)
            neighbours = [cell]
            if row:
                neighbours.append(cell - cols)
            if row + 1 < rows:
                neighbours.append(cell + cols)
            if column:
                neighbours.append(cell - 1)
            if column + 1 < cols:
                neighbours.append(cell + 1)
            cell = rng.choice(neighbours)
            path.append(cell)
        agents.append({"id": agent, "path": path, "conflict_degree": 0})

    events = reconstruct_conflicts(agents)
    pair_set = {
        tuple(sorted((int(event.left), int(event.right)))) for event in events
    }
    degrees: collections.Counter[int] = collections.Counter()
    for left, right in pair_set:
        degrees[left] += 1
        degrees[right] += 1
    for row in agents:
        row["conflict_degree"] = int(degrees[int(row["id"])])
    state = {
        "agents": agents,
        "conflict_edges": [list(pair) for pair in sorted(pair_set)],
        "num_of_colliding_pairs": len(pair_set),
        "rows": rows,
        "cols": cols,
        "obstacles": [0] * (rows * cols),
    }
    return state, analyze_state(state)


def _assert_exact_diagnostics(diagnostics: dict[str, int]) -> None:
    assert set(diagnostics) == _DIAGNOSTIC_KEYS
    assert all(type(value) is int for value in diagnostics.values())
    assert all(value >= 0 for value in diagnostics.values())


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"maximum_candidates": 1},
        {"maximum_neighborhood_size": 3},
        {"temporal_window": 0},
        {"maximum_jaccard_similarity": 0.0},
    ],
)
def test_full_result_exact_parity_for_fixed_state(limits: dict) -> None:
    state, analysis = _localized_state()
    anchor = {"candidate_id": "v2-anchor-fixed", "agents": [0, 1]}
    before = copy.deepcopy(state)
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=[copy.deepcopy(anchor)],
        **limits,
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[anchor],
        diagnostics=diagnostics,
        **limits,
    )

    assert asdict(observed) == asdict(expected)
    assert state == before
    _assert_exact_diagnostics(diagnostics)
    assert diagnostics["event_time_index_build_count"] == 1
    assert diagnostics["closure_call_count"] == observed.family_attempt_count


@pytest.mark.parametrize("seed", range(12))
def test_full_result_randomized_exact_parity(seed: int) -> None:
    state, analysis = _random_valid_state(70_000 + seed)
    rng = random.Random(80_000 + seed)
    anchor_agents = sorted(
        rng.sample(
            range(len(state["agents"])),
            rng.randint(1, min(3, len(state["agents"]))),
        )
    )
    anchors = [
        {
            "candidate_id": f"random-anchor-{seed}",
            "agents": anchor_agents,
        }
    ]
    limits = {
        "maximum_candidates": rng.randint(1, 12),
        "maximum_neighborhood_size": rng.randint(1, 10),
        "temporal_window": rng.randint(0, 2),
        "maximum_jaccard_similarity": rng.choice((0.0, 0.5, 0.9)),
    }
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=copy.deepcopy(anchors),
        **limits,
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=copy.deepcopy(anchors),
        diagnostics=diagnostics,
        **limits,
    )

    assert asdict(observed) == asdict(expected)
    _assert_exact_diagnostics(diagnostics)
    assert diagnostics["event_time_index_build_count"] == 1
    assert diagnostics["closure_call_count"] == observed.family_attempt_count


def test_same_core_at_multiple_seed_times_never_reuses_localized_identity() -> None:
    agents = [
        {"id": 0, "path": [0, 1, 2, 3, 4], "conflict_degree": 1},
        {"id": 1, "path": [5, 1, 6, 3, 7], "conflict_degree": 1},
        {"id": 2, "path": [8, 9, 10, 11, 12], "conflict_degree": 0},
    ]
    events = [
        ConflictEvent(1, "vertex", 0, 1, (1,)),
        ConflictEvent(3, "vertex", 0, 1, (3,)),
    ]
    state = {"agents": agents, "conflict_edges": [[0, 1]]}
    analysis = _analysis(agents, events=events, articulation={1, 3})
    anchor = {"candidate_id": "same-core-two-times", "agents": [0, 1]}
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=[copy.deepcopy(anchor)],
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[anchor],
        diagnostics=diagnostics,
    )

    assert asdict(observed) == asdict(expected)
    # The same agent tuple occurs as both an anchor and event component at two
    # seed times: four independently localized cores, five families per core.
    assert observed.core_attempt_count == 4
    assert observed.family_attempt_count == 20
    assert diagnostics["direct_completion_count"] == 4
    assert diagnostics["closure_call_count"] == 20


def test_initial_localized_oversize_never_runs_direct_completion() -> None:
    agents = [
        {"id": 0, "path": [0, 1, 2], "conflict_degree": 1},
        {"id": 1, "path": [3, 1, 4], "conflict_degree": 1},
    ]
    events = [ConflictEvent(1, "vertex", 0, 1, (1,))]
    state = {"agents": agents, "conflict_edges": [[0, 1]]}
    analysis = _analysis(agents, events=events, articulation={1})
    anchor = {"candidate_id": "single-agent-anchor", "agents": [0]}
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=[copy.deepcopy(anchor)],
        maximum_neighborhood_size=1,
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[anchor],
        maximum_neighborhood_size=1,
        diagnostics=diagnostics,
    )

    assert asdict(observed) == asdict(expected)
    assert observed.core_attempt_count == 2
    assert observed.oversized_core_count == 1
    assert observed.family_attempt_count == 5
    assert observed.oversized_family_count == 5
    assert diagnostics["closure_call_count"] == 5
    assert diagnostics["oversized_closure_call_count"] == 5
    assert diagnostics["direct_completion_count"] == 0


def test_same_time_equal_sort_keys_preserve_original_event_order() -> None:
    agents = [
        {"id": 0, "path": [0, 1, 2], "conflict_degree": 1},
        {"id": 1, "path": [3, 1, 4], "conflict_degree": 1},
        {"id": 2, "path": [5, 6, 7], "conflict_degree": 0},
    ]
    first = ConflictEvent(1, "vertex", 0, 1, (99,))
    second = ConflictEvent(1, "vertex", 0, 1, (3,))
    events = [first, second]
    state = {"agents": agents, "conflict_edges": [[0, 1]]}
    analysis = _analysis(agents, events=events, articulation={1})
    context = reference._with_events(
        reference._causal_context(
            state,
            relevant_times=frozenset({0, 1, 2}),
            horizon=3,
        ),
        analysis,
    )
    assert context.events_by_agent[0] == (first, second)
    assert context.events_by_agent[1] == (first, second)

    anchor = {"candidate_id": "same-time-order", "agents": [0]}
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=[copy.deepcopy(anchor)],
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[anchor],
        diagnostics=diagnostics,
    )
    assert asdict(observed) == asdict(expected)
    # Both events are indexed once for each endpoint; slicing cannot reorder
    # the stable same-key tuple produced by the frozen `_with_events` helper.
    assert diagnostics["event_time_index_entry_count"] == 4


def test_empty_events_return_before_event_index_build() -> None:
    state, analysis = _localized_state()
    analysis = copy.deepcopy(analysis)
    analysis.events = []
    analysis.pair_set = set()
    diagnostics: dict[str, int] = {}
    expected = reference.generate_causalclosure_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        v2_anchors=[],
    )
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[],
        diagnostics=diagnostics,
    )

    assert asdict(observed) == asdict(expected)
    assert diagnostics == {key: 0 for key in _DIAGNOSTIC_KEYS}


def test_diagnostics_have_exact_schema_and_native_integer_values() -> None:
    state, analysis = _localized_state(extra_nearby_agents=3)
    diagnostics: dict[str, int] = {}
    observed = exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[{"candidate_id": "diagnostics", "agents": [0, 1]}],
        diagnostics=diagnostics,
    )

    _assert_exact_diagnostics(diagnostics)
    assert diagnostics["event_time_index_build_count"] == 1
    assert diagnostics["event_time_index_entry_count"] == 2 * len(analysis.events)
    assert diagnostics["closure_call_count"] == observed.family_attempt_count
    assert diagnostics["event_slice_returned_count"] > 0


def test_reference_module_globals_retain_exact_identity() -> None:
    state, analysis = _localized_state()
    function = reference.generate_causalclosure_candidates
    frozen = {
        name: function.__globals__[name]
        for name in (
            "_with_events",
            "_localized_conflict_closure",
            "_closure_draft",
        )
    }

    exact.generate_causalclosure_candidates_exact_v7(
        state,
        analysis,
        v2_anchors=[{"candidate_id": "identity", "agents": [0, 1]}],
    )

    for name, original in frozen.items():
        assert function.__globals__[name] is original
