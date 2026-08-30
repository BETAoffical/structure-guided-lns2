from __future__ import annotations

import collections
import copy
import inspect
import random
from dataclasses import asdict
from unittest import mock

import pytest

from experiments.state_analysis import (
    ConflictEvent,
    StateAnalysis,
    analyze_state,
    reconstruct_conflicts,
)
from lns2_selector.runtime import repairdependencypool as reference
from lns2_selector.runtime import repairdependencypool_exact_v7 as exact


def _analysis(
    agents: list[dict],
    *,
    events: list[ConflictEvent],
    corridor: set[int],
    cell_count: int = 16,
) -> StateAnalysis:
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for row in agents:
        visit_heat.update(row["path"])
        agent_heat.update(set(row["path"]))
    pair_set = {
        tuple(sorted((int(event.left), int(event.right)))) for event in events
    }
    known = sorted(int(row["id"]) for row in agents)
    active = {agent for pair in pair_set for agent in pair}
    component_id = {agent: 0 for agent in active}
    component_members = {0: set(active)} if active else {}
    return StateAnalysis(
        rows=4,
        cols=max(4, (cell_count + 3) // 4),
        free_cells=set(range(cell_count)),
        degrees={cell: (2 if cell in corridor else 3) for cell in range(cell_count)},
        articulation=set(corridor),
        obstacle_rate_2={},
        obstacle_rate_4={},
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=events,
        pair_set=pair_set,
        component_id=component_id,
        component_members=component_members,
    )


def _fixed_state() -> tuple[dict, StateAnalysis]:
    agents = [
        {"id": 0, "path": [0, 1, 2, 3], "conflict_degree": 1},
        {"id": 1, "path": [4, 1, 5, 6], "conflict_degree": 1},
        {"id": 2, "path": [3, 2, 1, 7], "conflict_degree": 0},
        {"id": 3, "path": [8, 9, 10, 11], "conflict_degree": 0},
    ]
    events = [ConflictEvent(1, "vertex", 0, 1, (1,))]
    state = {
        "agents": agents,
        "conflict_edges": [[0, 1]],
        "num_of_colliding_pairs": 1,
        "rows": 4,
        "cols": 4,
        "obstacles": [0] * 16,
    }
    return state, _analysis(agents, events=events, corridor={1, 2})


def _random_valid_state(seed: int) -> tuple[dict, StateAnalysis]:
    rng = random.Random(seed)
    rows = cols = 4
    agent_count = rng.randint(3, 6)
    horizon = rng.randint(2, 7)
    agents: list[dict] = []
    for agent in range(agent_count):
        cell = 5 if agent < 2 else rng.randrange(rows * cols)
        path = [cell]
        explicit_length = rng.randint(1, horizon)
        for _ in range(1, explicit_length):
            row, col = divmod(cell, cols)
            neighbours = [cell]
            if row:
                neighbours.append(cell - cols)
            if row + 1 < rows:
                neighbours.append(cell + cols)
            if col:
                neighbours.append(cell - 1)
            if col + 1 < cols:
                neighbours.append(cell + 1)
            cell = rng.choice(neighbours)
            path.append(cell)
        agents.append({"id": agent, "path": path, "conflict_degree": 0})

    events = reconstruct_conflicts(agents)
    pair_set = {tuple(sorted((event.left, event.right))) for event in events}
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


def test_temporal_evidence_matches_reference_on_explicit_boundary_cases() -> None:
    agents = [
        {"id": 0, "path": [0, 1, 2, 2]},
        {"id": 1, "path": [2, 1, 0]},
        {"id": 2, "path": [1]},
        {"id": 3, "path": [3, 2, 1, 0, 0]},
        {"id": 4, "path": [7, 7, 7]},
    ]
    analysis = _analysis(
        agents,
        events=[ConflictEvent(1, "vertex", 0, 1, (1,))],
        corridor={0, 1, 2},
    )
    state = {"agents": agents}
    cores = ((0,), (0, 2), (0, 1, 2), (0, 1, 2, 3, 4))
    observed, diagnostics = exact._rolling_temporal_evidence_for_cores(
        state, analysis, cores
    )
    expected = tuple(
        reference._temporal_corridor_evidence(state, analysis, set(core))
        for core in cores
    )
    assert observed == expected
    assert diagnostics["context_build_count"] == 1
    assert diagnostics["core_count"] == len(cores)
    assert diagnostics["padded_path_cell_count"] == len(agents) * 5
    assert (
        diagnostics["max_rolling_transition_entries"]
        <= diagnostics["max_rolling_occupancy_entries"]
        <= 3 * len(cores) * len(agents)
    )


@pytest.mark.parametrize("seed", range(12))
def test_random_temporal_boundary_parity(seed: int) -> None:
    rng = random.Random(10_000 + seed)
    agent_count = rng.randint(2, 7)
    horizon = rng.randint(1, 8)
    cell_count = rng.randint(4, 12)
    agents = [
        {
            "id": agent,
            "path": [rng.randrange(cell_count) for _ in range(rng.randint(1, horizon))],
        }
        for agent in range(agent_count)
    ]
    corridor = {
        cell for cell in range(cell_count) if rng.choice((False, True))
    }
    analysis = _analysis(
        agents,
        events=[ConflictEvent(0, "vertex", 0, 1, (agents[0]["path"][0],))],
        corridor=corridor,
        cell_count=cell_count,
    )
    state = {"agents": agents}
    cores = []
    for _ in range(rng.randint(1, 6)):
        size = rng.randint(1, agent_count)
        cores.append(tuple(sorted(rng.sample(range(agent_count), size))))
    observed, _diagnostics = exact._rolling_temporal_evidence_for_cores(
        state, analysis, tuple(cores)
    )
    expected = tuple(
        reference._temporal_corridor_evidence(state, analysis, set(core))
        for core in cores
    )
    assert observed == expected


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"maximum_candidates": 1},
        {"maximum_added_agents": 1},
        {"maximum_neighborhood_size": 2},
        {"maximum_jaccard_similarity": 0.0},
    ],
)
def test_full_result_exact_parity_for_fixed_state(limits: dict) -> None:
    state, analysis = _fixed_state()
    existing = [{"agents": [0, 1]}]
    before = copy.deepcopy(state)
    expected = reference.generate_repairdependency_candidates(
        state,
        analysis,
        existing_candidates=existing,
        **limits,
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_repairdependency_candidates(
        state,
        analysis,
        existing_candidates=existing,
        diagnostics=diagnostics,
        **limits,
    )
    assert asdict(observed) == asdict(expected)
    assert state == before
    assert set(diagnostics) == exact._DIAGNOSTIC_KEYS
    assert diagnostics["context_build_count"] == 1
    assert diagnostics["core_count"] == len(reference._core_drafts(state, analysis))
    assert diagnostics["padded_path_cell_count"] == 16


@pytest.mark.parametrize("seed", range(8))
def test_full_result_randomized_exact_parity(seed: int) -> None:
    state, analysis = _random_valid_state(seed)
    rng = random.Random(20_000 + seed)
    limits = {
        "maximum_candidates": rng.randint(1, 6),
        "maximum_added_agents": rng.randint(1, 4),
        "maximum_neighborhood_size": rng.randint(2, 12),
        "maximum_jaccard_similarity": rng.choice((0.0, 0.5, 0.9)),
    }
    expected = reference.generate_repairdependency_candidates(
        copy.deepcopy(state), analysis, **limits
    )
    diagnostics: dict[str, int] = {}
    observed = exact.generate_repairdependency_candidates(
        copy.deepcopy(state), analysis, diagnostics=diagnostics, **limits
    )
    assert asdict(observed) == asdict(expected)
    if analysis.events and diagnostics["core_count"]:
        assert diagnostics["context_build_count"] == 1
    else:
        assert diagnostics["context_build_count"] == 0


def test_empty_events_return_before_context_build() -> None:
    state, analysis = _fixed_state()
    analysis = copy.deepcopy(analysis)
    analysis.events = []
    diagnostics: dict[str, int] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        build = mock.Mock(side_effect=AssertionError("context must not be built"))
        monkeypatch.setattr(exact, "_build_temporal_context", build)
        observed = exact.generate_repairdependency_candidates(
            state, analysis, diagnostics=diagnostics
        )
    expected = reference.generate_repairdependency_candidates(state, analysis)
    assert asdict(observed) == asdict(expected)
    assert diagnostics == {
        "context_build_count": 0,
        "core_count": 0,
        "max_rolling_occupancy_entries": 0,
        "max_rolling_transition_entries": 0,
        "padded_path_cell_count": 0,
    }


def test_invocation_fails_closed_on_state_or_analysis_identity_change() -> None:
    state, analysis = _fixed_state()
    invocation = exact._ExactInvocation(state, analysis)
    with pytest.raises(RuntimeError, match="state/analysis identity changed"):
        invocation.core_drafts(copy.deepcopy(state), analysis)
    with pytest.raises(RuntimeError, match="state/analysis identity changed"):
        invocation.core_drafts(state, copy.deepcopy(analysis))


def test_reference_globals_are_never_monkeypatched() -> None:
    state, analysis = _fixed_state()
    function = reference.generate_repairdependency_candidates
    frozen_core = function.__globals__["_core_drafts"]
    frozen_temporal = function.__globals__["_temporal_corridor_evidence"]
    exact.generate_repairdependency_candidates(state, analysis)
    assert function.__globals__["_core_drafts"] is frozen_core
    assert function.__globals__["_temporal_corridor_evidence"] is frozen_temporal


def test_exact_v7_named_entrypoint_is_an_explicit_signature_exact_alias() -> None:
    state, analysis = _fixed_state()
    existing = [{"agents": [0, 1]}]
    legacy_diagnostics: dict[str, int] = {}
    named_diagnostics: dict[str, int] = {}

    expected = exact.generate_repairdependency_candidates(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        existing_candidates=copy.deepcopy(existing),
        diagnostics=legacy_diagnostics,
    )
    observed = exact.generate_repairdependency_candidates_exact_v7(
        copy.deepcopy(state),
        copy.deepcopy(analysis),
        existing_candidates=copy.deepcopy(existing),
        diagnostics=named_diagnostics,
    )

    assert asdict(observed) == asdict(expected)
    assert named_diagnostics == legacy_diagnostics
    assert (
        inspect.signature(exact.generate_repairdependency_candidates_exact_v7)
        == inspect.signature(exact.generate_repairdependency_candidates)
    )
    assert (
        exact.generate_repairdependency_candidates_exact_v7
        is not exact.generate_repairdependency_candidates
    )
    assert "generate_repairdependency_candidates_exact_v7" in exact.__all__


def test_diagnostics_must_be_an_empty_plain_dict() -> None:
    state, analysis = _fixed_state()
    with pytest.raises(TypeError, match="must be a dict"):
        exact.generate_repairdependency_candidates(
            state, analysis, diagnostics=[]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be empty"):
        exact.generate_repairdependency_candidates(
            state, analysis, diagnostics={"old": 1}
        )
