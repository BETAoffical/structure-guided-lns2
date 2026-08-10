from __future__ import annotations

from experiments.state_analysis import analyze_state
from experiments.stride_marginalpool_root_diagnostic import (
    bottleneck_order_change_fraction,
    candidate_pool_diagnostic,
    closure_diagnostic,
    path_response_diagnostic,
    time_aligned_path_change_ratio,
)


def _candidate(identifier: str, agents: list[int], score: float, family: str):
    return {
        "candidate_id": identifier,
        "agents": agents,
        "score": score,
        "selection_families": [family],
    }


def _state(paths: dict[int, list[int]], edges: list[list[int]]):
    return {
        "rows": 2,
        "cols": 3,
        "obstacles": [0, 0, 0, 0, 0, 0],
        "agents": [
            {
                "id": agent_id,
                "start": path[0],
                "goal": path[-1],
                "path": path,
                "path_cost": len(path) - 1,
                "shortest_path_cost": max(1, len(path) - 1),
                "delay": 0,
                "conflict_degree": sum(agent_id in edge for edge in edges),
            }
            for agent_id, path in sorted(paths.items())
        ],
        "conflict_edges": edges,
        "num_of_colliding_pairs": len(edges),
        "iteration": 0,
        "sum_of_costs": sum(len(path) - 1 for path in paths.values()),
        "low_level": {},
        "context": {},
    }


def test_time_aligned_path_change_pads_with_goal() -> None:
    assert time_aligned_path_change_ratio([0, 1], [0, 1, 1]) == 0.0
    assert time_aligned_path_change_ratio([0, 1, 1], [0, 2, 1]) == 1 / 3


def test_candidate_pool_separates_generator_collapse_and_ranker_lock() -> None:
    pool = [
        _candidate("winner", [0, 1, 2, 3, 4], 3.0, "structpool-hotspot:8"),
        _candidate("near", [0, 1, 2, 3, 4, 5], 2.0, "structpool-component:8"),
        _candidate("base", [6, 7], 1.0, "collision:4"),
    ]
    result = candidate_pool_diagnostic(pool, "winner", maximum_diverse_jaccard=0.8)
    assert result["generator_collapse"] is True
    assert result["ranker_lock"] is True
    assert result["diverse_alternative_count"] == 1
    assert result["diverse_structural_alternative_count"] == 0


def test_candidate_pool_detects_diverse_structural_alternative() -> None:
    pool = [
        _candidate("winner", [0, 1], 3.0, "structpool-hotspot:8"),
        _candidate("other", [2, 3], 2.0, "structpool-component:8"),
    ]
    result = candidate_pool_diagnostic(pool, "winner")
    assert result["generator_collapse"] is False
    assert result["ranker_lock"] is True


def test_path_response_distinguishes_latent_change_from_noop() -> None:
    before = _state({0: [0, 1, 2], 1: [3, 4, 5]}, [])
    unchanged = _state({0: [0, 1, 2], 1: [3, 4, 5]}, [])
    changed = _state({0: [0, 0, 1, 2], 1: [3, 4, 5]}, [])
    noop = path_response_diagnostic(before, unchanged, {0}, {0, 1, 2})
    latent = path_response_diagnostic(before, changed, {0}, {0, 1, 2})
    assert noop["path_descriptor_changed"] is False
    assert latent["path_descriptor_changed"] is True
    assert latent["selected_changed_agent_count"] == 1


def test_bottleneck_order_change_detects_reversal() -> None:
    before = _state({0: [0, 1, 2], 1: [3, 4, 1, 2]}, [])
    after = _state({0: [0, 0, 1, 2], 1: [3, 1, 2]}, [])
    assert bottleneck_order_change_fraction(before, after, {0, 1}, {1}) == 1.0


def test_closure_reports_missing_persistent_endpoint() -> None:
    state = _state({0: [0, 1, 2], 1: [4, 1, 2], 2: [5, 4, 3]}, [[0, 1]])
    analysis = analyze_state(state)
    result = closure_diagnostic(state, state, {0}, analysis)
    assert result["persistent_touch_coverage"] == 1.0
    assert result["persistent_full_coverage"] == 0.0
    assert result["closure_deficit_agents"] == [1]
    assert result["truncated_component_count"] == 1
