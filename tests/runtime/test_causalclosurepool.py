from __future__ import annotations

import collections
import unittest

from experiments.state_analysis import ConflictEvent, StateAnalysis
from lns2_selector.runtime.causalclosurepool import (
    CAUSALCLOSUREPOOL_ID,
    CausalContactEvidence,
    _causal_context,
    _temporal_neighbors,
    _with_events,
    generate_causalclosure_candidates,
)


def _localized_state(extra_nearby_agents: int = 1) -> tuple[dict, StateAnalysis]:
    agents = [
        {"id": 0, "path": [0, 1, 2, 3, 4, 5], "conflict_degree": 1},
        {"id": 1, "path": [6, 1, 7, 8, 9, 10], "conflict_degree": 1},
        # This agent shares the conflict cell only far outside the event window.
        {"id": 2, "path": [11, 12, 13, 14, 15, 1], "conflict_degree": 0},
    ]
    for offset in range(extra_nearby_agents):
        agents.append(
            {
                "id": 3 + offset,
                "path": [20 + offset, 30 + offset, 1, 40 + offset, 50 + offset, 60 + offset],
                "conflict_degree": 0,
            }
        )
    event = ConflictEvent(1, "vertex", 0, 1, (1,))
    paths = [row["path"] for row in agents]
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for path in paths:
        visit_heat.update(path)
        agent_heat.update(set(path))
    state = {
        "agents": agents,
        "conflict_edges": [[0, 1]],
    }
    analysis = StateAnalysis(
        rows=8,
        cols=16,
        free_cells=set(range(128)),
        degrees={cell: (2 if cell == 1 else 3) for cell in range(128)},
        articulation={1},
        obstacle_rate_2={},
        obstacle_rate_4={},
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=[event],
        pair_set={(0, 1)},
        component_id={0: 0, 1: 0},
        component_members={0: {0, 1}},
    )
    return state, analysis


class CausalClosurePoolTest(unittest.TestCase):
    def test_sparse_context_materializes_terminal_waits_once(self) -> None:
        state = {
            "agents": [
                {"id": 0, "path": [7]},
                {"id": 1, "path": [8, 9, 10]},
            ],
            "conflict_edges": [],
        }
        context = _causal_context(
            state,
            relevant_times=frozenset({0, 1, 2}),
            horizon=3,
        )

        self.assertEqual(context.paths[0], (7, 7, 7))
        self.assertEqual(context.paths[1], (8, 9, 10))
        self.assertEqual(context.occupancy[(2, 7)], frozenset({0}))

    def test_integer_temporal_accumulator_matches_reference_evidence(self) -> None:
        state, analysis = _localized_state(extra_nearby_agents=3)
        horizon = max(len(row["path"]) for row in state["agents"])
        times = frozenset(range(horizon))
        context = _with_events(
            _causal_context(state, relevant_times=times, horizon=horizon),
            analysis,
        )

        def reference(agent: int, radius: int, bottleneck_only: bool):
            result: dict[int, list[int]] = {}
            path = context.paths[agent]
            for source_time in times:
                cell = path[source_time]
                bottleneck = (
                    cell in analysis.articulation
                    or int(analysis.degrees.get(cell, 4)) <= 2
                )
                if bottleneck_only and not bottleneck:
                    continue
                for delta in range(-radius, radius + 1):
                    other_time = source_time + delta
                    if other_time not in times:
                        continue
                    for other in context.occupancy.get((other_time, cell), ()):
                        if other == agent:
                            continue
                        values = result.setdefault(other, [0, 0, 0, 0, 0])
                        values[1] += int(delta == 0)
                        values[2] += int(delta != 0)
                        values[4] += int(bottleneck)
                    if source_time == 0 or other_time == 0:
                        continue
                    previous = path[source_time - 1]
                    if previous == cell:
                        continue
                    for other in context.transitions.get(
                        (other_time, cell, previous), ()
                    ):
                        if other == agent:
                            continue
                        values = result.setdefault(other, [0, 0, 0, 0, 0])
                        values[1] += int(delta == 0)
                        values[2] += int(delta != 0)
                        values[3] += 1
                        values[4] += int(bottleneck)
            return {
                other: CausalContactEvidence(*values)
                for other, values in result.items()
            }

        for agent in range(len(state["agents"])):
            for radius in (0, 2):
                for bottleneck_only in (False, True):
                    self.assertEqual(
                        _temporal_neighbors(
                            context,
                            analysis,
                            agent,
                            times,
                            temporal_radius=radius,
                            bottleneck_only=bottleneck_only,
                        ),
                        reference(agent, radius, bottleneck_only),
                    )

    def test_far_full_path_overlap_cannot_enter_local_closure(self) -> None:
        state, analysis = _localized_state()
        anchor = {"candidate_id": "v2-anchor", "agents": [0, 1]}
        first = generate_causalclosure_candidates(
            state, analysis, v2_anchors=[anchor], maximum_candidates=12
        )
        second = generate_causalclosure_candidates(
            state, analysis, v2_anchors=[anchor], maximum_candidates=12
        )
        self.assertEqual(first, second)
        self.assertTrue(first.candidates)
        for candidate in first.candidates:
            self.assertEqual(candidate["causalclosurepool_id"], CAUSALCLOSUREPOOL_ID)
            self.assertNotIn(2, candidate["agents"])
            self.assertNotIn(
                "shared_path_cells",
                next(iter(candidate["causalclosure_evidence_by_agent"].values())),
            )
        self.assertTrue(any(3 in row["agents"] for row in first.candidates))

    def test_oversized_causal_family_is_rejected_not_truncated(self) -> None:
        state, analysis = _localized_state(extra_nearby_agents=5)
        anchor = {"candidate_id": "v2-anchor", "agents": [0, 1]}
        result = generate_causalclosure_candidates(
            state,
            analysis,
            v2_anchors=[anchor],
            maximum_candidates=12,
            maximum_neighborhood_size=4,
        )
        self.assertGreater(result.oversized_family_count, 0)
        self.assertTrue(
            all(candidate["actual_size"] <= 4 for candidate in result.candidates)
        )
        self.assertTrue(
            any(
                attempt.get("rejection_reason") == "causal_closure_oversized"
                and int(attempt["proposed_size"]) > 4
                for attempt in result.attempts
            )
        )

    def test_invalid_limits_are_rejected(self) -> None:
        state, analysis = _localized_state()
        anchor = {"candidate_id": "v2-anchor", "agents": [0, 1]}
        with self.assertRaisesRegex(ValueError, "limits"):
            generate_causalclosure_candidates(
                state, analysis, v2_anchors=[anchor], maximum_candidates=0
            )
        with self.assertRaisesRegex(ValueError, "temporal window"):
            generate_causalclosure_candidates(
                state, analysis, v2_anchors=[anchor], temporal_window=-1
            )


if __name__ == "__main__":
    unittest.main()
