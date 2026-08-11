from __future__ import annotations

import unittest

from lns2_selector.runtime.paretopool import (
    PARETOPOOL_BUDGETS,
    generate_paretopool_candidates,
    natural_structural_breakpoints,
)
from lns2_selector.runtime.temporal_state import temporal_history_context
from lns2_selector.runtime.topology_candidates import (
    _SCALEPOOL_VARIANT_ORDER,
    _jaccard,
    _structural_candidate_context,
)
from tests.runtime.test_topology_candidates import TopologyCandidatesTest


class ParetoPoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state, self.analysis = TopologyCandidatesTest._structpool_state()
        self.anchor = tuple(range(8))

    def test_natural_breakpoints_are_state_derived(self) -> None:
        context = _structural_candidate_context(self.state, self.analysis)
        values = {
            variant: natural_structural_breakpoints(
                context, variant, maximum_neighborhood_size=64
            )
            for variant in _SCALEPOOL_VARIANT_ORDER
        }
        self.assertEqual(values["bottleneck_crossing"], (36, 40))
        self.assertEqual(values["conflict_component"], (32, 40))
        self.assertEqual(values["topology_boundary_articulation"], (8, 10, 24, 36))
        self.assertEqual(values["spatiotemporal_hotspot"], (4, 8, 10, 32))
        self.assertNotEqual(
            set(values["bottleneck_crossing"]),
            {8, 16, 24, 32},
        )

    def test_generation_is_deterministic_and_respects_budget(self) -> None:
        for budget in PARETOPOOL_BUDGETS:
            first = generate_paretopool_candidates(
                self.state,
                self.analysis,
                v2_anchor_agents=self.anchor,
                maximum_candidates=budget,
            )
            second = generate_paretopool_candidates(
                self.state,
                self.analysis,
                v2_anchor_agents=self.anchor,
                maximum_candidates=budget,
            )
            self.assertEqual(first, second)
            self.assertLessEqual(len(first.candidates), budget)
            self.assertGreater(first.raw_candidate_count, len(first.candidates))
            for candidate in first.candidates:
                self.assertLessEqual(candidate["v2_anchor_jaccard"], 0.9)
            for index, candidate in enumerate(first.candidates):
                for previous in first.candidates[:index]:
                    self.assertLessEqual(
                        _jaccard(candidate["agents"], previous["agents"]),
                        0.8,
                    )
        with self.assertRaisesRegex(ValueError, "budget"):
            generate_paretopool_candidates(
                self.state,
                self.analysis,
                v2_anchor_agents=self.anchor,
                maximum_candidates=7,
            )

    def test_history_adds_at_most_two_conditioned_candidates(self) -> None:
        history = temporal_history_context(
            recent_neighborhoods=[range(8), range(8, 16)],
            recent_neighborhood_exact_repeat=[False, False],
            recent_neighborhood_max_jaccard=[0.2, 0.3],
            persistent_conflict_edges=[[0, 1], [8, 9], [16, 17]],
            conflict_signature_streak=5,
            agent_repair_counts={0: 5, 1: 4, 8: 0, 9: 0},
            recent_pp_history=[(False, True, False)],
        )
        result = generate_paretopool_candidates(
            self.state,
            self.analysis,
            v2_anchor_agents=self.anchor,
            history=history,
            maximum_candidates=12,
        )
        self.assertGreaterEqual(result.history_candidate_count, 1)
        self.assertLessEqual(result.history_candidate_count, 2)
        conditioned = [
            candidate
            for candidate in result.candidates
            if candidate["paretopool_history_conditioned"]
        ]
        self.assertTrue(conditioned)
        self.assertTrue(
            all("history" in candidate["paretopool_sources"] for candidate in conditioned)
        )

    def test_empty_history_does_not_create_history_candidates(self) -> None:
        result = generate_paretopool_candidates(
            self.state,
            self.analysis,
            v2_anchor_agents=self.anchor,
            history=temporal_history_context(),
            maximum_candidates=6,
        )
        self.assertEqual(result.history_candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
