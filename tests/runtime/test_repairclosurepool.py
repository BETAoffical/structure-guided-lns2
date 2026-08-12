from __future__ import annotations

import unittest

from lns2_selector.runtime.repairclosurepool import (
    DependencyEvidence,
    REPAIRCLOSUREPOOL_ID,
    _evidence_fronts,
    generate_repairclosure_candidates,
)
from lns2_selector.runtime.topology_candidates import _jaccard
from tests.runtime import test_topology_candidates as topology_tests


class RepairClosurePoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state, self.analysis = topology_tests.TopologyCandidatesTest._structpool_state()
        self.anchor = {"candidate_id": "v2-anchor", "agents": list(range(8))}

    def test_dependency_fronts_do_not_scalarize_evidence(self) -> None:
        fronts = _evidence_fronts(
            {
                1: DependencyEvidence(2, 0, 0, 0),
                2: DependencyEvidence(0, 2, 0, 0),
                3: DependencyEvidence(1, 0, 0, 0),
                4: DependencyEvidence(0, 1, 0, 0),
            }
        )
        self.assertEqual(fronts, [(1, 2), (3, 4)])

    def test_generation_is_deterministic_bounded_and_conflict_touching(self) -> None:
        first = generate_repairclosure_candidates(
            self.state,
            self.analysis,
            v2_anchors=[self.anchor],
            maximum_candidates=12,
        )
        second = generate_repairclosure_candidates(
            self.state,
            self.analysis,
            v2_anchors=[self.anchor],
            maximum_candidates=12,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.candidates), 12)
        self.assertGreater(first.raw_candidate_count, len(first.candidates))
        active = {agent for edge in self.analysis.pair_set for agent in edge}
        anchor_set = set(self.anchor["agents"])
        for candidate in first.candidates:
            self.assertEqual(candidate["repairclosurepool_id"], REPAIRCLOSUREPOOL_ID)
            self.assertTrue(set(candidate["agents"]) & active)
            self.assertLessEqual(candidate["actual_size"], 64)
            self.assertNotEqual(set(candidate["agents"]), anchor_set)
        for index, candidate in enumerate(first.candidates):
            for previous in first.candidates[:index]:
                self.assertLessEqual(
                    _jaccard(candidate["agents"], previous["agents"]), 0.9
                )

    def test_sizes_come_from_dependency_fronts_not_fixed_grid(self) -> None:
        result = generate_repairclosure_candidates(
            self.state,
            self.analysis,
            v2_anchors=[self.anchor],
            maximum_candidates=12,
        )
        sizes = {int(candidate["actual_size"]) for candidate in result.candidates}
        self.assertTrue(sizes - {8, 16, 24, 32})
        self.assertIn(5, sizes)
        self.assertIn(6, sizes)

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "limits"):
            generate_repairclosure_candidates(
                self.state,
                self.analysis,
                v2_anchors=[self.anchor],
                maximum_candidates=0,
            )
        with self.assertRaisesRegex(ValueError, "temporal window"):
            generate_repairclosure_candidates(
                self.state,
                self.analysis,
                v2_anchors=[self.anchor],
                temporal_window=-1,
            )


if __name__ == "__main__":
    unittest.main()
