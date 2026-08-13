from __future__ import annotations

import unittest

from lns2_selector.runtime.causaltopopool import generate_causaltopopool_candidates
from tests.runtime import test_topology_candidates as topology_fixture


class CausalTopoPoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state, self.analysis = topology_fixture.TopologyCandidatesTest._structpool_state()

    def test_generation_is_deterministic_natural_and_bounded(self) -> None:
        first = generate_causaltopopool_candidates(
            self.state,
            self.analysis,
            maximum_candidates=12,
            maximum_neighborhood_size=64,
        )
        second = generate_causaltopopool_candidates(
            self.state,
            self.analysis,
            maximum_candidates=12,
            maximum_neighborhood_size=64,
        )
        self.assertEqual(first, second)
        self.assertTrue(first.candidates)
        self.assertLessEqual(len(first.candidates), 12)
        for candidate in first.candidates:
            self.assertTrue(candidate["causaltopo_exact_natural_closure"])
            self.assertFalse(candidate["causaltopo_truncated"])
            self.assertLessEqual(candidate["actual_size"], 64)

    def test_oversized_closures_are_rejected_not_truncated(self) -> None:
        result = generate_causaltopopool_candidates(
            self.state,
            self.analysis,
            maximum_candidates=12,
            maximum_neighborhood_size=12,
        )
        self.assertGreater(result.oversized_closure_count, 0)
        self.assertTrue(
            any(
                row.get("rejection_reason") == "oversized_exact_closure"
                and int(row["proposed_size"]) > 12
                for row in result.attempts
            )
        )
        self.assertTrue(all(row["actual_size"] <= 12 for row in result.candidates))

    def test_existing_exact_candidates_are_not_reemitted(self) -> None:
        baseline = generate_causaltopopool_candidates(
            self.state,
            self.analysis,
            maximum_candidates=12,
        )
        excluded = baseline.candidates[0]
        result = generate_causaltopopool_candidates(
            self.state,
            self.analysis,
            existing_candidates=[excluded],
            maximum_candidates=12,
        )
        self.assertNotIn(
            excluded["candidate_id"],
            {candidate["candidate_id"] for candidate in result.candidates},
        )
        self.assertGreaterEqual(result.exact_existing_duplicate_count, 1)


if __name__ == "__main__":
    unittest.main()
