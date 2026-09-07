import unittest
from unittest.mock import patch

from generators.models import MapData
from lns2_selector.evaluation.pressure_capacity import analyze_failure, eligible_edges, matching_capacity


class CapacityTests(unittest.TestCase):
    def test_matching_respects_shared_endpoints(self):
        self.assertEqual(matching_capacity({((0, 0), (1, 0)), ((0, 0), (1, 1))}), 1)
        self.assertEqual(matching_capacity(set()), 0)
        self.assertEqual(matching_capacity({((0, 0), (1, 0)), ((0, 1), (1, 1))}), 2)

    def test_shortest_route_eligibility(self):
        grid = MapData("open", 1, ["...", "..."])
        pools = {"left": [(0, 0), (1, 0)], "right": [(0, 2), (1, 2)]}
        edges = eligible_edges(grid, pools, {"left->right": 1}, [(0, 1)], 1, 4)["left->right"]
        self.assertIn(((0, 0), (0, 2)), edges)
        self.assertNotIn(((1, 0), (1, 2)), edges)

    def test_capacity_failure_takes_precedence_over_next_pair(self):
        frame = {"hotspot_skew": 0, "od_matrix": {"a->b": 1}, "agent": 0,
                 "bottleneck_agent_count": 2, "pools": {}, "bottlenecks": [],
                 "minimum_distance": 1, "maximum_distance": 5,
                 "used_starts": set(), "used_goals": set(), "assignments": [], "agent_count": 3}
        with patch("lns2_selector.evaluation.pressure_capacity.failure_frame", return_value=frame), patch(
                "lns2_selector.evaluation.pressure_capacity.eligible_edges",
                return_value={"a->b": {((0, 0), (0, 1))}}):
            result = analyze_failure(None, {}, {}, "expected")
        self.assertEqual(result["decision"], "constrained_endpoint_capacity_insufficient")
        self.assertTrue(result["valid_next_pair_exists_but_was_not_sampled"])


if __name__ == "__main__":
    unittest.main()
