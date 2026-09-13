from copy import deepcopy
import unittest

from scripts import audit_sa_node_allocation as audit


def lane(costs, solved=True):
    return dict(initial_fingerprint="root", censored=False,
                transitions=[dict(generated_delta=c) for c in costs],
                conflicts=([1] * len(costs) + [0 if solved else 1]))


class NodeAllocationTest(unittest.TestCase):
    def test_uses_observed_cost_not_equal_calls(self):
        r = audit.simulate([lane([100, 1]), lane([1, 1, 1])], "nodes")
        self.assertEqual(r["used"], [1, 3])
        self.assertEqual(r["generated"], 103)
        self.assertEqual(r["winner"], "uniform_greedy")

    def test_alternation_and_losing_lane_cost(self):
        r = audit.simulate([lane([100, 1]), lane([1, 1, 1])], "alternating")
        self.assertEqual(r["used"], [2, 1])
        self.assertEqual(r["generated"], 102)

    def test_no_future_cost_affects_current_selection(self):
        sources = [lane([9, 10, 20]), lane([4, 5, 6])]
        changed = deepcopy(sources)
        changed[0]["transitions"][1]["generated_delta"] = 9999
        a, b = [audit.simulate(s, "nodes", budget=3) for s in (sources, changed)]
        self.assertEqual(a, b)

    def test_zero_cost_tie_and_budget(self):
        r = audit.simulate([lane([0]*5, False), lane([0]*5, False)], "nodes", budget=4)
        self.assertEqual(r["used"], [2, 2])
        self.assertFalse(r["feasible"])
        self.assertFalse(r["unknown"])

    def test_missing_prefix_is_unknown_not_free_switch(self):
        r = audit.simulate([lane([0], False), lane([10], False)], "nodes")
        self.assertTrue(r["unknown"])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["used"], [1, 1])

    def test_initial_feasible_and_zero_budget(self):
        r = audit.simulate([lane([]), lane([])], "nodes")
        self.assertTrue(r["feasible"])
        self.assertEqual(r["generated"], 0)
        self.assertFalse(audit.simulate([lane([2]), lane([1])], "nodes", budget=0)["feasible"])

    def test_identity_censor_invalid_cost_and_mode(self):
        for key, value in [("initial_fingerprint", "different"), ("censored", True)]:
            b = lane([1]); b[key] = value
            with self.assertRaises(ValueError): audit.simulate([lane([1]), b], "nodes")
        for cost in [-1, 1.5, True]:
            with self.assertRaises(ValueError): audit.simulate([lane([cost]), lane([1])], "nodes")
        with self.assertRaises(ValueError): audit.simulate([lane([1]), lane([1])], "unknown")

    def test_node_gate_boundary_not_rounded(self):
        def row(cost):
            base = dict(feasible=True, generated=100, repair_calls=100)
            return dict(rank_sa=base, uniform_greedy=base, alternating=base,
                        nodes=dict(feasible=True, generated=cost, repair_calls=95, unknown=False))
        self.assertTrue(audit.summarize([row(110)])["gates"]["nodes_within_110pct_uniform"])
        self.assertFalse(audit.summarize([row(111)])["gates"]["nodes_within_110pct_uniform"])


if __name__ == "__main__":
    unittest.main()
