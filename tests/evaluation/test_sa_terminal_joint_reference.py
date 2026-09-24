import copy
import unittest
from scripts import check_sa_terminal_joint_reference as p
from tests.evaluation.test_local_path_compatibility import state,agent


class JointReferenceTests(unittest.TestCase):
    def test_three_agents_noncontinuous_and_order_invariance(self):
        s = state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0]),agent(105,[3,4,5])])
        original = copy.deepcopy(s)
        a = p.solve(s,[17,93,105],p.ref.Budget(seconds=10,max_expanded=100000))
        self.assertEqual(a["status"],"feasible")
        self.assertTrue(p.ref.validate_witness(s,a["paths"])["globally_feasible"])
        self.assertEqual(s,original)
        self.assertEqual(a,p.solve(dict(s,agents=list(reversed(s["agents"]))),[105,93,17],p.ref.Budget()))

    def test_matches_old_pair_reference(self):
        s = state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        a = p.solve(s,[17,93],p.ref.Budget())
        b = p.ref.cbs_pair(s,[17,93],p.ref.Budget())
        self.assertEqual(a["status"],b["status"])
        self.assertEqual(sum(len(v)-1 for v in a["paths"].values()),sum(len(v)-1 for v in b["paths"].values()))

    def test_fixed_outsider_exhaustion_is_scoped(self):
        s = state(1,4,[agent(17,[0,1,2]),agent(93,[2,1,0]),agent(52,[1])])
        a = p.solve(s,[17,93],p.ref.Budget())
        self.assertEqual(a["status"],"infeasible")
        self.assertEqual(a["reason"],"single_agent_fixed_outside_exhausted")
        self.assertIn(a["agent"],[17,93])

    def test_exhausted_budget_unknown_not_infeasible(self):
        s = state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        for budget in (p.ref.Budget(seconds=0),p.ref.Budget(max_expanded=1)):
            a = p.solve(s,[17,93],budget)
            self.assertEqual(a["status"],"unknown")

    def test_external_conflicts_not_silently_counted_as_full_success(self):
        s = state(2,4,[agent(7,[0,1]),agent(8,[1,0]),agent(31,[6,7]),agent(32,[7,6])])
        a = p.solve(s,[7,8],p.ref.Budget())
        self.assertEqual(a["status"],"feasible")
        self.assertFalse(a["witness_validation"]["globally_feasible"])
        self.assertEqual(a["witness_validation"]["external_conflict_pairs"],[[31,32]])

    def test_unknown_duplicate_and_large_sets_rejected(self):
        s = state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        for selected in ([],[17,17],[17,99],list(range(6))):
            with self.assertRaises(ValueError): p.solve(s,selected,p.ref.Budget())


if __name__ == "__main__": unittest.main()
