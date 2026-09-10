import unittest
from scripts.audit_witness_path_ablation import (
    compile_choices, enumerate_choices, full_edges, pair_conflicts, selected_edges,
)


class WitnessPathTests(unittest.TestCase):
    def test_vertex_swap_following_and_terminal_wait(self):
        for left,right,expected in [([0,1],[2,1],True), ([0,1],[1,0],True),
                                    ([0,1],[1,2],False), ([0,1],[3,2,1],True)]:
            self.assertEqual(pair_conflicts(left,right), expected)
            self.assertEqual(bool(full_edges({2:left,19:right})), expected)

    def test_all_mixed_masks_match_independent_reconstruction(self):
        old = {2:[0,1,2], 19:[2,1,0], 35:[3,4,5], 80:[6]}
        new = {2:[0,0,1,2], 19:[2,3,0], 35:[3,1,5], 80:[6]}
        variables,fixed,table,_,target = compile_choices(old,new)
        for mask in range(1<<len(variables)):
            mixed = dict(old)
            for i,a in enumerate(variables):
                if mask>>i&1:
                    mixed[a]=new[a]
            self.assertEqual(selected_edges(mask,fixed,table),full_edges(mixed))
        summary,_ = enumerate_choices(variables,fixed,table,target)
        self.assertEqual(summary['combinations'],8)
        self.assertGreaterEqual(summary['exact_post_edges_count'],1)

    def test_unchanged_witness_has_empty_minimal_set(self):
        variables,fixed,table,_,target = compile_choices({7:[0],20:[1]},{7:[0],20:[1]})
        summary,_ = enumerate_choices(variables,fixed,table,target)
        self.assertEqual(summary['exact_minimal_examples'],[[]])

    def test_agent_identity_and_budget_are_strict(self):
        with self.assertRaises(ValueError):
            compile_choices({1:[0]},{2:[0]})
        with self.assertRaises(ValueError):
            compile_choices({i:[i] for i in range(17)},{i:[i,i] for i in range(17)})


if __name__ == '__main__':
    unittest.main()
