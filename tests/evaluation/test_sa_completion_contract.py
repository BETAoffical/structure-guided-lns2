import copy
import unittest

from experiments.sa_completion_contract import signatures, split_diagnostics, compare_branches, pool_signature, aggregate
from lns2_selector.runtime.online_selection import proposal_random_seed


def state():
    return dict(initialized=True,initial_solution_complete=True,feasible=False,done=False,iteration=1,
                rows=2,cols=2,sum_of_costs=2,num_of_colliding_pairs=1,low_level=dict(generated=10),
                obstacles=[0]*4,conflict_edges=[[3,9]],agents=[dict(id=3,path=[0,1]),dict(id=9,path=[2,1])])


def trace(cid,full="same",counter_free="path",next_pool=None,completed=False):
    sig=dict(full=full,counter_free=counter_free,physical=counter_free,conflicts=1)
    return dict(candidate_id=cid,trial=0,states=[sig],next_pool=next_pool,completed=completed)


class CompletionContractTests(unittest.TestCase):
    def test_counter_views_are_only_for_audit(self):
        s=state(); original=copy.deepcopy(s); a=signatures(s)
        t=copy.deepcopy(s); t["low_level"]["generated"]+=1; b=signatures(t)
        self.assertEqual(s,original)
        self.assertNotEqual(a["full"],b["full"])
        self.assertEqual(a["counter_free"],b["counter_free"])
        self.assertNotEqual(proposal_random_seed("task",1,a["full"],1,3,"Target",8,0),
                            proposal_random_seed("task",1,b["full"],1,3,"Target",8,0))
        t["iteration"]+=1
        self.assertNotEqual(a["counter_free"],signatures(t)["counter_free"])
        self.assertEqual(a["physical"],signatures(t)["physical"])
        t["agents"][0]["path"]=[0,2,1]
        self.assertNotEqual(a["physical"],signatures(t)["physical"])

    def test_all_35_partitions_no_winner_selection(self):
        r=split_diagnostics({"a":[1]*8,"b":[0]*8},"b")
        self.assertEqual(len(r["partitions"]),35)
        self.assertEqual(r["cross_half_rate"],1)
        self.assertEqual(r["mean_jaccard"],1)
        self.assertTrue(all(p["strict_pairs"]==p["concordant_pairs"]==1 for p in r["partitions"]))

    def test_single_partition_can_mislead(self):
        r=split_diagnostics({"a":[1]*4+[0]*4,"b":[0]*4+[1]*4},"a")
        self.assertEqual(r["partitions"][0]["jaccard"],0)
        self.assertEqual(r["partitions"][0]["cross_half_rate"],0)
        self.assertGreater(r["cross_half_rate"],0)
        self.assertGreater(r["variance_pairs"][0]["paired_variance"],r["variance_pairs"][0]["independent_variance"])

    def test_constant_labels_have_no_variance_or_choice_opportunity(self):
        r=split_diagnostics({"a":[0]*8,"b":[0]*8},"b")
        self.assertFalse(r["informative"])
        self.assertEqual((r["cross_half_rate"],r["mean_jaccard"]),(0,1))
        self.assertEqual(r["variance_pairs"][0]["paired_variance"],0)
        with self.assertRaises(ValueError):
            split_diagnostics({"a":[1]*7,"b":[0]*8},"a")

    def test_candidate_order_and_trial_permutation_invariance(self):
        v={"a":[1,0,1,1,0,0,1,0],"b":[0,1,1,1,0,0,0,0]}
        r=split_diagnostics(v,"b")
        p=split_diagnostics({k:list(reversed(v[k])) for k in reversed(v)},"b")
        self.assertAlmostEqual(r["cross_half_rate"],p["cross_half_rate"])
        self.assertAlmostEqual(r["mean_jaccard"],p["mean_jaccard"])

    def test_counters_only_does_not_mean_full_state_or_causal_proof(self):
        p=dict(full="a",members=[(3,9)],selected=(3,9),proposal_seeds=[1])
        q=dict(full="b",members=[(3,)],selected=(3,),proposal_seeds=[2])
        r=compare_branches(trace("a","one",next_pool=p),trace("b","two",next_pool=q,completed=True))
        self.assertTrue(r["counters_only_with_continuation"] and r["next_members_differ"])
        self.assertTrue(r["different_completion"])
        self.assertFalse(r["same_first_full_state"])
        with self.assertRaisesRegex(ValueError,"same full state"):
            compare_branches(trace("a",next_pool=p),trace("b",next_pool=q))

    def test_terminal_and_full_state_merges_not_invented(self):
        a=trace("a","a","a"); b=trace("b","b","b")
        a["states"].append(dict(full="x",counter_free="p",physical="p",conflicts=0))
        b["states"].append(dict(full="x",counter_free="p",physical="p",conflicts=0))
        r=compare_branches(a,b)
        self.assertFalse(r["later_nonterminal_merge_after_first_difference"])
        a["states"][1]["conflicts"]=b["states"][1]["conflicts"]=1
        r=compare_branches(a,b)
        self.assertTrue(r["later_nonterminal_full_merge"])

    def test_pool_membership_separate_from_scores_and_provenance(self):
        e=dict(pool=[dict(agents=[9,3],score=1,proposal_seeds=[2])],action=dict(agents=[3,9]))
        f=copy.deepcopy(e); f["pool"][0]["score"]=7
        self.assertEqual(pool_signature(e)["members"],pool_signature(f)["members"])
        self.assertNotEqual(pool_signature(e)["full"],pool_signature(f)["full"])

    def test_aggregate_keeps_partitions_distinct_from_root_count(self):
        split=split_diagnostics({"a":[1]*8,"b":[0]*8},"b")
        pair=compare_branches(trace("a"),trace("b"))
        root=dict(id="root",map_id="map",split=split,pairs=[pair],
                  branches=[dict(unchanged_first_paths=False,first_accepted=True,physical_recurrence=False)])
        summary=aggregate([root])
        self.assertEqual((summary["roots"],summary["maps"],len(summary["partitions"])),(1,1,35))
        self.assertEqual(summary["cross_half_rate"],1)
        self.assertEqual(summary["informative_jaccard"],1)
        self.assertEqual(summary["strict_pair_partition_comparisons"],35)
        with self.assertRaises(ValueError):
            aggregate([root,root])


if __name__ == "__main__":
    unittest.main()
