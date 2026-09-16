import copy
from pathlib import Path
import tempfile
import unittest

from experiments.sa_history_information import (OrderedHistory, TEMPORAL, RESOURCE,
    bag_features, ordered_features, future_labels, profile_features, episode_weights, metric, peers)
from scripts.audit_sa_history_information import fit_fold, comparison, atomic, save_result, receipt_result, source_finished


def state():
    return dict(num_of_colliding_pairs=1, conflict_edges=[[3, 9]], agents=[
        dict(id=3,path=[0,1,2],goal=2), dict(id=9,path=[4,1,0],goal=0),
        dict(id=20,path=[7,8,5],goal=5)])


def records():
    return [{k:float(i) for k in TEMPORAL+RESOURCE} for i in range(32)]


def rows():
    return [dict(id=f"s{i}",episode=f"e{i//2}",map_id=f"m{i//4}",
                 base={"x":i%3,"history.count":i%2},records=records(),
                 labels={"completion":i%2,"sustained_progress":i%2}) for i in range(12)]


class InformationTests(unittest.TestCase):
    def test_valid_unsolved_source_retained(self):
        self.assertTrue(source_finished({"status":"completed"}))
        self.assertTrue(source_finished({"status":"no_feasible_solution"}))
        self.assertFalse(source_finished({"status":"error"}))
        self.assertFalse(source_finished({"status":"running"}))

    def test_prefix_order_separates_same_bag(self):
        r=records()
        self.assertEqual(bag_features(r),bag_features(list(reversed(r))))
        self.assertNotEqual(ordered_features(r),ordered_features(list(reversed(r))))

    def test_shuffle_determinism_and_dimension(self):
        r=records()
        self.assertEqual(ordered_features(r,True,123),ordered_features(r,True,123))
        self.assertEqual(ordered_features(r,True).keys(),ordered_features(r,True,123).keys())
        self.assertNotEqual(ordered_features(r,True),ordered_features(r,True,123))

    def test_missing_history_is_explicit(self):
        f=ordered_features(records()[:2])
        self.assertEqual(f["ordered.0-1.available"],1)
        self.assertEqual(f["ordered.1-4.available"],1/3)
        self.assertEqual(f["ordered.16-32.available"],0)

    def test_noncontinuous_ids_and_relevant_peers(self):
        s=state()
        self.assertEqual(peers(s,[3]),({9},{9}))
        h=OrderedHistory()
        a=copy.deepcopy(s)
        a["agents"][1]["path"]=[4,4,1,0]
        event=dict(metrics=dict(neighborhood=[9],conflicts_before=1,conflicts_after=1,
                                pp_rolled_back=False,acceptance_evaluated=True))
        h.observe(s,event,a,1)
        r=h.records(a,dict(agents=[3]))[0]
        self.assertEqual(r["boundary_changed"],1)
        self.assertEqual(r["selected_changed"],0)
        self.assertEqual(r["boundary_path_difference"],0)
        self.assertEqual(h.records(s,dict(agents=[3]))[0]["boundary_path_difference"],1)

    def test_unknown_agent_rejected(self):
        with self.assertRaises(ValueError):
            OrderedHistory().records(state(),dict(agents=[99]))

    def test_candidate_order_invariant(self):
        h=OrderedHistory()
        self.assertEqual(h.records(state(),dict(agents=[3,9])),h.records(state(),dict(agents=[9,3])))

    def test_completion_absorbing(self):
        self.assertEqual(future_labels([4,3,0],1,3),dict(completion=1,sustained_progress=1))

    def test_censored_not_failure(self):
        self.assertIsNone(future_labels([4,3,2],1,3))

    def test_one_step_drop_not_sustained(self):
        c=[1]+[5]*31
        self.assertEqual(future_labels(c,0,4)["sustained_progress"],0)
        c=[5]*24+[3]*8
        self.assertEqual(future_labels(c,0,4)["sustained_progress"],1)

    def test_sustained_relative_to_best_not_current(self):
        self.assertEqual(future_labels([3]*32,0,2)["sustained_progress"],0)

    def test_future_boundary(self):
        self.assertEqual(future_labels([5]*32+[0],0,6)["completion"],0)
        self.assertEqual(future_labels([5]*32+[0],1,6)["completion"],1)

    def test_illegal_post_completion(self):
        with self.assertRaises(ValueError):
            future_labels([0,2],0,4)

    def test_inputs_independent_of_outcome_and_identity(self):
        a=rows()[0]
        b=copy.deepcopy(a)
        b["labels"]={"completion":1,"sustained_progress":1}
        b["map_id"]="unseen-map"
        b["runtime"]=1e99
        for profile in ("dynamic","aggregate","ordered","resource_ordered","resource_bag","shuffled_0"):
            self.assertEqual(profile_features(a,profile),profile_features(b,profile))

    def test_equal_episode_weight(self):
        r=rows()[:4]
        r[0]["episode"]=r[1]["episode"]=r[2]["episode"]="one"
        w=episode_weights(r)
        self.assertAlmostEqual(sum(w[:3]),w[3])
        self.assertAlmostEqual(sum(w),len(r))

    def test_metric_probability_validation(self):
        with self.assertRaises(ValueError):
            metric(rows(),[float("nan")]*12,"completion")

    def test_map_bootstrap_deterministic(self):
        r=rows()
        a={x["id"]:.8 if x["labels"]["completion"] else .2 for x in r}
        b={x["id"]:.5 for x in r}
        c=comparison(r,a,b,"completion",100,1)
        self.assertEqual(c,comparison(r,a,b,"completion",100,1))
        self.assertLess(c["ci95"][1],0)

    def test_held_map_labels_not_used_in_fit(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn unavailable")
        r=rows()
        job=dict(rows=r,map_id="m2",profile="dynamic",target="completion",
                 model=dict(max_iter=3,min_samples_leaf=2,random_state=1,early_stopping=False))
        a=fit_fold(job)
        for row in r:
            if row["map_id"]=="m2":
                row["labels"]["completion"]=1-row["labels"]["completion"]
        self.assertEqual(a,fit_fold(job))

    def test_episode_leakage_rejected(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn unavailable")
        r=rows()
        r[-1]["episode"]=r[0]["episode"]
        with self.assertRaisesRegex(ValueError,"episode leakage"):
            fit_fold(dict(rows=r,map_id="m2",profile="dynamic",target="completion",model={}))

    def test_atomic_receipt_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"r.json"
            save_result(path,"a",{"x":1})
            self.assertEqual(receipt_result(path,"a"),{"x":1})
            with self.assertRaises(ValueError):
                save_result(path,"a",{})
            atomic(path,{"binding":"a","result":{"x":2}})
            with self.assertRaises(ValueError):
                receipt_result(path,"a")


if __name__=="__main__":
    unittest.main()
