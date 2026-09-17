import copy
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import SCHEMA
from experiments.sa_low_complexity_ranker import (
    METHODS, coarse_labels, completion_rates, fit, predict, prepare_matrix, summarize,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    states = []
    for i, m in enumerate(("m0", "m0", "m1", "held")):
        candidates = []
        for j, successes in enumerate((7, 4, 2, 0)):
            candidates.append(dict(candidate_id=str(j), agents=[10+j],
                features={"realized.signal":float(3-j), "sa.log_temperature":2.},
                trials=[dict(trial=t, randomization_key=json_fingerprint(["trial", t]),
                    stop="feasible" if t<successes else "horizon", steps=1 if t<successes else 32,
                    final_conflicts=0 if t<successes else 1, completed=t<successes) for t in range(8)]))
        states.append(dict(state_id="s"+str(i), map_id=m, episode="e"+str(i), decision=16,
            state_fingerprint=json_fingerprint(["state", i]), history_fingerprint=json_fingerprint(["history", i]),
            agent_ids=[10,11,12,13], anchor_id="1", candidates=candidates))
    return dict(schema=SCHEMA, role="prospective_development", sampling="outcome_blind_same_state",
        source_kind="registered_prospective_collection", continuation_binding="a"*64,
        horizon=32, trial_count=8, feature_names=["realized.signal", "sa.log_temperature"], states=states)


def model():
    return dict(feature_names=["realized.signal", "sa.log_temperature"],
                mean=[0.,0.], scale=[1.,1.], coefficients=[1.,0.])


class LowComplexityTests(unittest.TestCase):
    def test_frozen_scope(self):
        cfg = json.loads((ROOT/"configs/sa_low_complexity_ranker.json").read_text())
        self.assertEqual((cfg["states"],cfg["maps"],cfg["ridge_alpha"],cfg["svm_C"]),(47,8,1.,1.))
        self.assertEqual(cfg["new_solver_calls"],0)
        self.assertFalse(cfg["automatic_promotion"])
        self.assertEqual(METHODS,("linear_difference","coarse_rank_svm"))

    def test_coarse_ties_and_quantile_groups(self):
        self.assertEqual(coarse_labels(dict(a=0.,b=.25,c=.5,d=1.)),dict(a=0,b=0,c=1,d=2))
        self.assertEqual(len(set(coarse_labels(dict(a=.5,b=.5,c=.5,d=.5)).values())),1)
        self.assertEqual(coarse_labels(dict(a=0.,b=0.,c=1.,d=1.)),dict(a=0,b=0,c=2,d=2))

    def test_map_balancing_state_weights_and_reverse_pairs(self):
        for method in METHODS:
            matrix = prepare_matrix(fixture(),"held",method)
            self.assertEqual(matrix["train_maps"],["m0","m1"])
            totals = defaultdict(float)
            for sid,w in zip(matrix["owners"],matrix["weights"]):
                totals[sid] += w
            for sid,w in matrix["state_weights"].items():
                self.assertAlmostEqual(totals[sid],w)
            self.assertAlmostEqual(totals["s0"]+totals["s1"],totals["s2"])
            np.testing.assert_array_equal(matrix["x"][::2],-matrix["x"][1::2])
            np.testing.assert_array_equal(matrix["y"][::2],-matrix["y"][1::2])
            self.assertEqual(len(matrix["owners"]),matrix["constraints"]["s0"]*3)

    def test_holdout_features_and_labels_do_not_change_training(self):
        d = fixture()
        before = prepare_matrix(d,"held",METHODS[0])
        for c in d["states"][-1]["candidates"]:
            c["features"]["realized.signal"] = 1e8
            c["trials"] = copy.deepcopy(d["states"][0]["candidates"][0]["trials"])
        after = prepare_matrix(d,"held",METHODS[0])
        for key in ("x","y","mean","scale","weights"):
            np.testing.assert_array_equal(before[key],after[key])

    def test_tie_state_has_no_svm_constraints_but_stays_in_evaluation(self):
        d = fixture()
        for c in d["states"][0]["candidates"]:
            c["trials"] = copy.deepcopy(d["states"][1]["candidates"][1]["trials"])
        svm = prepare_matrix(d,"held",METHODS[1])
        ridge = prepare_matrix(d,"held",METHODS[0])
        self.assertEqual(svm["constraints"]["s0"],0)
        self.assertEqual(ridge["constraints"]["s0"],12)
        self.assertIn("s0",svm["train_ids"])

    def test_censor_and_invalid_holdout_rejected(self):
        d = fixture()
        d["states"][0]["candidates"][0]["trials"][0].update(
            stop="wall_safety",steps=2,final_conflicts=1,completed=None)
        with self.assertRaisesRegex(ValueError,"censor"):
            prepare_matrix(d,"held",METHODS[0])
        with self.assertRaisesRegex(ValueError,"held map"):
            prepare_matrix(fixture(),"missing",METHODS[0])

    def test_prediction_has_no_label_inputs_order_or_irrelevant_candidate_dependence(self):
        s = fixture()["states"][0]
        expected = predict(model(),s)
        s["candidates"].reverse()
        for c in s["candidates"]:
            c["trials"] = []
            c["runtime"] = 1e9
            c["future_paths"] = [999]
        self.assertEqual(expected,predict(model(),s))
        s["candidates"] = [c for c in s["candidates"] if c["candidate_id"]!="3"]
        self.assertEqual(predict(model(),s)["selected"],expected["selected"])
        s["candidates"][0]["features"]["outcome.completion"] = 1.
        with self.assertRaisesRegex(ValueError,"schema"):
            predict(model(),s)

    def test_scalar_transitivity_and_anchor_tie(self):
        s = fixture()["states"][0]
        scores = predict(model(),s)["scores"]
        self.assertAlmostEqual((scores["0"]-scores["1"])+(scores["1"]-scores["2"]),scores["0"]-scores["2"])
        m = model()
        m["coefficients"] = [0.,0.]
        self.assertEqual(predict(m,s)["selected"],s["anchor_id"])

    def test_positive_exploration_does_not_require_all_maps_or_promotion(self):
        d = fixture()
        predictions = {s["state_id"]:{"gbdt":"1",METHODS[0]:"0",METHODS[1]:"1"} for s in d["states"]}
        predictions["s3"][METHODS[0]] = "2"
        cfg = dict(seed=42,bootstrap=500)
        report = summarize(d,predictions,cfg)
        self.assertEqual(report,summarize(d,predictions,cfg))
        self.assertEqual(report["positive_point_estimates"],[METHODS[0]])
        self.assertEqual(report["contrasts"][METHODS[0]]["frozen"]["losses"],1)
        self.assertFalse(report["promotion"])
        self.assertFalse(report["independent_confirmation"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"sklearn unavailable; no installation")
    def test_fit_repeatable_and_json_reload_exact(self):
        import sklearn
        if sklearn.__version__!="1.5.0":
            self.skipTest("registered sklearn required")
        cfg = json.loads((ROOT/"configs/sa_low_complexity_ranker.json").read_text())
        d = fixture()
        for method in METHODS:
            a,b = fit(d,"held",method,cfg),fit(d,"held",method,cfg)
            self.assertEqual(a,b)
            reloaded = json.loads(json.dumps(a["model"]))
            self.assertNotIn("held",reloaded["train_maps"])
            self.assertFalse(reloaded["runtime_integration_allowed"])
            self.assertEqual(predict(reloaded,d["states"][-1])["selected"],"0")
            self.assertEqual(predict(reloaded,d["states"][-1]),{k:a["rows"][-1][k] for k in ("selected","scores")})


if __name__=="__main__":
    unittest.main()
