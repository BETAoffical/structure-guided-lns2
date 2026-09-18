import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_sa_source_matched_prediction as a


def fixture():
    records = [dict(state_id="s", map_id="m", pair_ids=["a","b"], anchor_id="z",
                    values=dict(a=[True]*8,b=[False]*8,z=[True]*4+[False]*4))]
    predictions = [dict(state_id="s",map_id="m",pair_ids=["a","b"],anchor_id="z",
                        historical_model=dict(selected="a"),frozen_pool=dict(selected="b"),
                        historical_union=dict(selected="z"))]
    return records,predictions


class FrozenPredictionTests(unittest.TestCase):
    def test_coverage_does_not_substitute_missing_scores(self):
        roots=[dict(state_id="s",pair_ids=["a","b"],candidates=[dict(candidate_id=c) for c in "abz"])]
        result=a.score_coverage(roots,dict(s=dict(scores=dict(a=.1,z=.2))))
        self.assertEqual(result["complete_pairs"],0)
        self.assertEqual(result["pair_candidate_scores"],1)
        self.assertEqual(result["rows"][0]["missing_pair_ids"],["b"])

    def test_fold_all_training_maps_are_excluded(self):
        training=[dict(state_id=str(i),map_id="m"+str(i//2)) for i in range(16)]
        maps=sorted({s["map_id"] for s in training})
        folds=[dict(map_id=m,train_ids=sorted(s["state_id"] for s in training if s["map_id"]!=m),
                    train_maps=[x for x in maps if x!=m]) for m in maps]
        a.check_folds(folds,training,maps)
        folds[0]["train_ids"].append("0")
        with self.assertRaisesRegex(ValueError,"leakage"):
            a.check_folds(folds,training,maps)

    def test_pair_scoring_never_fits_and_uses_no_trials(self):
        class Estimator:
            def fit(self,*args,**kwargs):
                raise AssertionError("fit forbidden")
            def predict(self,x):
                self.x=x
                return [.8,-.4]
        class Model:
            feature_names=["x"]
            estimator=Estimator()
        state=dict(anchor_id="z",candidates=[dict(candidate_id="a",features=dict(x=2),trials="poison"),
                                             dict(candidate_id="b",features=dict(x=4),outcome="poison")])
        result=a.paired_preference(Model,state,["b","a"])
        self.assertEqual(result["selected"],"a")
        self.assertAlmostEqual(result["scores"]["a"],.6)
        self.assertEqual(Model.estimator.x,[[-2,3.0],[2,3.0]])

    def test_ties_keep_each_historical_rule(self):
        self.assertEqual(a.choose(dict(a=1,b=1),"b"),"b")
        self.assertEqual(a.choose(dict(a=1,b=1),"z"),"a")
        self.assertEqual(a.choose(dict(a=1,b=1),"b",rounded=True),"a")
        self.assertEqual(a.choose(dict(a=1,b=1+1e-14),"b",rounded=True),"a")
        with self.assertRaises(ValueError):
            a.choose(dict(a=float("nan")),"a")

    def test_analysis_keeps_full_cohort_and_separates_anchor(self):
        records,predictions=fixture()
        result=a.evaluate(records,predictions,100,7)
        self.assertEqual(result["metrics"][a.HISTORICAL+"_minus_uniform"]["mean"],.5)
        self.assertEqual(result["metrics"][a.POOL+"_minus_uniform"]["mean"],-.5)
        self.assertEqual(result["metrics"]["historical_union_minus_anchor"]["mean"],0)
        self.assertEqual(result["states"],1)
        self.assertFalse(result["promotion_allowed"])

    def test_map_equal_not_trial_or_state_count_weighted(self):
        records,predictions=fixture()
        extra=copy.deepcopy(records[0]); extra.update(state_id="s2",map_id="m2")
        extra["values"]["a"]=[False]*8; extra["values"]["b"]=[True]*8
        p2=copy.deepcopy(predictions[0]); p2.update(state_id="s2",map_id="m2")
        duplicate=copy.deepcopy(records[0]); duplicate["state_id"]="s3"
        p3=copy.deepcopy(predictions[0]); p3["state_id"]="s3"
        result=a.evaluate(records+[extra,duplicate],predictions+[p2,p3],100,7)
        self.assertEqual(result["metrics"][a.HISTORICAL+"_minus_uniform"]["mean"],0)

    def test_unknown_is_not_failure_or_removed(self):
        records,predictions=fixture()
        records[0]["values"]["b"][0]=None
        with self.assertRaisesRegex(ValueError,"incomplete"):
            a.evaluate(records,predictions)

    def test_wrong_pair_map_or_selection_rejected(self):
        for field,value in (("map_id","other"),("pair_ids",["b","a"]),("anchor_id","a")):
            records,predictions=fixture(); predictions[0][field]=value
            with self.assertRaisesRegex(ValueError,"identity"):
                a.evaluate(records,predictions)
        records,predictions=fixture(); predictions[0]["historical_model"]["selected"]="z"
        with self.assertRaisesRegex(ValueError,"outside"):
            a.evaluate(records,predictions)

    def test_bootstrap_deterministic_and_input_unchanged(self):
        records,predictions=fixture(); before=copy.deepcopy((records,predictions))
        self.assertEqual(a.evaluate(records,predictions),a.evaluate(records,predictions))
        self.assertEqual((records,predictions),before)

    def test_new_features_reproduce_stored_pre_action_features(self):
        candidates=[dict(candidate_id=c,agents=[i],score=.3) for i,c in enumerate("ab")]
        golden=dict(x=2.,**{"source.score":.3,"sa.log_decision":math.log1p(4),"sa.log_temperature":math.log1p(10)})
        names=sorted(golden)
        root=dict(state={},state_fingerprint="fp",source=dict(decision=4),control_event=dict(temperature=10),
                  candidates=candidates,feature_rows=[golden|{"history.foo":99}]*2,
                  future="must not enter features")
        class Engine:
            def __init__(self,*args,**kwargs): pass
            def realized_rows(self,candidates,**kwargs):
                return [dict(candidate_id=c["candidate_id"],features=dict(realized_dynamic=dict(x=2.))) for c in candidates],{}
        with patch("experiments.repair_collection.state_fingerprint",return_value="fp"):
            output=a.feature_state(root,candidates,names,Engine)
            self.assertEqual(output["a"]["features"],golden)
            root["feature_rows"][0]=golden|dict(x=3.)
            with self.assertRaisesRegex(ValueError,"golden"):
                a.feature_state(root,candidates,names,Engine)

    def test_changed_or_missing_feature_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            plan=dict(binding="b",roots=[dict(state_id="s")])
            a.atomic(out/"features/s.json",dict(before=True))
            receipt=dict(binding="b",files={"features/s.json":a.sha256_file(out/"features/s.json")})
            a.atomic(out/"feature_receipt.json",receipt)
            a.verify_features(plan,out)
            a.atomic(out/"features/s.json",dict(before=False))
            with self.assertRaisesRegex(ValueError,"features changed"):
                a.verify_features(plan,out)
            a.atomic(out/"feature_receipt.json",dict(binding="b",files={}))
            with self.assertRaisesRegex(ValueError,"coverage"):
                a.verify_features(plan,out)


if __name__=="__main__":
    unittest.main()
