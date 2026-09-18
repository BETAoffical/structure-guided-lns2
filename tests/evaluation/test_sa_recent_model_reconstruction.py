import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, training_matrix
from experiments.sa_unbalanced_coverage import state_weights
from scripts import reconstruct_sa_recent_model as a
from tests.evaluation.test_sa_paired_completion import fixture, LinearEstimator
from tests.evaluation.test_sa_source_matched_prediction import fixture as prediction_fixture


class SpyEstimator(LinearEstimator):
    def __init__(self, **params):
        self.params = params

    def fit(self, x, y, sample_weight):
        self.fitted = (x,y,sample_weight)
        return self


def golden(data, held="held"):
    model = PairedCompletionModel(data["feature_names"], LinearEstimator())
    return dict(method="gbdt", held=held, seed=MODEL_PARAMS["random_state"], sklearn="1.5.0",
                train_ids=sorted(s["state_id"] for s in data["states"] if s["map_id"] != held),
                rows=a.old_predictions(model,data,held))


class RecentReconstructionTests(unittest.TestCase):
    def test_fixed_parameters_weighting_and_exact_predictions(self):
        data=fixture(); before=copy.deepcopy(data); g=golden(data)
        model,rows=a.fit_exact(data,"held",g,SpyEstimator)
        m=training_matrix(data,{"held"}); w=state_weights(data["states"],"held")
        self.assertEqual(model.estimator.params,dict(loss="squared_error",**MODEL_PARAMS))
        self.assertEqual(model.estimator.fitted,(m["x"],m["y"],[v*w[s] for v,s in zip(m["weights"],m["state_ids"])]))
        self.assertEqual(rows,g["rows"]); self.assertEqual(data,before)

    def test_held_labels_never_enter_fit(self):
        data=fixture(); g=golden(data)
        first,_=a.fit_exact(data,"held",g,SpyEstimator)
        for c in data["states"][1]["candidates"]:
            for t in c["trials"]:
                t.update(completed=True,stop="feasible",steps=1,final_conflicts=0)
        second,_=a.fit_exact(data,"held",g,SpyEstimator)
        self.assertEqual(first.estimator.fitted,second.estimator.fitted)

    def test_tiny_score_mismatch_rejected(self):
        data=fixture(); g=golden(data); g["rows"][0]["scores"]["a"]+=1e-14
        with self.assertRaisesRegex(ValueError,"prediction mismatch"):
            a.fit_exact(data,"held",g,SpyEstimator)

    def test_selection_mismatch_rejected(self):
        data=fixture(); g=golden(data); g["rows"][0]["selected"]="b"
        with self.assertRaisesRegex(ValueError,"prediction mismatch"):
            a.fit_exact(data,"held",g,SpyEstimator)

    def test_fold_leakage_and_missing_candidate_rejected(self):
        data=fixture(); g=golden(data); g["train_ids"].append("s1")
        with self.assertRaisesRegex(ValueError,"leakage"):
            a.fit_exact(data,"held",g,SpyEstimator)
        g=golden(data); del g["rows"][0]["scores"]["b"]
        with self.assertRaisesRegex(ValueError,"candidate coverage"):
            a.fit_exact(data,"held",g,SpyEstimator)

    def test_fixed_contract_no_new_label_training(self):
        cfg=json.loads((a.ROOT/"configs/sa_recent_model_reconstruction.json").read_text())
        a.check_contract(cfg)
        cfg["new_labels_allowed_in_fit"]=True
        with self.assertRaisesRegex(ValueError,"boundary"):
            a.check_contract(cfg)

    def test_model_receipt_missing_or_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); p=dict(binding="x",folds=[dict(held="m")])
            a.atomic(out/"model_receipt.json",a.sealed(dict(binding="x",files={})))
            with self.assertRaisesRegex(ValueError,"coverage"):
                a.verify_models(p,out)
            a.atomic(out/"folds/m.json",dict(a=1))
            (out/"folds/m.pkl").write_bytes(b"not-a-model")
            names=["folds/m.json","folds/m.pkl"]
            a.atomic(out/"model_receipt.json",a.sealed(dict(binding="x",files={n:a.sha256_file(out/n) for n in names})))
            a.verify_models(p,out)
            (out/"folds/m.pkl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"output changed"):
                a.verify_models(p,out)

    def test_pickle_cannot_load_before_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/"folds").mkdir(); (out/"folds/m.pkl").write_bytes(b"bad")
            p=dict(binding="x",inputs={"golden":"g"}); fold=dict(held="m",golden="golden")
            a.atomic(out/"folds/m.json",a.sealed(dict(binding="x",held="m",golden_sha256="g",model_sha256="wrong")))
            with patch.object(a.pickle,"load",side_effect=AssertionError("unsafe load")):
                with self.assertRaisesRegex(ValueError,"before load"):
                    a.load_fold(out,fold,p)

    def test_unfinished_fold_is_not_silently_retrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/"folds").mkdir(); (out/"folds/m.pkl").write_bytes(b"partial")
            with patch.object(a,"fit_exact",side_effect=AssertionError("unexpected refit")):
                with self.assertRaisesRegex(ValueError,"interrupted fold"):
                    a.reconstruct_job(({},dict(held="m"),str(out)))

    def test_missing_full_receipt_blocks_prediction_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                a.verify_models(dict(binding="x",folds=[]),Path(tmp))

    def test_evaluation_model_identity_and_determinism(self):
        records,oldp=prediction_fixture()
        old=a.previous.evaluate(records,oldp,100,7)
        newp=[dict(oldp[0],recent_model=dict(selected="b"),recent_union=dict(selected="z"))]
        first=a.evaluate_recent(records,newp,old,100,7)
        self.assertEqual(first,a.evaluate_recent(records,newp,old,100,7))
        self.assertEqual(first["metrics"]["recent_minus_historical"]["mean"],-1)
        self.assertEqual(first["metrics"][a.RECENT+"_minus_uniform"]["mean"],-.5)
        self.assertNotIn(a.previous.HISTORICAL,first["metrics"])
        self.assertFalse(first["promotion_allowed"])
        old["rows"][0]["rates"]["a"]=.9
        with self.assertRaisesRegex(ValueError,"drift"):
            a.evaluate_recent(records,newp,old,100,7)


if __name__ == "__main__":
    unittest.main()
