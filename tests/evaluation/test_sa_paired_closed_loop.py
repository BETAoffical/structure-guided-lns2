import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from experiments._common import read_json, json_fingerprint
from experiments.sa_paired_closed_loop import choose, subset, gate, stop_reason, portable_model, portable_payload
from experiments.sa_paired_completion import PairedCompletionModel, MODEL_PARAMS
from scripts.run_sa_paired_closed_loop import features_for, once, receipt_valid


class Estimator:
    def predict(self, vectors):
        return [v[0] for v in vectors]


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.pool=[dict(candidate_id="a",agents=[2],score=1),dict(candidate_id="b",agents=[9],score=0)]
        self.features=[{"x":0.},{"x":1.}]
        self.model=PairedCompletionModel(["x"],Estimator())

    def test_anchor_vs_paired(self):
        self.assertEqual(choose("frozen",self.pool,"a",self.model,self.features,[2,9],"k",3)["selected"],"a")
        self.assertEqual(choose("paired",self.pool,"a",self.model,self.features,[2,9],"k",3)["selected"],"b")

    def test_uniform_deterministic_order_independent(self):
        a=choose("uniform",self.pool,"a",self.model,self.features,[2,9],"k",3)
        b=choose("uniform",self.pool[::-1],"a",self.model,self.features[::-1],[2,9],"k",3)
        self.assertEqual(a,b)

    def test_tie_prefers_anchor(self):
        f=[{"x":0.},{"x":0.}]
        self.assertEqual(choose("paired",self.pool,"a",self.model,f,[2,9],"k",3)["selected"],"a")

    def test_single_candidate(self):
        for arm in ("frozen","uniform","paired"):
            self.assertEqual(choose(arm,self.pool[:1],"a",self.model,self.features[:1],[2,9],"k",3)["selected"],"a")
        self.assertEqual(subset(self.pool[:1],"a","k",3,4),["a"])

    def test_unknown_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            choose("paired",self.pool,"a",self.model,self.features,[2],"k",3)
        with self.assertRaises(ValueError):
            subset([self.pool[0],self.pool[0]],"a","k",3,4)

    def test_subset_outcome_blind(self):
        original=subset(self.pool,"a","k",3,4)
        other=copy.deepcopy(self.pool)
        other[1].update(score=1e9,completed=True)
        self.assertEqual(original,subset(other,"a","k",3,4))

    def test_stop_priority(self):
        cfg=dict(node_budget=10,max_decisions=4,episode_seconds=5)
        self.assertIsNone(stop_reason({"feasible":False},3,9,4,cfg))
        self.assertEqual(stop_reason({"feasible":False},4,9,4,cfg),"decision_budget")
        self.assertEqual(stop_reason({"feasible":False},4,11,8,cfg),"node_budget")
        self.assertEqual(stop_reason({"feasible":True},4,11,8,cfg),"feasible")

    def test_gate_no_posthoc_rounding(self):
        cfg=dict(minimum_success_gain=.05,minimum_map_nonlosses=6,maximum_common_success_node_ratio=1.1)
        c=dict(frozen=dict(delta=.04999,ci95=[0,.1],map_nonlosses=8),uniform=dict(delta=.1,ci95=[0,.2]))
        self.assertFalse(gate(c,1,0,cfg)["passed"])
        c["frozen"]["delta"]=.05
        self.assertTrue(gate(c,1,0,cfg)["passed"])
        self.assertFalse(gate(c,1,1,cfg)["passed"])
        self.assertFalse(gate(c,None,0,cfg)["passed"])

    def test_state_prepared_before_features(self):
        class Engine:
            def prepare(self,state):
                self.state=state
            def realized_rows(self,candidates,state_hash):
                return [{"features":{"realized_dynamic":{"x":self.state["x"]}}}],{}
        class History:
            def features(self,*args):
                return {}
        from unittest.mock import patch
        engine=Engine()
        with patch("scripts.run_sa_paired_closed_loop.profile_features",side_effect=lambda row,profile:row["base"]):
            first=features_for({"x":1},self.pool[:1],engine,History(),1,"a")
            second=features_for({"x":2},self.pool[:1],engine,History(),1,"b")
        self.assertEqual(first[0]["x"],1)
        self.assertEqual(second[0]["x"],2)

    def test_atomic_resume_rejects_changed_results(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.json"
            once(p,{"x":1})
            once(p,{"x":1})
            with self.assertRaises(ValueError):
                once(p,{"x":2})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"Windows sklearn test")
    def test_sklearn_python_portable_exact(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        x=[[i/100.,.5] for i in range(-50,50)]
        estimator=HistGradientBoostingRegressor(loss="squared_error",**MODEL_PARAMS).fit(x,[r[0] for r in x])
        original=PairedCompletionModel(["x"],estimator)
        payload=portable_payload(original)
        portable=portable_model(payload,["x"],native=False)
        s=dict(anchor_id="a",agent_ids=[2,9],candidates=[dict(c,features=f) for c,f in zip(self.pool,self.features)])
        self.assertEqual(original.rank(s),portable.rank(s))
        payload["baseline"]+=1
        with self.assertRaises(ValueError):
            portable_model(payload,["x"])

    def test_contract_has_no_ttf_or_automatic_promotion(self):
        cfg=read_json(Path(__file__).resolve().parents[2]/"configs/sa_paired_closed_loop.json")
        self.assertFalse(cfg["formal_ttf"])
        self.assertFalse(cfg["automatic_promotion"])
        self.assertEqual(cfg["candidate_limit"],4)
        self.assertEqual(cfg["arms"],["frozen","uniform","paired"])

    @unittest.skipUnless(importlib.util.find_spec("lns2_env"),"frozen WSL native fixture")
    def test_native_micro_paths_and_each_arm(self):
        import lns2_env
        from experiments.online_feature_engine import OnlineFeatureEngine
        from experiments.sa_history_selector import History
        from scripts import run_sa_path_quality as q
        from scripts.run_feedback_exploration_diagnostics import validate_final
        with tempfile.TemporaryDirectory() as d:
            m,s=Path(d)/"tiny.map",Path(d)/"tiny.scen"
            m.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n",encoding="utf8")
            s.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n",encoding="utf8")
            initial=[]
            for arm in ("frozen","uniform","paired"):
                env=lns2_env.LNS2RepairEnv(str(m),str(s),2)
                state=q._plain(env.reset_paths([[0,1,2,3],[3,2,1,0]],seed=7))
                initial.append(q.state_fingerprint(state))
                c=[dict(candidate_id="one",agents=[0,1],score=0,provenance=[])]
                history=History(state)
                # Native fixture exercises actual state/feature and SA transitions.
                engine=OnlineFeatureEngine(state,backend="native")
                f=features_for(state,c,engine,history,q.temperature(0),q.state_fingerprint(state))
                self.assertEqual(len(f[0]),127)
                chosen=choose(arm,c,"one",self.model,f,[0,1],"tiny",3)
                self.assertEqual(chosen["selected"],"one")
                action=dict(mode="explicit_neighborhood",agents=[0,1],random_seed=77)
                raw=q._plain(env.step_experimental_pp(action,1.,"annealed",q.temperature(0),.1))
                q.validate_transition(state,raw["observation"],raw["metrics"],[0,1],"annealed",q.temperature(0),.1)
                validate_final(raw["observation"])
                self.assertEqual(q.state_fingerprint(q.apply_state_delta(state,q.encode_state_delta(state,raw["observation"]))),q.state_fingerprint(raw["observation"]))
            self.assertEqual(len(set(initial)),1)


if __name__=="__main__":
    unittest.main()
