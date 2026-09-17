import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.sa_linear_closed_loop import ARMS,LinearRanker,training_arrays,fit_full,choose,summarize
from experiments.sa_low_complexity_ranker import prepare_matrix
from tests.evaluation.test_sa_low_complexity_ranker import fixture
from scripts.run_sa_paired_closed_loop import once


def linear(names=None):
    names=names or ["realized.signal","sa.log_temperature"]
    return LinearRanker(dict(schema="lns2.sa.linear_h32_runtime_probe.v1",feature_names=names,
        mean=[0.]*len(names),scale=[1.]*len(names),coefficients=[1.]+[0.]*(len(names)-1)))


def episodes():
    result=[]
    for i in range(4):
        for arm in ARMS:
            success=arm!="frozen" or i<2
            result.append(dict(pair_id=str(i),map_id="m"+str(i//2),arm=arm,initial_fingerprint=str(i),
                success=success,stop="feasible" if success else "decision_budget",decisions=20 if success else 512,
                generated=100,physical_revisits=0))
    return result


class LinearLoopTests(unittest.TestCase):
    def test_case_adapter_and_job_error_preserve_identity(self):
        from scripts.run_sa_linear_closed_loop import runtime_case,job_failure
        from lns2_selector.evaluation.path_quality_execution import controller_job
        case=dict(task_id="t",map_id="m",family="warehouse",static_audit=dict(agent_count=2),
            files=dict(map_file="a.map",scenario_file="a.scen",task_file="a.json"))
        ready=runtime_case(case)
        self.assertNotIn("status",case)
        self.assertEqual({k:v for k,v in ready.items() if k!="status"},case)
        template=dict(proposal={},environment=dict(replan_algorithm="PP",use_sipp=True),frozen_models="frozen",model_registration={})
        job=controller_job(Path("."),ready,dict(controller="official_adaptive",task_id="t",solver_seed=7,budget_seconds=1),template,Path("out"),"binding")
        self.assertEqual(job["row"]["agent_count"],2)
        self.assertEqual(job_failure(dict(job_id="test"),"error","original cause"),
                         dict(job_id="test",status="error",error="original cause"))
        with self.assertRaises(ValueError):
            runtime_case(dict(case,status="quarantined"))

    def test_full_training_matches_registered_fold_recipe(self):
        d=fixture()
        old=prepare_matrix(d,"held","linear_difference")
        new=training_arrays(d,"held")
        for a,b in (("x","linear"),("y","y"),("weights","weights"),("mean","mean"),("scale","scale")):
            np.testing.assert_array_equal(old[a],new[b])
        self.assertEqual(old["state_weights"],new["state_weights"])
        full=training_arrays(d)
        self.assertEqual(len(full["train_ids"]),4)
        totals={m:sum(full["state_weights"][s["state_id"]] for s in d["states"] if s["map_id"]==m) for m in full["train_maps"]}
        self.assertTrue(all(abs(v-4/3)<1e-12 for v in totals.values()))

    def test_portable_order_ties_and_no_outcome_access(self):
        s=fixture()["states"][0]
        model=linear()
        expected=model.rank(s)
        for c in s["candidates"]:
            c.pop("trials")
            c["outcome"]=1e10
        s["candidates"].reverse()
        self.assertEqual(expected,model.rank(s))
        model.payload["coefficients"]=[0.,0.]
        self.assertEqual(model.rank(s)["selected"],s["anchor_id"])

    def test_choice_identity_and_three_arms(self):
        s=fixture()["states"][0]
        cs=s["candidates"]
        models=dict(gbdt=linear(),linear=linear())
        fs=[c["features"] for c in cs]
        self.assertEqual(choose("frozen",cs,"1",models,fs,s["agent_ids"])["selected"],"1")
        for arm in ("gbdt","linear"):
            self.assertEqual(choose(arm,cs,"1",models,fs,s["agent_ids"])["selected"],"0")
        with self.assertRaises(ValueError):
            choose("linear",cs,"1",models,fs,[10])

    def test_censoring_bounds_do_not_become_failure_labels(self):
        rows=episodes()
        one=next(r for r in rows if r["pair_id"]=="3" and r["arm"]=="linear")
        one.update(success=False,stop="wall_safety")
        r=summarize(rows,dict(master_seed=1,bootstrap=100))
        self.assertEqual(r["summary"]["linear"]["censored"],1)
        self.assertEqual(r["decision"],"resource_censored_inconclusive")
        self.assertEqual(r["contrasts"]["linear_vs_frozen"]["censor_delta_bounds"],[.25,.5])

    def test_missing_duplicate_pair_and_initial_mismatch_rejected(self):
        for change in (lambda r:r.pop(),lambda r:r[-1].update(arm="frozen"),lambda r:r[-1].update(initial_fingerprint="wrong")):
            r=episodes()
            change(r)
            with self.assertRaises(ValueError):
                summarize(r,dict(master_seed=1,bootstrap=100))

    def test_positive_signal_not_all_map_dominance_or_automatic_promotion(self):
        r=summarize(episodes(),dict(master_seed=1,bootstrap=100))
        self.assertEqual(r["decision"],"bounded_positive_signal")
        self.assertEqual(r["contrasts"]["linear_vs_frozen"]["map_ties"],1)
        self.assertFalse(r["automatic_promotion"])
        self.assertTrue(r["no_ttf"])

    def test_scope_and_no_parameter_search(self):
        cfg=json.loads((Path(__file__).resolve().parents[2]/"configs/sa_linear_closed_loop.json").read_text())
        self.assertEqual(cfg["arms"],list(ARMS))
        self.assertEqual((cfg["maps"],cfg["workers"],cfg["max_decisions"]),(8,20,512))
        self.assertEqual(cfg["solver_seeds"],[227,229])
        self.assertFalse(cfg["formal_ttf"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"registered Windows training only")
    def test_full_fit_reload_and_gbdt_portable(self):
        import sklearn
        if sklearn.__version__!="1.5.0":
            self.skipTest("registered sklearn required")
        from experiments.sa_paired_closed_loop import portable_payload,portable_model
        d=fixture()
        payload,gbdt=fit_full(d)
        p=portable_model(portable_payload(gbdt),d["feature_names"],native=False)
        model=LinearRanker(json.loads(json.dumps(payload)))
        for s in d["states"]:
            self.assertEqual(model.rank(s)["selected"],"0")
            self.assertEqual(p.rank(s),gbdt.rank(s))

    @unittest.skipUnless(importlib.util.find_spec("lns2_env"),"frozen WSL native")
    def test_micro_episode_trace_audit_resume_and_zero_conflict(self):
        import lns2_env
        from scripts import run_sa_linear_closed_loop as runner
        from scripts import run_sa_path_quality as q
        from experiments.online_feature_engine import OnlineFeatureEngine
        from experiments.sa_history_selector import History
        from scripts.run_sa_paired_closed_loop import features_for
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            m,s=root/"tiny.map",root/"tiny.scen"
            m.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n",encoding="utf8")
            s.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n",encoding="utf8")
            def reset(case,seed,plan):
                env=lns2_env.LNS2RepairEnv(str(m),str(s),2)
                paths=[[0,1,2,3],[3,2,1,0]] if not case.get("zero") else [[0,1,2,3],[3,7,6,5,4,0]]
                state=q._plain(env.reset_paths(paths,seed=seed))
                return env,state,dict(case_id="tiny-seed7")
            class Selector:
                def __init__(self,source): pass
                def select(self,env,state,decision):
                    return 0,[dict(candidate_id="all",agents=[0,1],score=1,provenance=[]),
                              dict(candidate_id="one",agents=[0],score=0,provenance=[])]
            _,state,_=reset({},7,{})
            _,cs=Selector({}).select(None,state,0)
            fs=features_for(state,cs,OnlineFeatureEngine(state,backend="native"),History(state),q.temperature(0),q.state_fingerprint(state))
            names=sorted(fs[0])
            model=linear(names)
            model.estimator=SimpleNamespace(model=SimpleNamespace(inference_backend="native-portable-tree"))
            once(root/"model/metadata.json",dict(feature_names=names,ranges={n:[-1e20,1e20] for n in names}))
            config=dict(output=".",candidate_seed=3,candidate_limit=4,max_decisions=4,node_budget=100000,
                episode_seconds=10,pp_seconds=1.)
            plan=dict(binding="a"*64,config=config)
            with patch.object(runner,"ROOT",root),patch.object(runner,"model_receipt",return_value={}),\
                 patch.object(runner,"models_load",return_value=dict(gbdt=model,linear=model)),\
                 patch.object(runner,"reset_case",side_effect=reset),patch.object(q,"SingleFullCheckPool",Selector),\
                 patch("scripts.train_sa_history_selector.die_with_parent"):
                for arm in ARMS:
                    job=dict(parent_pid=os.getpid(),plan=plan,case=dict(map_id="tiny"),arm=arm,job_id="tiny-s7-"+arm,
                        pair_id="tiny-s7",solver_seed=7,expected_initial=q.state_fingerprint(state))
                    r=runner.episode_worker(job)
                    self.assertEqual(r["status"],"ok")
                    audit=runner.audit_episode(dict(folder=str(root/"episodes"/job["job_id"]),plan=plan))
                    self.assertLessEqual(audit["decisions"],4)
                    self.assertTrue(runner.episode_worker(job)["resumed"])
                _,zero,_=reset(dict(zero=True),7,plan)
                job.update(case=dict(map_id="tiny",zero=True),job_id="zero",expected_initial=q.state_fingerprint(zero))
                self.assertTrue(zero["feasible"])
                r=runner.episode_worker(job)
                self.assertEqual(r["decisions"],0)
                # Hash changes must block resume even when the episode was successful.
                path=root/"episodes"/job["job_id"]/"result.json"
                path.write_text("{}")
                with self.assertRaises(ValueError):
                    runner.episode_worker(job)


if __name__=="__main__":
    unittest.main()
