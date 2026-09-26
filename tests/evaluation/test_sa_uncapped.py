from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from tests.native_support import isolated_sa_native
from unittest.mock import patch

from experiments import sa_uncapped_runtime as runtime
from scripts import compare_sa_uncapped as compare
from scripts import run_sa_onpolicy as run


def plan():
    return dict(binding="b"*64, config=dict(stream_seed=123), proposal=dict(max_decisions=None,
        decision_feature_reference=256,node_budget=25000000,episode_safety_seconds=900.,pp_safety_seconds=20.))


class UncappedTests(unittest.TestCase):
    def test_no_decision_limit_at_any_horizon(self):
        cfg=plan()["proposal"]
        for d in (0,255,256,257,100000,10**12):
            self.assertIsNone(runtime.work_stop(False,d,24999999,cfg))
        self.assertEqual(runtime.work_stop(False,300,25000000,cfg),"node_budget")
        self.assertEqual(runtime.work_stop(True,300,26000000,cfg),"feasible")
        with self.assertRaises(ValueError): runtime.work_stop(False,3,4,dict(cfg,max_decisions=256))

    def test_frozen_features_exact_and_clipped_without_execution_cap(self):
        p=plan()
        original=deepcopy(p)
        f=runtime.feature_plan(p)
        self.assertEqual(p,original)
        for d in (0,50,255,256,257,10000):
            values=run.budget_features({},d,123,f["proposal"])
            self.assertEqual(values,run.budget_features({},d,123,dict(p["proposal"],max_decisions=256)))
            self.assertEqual(values["budget.remaining_decision_fraction"],max(0,256-d)/256)
        with self.assertRaises(ValueError):runtime.feature_plan(dict(p,proposal=dict(p["proposal"],decision_feature_reference=1000)))

    def test_terminal_rejects_decision_budget_but_accepts_atomic_overshoot(self):
        cfg=plan()["proposal"]
        row=dict(status="ok",success=False,decisions=50000,generated=25000005,stop="node_budget")
        runtime.validate_terminal(row,cfg)
        runtime.validate_terminal(dict(row,success=True,stop="feasible"),cfg)
        for bad in (dict(row,stop="decision_budget"),dict(row,generated=100),dict(row,decisions=-1)):
            with self.assertRaises(ValueError):runtime.validate_terminal(bad,cfg)
        runtime.validate_terminal(dict(row,status="censored",stop="wall_safety",generated=10),cfg)

    def test_real_loop_continues_past_256(self):
        state=dict(iteration=0,agents=[dict(id=0,path=[0,1]),dict(id=1,path=[1,0])],
                   low_level=dict(generated=0),conflict_edges=[[0,1]],num_of_colliding_pairs=1,sum_of_costs=2,feasible=False)
        cursor=[state]
        def step(*args):
            before=cursor[0]
            after=deepcopy(before)
            after["iteration"]+=1
            after["low_level"]["generated"]+=1
            if after["iteration"]==300:
                after.update(feasible=True,conflict_edges=[],num_of_colliding_pairs=0)
            cursor[0]=after
            return dict(observation=after,metrics=dict(neighborhood=[0,1],acceptance_evaluated=True,
                pp_failure_reason="none",conflicts_before=before["num_of_colliding_pairs"],
                conflicts_after=after["num_of_colliding_pairs"],pp_rolled_back=False))
        q=SimpleNamespace(_plain=lambda x:x,temperature=lambda d:1.,state_fingerprint=run.json_fingerprint,
                          encode_state_delta=lambda a,b:b,validate_transition=lambda *args:None)
        job=dict(plan=plan(),arm="official_sa",phase="test",pair_id="p",replica=0,job_id="j",split="test",
                 comparison_binding="x"*64,case=dict(map_id="tiny"))
        with tempfile.TemporaryDirectory() as tmp,patch("scripts.run_feedback_exploration_diagnostics.validate_final"):
            row=runtime.episode_loop(job,q,SimpleNamespace(step_experimental_pp=step),state,{},Path(tmp),None)
            self.assertEqual((row["decisions"],row["stop"],row["generated"]),(300,"feasible",300))
            self.assertEqual(len(list(run.trace_read(Path(tmp)))),300)

    def test_prefix_exactness_and_changed_actions_rejected(self):
        before=dict(decisions=2,stop="decision_budget")
        old=[dict(decision=i,action=i,metrics=dict(runtime=1)) for i in range(2)]
        new=[dict(decision=i,action=i,metrics=dict(runtime=3)) for i in range(3)]
        self.assertEqual(compare.prefix_check(old,new,before,dict(decisions=3)),2)
        with self.assertRaises(ValueError):compare.prefix_check(old,new,before,dict(decisions=2))
        new[0]["action"]=9
        with self.assertRaises(ValueError):compare.prefix_check(old,new,before,dict(decisions=3))

    def test_uncapped_plan_does_not_change_rng_models_or_pp(self):
        old=dict(config=run.read_json(run.ROOT/compare.prior.CONFIG))
        cfg=compare.configuration(old)
        p=plan()
        p["proposal"]["max_decisions"]=256
        for arm in compare.prior.ARMS:
            current=compare.runtime_plan(p,cfg,arm)
            self.assertIsNone(current["proposal"]["max_decisions"])
            self.assertEqual(current["binding"],p["binding"])
            for d in (0,255,256,10000):
                for purpose in ("pp","select","accept"):
                    self.assertEqual(run.stream_draw(current,cfg["phase"],"p",0,d,purpose),
                                     run.stream_draw(p,cfg["phase"],"p",0,d,purpose))
            self.assertEqual(current["proposal"]["pp_safety_seconds"],20.)

    def test_safe_batch_stop_resume_and_partial_refusal(self):
        from contextlib import nullcontext
        old=dict(config=run.read_json(run.ROOT/compare.prior.CONFIG))
        cfg=compare.configuration(old)
        jobs=[dict(job_id=str(i),comparison_arm="official_sa",phase=cfg["phase"]) for i in range(40)]
        reg=dict(config=cfg,jobs=jobs,binding="c"*64)
        calls=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(compare.prior,"ROOT",Path(tmp)):
            out=Path(tmp)/cfg["output"]
            def pool(worker,batch,*args,**kwargs):
                calls.append(len(batch))
                for j in batch:
                    run.once(compare.prior.folder_for(cfg,j)/"result.json",dict(status="ok"))
                if len(calls)==1:run.write_json(out/"STOP_AFTER_BATCH",{})
                return [dict(status="ok") for _ in batch]
            actual_require=run.require
            def platform_only(condition,message):
                if message!="use frozen WSL native":actual_require(condition,message)
            with patch.object(run,"require",side_effect=platform_only),patch.object(compare,"verify",return_value=(reg,plan(),out)), \
                 patch.object(compare,"read_result",return_value=dict(status="ok")), \
                 patch.object(compare.recovery,"strict_lock",return_value=nullcontext()), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                self.assertEqual(compare.collect()["status"],"paused")
                self.assertEqual(calls,[20])
                self.assertEqual(compare.collect(True)["status"],"completed")
                self.assertEqual(calls,[20,20])
                self.assertEqual(compare.collect(True)["status"],"completed")
                self.assertEqual(calls,[20,20])
                partial=compare.prior.folder_for(cfg,jobs[0])
                (partial/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):
                    compare.collect(True)

    @isolated_sa_native
    def test_frozen_native_micro(self):
        try:
            import lns2_env
        except ImportError:
            self.skipTest("frozen WSL native required")
        cfg=run.read_json(run.ROOT/run.CONFIG)
        p=run.read_json(run.ROOT/cfg["output"]/"plan.json")
        p=dict(p,proposal=dict(p["proposal"],max_decisions=None,decision_feature_reference=256))
        q=run.native_runtime(p)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"tiny.map").write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n",encoding="utf8")
            (root/"tiny.scen").write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n",encoding="utf8")
            env=lns2_env.LNS2RepairEnv(str(root/"tiny.map"),str(root/"tiny.scen"),2)
            initial=q._plain(env.reset_paths([[0,1,2,3],[3,2,1,0]],seed=11))
            job=dict(plan=p,arm="official_sa",phase="micro",pair_id="tiny",replica=0,job_id="tiny",split="micro",
                     comparison_binding="x"*64,case=dict(map_id="tiny"))
            row=runtime.episode_loop(job,q,env,initial,{},root/"episode",None)
            self.assertEqual(row["stop"],"feasible")
            self.assertTrue(row["success"])


if __name__=="__main__":
    unittest.main()
