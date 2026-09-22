from copy import deepcopy
import ast
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_sa_crossfit as compare
from scripts import run_sa_onpolicy as run


def fixtures():
    cfg = run.read_json(run.ROOT/compare.CONFIG)
    original = run.read_json(run.ROOT/run.CONFIG)
    design = run.read_json(run.ROOT/original["design"])
    source = dict(config=original, proposal=design["proposal"], binding="b"*64,
                  split=dict(train_maps=[f"m{i}" for i in range(6)]),
                  cases=[dict(task_id=f"m{i}-t{t}", map_id=f"m{i}",solver_seeds=[227,229])
                         for i in range(8) for t in range(2)])
    return source,cfg


class ComparisonTests(unittest.TestCase):
    def test_schedule_disjoint_paired_and_no_extra_iteration(self):
        source,cfg=fixtures()
        jobs=compare.schedule(source,cfg)
        self.assertEqual((len(jobs),len({j["job_id"] for j in jobs})),(80,80))
        self.assertEqual({j["case"]["map_id"] for j in jobs},{"m6","m7"})
        self.assertEqual({j["split"] for j in jobs},{"development_holdout"})
        for arm in compare.ARMS:
            rows=[j for j in jobs if j["comparison_arm"]==arm]
            self.assertEqual(len(rows),16)
            self.assertEqual({j["iteration"] for j in rows},{1 if arm.startswith("bounded_") else 0})
        cfg["replicas"]=3
        with self.assertRaises(ValueError): compare.schedule(source,cfg)

    def test_runtime_preserves_engine_and_scientific_binding(self):
        source,cfg=fixtures()
        old=deepcopy(source)
        plans=[compare.runtime_plan(source,cfg,a) for a in compare.ARMS]
        self.assertEqual(source,old)
        self.assertEqual(len({p["config"]["output"] for p in plans}),5)
        for p in plans:
            self.assertEqual(p["binding"],source["binding"])
            for k in ("max_decisions","node_budget","pp_safety_seconds"):
                self.assertEqual(p["proposal"][k],source["proposal"][k])
            self.assertEqual(p["proposal"]["episode_safety_seconds"],900.)

    def test_same_draws_across_arms_distinct_replica_and_phase(self):
        source,cfg=fixtures()
        draws=[]
        for arm in compare.ARMS:
            p=compare.runtime_plan(source,cfg,arm)
            draws.append([run.stream_draw(p,cfg["phase"],"task",0,d,purpose)
                          for d in range(5) for purpose in ("select","pp","accept")])
        self.assertTrue(all(x==draws[0] for x in draws))
        self.assertNotEqual(run.stream_draw(source,cfg["phase"],"task",0,0,"select"),
                            run.stream_draw(source,cfg["phase"],"task",1,0,"select"))

    def test_distinct_model_receipts_no_training_alias(self):
        _,cfg=fixtures()
        self.assertNotEqual(compare.expected_policy(cfg,dict(comparison_arm="bounded_condition",arm="trained_actor")),
                            compare.expected_policy(cfg,dict(comparison_arm="bounded_state",arm="trained_actor")))
        self.assertEqual(compare.engine_arm("official_sa"),"official_sa")
        self.assertEqual(compare.engine_arm("dual16_sa"),"dual16_sa")
        with self.assertRaises(ValueError): compare.engine_arm("unknown")

    def test_failure_and_partial_refuse_silent_retry(self):
        source,cfg=fixtures()
        j=compare.schedule(source,cfg)[0]
        reg=dict(config=cfg,jobs=[j],binding="c"*64)
        with tempfile.TemporaryDirectory() as tmp,patch.object(compare,"ROOT",Path(tmp)):
            out=Path(tmp)/cfg["output"]
            folder=compare.folder_for(cfg,j)
            folder.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"partial episode"):
                compare.pending_jobs(reg,source,out,True)
            folder.rmdir()
            run.once(out/"failures"/(j["job_id"]+".json"),dict(status="censored"))
            with self.assertRaisesRegex(ValueError,"recorded failure"):
                compare.pending_jobs(reg,source,out,True)

    def test_unknown_bounds_and_common_success_not_failure_imputation(self):
        def row(success,status="ok"):
            return dict(success=success,status=status,decisions=5,generated=30,soc=100,makespan=20,wait_steps=4)
        left={"a":row(True),"b":row(False),"c":row(False,"censored"),"d":row(True)}
        right={"a":row(False),"b":row(True),"c":row(False),"d":row(True)}
        r=compare.paired_comparison(left,right)
        self.assertEqual((r["wins"],r["losses"],r["ties"],r["unknown_pairs"]),(1,1,1,1))
        self.assertEqual(r["net_success_bounds"],[0,1])
        self.assertEqual(r["common_success"],1)
        with self.assertRaises(ValueError):compare.paired_comparison(left,{})

    def test_safe_stop_finishes_batch_and_resume_does_not_duplicate(self):
        source,cfg=fixtures()
        jobs=compare.schedule(source,cfg)
        reg=dict(config=cfg,jobs=jobs,binding="c"*64)
        completed=set()
        calls=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(compare,"ROOT",Path(tmp)):
            out=Path(tmp)/cfg["output"]
            def pending(*args):return [j for j in jobs if j["job_id"] not in completed]
            def pool(worker,batch,*args,**kwargs):
                calls.append(len(batch))
                rows=[]
                for j in batch:
                    folder=compare.folder_for(cfg,j)
                    run.once(folder/"result.json",dict(status="ok"))
                    completed.add(j["job_id"])
                    r=dict(status="ok",job_id=j["job_id"])
                    kwargs["on_result"](r)
                    rows.append(r)
                if len(calls)==1:run.write_json(out/"STOP_AFTER_BATCH",{})
                return rows
            from contextlib import nullcontext
            actual_require=compare.require
            def platform_only(condition,message):
                if message!="use frozen WSL native": actual_require(condition,message)
            with patch.object(compare,"require",side_effect=platform_only),patch.object(compare,"verify",return_value=(reg,source,out)), \
                 patch.object(compare,"pending_jobs",side_effect=pending), \
                 patch.object(compare,"read_result",return_value=dict(status="ok")), \
                 patch.object(compare.recovery,"strict_lock",return_value=nullcontext()), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                self.assertEqual(compare.collect()["status"],"paused")
                self.assertEqual(calls,[20])
                self.assertEqual(compare.collect(True)["status"],"completed")
                self.assertEqual(calls,[20,20,20,20])
                self.assertEqual(compare.collect(True)["status"],"completed")
                self.assertEqual(calls,[20,20,20,20])

    def test_worker_keeps_engine_label_but_registers_actual_arm(self):
        source,cfg=fixtures()
        j=next(j for j in compare.schedule(source,cfg) if j["comparison_arm"]=="bounded_state")
        j.update(comparison_config=cfg,comparison_binding="c"*64)
        with tempfile.TemporaryDirectory() as tmp,patch.object(compare,"ROOT",Path(tmp)):
            folder=compare.folder_for(cfg,j)
            run.once(folder/"result.json",dict(arm="trained_actor"))
            with patch.object(compare.noop,"episode_worker",return_value=dict(status="ok",job_id=j["job_id"])) as mocked:
                r=compare.comparison_worker(j)
            self.assertEqual(mocked.call_args.args[0]["arm"],"trained_actor")
            self.assertEqual(r["comparison_arm"],"bounded_state")
            receipt=run.check_seal(run.read_json(folder/"comparison_receipt.json"))
            self.assertEqual(receipt["comparison_arm"],"bounded_state")

    def test_copied_loop_only_changes_interruption_classification(self):
        original=ast.parse(inspect.getsource(run.episode_loop))
        current=ast.parse(inspect.getsource(compare.noop.episode_loop))
        class Restore(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=="incomplete" for t in node.targets): return None
                return self.generic_visit(node)
            def visit_If(self,node):
                if isinstance(node.test,ast.Name) and node.test.id=="incomplete":
                    node.test=ast.parse('metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]',mode="eval").body
                return self.generic_visit(node)
        self.assertEqual(ast.dump(original),ast.dump(Restore().visit(current)))

    def test_legal_noop_counts_decision_but_not_search_work(self):
        before=dict(iteration=2,agents=[dict(id=17,conflict_degree=0,path=[1,2])],low_level=dict(generated=30),
                    conflict_edges=[[0,1]],sum_of_costs=1,num_of_colliding_pairs=1)
        after=deepcopy(before)
        after["iteration"]=3
        metrics=dict(pp_failure_reason="not_run",acceptance_evaluated=False,action_valid=True,step_applied=True,
                     pp_attempted_agent_count=0,pp_inserted_agent_count=0,pp_rolled_back=False,repair_order=[],
                     neighborhood=[17],generated=True)
        self.assertFalse(compare.noop.pp_incomplete(before,after,metrics))
        for key,val in (("pp_attempted_agent_count",1),("pp_rolled_back",True),("neighborhood",[99])):
            broken=dict(metrics,**{key:val})
            with self.assertRaises(ValueError): compare.noop.pp_incomplete(before,after,broken)
        bad=deepcopy(after)
        bad["low_level"]["generated"]+=1
        with self.assertRaises(ValueError): compare.noop.pp_incomplete(before,bad,metrics)
        before["agents"][0]["conflict_degree"]=1
        after=deepcopy(before)
        after["iteration"]+=1
        with self.assertRaises(ValueError): compare.noop.pp_incomplete(before,after,metrics)
        self.assertFalse(compare.noop.pp_incomplete(before,after,dict(metrics,generated=False)))
        self.assertTrue(compare.noop.pp_incomplete(before,after,dict(metrics,pp_failure_reason="time_limit")))
        self.assertTrue(compare.noop.pp_incomplete(before,after,dict(metrics,pp_failure_reason="other_failure")))


if __name__=="__main__":unittest.main()
