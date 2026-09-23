from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import sa_terminal_sampler as sampler
from scripts import probe_sa_terminal_sampling as current

run = current.run


def parent_bundle():
    return dict(schema="lns2.sa.onpolicy_actor.v1",binding="s",iteration=1,feature_names=["x"],
        mean=[0.],scale=[1.],w1=[[.01]]*32,b1=[0.]*32,w2=[[.1]*32],b2=[0.],epsilon=.1,residual_bound=2.)


def jobs_fixture():
    cfg = run.read_json(current.ROOT/current.CONFIG)
    maps = [f"M{i}" for i in range(6)]
    jobs = [dict(job_id=f"{m}-{seed}-{r}",pair_id=f"{m}-{seed}",replica=r,solver_seed=seed,split="train",
                 comparison_arm=cfg["parent_arm"],case=dict(map_id=m,task_variant="bottleneck_d25"))
            for m in maps for seed in (233,239) for r in range(4)]
    return cfg,maps,jobs


class SamplingTests(unittest.TestCase):
    def test_sampler_identity_probability_and_parent_immutability(self):
        p = parent_bundle()
        saved = deepcopy(p)
        bundle = sampler.make_sampler(p)
        actor = sampler.TerminalSampler(bundle)
        ids, features = ["a","b","c"],[dict(x=v) for v in (0.,1.,2.)]
        before = sampler.NumpyActor(p).probabilities(ids,"a",features)
        after = actor.probabilities(ids,"a",features)
        self.assertEqual(after,{cid:.5*v+.5/3 for cid,v in before.items()})
        self.assertAlmostEqual(math.fsum(after.values()),1.)
        self.assertEqual(p,saved)
        self.assertNotEqual(actor.sha,sampler.validate_bundle(p))
        with self.assertRaises(ValueError): sampler.validate_bundle(bundle)
        for key,value in (("uniform_mix",.6),("parent_policy","bad"),("trainable",True)):
            with self.assertRaises(ValueError): sampler.validate_sampler(dict(bundle,**{key:value}))

    def test_uniform_mixture_support_and_singleton(self):
        self.assertEqual(sampler.mixture({"a":1.}),{"a":1.})
        self.assertEqual(sampler.mixture({"a":.9,"b":.1}),{"a":.7,"b":.3})
        for p in ({},{"a":0.},{"a":float("nan")},{"a":.9}):
            with self.assertRaises(ValueError): sampler.mixture(p)

    def test_adapter_is_scoped_rejects_nesting_and_restores_on_error(self):
        actor,validate = run.NumpyActor,run.validate_bundle
        with self.assertRaisesRegex(RuntimeError,"test failure"):
            with sampler.sampler_runtime():
                self.assertIs(run.NumpyActor,sampler.TerminalSampler)
                self.assertEqual(run.validate_bundle(sampler.make_sampler(parent_bundle())),
                                 sampler.validate_sampler(sampler.make_sampler(parent_bundle())))
                with self.assertRaises(ValueError):
                    with sampler.sampler_runtime(): pass
                raise RuntimeError("test failure")
        self.assertIs(run.NumpyActor,actor)
        self.assertIs(run.validate_bundle,validate)

    def test_all_high_density_conditions_included_not_only_failures(self):
        cfg,maps,jobs = jobs_fixture()
        self.assertEqual(current.selected_controls(jobs,cfg,maps),jobs)
        easy = dict(jobs[0],job_id="easy",case=dict(map_id=maps[0],task_variant="bottleneck_d20"))
        self.assertEqual(current.selected_controls(jobs+[easy],cfg,maps),jobs)
        for bad in (jobs[:-1],jobs+[jobs[0]], [dict(jobs[0],split="development_holdout")]+jobs[1:]):
            with self.assertRaises(ValueError): current.selected_controls(bad,cfg,maps)

    def test_more_mixed_groups_from_losing_successes_is_not_good_credit(self):
        r = sampler.coverage({"hard":0,"easy":4},{"hard":0,"easy":2})
        self.assertEqual(r["sampled_mixed"],1)
        self.assertEqual(r["decision"],"no_new_hard_terminal_support_stop_this_sampler")
        self.assertEqual(r["damaged_all_success"],["easy"])

    def test_recovery_with_losses_is_explicit_tradeoff_not_all_wins_gate(self):
        r = sampler.coverage({"hard":0,"easy":4},{"hard":2,"easy":3})
        self.assertEqual(r["decision"],"terminal_coverage_signal_needs_independent_sampling_confirmation")
        r = sampler.coverage({"hard":0,"easy":4},{"hard":1,"easy":1})
        self.assertEqual(r["decision"],"coverage_efficiency_tradeoff_no_training_authorization")
        self.assertEqual(r["new_mixed_from_all_failure"],["hard"])
        with self.assertRaises(ValueError): sampler.coverage({"a":4},{"a":5})
        with self.assertRaises(ValueError): sampler.coverage({"a":4},{"b":4})

    def test_no_decision_limit_and_unchanged_random_stream(self):
        cfg,_,_ = jobs_fixture()
        cfg.update(arms=[current.ARM])
        source = dict(config=dict(stream_seed=7),proposal=dict(pp_safety_seconds=20.),
                      template=dict(environment=dict(max_repair_iterations=0)))
        plan = current.compare.runtime_plan(source,cfg,current.ARM)
        self.assertIsNone(current.runtime.work_stop(False,10**9,1,plan["proposal"]))
        for d in (0,256,10000):
            for purpose in ("select","pp","accept"):
                self.assertEqual(run.stream_draw(plan,"phase","pair",0,d,purpose),
                                 run.stream_draw(source,"phase","pair",0,d,purpose))

    def test_trace_diagnostics_use_accepted_global_conflicts(self):
        rows = [dict(selected_id="b",anchor_id="a",probabilities={"a":.6,"b":.4},
                     metrics=dict(conflicts_after=i)) for i in (4,6,2)]
        with patch.object(run,"trace_read",return_value=iter(rows)):
            r = current.summarize_trace(Path("unused"))
        self.assertEqual(r["minimum_conflicts"],2)
        self.assertEqual(r["last50_conflicts"],[4,6,2])
        self.assertEqual(r["nonanchor"],3)

    def test_float_roundoff_is_not_a_selection_mismatch(self):
        samples = [dict(job_id=str(i),decision=0,selections=["a","b"],probabilities_sha256="a") for i in range(48)]
        a = dict(binding="test",max_error=1.11e-16,samples=samples)
        b = dict(binding="test",max_error=0.,samples=[dict(s,probabilities_sha256="b") for s in samples])
        current.check_parity([a,b],"test")
        with self.assertRaises(ValueError): current.check_parity([dict(a,max_error=2e-12),b],"test")
        b["samples"][0]["selections"] = ["b","a"]
        with self.assertRaises(ValueError): current.check_parity([a,b],"test")

    def test_batch_safe_stop_resume_and_partial_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            jobs = [dict(job_id=str(i)) for i in range(22)]
            reg = dict(binding="test",config=dict(workers=20))
            calls = []
            def folder(j): return out/"episodes"/j["job_id"]
            def pool(worker,js,*args,**kwargs):
                calls.append(len(js))
                for j in js: run.once(folder(j)/"result.json",dict(status="ok"))
                if len(calls)==1: run.write_json(out/"STOP_AFTER_BATCH",{})
                return [dict(status="ok") for _ in js]
            require = run.require
            def platform(condition,message):
                if message != "use frozen WSL native": require(condition,message)
            with patch.object(run,"require",side_effect=platform), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                args = (reg,out,jobs,None,folder,lambda j:run.read_json(folder(j)/"result.json"),"collection")
                self.assertFalse(current.batch.execute_batches(*args,False,960.))
                with self.assertRaisesRegex(ValueError,"explicit resume"):
                    current.batch.execute_batches(*args,False,960.)
                self.assertTrue(current.batch.execute_batches(*args,True,960.))
                self.assertEqual(calls,[20,2])
                (folder(jobs[0])/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):
                    current.batch.execute_batches(*args,True,960.)


if __name__ == "__main__": unittest.main()
