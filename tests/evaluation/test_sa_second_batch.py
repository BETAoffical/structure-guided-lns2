from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import collect_sa_second_batch as batch

run = batch.run


def fixture():
    cfg = run.read_json(batch.ROOT/batch.CONFIG)
    entries = {}
    for m in range(6):
        for t in range(2):
            for seed in (227,229):
                for r in range(4):
                    task = f"m{m}-t{t}"
                    j = dict(case=dict(task_id=task,map_id=f"m{m}",solver_seeds=[227,229],files={"map_file":f"m{m}.map"}),
                        split="train",pair_id=f"{task}-s{seed}",solver_seed=seed,replica=r,phase="train-0")
                    entries[f"{task}-{seed}-{r}"] = dict(job=j)
    return dict(entries=entries),cfg


def episode(pair,map_id,r,success):
    sha = "b"*64
    return dict(episode_id=f"{pair}-{r}",policy_sha256=sha,split="train",pair_id=pair,map_id=map_id,
        replica=r,initial_fingerprint=run.json_fingerprint(pair),rng_stream_id=run.json_fingerprint([pair,r]),
        status="ok",stop="feasible" if success else "node_budget",success=success,
        generated=1 if success else 25000000,decisions=1,final_conflicts=0 if success else 2,
        steps=[dict(decision=0,policy_sha256=sha,probabilities={"x":1.},selected_id="x",behavior_log_probability=0.)])


class SecondBatchTests(unittest.TestCase):
    def test_only_new_initial_solver_seeds_not_new_maps(self):
        old,cfg = fixture()
        roots = batch.conditions(old,cfg)
        self.assertEqual(len(roots),24)
        self.assertEqual({c["solver_seed"] for c in roots},{233,239})
        self.assertEqual(Counter(c["case"]["map_id"] for c in roots),{f"m{i}":4 for i in range(6)})
        originals = {e["job"]["case"]["task_id"]:e["job"]["case"] for e in old["entries"].values()}
        for c in roots:
            before = originals[c["case"]["task_id"]]
            self.assertEqual({k:v for k,v in c["case"].items() if k != "solver_seeds"},
                             {k:v for k,v in before.items() if k != "solver_seeds"})
        jobs = batch.schedule(roots,cfg)
        self.assertEqual(len(jobs),192)
        self.assertEqual(Counter(j["comparison_arm"] for j in jobs),{a:96 for a in batch.ARMS})
        self.assertEqual(Counter(j["replica"] for j in jobs),{r:48 for r in range(4)})

    def test_holdout_relabel_old_seed_or_update_not_allowed(self):
        old,cfg = fixture()
        for k,v in (("solver_seeds",[227,229]),("max_decisions",100000),
                    ("maximum_updates_this_stage",1),("heldout_evaluation",True),("formal_ttf",True)):
            with self.assertRaises(ValueError):batch.conditions(old,dict(cfg,**{k:v}))
        bad = deepcopy(old)
        next(iter(bad["entries"].values()))["job"]["split"] = "development_holdout"
        with self.assertRaisesRegex(ValueError,"Train only"):batch.conditions(bad,cfg)

    def test_runtime_uncapped_and_models_do_not_change_paired_streams(self):
        old,cfg = fixture()
        source = dict(binding="science",config=dict(stream_seed=123),proposal=dict(max_decisions=256,pp_safety_seconds=20.),
                      template=dict(environment=dict(max_repair_iterations=0)))
        jobs = batch.schedule(batch.conditions(old,cfg),cfg)
        a,b = jobs[:2]
        plans = [batch.compare.runtime_plan(source,cfg,j["comparison_arm"]) for j in (a,b)]
        for p in plans:
            self.assertIsNone(p["proposal"]["max_decisions"])
            self.assertIsNone(batch.runtime.work_stop(False,10**12,1,p["proposal"]))
        for d in (0,255,256,9999):
            for purpose in ("select","pp","accept"):
                self.assertEqual(run.stream_draw(plans[0],a["phase"],a["pair_id"],a["replica"],d,purpose),
                                 run.stream_draw(plans[1],b["phase"],b["pair_id"],b["replica"],d,purpose))

    def test_condition_registry_rejects_any_phase_reuse(self):
        old,cfg = fixture()
        roots = batch.conditions(old,cfg)
        with tempfile.TemporaryDirectory() as tmp,patch.object(batch,"ROOT",Path(tmp)):
            out = Path(tmp)/"build"/"sa-current"
            run.write_json(out/"registration.json",dict(conditions=roots))
            batch.require_unused(roots,out)
            run.write_json(Path(tmp)/"build"/"sa-old"/"registration.json",dict(entries={"x":dict(job=roots[0])}))
            with self.assertRaisesRegex(ValueError,"previously registered"):batch.require_unused(roots,out)

    def test_path_novelty_ignores_clock_counter_and_agent_order(self):
        a = dict(agents=[dict(id=5,path=[1,2]),dict(id=1,path=[3,4])],runtime=1,low_level=dict(generated=2))
        b = dict(a,agents=list(reversed(a["agents"])),runtime=999,low_level=dict(generated=999))
        self.assertEqual(batch.paths_signature(a,"m"),batch.paths_signature(b,"m"))
        b["agents"][0] = dict(id=1,path=[3,3,4])
        self.assertNotEqual(batch.paths_signature(a,"m"),batch.paths_signature(b,"m"))

    def test_credit_is_current_policy_only_and_map_equal(self):
        rows = [episode("a","m0",r,r<2) for r in range(4)]+[episode("b","m1",r,True) for r in range(4)]
        result = batch.credit_summary(rows,"b"*64,{"a":"m0","b":"m1"})
        self.assertEqual(result["mixed_conditions"],{"a":2})
        self.assertEqual(result["effective_maps"],["m0"])
        self.assertEqual(result["nonzero_credit_episodes"],4)
        self.assertEqual(result["credited_decisions"],4)
        self.assertTrue(result["one_update_eligible"])
        self.assertAlmostEqual(sum(c["episode_weight"] for c in result["coefficients"]),1.)
        for key,value in (("policy_sha256","c"*64),("split","development_holdout")):
            bad = deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):batch.credit_summary(bad,"b"*64,{"a":"m0","b":"m1"})

    def test_zero_conflict_retained_and_no_signal_means_no_update(self):
        rows = [episode("a","m",r,True) for r in range(4)]
        for row in rows:row.update(decisions=0,steps=[],generated=0)
        result = batch.credit_summary(rows,"b"*64,{"a":"m"})
        self.assertEqual(len(result["coefficients"]),4)
        self.assertFalse(result["one_update_eligible"])
        self.assertEqual(result["decisions"],0)

    def test_censor_and_missing_replica_never_become_failure_label(self):
        rows = [episode("a","m",r,r<2) for r in range(4)]
        rows[0].update(status="censored",stop="wall_safety")
        with self.assertRaisesRegex(ValueError,"censored batch"):batch.credit_summary(rows,"b"*64,{"a":"m"})
        with self.assertRaisesRegex(ValueError,"incomplete batch"):batch.credit_summary(rows[1:],"b"*64,{"a":"m"})

    def test_safe_batch_stop_resume_does_not_repeat_completed_work(self):
        _,cfg = fixture()
        reg = dict(binding="b",config=cfg)
        jobs = [dict(job_id=str(i)) for i in range(42)]
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = lambda j:out/"episodes"/j["job_id"]
            def pool(worker,batch_jobs,*args,**kwargs):
                calls.append(len(batch_jobs))
                for j in batch_jobs:run.once(folder(j)/"result.json",dict(status="ok"))
                if len(calls) == 1:run.write_json(out/"STOP_AFTER_BATCH",{})
                return [dict(status="ok") for j in batch_jobs]
            original_require = run.require
            def require(condition,message):
                if message != "use frozen WSL native":original_require(condition,message)
            with patch.object(run,"require",side_effect=require),patch("experiments.repair_collection._run_jobs",side_effect=pool):
                reader = lambda j:run.read_json(folder(j)/"result.json")
                args = (reg,out,jobs,object(),folder,reader,"test")
                self.assertFalse(batch.execute_batches(*args,False,960))
                self.assertEqual(calls,[20])
                self.assertTrue(batch.execute_batches(*args,True,960))
                self.assertTrue(batch.execute_batches(*args,True,960))
                self.assertEqual(calls,[20,20,2])
                (folder(jobs[0])/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):batch.execute_batches(*args,True,960)


if __name__ == "__main__":unittest.main()
