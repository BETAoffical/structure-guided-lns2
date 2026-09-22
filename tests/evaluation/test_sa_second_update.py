from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_sa_second_update as current

run = current.run


def fixture():
    cfg = run.read_json(current.ROOT/current.CONFIG)
    model = dict(path="parent", iteration=1, file_sha256="a"*64, policy_sha256="b"*64)
    old = dict(binding="old", config=dict(output=cfg["source_evaluation"], models={
        "uncapped_condition":model, "untrained_exploration":dict(model, path="explore", iteration=0)}), jobs=[
        dict(phase="crossfit-comparison", pair_id=f"p{i}", replica=r, split="development_holdout",
             case=dict(map_id=f"M0{6+i//4}", task_id=f"task{i}"), solver_seed=227+i%2,
             expected_initial=run.json_fingerprint(i)) for i in range(8) for r in (2,3)])
    source = dict(binding="s", config=dict(stream_seed=123), template=dict(environment=dict(max_repair_iterations=0)),
        proposal=dict(max_decisions=256, pp_safety_seconds=20., node_budget=25000000))
    return old, cfg, source


class SecondUpdateTests(unittest.TestCase):
    def test_complete_frozen_schedule_and_lineage(self):
        old, override, _ = fixture()
        cfg = current.configuration(old, override)
        jobs = current.schedule(old, cfg)
        self.assertEqual(len(jobs), 64)
        self.assertEqual(Counter(j["comparison_arm"] for j in jobs), {a:16 for a in current.ARMS})
        self.assertEqual(Counter(j["replica"] for j in jobs), {4:32, 5:32})
        self.assertEqual(len({j["job_id"] for j in jobs}), 64)
        self.assertEqual(cfg["models"]["uncapped_condition"], old["config"]["models"]["uncapped_condition"])
        self.assertEqual(cfg["models"][current.ARM]["iteration"], 2)
        self.assertEqual(set(cfg["models"]), {"uncapped_condition", "untrained_exploration", current.ARM})
        for j in jobs:
            expected = next(x for x in old["jobs"] if x["pair_id"] == j["pair_id"])
            for k in ("case", "solver_seed", "expected_initial", "phase", "split"):
                self.assertEqual(j[k], expected[k])
            self.assertEqual(j["arm"], "trained_actor" if j["comparison_arm"] in (current.ARM, "uncapped_condition") else j["comparison_arm"])

    def test_reject_changed_scope_and_source_conditions(self):
        old, cfg, _ = fixture()
        for key, value in (("replica_ids", [2,3]), ("expected_jobs", 80), ("node_budget", 50000000),
            ("max_decisions",100000), ("training",True), ("automatic_promotion",True), ("formal_ttf",True)):
            with self.assertRaises(ValueError):
                current.configuration(old, dict(cfg, **{key:value}))
        cfg = current.configuration(old, cfg)
        for key, value in (("split","train"), ("expected_initial","wrong"), ("solver_seed",999)):
            bad = deepcopy(old)
            bad["jobs"][0][key] = value
            with self.assertRaises(ValueError):
                current.schedule(bad, cfg)

    def test_paired_stream_and_no_hidden_decision_cap(self):
        old, override, source = fixture()
        cfg = current.configuration(old, override)
        for d in (0,255,256,10000):
            for purpose in ("select","pp","accept"):
                values = []
                for arm in current.ARMS:
                    plan = current.previous.runtime_plan(source,cfg,arm)
                    self.assertIsNone(plan["proposal"]["max_decisions"])
                    self.assertEqual(plan["template"]["environment"]["max_repair_iterations"],0)
                    self.assertIsNone(current.runtime.work_stop(False,10**9,1,plan["proposal"]))
                    self.assertEqual(current.runtime.work_stop(False,10**9,25000000,plan["proposal"]),"node_budget")
                    values.append(run.stream_draw(plan,cfg["phase"],"p0",4,d,purpose))
                self.assertEqual(len(set(values)),1)
                self.assertNotEqual(values[0], run.stream_draw(source,cfg["phase"],"p0",2,d,purpose))

    def test_new_stream_inventory_and_training_registration(self):
        old, cfg, _ = fixture()
        jobs = current.schedule(old,current.configuration(old,cfg))
        with tempfile.TemporaryDirectory() as tmp, patch.object(current.fresh,"ROOT",Path(tmp)):
            root = Path(tmp)/"build"
            run.write_json(root/"sa-old"/"registration.json",old)
            run.write_json(root/"sa-new"/"registration.json",dict(jobs=jobs))
            run.write_json(root/"sa-training"/"training_registration.json",dict(references=jobs))
            result = current.fresh.inventory(jobs,root/"sa-new")
            self.assertEqual(len(result["fresh_streams"]),16)
            run.write_json(root/"sa-other"/"registration.json",dict(jobs=[jobs[0]]))
            with self.assertRaisesRegex(ValueError,"previously registered"):
                current.fresh.inventory(jobs,root/"sa-new")

    def test_collect_safe_batch_pause_resume_and_partial_refusal(self):
        old, override, source = fixture()
        cfg = current.configuration(old,override)
        reg = dict(config=cfg,jobs=current.schedule(old,cfg),binding="r")
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(current.previous.prior,"ROOT",Path(tmp)):
            out = Path(tmp)/cfg["output"]
            def pool(worker,jobs,*args,**kwargs):
                calls.append(len(jobs))
                for j in jobs:
                    self.assertIsNone(j["plan"]["proposal"]["max_decisions"])
                    run.once(current.previous.prior.folder_for(cfg,j)/"result.json",dict(status="ok"))
                if len(calls)==1:
                    run.write_json(out/"STOP_AFTER_BATCH",{})
                return [dict(status="ok") for _ in jobs]
            require = run.require
            def platform(condition,message):
                if message != "use frozen WSL native":
                    require(condition,message)
            with patch.object(current,"verify",return_value=(reg,source,out)), \
                 patch.object(run,"require",side_effect=platform), \
                 patch.object(current.recovery,"strict_lock",return_value=nullcontext()), \
                 patch.object(current.previous,"read_result",return_value=dict(status="ok")), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                self.assertEqual(current.collect()["status"],"paused")
                self.assertEqual(calls,[20])
                with self.assertRaisesRegex(ValueError,"explicit resume"):
                    current.collect()
                self.assertEqual(current.collect(True)["collected"],64)
                self.assertEqual(current.collect(True)["collected"],64)
                self.assertEqual(calls,[20,20,20,4])
                (current.previous.prior.folder_for(cfg,reg["jobs"][0])/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):
                    current.collect(True)

    def test_unknown_and_failure_are_not_automatic_retries(self):
        old, cfg, source = fixture()
        cfg = current.configuration(old,cfg)
        reg = dict(config=cfg,jobs=current.schedule(old,cfg),binding="r")
        with tempfile.TemporaryDirectory() as tmp, patch.object(current.previous.prior,"ROOT",Path(tmp)):
            out = Path(tmp)/cfg["output"]
            job = reg["jobs"][0]
            run.write_json(current.previous.prior.folder_for(cfg,job)/"result.json",{})
            with patch.object(current.previous,"read_result",return_value=dict(status="censored")):
                with self.assertRaisesRegex(ValueError,"censored"):
                    current.fresh.pending_jobs(reg,source,out,True)
            run.write_json(out/"failures"/(job["job_id"]+".json"),{})
            with self.assertRaisesRegex(ValueError,"recorded failure"):
                current.fresh.pending_jobs(reg,source,out,True)

    def test_pairing_missing_identity_and_unknown_rejected(self):
        data = {a:{(f"p{i//2}",4+i%2):dict(status="ok",split="development_holdout",initial_fingerprint=f"p{i//2}",
                 rng_stream_id=f"r{i}",map_id=f"M{6+i//8}") for i in range(16)} for a in current.ARMS}
        current.validate_pairs(data)
        for field,value in (("status","censored"),("initial_fingerprint","changed"),("rng_stream_id","changed"),("split","train")):
            bad = deepcopy(data)
            bad[current.ARM][("p0",4)][field]=value
            with self.assertRaises(ValueError):
                current.validate_pairs(bad)
        bad = deepcopy(data)
        bad[current.ARM].pop(("p0",4))
        with self.assertRaises(ValueError):
            current.validate_pairs(bad)

    def test_frozen_net_gain_criterion_allows_case_losses(self):
        def comp(parent,explore,dual):
            return {a:dict(overall=dict(net_success_bounds=[n,n],wins=3+n,losses=3)) for a,n in
                    (("uncapped_condition",parent),("untrained_exploration",explore),("dual16_sa",dual))}
        self.assertEqual(current.interpretation(comp(1,1,0)),"second_update_development_net_gain_needs_confirmation")
        self.assertEqual(current.interpretation(comp(0,1,1)),"second_update_mixed_or_no_net_gain_do_not_promote")
        self.assertEqual(current.interpretation(comp(-1,1,-1)),"second_update_development_regression_do_not_promote")
        unknown = comp(1,1,0)
        unknown["dual16_sa"]["overall"]["net_success_bounds"]=[-1,0]
        with self.assertRaises(ValueError):
            current.interpretation(unknown)

    def test_common_success_means_do_not_include_failed_paths(self):
        base = dict(status="ok",success=True,decisions=10,generated=100,soc=50,makespan=9,wait_steps=3)
        left = {1:base,2:dict(base,success=False,generated=99999)}
        right = {1:dict(base,generated=200),2:base}
        result = current.previous.prior.paired_comparison(left,right)
        self.assertEqual(result["common_success"],1)
        self.assertEqual(result["losses"],1)
        self.assertEqual(result["common_success_means"]["generated"],dict(left=100,right=200))


if __name__ == "__main__":
    unittest.main()
