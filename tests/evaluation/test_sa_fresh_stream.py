from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_sa_fresh_stream as fresh

run = fresh.run


def fixture():
    override = run.read_json(fresh.ROOT/fresh.CONFIG)
    model = dict(path="model", iteration=1, file_sha256="a"*64, policy_sha256="b"*64)
    prior = dict(binding="p", config=dict(output=override["source_comparison"],
        models={fresh.last.ARM:model}, phase="crossfit-comparison"), jobs=[
            dict(pair_id=f"p{i}", replica=r, phase="crossfit-comparison", split="development_holdout",
                 comparison_arm=fresh.last.ARM, arm="trained_actor", iteration=1,
                 expected_initial=run.json_fingerprint(i), case=dict(map_id=f"m{6+i//4}"))
            for i in range(8) for r in (0, 1)])
    baseline = dict(config=dict(models={"untrained_exploration":dict(model, iteration=0), "bounded_condition":model}))
    source = dict(binding="s", config=dict(stream_seed=123), proposal=dict(max_decisions=256,
        pp_safety_seconds=20., node_budget=25000000), template=dict(environment=dict(max_repair_iterations=0)))
    return prior, baseline, override, source


class FreshStreamTests(unittest.TestCase):
    def test_complete_new_schedule_preserves_task_and_solver_identity(self):
        prior, baseline, override, _ = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        jobs = fresh.schedule(prior, cfg)
        self.assertEqual(len(jobs), 80)
        self.assertEqual(len({j["job_id"] for j in jobs}), 80)
        self.assertEqual(Counter(j["comparison_arm"] for j in jobs), {a:16 for a in fresh.ARMS})
        self.assertEqual(Counter(j["replica"] for j in jobs), {2:40, 3:40})
        self.assertFalse(set(map(fresh.stream_key, jobs)) & set(map(fresh.stream_key, prior["jobs"])))
        roots = {j["pair_id"]:j for j in prior["jobs"]}
        for job in jobs:
            for key in ("case", "expected_initial", "phase", "split"):
                self.assertEqual(job[key], roots[job["pair_id"]][key])
            arm = job["comparison_arm"]
            self.assertEqual(job["arm"], "trained_actor" if arm in ("bounded_condition", fresh.last.ARM) else arm)
        self.assertEqual(cfg["models"][fresh.last.ARM], prior["config"]["models"][fresh.last.ARM])
        self.assertEqual(cfg["models"]["bounded_condition"], baseline["config"]["models"]["bounded_condition"])

    def test_only_replica_changes_paired_random_streams(self):
        prior, baseline, override, source = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        selected = [j for j in fresh.schedule(prior, cfg) if j["pair_id"] == "p0" and j["replica"] == 2]
        for d in (0, 255, 256, 10000):
            for purpose in ("select", "pp", "accept"):
                draws = [run.stream_draw(fresh.previous.runtime_plan(source,cfg,j["comparison_arm"]),
                    j["phase"],j["pair_id"],j["replica"],d,purpose) for j in selected]
                self.assertEqual(len(set(draws)), 1)
                self.assertNotEqual(draws[0], run.stream_draw(source,"crossfit-comparison","p0",0,d,purpose))

    def test_uncapped_contract_keeps_nodes_and_feature_reference(self):
        prior, baseline, override, source = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        for arm in fresh.ARMS:
            plan = fresh.previous.runtime_plan(source, cfg, arm)
            self.assertIsNone(plan["proposal"]["max_decisions"])
            self.assertEqual(plan["proposal"]["decision_feature_reference"], 256)
            self.assertEqual(plan["template"]["environment"]["max_repair_iterations"], 0)
            self.assertIsNone(fresh.runtime.work_stop(False, 10**12, 1, plan["proposal"]))
            self.assertEqual(fresh.runtime.work_stop(False, 10**12, 25000000, plan["proposal"]), "node_budget")

    def test_scope_rejects_old_streams_extra_trials_training_and_cap(self):
        prior, baseline, override, _ = fixture()
        for key, value in (("replica_ids",[0,1]), ("replica_ids",[2,3,4]), ("max_decisions",10**12),
            ("node_budget",50000000), ("training",True), ("formal_ttf",True), ("automatic_promotion",True)):
            with self.assertRaises(ValueError):fresh.configuration(prior, baseline, dict(override, **{key:value}))
        cfg = fresh.configuration(prior, baseline, override)
        for key, value in (("split","train"), ("phase","changed")):
            bad = deepcopy(prior)
            bad["jobs"][0][key] = value
            with self.assertRaises(ValueError):fresh.schedule(bad, cfg)

    def test_metadata_inventory_rejects_reuse_ignores_own_registration(self):
        prior, baseline, override, _ = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        jobs = fresh.schedule(prior, cfg)
        with tempfile.TemporaryDirectory() as tmp, patch.object(fresh,"ROOT",Path(tmp)):
            out = Path(tmp)/"build"/"sa-new"
            run.write_json(Path(tmp)/"build"/"sa-old"/"registration.json", dict(entries=[dict(job=j) for j in prior["jobs"]]))
            run.write_json(out/"registration.json", dict(jobs=jobs))
            report = fresh.inventory(jobs, out)
            self.assertEqual(len(report["fresh_streams"]), 16)
            self.assertEqual(len(report["prior_streams"]), 16)
            run.write_json(Path(tmp)/"build"/"sa-other"/"registration.json", dict(jobs=[jobs[0]]))
            with self.assertRaisesRegex(ValueError,"previously registered stream"):fresh.inventory(jobs, out)

    def test_batch_stop_resume_and_partial_refusal(self):
        prior, baseline, override, source = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        reg = dict(config=cfg, jobs=fresh.schedule(prior, cfg), binding="r")
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(fresh.previous.prior,"ROOT",Path(tmp)):
            out = Path(tmp)/cfg["output"]
            def pool(worker, jobs, *args, **kwargs):
                calls.append(len(jobs))
                self.assertTrue(all(j["plan"]["proposal"]["max_decisions"] is None for j in jobs))
                for j in jobs:
                    run.once(fresh.previous.prior.folder_for(cfg,j)/"result.json", dict(status="ok"))
                if len(calls) == 1:run.write_json(out/"STOP_AFTER_BATCH", {})
                return [dict(status="ok") for _ in jobs]
            require = run.require
            def platform(condition, message):
                if message != "use frozen WSL native":require(condition, message)
            with patch.object(fresh,"verify",return_value=(reg,source,out)), \
                 patch.object(run,"require",side_effect=platform), \
                 patch.object(fresh.recovery,"strict_lock",return_value=nullcontext()), \
                 patch.object(fresh.previous,"read_result",return_value=dict(status="ok")), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                self.assertEqual(fresh.collect()["status"], "paused")
                self.assertEqual(calls,[20])
                with self.assertRaisesRegex(ValueError,"explicit resume"):fresh.collect()
                self.assertEqual(fresh.collect(True)["collected"],80)
                self.assertEqual(fresh.collect(True)["collected"],80)
                self.assertEqual(calls,[20,20,20,20])
                (fresh.previous.prior.folder_for(cfg,reg["jobs"][0])/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):fresh.collect(True)

    def test_unknown_and_recorded_failure_never_auto_retry(self):
        prior, baseline, override, source = fixture()
        cfg = fresh.configuration(prior, baseline, override)
        reg = dict(config=cfg, jobs=fresh.schedule(prior,cfg), binding="r")
        with tempfile.TemporaryDirectory() as tmp, patch.object(fresh.previous.prior,"ROOT",Path(tmp)):
            out = Path(tmp)/cfg["output"]
            first = reg["jobs"][0]
            run.write_json(fresh.previous.prior.folder_for(cfg,first)/"result.json", {})
            with patch.object(fresh.previous,"read_result",return_value=dict(status="censored")):
                with self.assertRaisesRegex(ValueError,"censored"):fresh.pending_jobs(reg,source,out,True)
            run.write_json(out/"failures"/(first["job_id"]+".json"), {})
            with self.assertRaisesRegex(ValueError,"recorded failure"):fresh.pending_jobs(reg,source,out,True)

    def test_net_gain_criterion_allows_losses_but_cannot_pool_to_rescue(self):
        def comp(old, dual, explore):
            return {a:dict(overall=dict(net_success_bounds=[x,x],wins=x+3,losses=3))
                    for a,x in (("bounded_condition",old),("dual16_sa",dual),("untrained_exploration",explore))}
        self.assertEqual(fresh.interpretation(comp(1,0,1)), "net_gain_repeated_on_fresh_streams_not_independent_maps")
        self.assertEqual(fresh.interpretation(comp(0,0,1)), "net_gain_not_repeated_do_not_promote")
        self.assertEqual(fresh.interpretation(comp(-1,-1,0)), "fresh_stream_regression_do_not_promote")
        bad = comp(1,1,1)
        bad["dual16_sa"]["overall"]["net_success_bounds"] = [-1,1]
        with self.assertRaises(ValueError):fresh.interpretation(bad)


if __name__ == "__main__":unittest.main()
