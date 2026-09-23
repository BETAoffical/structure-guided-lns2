import copy
import unittest
from scripts import probe_sa_hard_official_reference as probe


class HardReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.run.read_json(probe.ROOT/probe.CONFIG)
        self.jobs = []
        for m in (self.cfg["hard_map"],self.cfg["control_map"]):
            for seed in (233,239):
                for r in range(4):
                    task = m + "__task_0001"
                    self.jobs.append(dict(case=dict(map_id=m,task_id=task,task_variant="bottleneck_d25"),
                        solver_seed=seed,pair_id=f"{task}-s{seed}",replica=r,phase="source-phase",split="train",
                        expected_initial=f"{m}-{seed}",job_id=f"{m}-{seed}-{r}",comparison_arm="half_uniform_sampling"))

    def test_schedule_is_full_identity_and_reuses_rng_phase(self):
        selected = probe.select_controls(dict(jobs=self.jobs),self.cfg)
        jobs = probe.schedule(selected)
        self.assertEqual(len({j["job_id"] for j in jobs}),16)
        for a,b in zip(jobs,self.jobs):
            self.assertEqual(a["arm"],"official_sa")
            for key in ("phase","replica","pair_id","expected_initial"):
                self.assertEqual(a[key],b[key])

    def test_reject_missing_or_duplicate(self):
        for js in (self.jobs[:-1], self.jobs[:-1]+[self.jobs[0]]):
            with self.assertRaises(ValueError): probe.select_controls(dict(jobs=js),self.cfg)

    def test_reject_wrong_task_seed_split_or_initial(self):
        for field,value in (("solver_seed",241),("split","test"),("expected_initial","different")):
            js = copy.deepcopy(self.jobs)
            js[0][field] = value
            with self.assertRaises(ValueError): probe.select_controls(dict(jobs=js),self.cfg)
        js = copy.deepcopy(self.jobs)
        js[0]["case"]["task_id"] = "different_task"
        with self.assertRaises(ValueError): probe.select_controls(dict(jobs=js),self.cfg)

    def test_heldout_maps_not_added(self):
        js = copy.deepcopy(self.jobs)
        js.append(dict(case=dict(map_id="sa_linear_v1_m06_station_centric_0000"),split="development_holdout"))
        self.assertEqual(probe.select_controls(dict(jobs=js),self.cfg),self.jobs)

    def test_no_cap_and_original_official_rng(self):
        self.assertIsNone(self.cfg["max_decisions"])
        self.assertIsNone(probe.runtime.work_stop(False,100000,200,self.cfg))
        import inspect
        source = inspect.getsource(probe.runtime.episode_loop)
        self.assertIn('action = dict(mode="official")',source)
        self.assertIn('if job["arm"] != "official_sa" else None',source)

    def test_distinct_signature_ignores_unused_draws_and_timing(self):
        e = dict(before="state",uniform=.1,delta=dict(version=1,top_set=dict(runtime=1,iteration=1)),
            metrics=dict(neighborhood=[2,9],repair_order=[9,2],pp_failure_reason="none",step_applied=True,
                         conflicts_after=2,step_runtime=.1))
        b = copy.deepcopy(e)
        b["uniform"] = .9
        b["delta"]["top_set"]["runtime"] = 99
        b["metrics"]["step_runtime"] = 99
        self.assertEqual(probe.semantic_step(e),probe.semantic_step(b))
        b["metrics"]["repair_order"].reverse()
        self.assertNotEqual(probe.semantic_step(e),probe.semantic_step(b))

    def test_unknown_is_not_infeasible(self):
        rows = {0:dict(map_id=self.cfg["hard_map"],status="censored",success=False),
                1:dict(map_id=self.cfg["control_map"],status="ok",success=True)}
        self.assertEqual(probe.reference_decision(rows,self.cfg),"incomplete_reference_no_causal_conclusion")
        rows[0]["status"] = "ok"
        self.assertEqual(probe.reference_decision(rows,self.cfg),"hard_failure_shared_at_fixed_budget_no_success_labels_for_retraining")
        rows[0]["success"] = True
        self.assertEqual(probe.reference_decision(rows,self.cfg),"official_success_witness_available_compare_candidate_coverage_next")

    def test_control_loss_is_not_global_shared_failure(self):
        rows = {0:dict(map_id=self.cfg["hard_map"],status="ok",success=False),
                1:dict(map_id=self.cfg["control_map"],status="ok",success=False)}
        self.assertEqual(probe.reference_decision(rows,self.cfg),"reference_also_loses_control_inspect_configuration_before_expansion")

    def test_pairing_rejects_unmatched_denominators(self):
        with self.assertRaises(ValueError): probe.compare.prior.paired_comparison({0:{}},{1:{}})


if __name__ == "__main__": unittest.main()
