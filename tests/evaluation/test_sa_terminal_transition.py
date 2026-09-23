import copy
import json
import unittest

from experiments.sa_terminal_transition import retained_ids, subset_document, signal_decision
from scripts import probe_sa_terminal_transition as probe


def task(n=43):
    meta = dict(flow_type="random",scenario_type="legacy_flow",swapped_agent_pairs=[],
        flow_assignments=[("left->right","right->left","storage->station")[i%3] for i in range(n)],
        required_bottlenecks=[[1,1] if i%4 == 0 else None for i in range(n)],
        required_intersections=[None]*n,required_intersection_component_ids=[None]*n,
        actual_shortest_distances=[2+i%5 for i in range(n)])
    return dict(task_id="map__task_0001",map_id="map",seed=1,metadata=meta,
                starts=[[0,i] for i in range(n)],goals=[[1,i] for i in range(n)])


class TerminalTransitionTests(unittest.TestCase):
    def test_exact_counts_and_largest_remainder(self):
        for n in (10,43,448,491,500,523,539):
            ids,strata = retained_ids(task(n),2026092301)
            self.assertEqual(len(ids),(9*n+5)//10)
            self.assertEqual(sum(s["retained"] for s in strata),len(ids))
            self.assertEqual(ids,sorted(set(ids)))
            for s in strata:
                self.assertIn(s["retained"],(9*s["source"]//10,(9*s["source"]+9)//10))

    def test_deterministic_not_simple_prefix(self):
        a = retained_ids(task(),2026092301)
        self.assertEqual(a,retained_ids(task(),2026092301))
        self.assertNotEqual(a[0],list(range(len(a[0]))))
        self.assertNotEqual(a[0],retained_ids(task(),2026092302)[0])

    def test_endpoints_metadata_mapping_and_source_immutable(self):
        parent = task()
        old = copy.deepcopy(parent)
        ids,strata = retained_ids(parent,7)
        child = subset_document(parent,ids,strata,seed=7,free_cells=100,source_sha256="a"*64)
        self.assertEqual(parent,old)
        self.assertEqual(child["starts"],[parent["starts"][i] for i in ids])
        self.assertEqual(child["goals"],[parent["goals"][i] for i in ids])
        self.assertEqual(child["metadata"]["parent_agent_ids"],ids)
        self.assertEqual(sum(child["metadata"]["realized_flow_counts"].values()),len(ids))
        self.assertEqual(child["metadata"]["agent_density_free_cells"],len(ids)/100)
        self.assertEqual(child["map_id"],parent["map_id"])

    def test_invalid_metadata_and_swap_pairs_rejected(self):
        for name,value in (("actual_shortest_distances",[2]),("swapped_agent_pairs",[[0,1]])):
            t = task()
            t["metadata"][name] = value
            with self.assertRaises(ValueError): retained_ids(t,7)
        with self.assertRaises(ValueError): retained_ids(task(9),7)

    def test_wrong_mapping_or_quotas_rejected(self):
        t = task()
        ids,strata = retained_ids(t,7)
        with self.assertRaises(ValueError): subset_document(t,ids[::-1],strata,seed=7,free_cells=100,source_sha256="a"*64)
        strata[0]["retained"] += 1
        with self.assertRaises(ValueError): subset_document(t,ids,strata,seed=7,free_cells=100,source_sha256="a"*64)

    def test_no_future_outcome_input(self):
        t = task()
        first = retained_ids(t,7)
        t["future_success"] = True
        t["metadata"]["future_conflicts"] = [999]*43
        self.assertEqual(first,retained_ids(t,7))

    def test_no_decision_cap_or_training(self):
        cfg = probe.run.read_json(probe.ROOT/probe.CONFIG)
        self.assertIsNone(cfg["max_decisions"])
        self.assertFalse(cfg["training"])
        self.assertIsNone(probe.runtime.work_stop(False,100000,1,cfg))
        self.assertEqual(cfg["expected_jobs"],48)

    def test_six_train_maps_no_heldout_and_complete_schedule(self):
        cfg = probe.run.read_json(probe.ROOT/probe.CONFIG)
        maps = [f"m{i}" for i in range(6)]
        conditions = [dict(case=dict(task_id=m+"__task_0001",map_id=m,task_variant="bottleneck_d25")) for m in maps]
        source = dict(split=dict(train_maps=maps))
        cases = probe.parent_cases(dict(conditions=conditions),source,cfg)
        jobs = probe.jobs_for(probe.condition_schedule(cases,cfg),cfg)
        self.assertEqual(len(jobs),48)
        self.assertEqual(len({j["job_id"] for j in jobs}),48)
        self.assertTrue(all(j["split"] == "train" and j["iteration"] == 1 for j in jobs))
        conditions[-1]["case"]["map_id"] = "m06-heldout"
        with self.assertRaises(ValueError): probe.parent_cases(dict(conditions=conditions),source,cfg)

    def test_partial_signal_preserved_without_training(self):
        info = dict(mixed_conditions={"p":1},effective_maps=["m"])
        r = signal_decision(info,1)
        self.assertEqual(r["decision"],"localized_terminal_signal_preserve_no_automatic_expansion")
        self.assertTrue(r["hard_task_subset_has_success"])
        self.assertFalse(r["original_hard_task_improved"])
        self.assertTrue(r["no_training_this_stage"])

    def test_coverage_gate_not_success_rate_gate(self):
        none = signal_decision(dict(mixed_conditions={},effective_maps=[]),8)
        self.assertTrue(none["decision"].startswith("no_within"))
        yes = signal_decision(dict(mixed_conditions={str(i):2 for i in range(4)},effective_maps=["a","b","c"]),0)
        self.assertTrue(yes["decision"].startswith("additional_terminal"))
        self.assertFalse(yes["hard_task_subset_has_success"])

    def test_sealed_json_roundtrip(self):
        t = task()
        ids,strata = retained_ids(t,7)
        value = subset_document(t,ids,strata,seed=7,free_cells=100,source_sha256="a"*64)
        sealed = probe.run.sealed(value)
        self.assertEqual(sealed,probe.run.check_seal(json.loads(json.dumps(sealed))))

    def test_real_parent_files_have_exact_od_mapping(self):
        p = probe.ROOT/"build/sa-onpolicy-execution-v1/plan.json"
        if not p.exists(): self.skipTest("local frozen source data required")
        source = probe.run.read_json(p)
        cfg = probe.run.read_json(probe.ROOT/probe.CONFIG)
        count = 0
        for c in source["cases"]:
            if c["map_id"] not in source["split"]["train_maps"] or c["task_variant"] != "bottleneck_d25": continue
            child,scen = probe.derive(c,cfg)
            parent = probe.run.read_json(probe.ROOT/c["files"]["task_file"])
            ids = child["metadata"]["parent_agent_ids"]
            self.assertEqual(child["starts"],[parent["starts"][i] for i in ids])
            self.assertEqual(len(scen.splitlines()),len(ids)+1)
            count += 1
        self.assertEqual(count,6)


if __name__ == "__main__": unittest.main()
