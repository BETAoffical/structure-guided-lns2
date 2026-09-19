import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import read_json
from experiments.sa_source_matched_analysis import analyze_record
from scripts import collect_sa_matched_remaining as c
from scripts import collect_sa_source_matched as old


def fixture():
    roots = [dict(state_id=f"s{i}", map_id=f"m{i}", decision=d, pair_ids=["a","b"], anchor_id="outside",
                  family="random:16", candidates=[dict(candidate_id=cid, actual_size=16,
                  agents=list(range(start,start+16)), selection_families=["random:16"])
                  for cid,start in (("a",0),("b",1))]) for i,d in enumerate((4,64))]
    return dict(roots=roots, config=dict(trials=8, trial_seconds=180, workers=20, bootstrap=100,
                                       bootstrap_seed=202609183), binding="test")


def write_unknown(job, stop="hard_fuse"):
    folder = Path(job["folder"])
    old.atomic(folder/"started.json", dict(binding=job["plan"]["binding"],job_id=job["job_id"]))
    old.atomic(folder/"result.json", dict(status="censored",binding=job["plan"]["binding"],
               root_id=job["root"]["state_id"],candidate_id=job["candidate_id"],trial=job["trial"],stop=stop))
    old.mark_receipt(folder,job,job["plan"])
    return dict(status="ok",job_id=job["job_id"],completion=None,stop=stop)


def fake_run(worker, jobs, workers, **kwargs):
    rows = [write_unknown(j) for j in jobs]
    for row in rows:
        kwargs["on_result"](row)
    return rows


class MatchedRemainingCollectionTests(unittest.TestCase):
    def phases(self, plan):
        jobs = old.schedule(plan)
        return [dict(name="short",workers=20,batches=[[j["job_id"] for j in jobs[:2]]]),
                dict(name="long",workers=4,batches=[[j["job_id"] for j in jobs[-2:]]])]

    def test_preflight_is_two_extreme_roots_and_pair_then_repeat(self):
        plan = fixture()
        primary,repeat = c.preflight_batches(plan)
        self.assertEqual(primary, ["s0-a-t0","s0-b-t0","s1-a-t0","s1-b-t0"])
        self.assertEqual(repeat,["s0-a-t0","s1-a-t0"])
        plan["roots"].reverse()
        self.assertEqual(c.preflight_batches(plan),(primary,repeat))

    def test_batch_resume_does_not_retry_censored(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,"_run_jobs",side_effect=fake_run) as runner, \
             patch.object(c,"verify",return_value=(plan,Path(tmp))):
            out=Path(tmp)
            first=c.execute(plan,out,self.phases(plan),limit_batches=1)
            self.assertEqual((first["status"],first["complete"]),("paused",2))
            second=c.execute(plan,out,self.phases(plan),resume=True)
            self.assertEqual((second["status"],second["complete"]),("completed",4))
            self.assertEqual([call.kwargs["workers"] for call in runner.call_args_list],[20,4])
            last=c.execute(plan,out,self.phases(plan),resume=True)
            self.assertEqual(last["complete"],4)
            self.assertEqual(runner.call_count,2)

    def test_stop_drains_active_batch_and_does_not_launch_next(self):
        plan=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,"verify",return_value=(plan,Path(tmp))):
            out=Path(tmp)
            def stop_after(worker,jobs,workers,**kwargs):
                rows=fake_run(worker,jobs,workers,**kwargs)
                old.atomic(out/"STOP_AFTER_BATCH",dict(binding=plan["binding"]))
                return rows
            with patch.object(c,"_run_jobs",side_effect=stop_after) as runner:
                result=c.execute(plan,out,self.phases(plan))
            self.assertEqual((result["status"],result["complete"]),("paused",2))
            self.assertEqual(runner.call_count,1)
            with patch.object(c,"_run_jobs",side_effect=fake_run):
                self.assertEqual(c.execute(plan,out,self.phases(plan),resume=True)["status"],"completed")

    def test_stop_before_start_and_duplicate_batches(self):
        plan=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,"_run_jobs") as runner:
            out=Path(tmp)
            old.atomic(out/"STOP_AFTER_BATCH",dict(binding=plan["binding"]))
            self.assertEqual(c.execute(plan,out,self.phases(plan))["status"],"paused")
            runner.assert_not_called()
            with self.assertRaisesRegex(ValueError,"duplicated"):
                c.execute(plan,out,self.phases(plan)*2)

    def test_partial_receipt_foreign_jobs_and_tampering_rejected(self):
        plan=fixture()
        jobs=old.schedule(plan)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            job=dict(jobs[0],folder=str(out/"trials"/jobs[0]["job_id"]),plan=plan)
            Path(job["folder"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"interrupted"):
                c.inspect_jobs(plan,out,{job["job_id"]})
            write_unknown(job)
            with self.assertRaisesRegex(ValueError,"unexpected completed"):
                c.inspect_jobs(plan,out,{jobs[1]["job_id"]})
            (Path(job["folder"])/"result.json").write_text("{}",encoding="utf8")
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                c.inspect_jobs(plan,out,{job["job_id"]})

    def test_error_is_not_censor_and_next_batch_is_not_launched(self):
        plan=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,"verify",return_value=(plan,Path(tmp))):
            def broken(worker,jobs,workers,**kwargs):
                return [dict(status="error",job_id=j["job_id"],error="fingerprint mismatch") for j in jobs]
            with patch.object(c,"_run_jobs",side_effect=broken) as runner, self.assertRaisesRegex(ValueError,"review required"):
                c.execute(plan,Path(tmp),self.phases(plan))
            self.assertEqual(runner.call_count,1)
            self.assertEqual(read_json(Path(tmp)/"run_status.json")["status"],"error")

    def test_preflight_unknown_cannot_allow_collection(self):
        plan=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,"verify",return_value=(plan,Path(tmp))), \
             patch.object(c,"_run_jobs",side_effect=fake_run):
            with self.assertRaisesRegex(ValueError,"preflight censored"):
                c.preflight()
            self.assertFalse((Path(tmp)/"preflight_report.json").exists())

    def record(self, unknown=False):
        return dict(state_id="s",map_id="m",pair_ids=["a","b"],anchor_id="outside",family="random:16",
                    values=dict(a=[True]*8,b=[None if unknown else False]*5+[True]*3))

    def test_pair_only_analysis_matches_frozen_math_without_anchor(self):
        record=self.record()
        new=c.pair_record(record)
        old_input=copy.deepcopy(record)
        old_input["values"]["outside"]=[False]*8
        expected=analyze_record(old_input)
        for field in ("pair_complete","paired_counts","all_trial_difference","absolute_all_trial_difference",
                      "pair_uniform_rate","crossfit_gain","half_direction_counts"):
            self.assertEqual(new[field],expected[field])
        self.assertFalse(new["anchor_sampled"])
        self.assertEqual(set(new["candidates"]),{"a","b"})
        self.assertNotIn("anchor_completion_rate",new)

    def test_unknown_not_false_and_not_filtered_from_mean(self):
        record=self.record(True)
        new=c.pair_record(record)
        self.assertIsNone(new["all_trial_difference"])
        self.assertIsNone(new["tie_neutral_crossfit"])
        self.assertEqual(new["paired_counts"]["unknown"],5)
        records=[record, self.record() | dict(state_id="other",map_id="m2")]
        report=c.summarize(records,fixture()["config"])
        self.assertEqual(report["states"],2)
        self.assertIsNone(report["metrics"]["pair_uniform_rate"]["mean"])
        self.assertIsNone(report["metrics"]["pair_uniform_rate"]["ci95"])

    def test_pair_validation_and_order_invariance(self):
        record=self.record()
        expected=c.pair_record(record)
        record["pair_ids"].reverse()
        self.assertEqual(expected,c.pair_record(record))
        for mutation in (lambda r:r["values"].update(outside=[False]*8),
                         lambda r:r["values"]["a"].pop(),lambda r:r["values"]["a"].__setitem__(0,1)):
            changed=self.record();mutation(changed)
            with self.assertRaises(ValueError): c.pair_record(changed)

    def test_anchor_in_pair_retained_and_bootstrap_deterministic(self):
        record=self.record() | dict(anchor_id="a")
        self.assertTrue(c.pair_record(record)["anchor_sampled"])
        first=c.summarize([record],fixture()["config"])
        self.assertEqual(first,c.summarize([record],fixture()["config"]))
        self.assertEqual(first["metrics"]["pair_uniform_rate"]["ci95"],[11/16,11/16])

    def test_repeat_semantics_include_actions_paths_and_nodes(self):
        a=dict(binding="a",events=[dict(action={"agents":[1,2]},metrics={"step_runtime":9,"generated":4})],final_fingerprint="fp")
        b=copy.deepcopy(a);b["binding"]="b";b["events"][0]["metrics"]["step_runtime"]=10
        self.assertEqual(c.semantic_row(a),c.semantic_row(b))
        b["events"][0]["metrics"]["generated"]=5
        self.assertNotEqual(c.semantic_row(a),c.semantic_row(b))

    def test_fixed_contract(self):
        cfg=read_json(c.CONFIG)
        fixed=read_json(c.ROOT/prep_config_path())
        roots=[]
        for i in range(31):
            root=copy.deepcopy(fixture()["roots"][0]);root["state_id"]=str(i);roots.append(root)
        prepared=dict(config=fixed,roots=roots)
        c.check_contract(cfg,prepared)
        for mutation in (dict(workers=16),dict(horizon=1),dict(trials=4),dict(source="other"),dict(sustain=4),
                         dict(training_allowed=True),dict(output="build"),dict(pp_seconds=10)):
            with self.assertRaises(ValueError): c.check_contract(cfg|mutation,prepared)
        prepared["roots"][0]["candidates"].append(dict(candidate_id="anchor"))
        with self.assertRaises(ValueError): c.check_contract(cfg,prepared)


def prep_config_path():
    return "configs/sa_matched_remaining_preparation.json"


if __name__ == "__main__": unittest.main()
