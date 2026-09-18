from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import json_fingerprint, read_json
from scripts import collect_sa_source_matched as c
from scripts.collect_sa_history_candidate_bridge import randomization


def fixture():
    root = dict(state_id="s", map_id="m", pair_ids=["a", "b"], anchor_id="b",
                candidates=[dict(candidate_id="a"), dict(candidate_id="b")])
    return dict(config=dict(trials=8, seed=202609182), roots=[root], binding="binding")


class SourceMatchedCollectionTests(unittest.TestCase):
    def test_trial_pairing_and_unique_jobs(self):
        jobs = c.schedule(fixture())
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({j["job_id"] for j in jobs}), 16)
        self.assertEqual([j["trial"] for j in jobs[:4]], [0, 0, 1, 1])

    def test_fresh_stream_no_candidate_dependency(self):
        cfg = fixture()["config"]
        values = {randomization("s", t, d, cfg)[0] for t in range(8) for d in range(4, 36)}
        self.assertEqual(len(values), 256)
        for old in (20260919, 20260921):
            prior = {randomization("s", t, d, dict(cfg, seed=old))[0] for t in range(8) for d in range(4, 36)}
            self.assertFalse(values & prior)
        self.assertEqual(randomization("s", 2, 9, cfg), randomization("s", 2, 9, cfg))

    def test_deadline_includes_prior_work(self):
        self.assertEqual(c.remaining(180, clock=lambda: 140), 40)
        with self.assertRaises(c.BudgetExpired):
            c.remaining(180, clock=lambda: 180)

    def test_prepared_contract(self):
        cfg = read_json(c.CONFIG)
        plan = dict(config=dict(trial_seed=202609182, planned_trials=8, planned_horizon=32,
                    planned_workers=20, planned_pp_safety_seconds=5, planned_trial_timeout_seconds=180))
        plan["binding"] = json_fingerprint(plan)
        with patch.object(c, "read_json", return_value=plan), \
             patch.object(c, "sha256_file", return_value=cfg["preparation_sha256"]):
            self.assertEqual(c.prepared(cfg), plan)
            for key, value in (("seed", 1), ("workers", 21), ("trials", 9), ("horizon", 1),
                               ("pp_seconds", 10), ("trial_seconds", 181), ("training_allowed", True)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.prepared(dict(cfg, **{key: value}))

    def make_result(self, folder, job, plan, stop="hard_fuse"):
        c.atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"]))
        row = dict(status="censored", binding=plan["binding"], root_id="s", candidate_id=job["candidate_id"],
                   trial=job["trial"], stop=stop)
        c.atomic(folder / "result.json", row)
        c.mark_receipt(folder, job, plan)
        return row

    def test_resume_and_isolated_output(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            jobs = c.schedule(plan)
            folder = out / "trials" / jobs[0]["job_id"]
            row = self.make_result(folder, jobs[0], plan)
            pending, complete = c.existing_jobs(plan, out)
            self.assertEqual((len(pending), len(complete)), (15, 1))
            self.assertEqual(c.receipt(folder, jobs[0], plan), row)
            self.assertIsNone(c.validate_trial(row, jobs[0], plan))
            with self.assertRaises(ValueError):
                c.receipt(folder, jobs[1], plan)

    def test_partial_trial_refuses_retry(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            job = c.schedule(plan)[0]
            (out / "trials" / job["job_id"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "requires review"):
                c.existing_jobs(plan, out)

    def test_tamper_and_unregistered_file(self):
        plan = fixture()
        job = c.schedule(plan)[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = self.make_result(folder, job, plan)
            c.atomic(folder / "unexpected.json", {})
            with self.assertRaises(ValueError):
                c.receipt(folder, job, plan)
            (folder / "unexpected.json").unlink()
            c.atomic(folder / "result.json", dict(row, trial=7))
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                c.receipt(folder, job, plan)

    def test_censor_not_false_label_and_wrong_trial_rejected(self):
        plan = fixture()
        job = c.schedule(plan)[0]
        row = dict(status="censored", root_id="s", candidate_id="a", trial=0, stop="hard_fuse")
        self.assertIsNone(c.validate_trial(row, job, plan))
        with self.assertRaises(ValueError):
            c.validate_trial(dict(row, trial=1), job, plan)

    def test_timeout_commits_unknown_not_failure(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            job = dict(c.schedule(plan)[0], plan=plan, folder=str(Path(tmp) / "job"))
            result = c.failure(job, "timeout", "180s")
            self.assertIsNone(result["completion"])
            self.assertEqual(result["status"], "ok")
            self.assertEqual(c.receipt(Path(job["folder"]), job, plan)["stop"], "hard_fuse")

    def test_completed_receipt_wins_timeout_race(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            job = dict(c.schedule(plan)[0], plan=plan, folder=str(folder))
            self.make_result(folder, job, plan, stop="prefix_or_total_safety")
            before = (folder / "result.json").read_bytes()
            self.assertEqual(c.failure(job, "timeout", "late_exit")["stop"], "prefix_or_total_safety")
            self.assertEqual(before, (folder / "result.json").read_bytes())

    def test_errors_are_not_censors(self):
        plan = fixture()
        result = c.failure(dict(c.schedule(plan)[0], plan=plan, folder="unused"), "error", "crash")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("completion", result)

    def test_interrupted_atomic_writes_preserved_on_timeout(self):
        plan = fixture()
        for name in ("started.json.tmp", "result.json.tmp", "receipt.json.tmp"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / "trials" / "job"
                job = dict(c.schedule(plan)[0], plan=plan, folder=str(folder))
                folder.mkdir(parents=True)
                (folder / name).write_bytes(b'{"partial":')
                result = c.failure(job, "timeout", "180s")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(c.receipt(folder, job, plan)["stop"], "hard_fuse")
                archived = list((Path(tmp) / "interrupted_writes").glob("*/*"))
                self.assertEqual(len(archived), 1)
                self.assertEqual(archived[0].read_bytes(), b'{"partial":')

    def test_corrupt_timeout_record_returns_error_and_does_not_raise(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            job = dict(c.schedule(plan)[0], plan=plan, folder=str(folder))
            c.atomic(folder / "result.json", {"invalid": True})
            self.assertEqual(c.failure(job, "timeout", "late")["status"], "error")

    def test_conflicting_or_changed_historical_hash_rejected(self):
        with patch.object(c, "sha256_file", return_value="new"):
            with self.assertRaisesRegex(ValueError, "conflicting"):
                c.merge_pinned({"model": "new"}, {"model": "old"})
            with self.assertRaisesRegex(ValueError, "changed"):
                c.merge_pinned({}, {"model": "old"})

    def test_stop_allowed_when_runtime_has_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "build/out"
            plan = dict(config=dict(output="build/out"))
            plan["binding"] = json_fingerprint(plan)
            c.atomic(out / "plan.json", plan)
            cfg = root / "config.json"
            c.atomic(cfg, plan["config"])
            with patch.object(c, "ROOT", root), patch.object(c, "CONFIG", cfg), \
                 patch.object(c, "verify", side_effect=ValueError("runtime drift")):
                self.assertTrue(c.request_stop()["stop_requested"])
            self.assertTrue((out / "STOP_AFTER_BATCH").exists())

    def test_smoke_safe_stop_does_not_schedule(self):
        plan = fixture()
        plan["config"].update(workers=20, trial_seconds=180)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c.atomic(out / "STOP_AFTER_BATCH", {})
            with patch.object(c, "verify", return_value=(plan, out)), \
                 patch("experiments.repair_collection._run_jobs") as run:
                result = c.collect()
            run.assert_not_called()
            self.assertEqual(result["status"], "paused")


if __name__ == "__main__":
    unittest.main()
