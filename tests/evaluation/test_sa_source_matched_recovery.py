import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import recover_sa_source_matched as r


def fixture():
    root = dict(state_id="s", candidates=[dict(candidate_id=c) for c in "abc"])
    original = dict(roots=[root], config=dict(trials=8))
    record = dict(state_id="s", values={c: [False] * 8 for c in "abc"})
    for c, indices in (("a", (2, 3, 4, 6)), ("b", (2, 3, 4, 5, 6, 7)), ("c", (1, 2, 3, 4, 5, 6))):
        for t in indices:
            record["values"][c][t] = None
    cfg = dict(root_id="s", unknown_jobs=16, control_trial=0)
    return original, [record], cfg


class RecoveryTests(unittest.TestCase):
    def test_scope_is_exact_unknowns_plus_predeclared_controls(self):
        original, records, cfg = fixture()
        jobs = r.scope(original, records, cfg)
        self.assertEqual(len(jobs), 19)
        self.assertEqual([j["role"] for j in jobs[:3]], ["control"] * 3)
        self.assertEqual([j["trial"] for j in jobs[:3]], [0] * 3)
        self.assertEqual(len({j["job_id"] for j in jobs}), 19)
        self.assertEqual({j["candidate_id"] for j in jobs[:3]}, set("abc"))

    def test_extra_unknown_or_changed_control_refused(self):
        original, records, cfg = fixture()
        records[0]["values"]["a"][0] = None
        with self.assertRaises(ValueError):
            r.scope(original, records, cfg)

    def test_recovery_never_overwrites_observed_labels_or_original(self):
        original, records, cfg = fixture()
        before = copy.deepcopy(records)
        jobs = r.scope(original, records, cfg)
        merged = r.merge_records(records, [(j, True) for j in jobs if j["role"] == "recovery"])
        self.assertEqual(records, before)
        self.assertEqual(sum(v is True for vs in merged[0]["values"].values() for v in vs), 16)
        with self.assertRaisesRegex(ValueError, "observed"):
            r.merge_records(records, [(jobs[0], True)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            r.merge_records(records, [(jobs[3], True), (jobs[3], False)])

    def test_remaining_unknown_is_not_false(self):
        original, records, cfg = fixture()
        job = r.scope(original, records, cfg)[3]
        self.assertEqual(r.merge_records(records, [(job, None)]), records)

    def test_counts_distinguish_sealed_valid_and_success(self):
        self.assertEqual(r.counts([dict(completion=x) for x in (True, False, None)]),
                         dict(sealed=3, valid=2, feasible=1, horizon_nonfeasible=1, censored=1))

    def test_equivalence_excludes_only_registered_timing(self):
        row = dict(binding="new", diagnostic_seconds=7, generated=12, events=[dict(
            metrics=dict(pp_replan_seconds=.1, requested_pp_time_limit_seconds=5, repair_order=[1, 2]),
            pool=[dict(score=.3, agents=[1, 2])], delta=dict(agents=[1]))])
        changed = copy.deepcopy(row)
        changed["binding"] = "old"
        changed["events"][0]["metrics"]["pp_replan_seconds"] = 99
        self.assertEqual(r.semantic_row(row), r.semantic_row(changed))
        for key, value in (("requested_pp_time_limit_seconds", 6), ("repair_order", [2, 1])):
            changed = copy.deepcopy(row)
            changed["events"][0]["metrics"][key] = value
            self.assertNotEqual(r.semantic_row(row), r.semantic_row(changed))
        changed = copy.deepcopy(row)
        changed["events"][0]["pool"][0]["score"] = .4
        self.assertNotEqual(r.semantic_row(row), r.semantic_row(changed))

    def sealed(self, folder, job, plan):
        r.atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"]))
        r.Phases(folder, plan["binding"], job["job_id"]).mark("prefix", decision=1)
        row = dict(status="censored", binding=plan["binding"], root_id=job["root"]["state_id"],
                   candidate_id=job["candidate_id"], trial=job["trial"], stop="hard_fuse")
        r.atomic(folder / "result.json", row)
        r.seal(folder, job, plan)
        return row

    def test_receipt_phase_hash_and_resume_isolation(self):
        original, records, cfg = fixture()
        jobs = r.scope(original, records, cfg)
        plan = dict(binding="binding", jobs=jobs)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = out / "trials" / jobs[0]["job_id"]
            row = self.sealed(folder, jobs[0], plan)
            self.assertEqual(r.receipt(folder, jobs[0], plan), row)
            pending, done = r.inventory(plan, out)
            self.assertEqual((len(pending), len(done)), (18, 1))
            r.Phases(folder, plan["binding"], jobs[0]["job_id"]).mark("tamper")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                r.receipt(folder, jobs[0], plan)

    def test_partial_trial_never_silently_retried(self):
        original, records, cfg = fixture()
        jobs = r.scope(original, records, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "trials" / jobs[0]["job_id"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "requires review"):
                r.inventory(dict(binding="b", jobs=jobs), out)

    def test_hard_timeout_seals_unknown_and_preserves_progress(self):
        original, records, cfg = fixture()
        job = r.scope(original, records, cfg)[3]
        plan = dict(binding="binding", runtime_config={})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trials" / job["job_id"]
            r.Phases(folder, "binding", job["job_id"]).mark("prefix_step", decision=127)
            phase_before = (folder / "phases.jsonl").read_bytes()
            result = r.failure(dict(job, folder=str(folder), plan=plan), "timeout", "180s")
            self.assertTrue(result["censored"])
            self.assertFalse(result["valid"])
            self.assertEqual((folder / "phases.jsonl").read_bytes(), phase_before)
            self.assertEqual(r.receipt(folder, job, plan)["stop"], "hard_fuse")

    def test_crash_is_not_budget_censor(self):
        self.assertEqual(r.failure(dict(job_id="j"), "error", "crash")["status"], "error")

    def test_timeout_log_salvage_only_allows_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phases.jsonl"
            path.write_text('{"phase":"prefix"}\n{"phase":', encoding="utf8")
            self.assertTrue(r.phase_records(path, True)["interrupted_final_line"])
            with self.assertRaises(ValueError):
                r.phase_records(path, False)
            path.write_text('{"phase":\n{"phase":"done"}\n', encoding="utf8")
            with self.assertRaises(ValueError):
                r.phase_records(path, True)

    def test_absolute_deadline_includes_reset_and_prefix(self):
        with self.assertRaises(r.old.BudgetExpired):
            r.old.remaining(180, clock=lambda: 180)
        self.assertEqual(r.old.remaining(180, clock=lambda: 170), 10)

    def test_phases_do_not_touch_candidate_or_random_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            p = r.Phases(folder, "b", "j", clock=lambda: 10)
            p.mark("reset")
            p.mark("prefix")
            lines = [r.json.loads(x) for x in (folder / "phases.jsonl").read_text().splitlines()]
            self.assertEqual([x["elapsed"] for x in lines], [0, 0])

    def test_instrumented_rollout_matches_frozen_loop(self):
        from scripts import run_sa_path_quality as q
        from scripts import collect_sa_history_candidate_bridge as frozen
        initial = dict(feasible=False, low_level=dict(generated=0), num_of_colliding_pairs=1, sum_of_costs=10)
        pool = [dict(candidate_id="a", agents=[1, 2]), dict(candidate_id="b", agents=[2, 3])]
        root = dict(state=initial, state_fingerprint="fp", source=dict(id="s", decision=128),
                    control_event=dict(pool=pool), previous_best=1)
        job = dict(root=dict(state_id="s"), candidate_id="a", trial=0)
        cfg = dict(horizon=4, pp_seconds=5, seed=202609182, node_budget=2**63-1, branch_seconds=100)
        plan = dict(binding="b", runtime_config=cfg)

        class Env:
            def __init__(self):
                self.state = copy.deepcopy(initial)
            def get_state(self):
                return self.state
            def step_experimental_pp(self, action, seconds, mode, temp, uniform):
                self.state = copy.deepcopy(self.state)
                self.state["low_level"]["generated"] += 1
                return dict(observation=self.state, metrics=dict(pp_failure_reason="none", acceptance_evaluated=True,
                             requested_pp_time_limit_seconds=seconds, neighborhood=action["agents"]))

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(q, "state_fingerprint", return_value="fp"), \
             patch.object(q, "_plain", side_effect=lambda x: x), \
             patch.object(q, "validate_transition"), \
             patch.object(q, "encode_state_delta", side_effect=lambda a, b: b), \
             patch.object(q, "SingleFullCheckPool") as selector, \
             patch("scripts.run_feedback_exploration_diagnostics.validate_final"), \
             patch.object(frozen, "die_with_parent"):
            selector.return_value.select.return_value = (1, pool)
            folder = Path(tmp)
            frozen.branch(Env(), root, {}, dict(target=root["source"]), pool[0], 0,
                          dict(plan, config=cfg), folder / "old.json", 1)
            row = r.rollout(Env(), root, {}, pool[0], job, plan, r.time.monotonic()+100,
                            r.Phases(folder, "b", "j"))
            self.assertEqual(r.semantic_row(row), r.semantic_row(r.read_json(folder / "old.json")))
            self.assertEqual(len(row["events"]), 4)

    def test_safe_stop_does_not_schedule_jobs(self):
        original, records, cfg = fixture()
        plan = dict(binding="b", jobs=r.scope(original, records, cfg), config=dict(workers=4))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r.atomic(out / "STOP_AFTER_BATCH", {})
            with patch.object(r, "verify", return_value=(plan, out)), \
                 patch("experiments.repair_collection._run_jobs") as run:
                result = r.collect()
            run.assert_not_called()
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["sealed"], 0)


if __name__ == "__main__":
    unittest.main()
