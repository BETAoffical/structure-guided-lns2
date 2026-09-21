from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import read_json
from experiments.neighborhood_candidates import candidate_id
from scripts import collect_sa_structural_focus as c
from scripts import collect_sa_source_matched as old
from scripts.collect_sa_history_candidate_bridge import randomization


def fixture():
    anchor = dict(candidate_id=candidate_id(list(range(16))), agents=list(range(16)), actual_size=16,
                  selection_families=["structpool-conflict-component:16"], score=1.0)
    alt = dict(candidate_id=candidate_id(list(range(16, 32))), agents=list(range(16, 32)), actual_size=16,
               selection_families=["diagnostic-second-core:structpool-conflict-component:16"])
    entry = dict(state_id="s", map_id="m", decision=4, source_root="root.json", anchor_id=anchor["candidate_id"],
                 alternate_id=alt["candidate_id"], candidates=[anchor, alt])
    root = dict(state=dict(agents=[dict(id=i) for i in range(32)], feasible=False, sum_of_costs=10,
                          low_level=dict(generated=100)),
                source=dict(id="s", decision=4), previous_best=1,
                control_event=dict(pool=[anchor], selected_index=0))
    plan = dict(binding="b", roots=[entry], config=dict(trials=8, seed=202609211, horizon=32, sustain=8))
    return root, entry, plan


class FocusContinuationTests(unittest.TestCase):
    def test_config_and_preregistered_scope(self):
        cfg = read_json(c.CONFIG)
        self.assertEqual((cfg["trials"], cfg["horizon"], cfg["workers"], cfg["pp_seconds"], cfg["trial_seconds"]), (8, 32, 20, 5, 180))
        self.assertTrue(cfg["no_ttf"])
        self.assertFalse(cfg["training_allowed"] or cfg["automatic_promotion"])

    def test_forced_root_keeps_original_anchor_and_does_not_mutate(self):
        root, entry, _ = fixture()
        before = deepcopy(root)
        self.assertIs(c.forced_root(root, entry, entry["anchor_id"]), root)
        changed = c.forced_root(root, entry, entry["alternate_id"])
        self.assertEqual(changed["control_event"]["pool"], root["control_event"]["pool"] + [entry["candidates"][1]])
        self.assertEqual(root, before)
        self.assertNotIn("score", changed["control_event"]["pool"][-1])

    def test_member_identity_and_unknown_agent_rejected(self):
        root, entry, _ = fixture()
        alt = entry["candidates"][1]
        for agents in (list(range(16, 31)) + [100], [16] * 16, list(range(17, 32))):
            changed = deepcopy(entry)
            changed["candidates"][1].update(agents=agents, candidate_id=candidate_id(agents))
            changed["alternate_id"] = changed["candidates"][1]["candidate_id"]
            with self.assertRaises(ValueError):
                c.forced_root(root, changed, changed["alternate_id"])
        with self.assertRaises(ValueError):
            c.forced_root(root, entry, "unregistered")
        self.assertEqual(len(alt["agents"]), 16)

    def test_wrong_original_choice_rejected(self):
        root, entry, _ = fixture()
        root["control_event"]["pool"][0] = dict(entry["candidates"][0], score=2)
        with self.assertRaisesRegex(ValueError, "original choice"):
            c.forced_root(root, entry, entry["alternate_id"])

    def test_private_binding_leaves_frozen_module_unchanged(self):
        before = old.trial_worker.__globals__["restore"]
        fake = lambda *args: "new"
        def function():
            return read_json("unused")
        self.assertEqual(c.private_call(function, dict(read_json=fake)), "new")
        self.assertIs(old.trial_worker.__globals__["restore"], before)

    def test_schedule_pairing_and_fresh_streams(self):
        _, entry, plan = fixture()
        jobs = old.schedule(plan)
        self.assertEqual(len(jobs), 16)
        self.assertEqual([j["trial"] for j in jobs[:4]], [0, 0, 1, 1])
        self.assertEqual([j["candidate_id"] for j in jobs[:2]], [entry["anchor_id"], entry["alternate_id"]])
        actual = {randomization("s", t, d, plan["config"]) for t in range(8) for d in range(4, 36)}
        self.assertEqual(len(actual), 256)
        for previous in (202609182, 20260919, 20260921):
            other = {randomization("s", t, d, dict(plan["config"], seed=previous)) for t in range(8) for d in range(4, 36)}
            self.assertFalse(actual & other)

    def test_unknown_not_failure(self):
        _, _, plan = fixture()
        job = old.schedule(plan)[0]
        row = dict(status="censored", root_id="s", candidate_id=job["candidate_id"], trial=0, stop="hard_fuse")
        self.assertIsNone(c.validate_trial(row, job, plan))
        with self.assertRaises(ValueError):
            c.validate_trial(dict(row, trial=1), job, plan)
        with self.assertRaises(ValueError):
            c.validate_trial(dict(row, stop="crash"), job, plan)

    def test_timeout_resume_and_tamper_refusal(self):
        _, _, plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            j = old.schedule(plan)[0]
            job = dict(j, folder=str(out / "trials" / j["job_id"]), plan=plan)
            self.assertIsNone(c.failure(job, "timeout", "fuse")["completion"])
            pending, done = c.inventory(plan, out, [v["job_id"] for v in old.schedule(plan)])
            self.assertEqual((len(pending), len(done)), (15, 1))
            path = Path(job["folder"]) / "result.json"
            old.atomic(path, dict(read_json(path), trial=7))
            receipt_before = (path.parent / "receipt.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                c.inventory(plan, out, [v["job_id"] for v in old.schedule(plan)])
            self.assertEqual(c.failure(job, "timeout", "late")["status"], "error")
            self.assertEqual((path.parent / "receipt.json").read_bytes(), receipt_before)

    def test_partial_output_requires_review(self):
        _, _, plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            j = old.schedule(plan)[0]
            (out / "trials" / j["job_id"]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "requires review"):
                c.inventory(plan, out, [j["job_id"]])

    def test_completed_scope_cannot_be_extended_by_resume(self):
        _, _, plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            j = old.schedule(plan)[0]
            job = dict(j, folder=str(out / "trials" / j["job_id"]), plan=plan)
            c.failure(job, "timeout", "fuse")
            with self.assertRaisesRegex(ValueError, "out-of-scope"):
                c.inventory(plan, out, [])

    def test_semantic_validation_rejects_incomplete_pp_and_summary_drift(self):
        from scripts import run_sa_path_quality as q
        root, entry, plan = fixture()
        job = old.schedule(plan)[0]
        after = dict(root["state"], low_level=dict(generated=103))
        event = dict(pool=root["control_event"]["pool"], delta={},
                     metrics=dict(pp_failure_reason="", acceptance_evaluated=True))
        row = dict(status="ok", root_id="s", candidate_id=entry["anchor_id"], trial=0, no_ttf=True,
                   events=[event], stop="wall_safety", final_cost=10, generated=3)
        with patch.object(c, "read_json", return_value=root), patch.object(c, "branch_labels", return_value=None), \
             patch.object(q, "apply_state_delta", return_value=after), \
             patch("scripts.run_feedback_exploration_diagnostics.validate_final"):
            self.assertIsNone(c.validate_trial(row, job, plan))
            for key, value in (("final_cost", 11), ("generated", 4), ("stop", "feasible")):
                with self.assertRaises(ValueError):
                    c.validate_trial(dict(row, **{key: value}), job, plan)
            changed = deepcopy(row)
            changed["events"][0]["metrics"]["acceptance_evaluated"] = False
            with self.assertRaisesRegex(ValueError, "incomplete PP"):
                c.validate_trial(changed, job, plan)

    def test_preflight_both_arms_at_early_and_latest_root(self):
        _, entry, plan = fixture()
        plan["roots"] = [dict(entry, state_id=f"s{i}", decision=d) for i, d in enumerate([4, 16, 32])]
        ids = c.preflight_jobs(plan)
        self.assertEqual(len(ids), 4)
        self.assertTrue(all("-t0" in v and not v.startswith("s1-") for v in ids))

    def records(self):
        return [dict(state_id=f"s{i}", map_id=f"m{i % 2}", decision=4, family="f", removed_members=2,
                     anchor=[False] * 8, alternate=[True] * 8) for i in range(4)]

    def test_map_bootstrap_not_trial_bootstrap_and_repeatable(self):
        cfg = dict(trials=8, bootstrap=5000, bootstrap_seed=1)
        a = c.summarize(self.records(), cfg)
        self.assertEqual(a, c.summarize(self.records(), cfg))
        self.assertEqual(a["map_bootstrap_ci95"], [1, 1])
        self.assertEqual(a["paired_trials"]["gains"], 32)
        self.assertEqual(a["decision"], "positive_development_signal")
        self.assertFalse(a["controller_promotion"] or a["training_allowed"])

    def test_unknown_bounds_keep_full_denominator(self):
        records = self.records()
        records[0]["alternate"][0] = None
        result = c.summarize(records, dict(trials=8, bootstrap=10, bootstrap_seed=1))
        self.assertEqual(result["arms"]["alternate"]["unknown"], 1)
        self.assertEqual(result["rows"][0]["delta_bounds"], [7 / 8, 1])
        self.assertEqual(result["decision"], "unknown_resource_censoring")

    def test_unbalanced_map_denominators_reported_separately(self):
        records = self.records()[:3]
        records[1]["alternate"] = [False] * 8
        result = c.summarize(records, dict(trials=8, bootstrap=10, bootstrap_seed=1))
        self.assertAlmostEqual(result["root_weighted_delta_bounds"][0], 2 / 3)
        self.assertEqual(result["map_balanced_delta_bounds"], [.5, .5])

    def test_safe_stop_drains_without_new_batch(self):
        _, _, plan = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old.atomic(out / "STOP_AFTER_BATCH", dict(binding=plan["binding"]))
            with patch.object(c, "verify"), patch.object(c, "_run_jobs") as run:
                result = c.execute(plan, out, [j["job_id"] for j in old.schedule(plan)], 20)
            self.assertEqual(result["status"], "paused")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
