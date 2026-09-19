import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_sa_matched_remaining as prep
from scripts.prepare_sa_source_matched import select, work_budget
from tests.evaluation.test_sa_source_matched import fixture


def complement_fixture():
    rows, cfg = fixture()
    cfg.update(universe_roots=6, excluded_roots=2, remaining_roots=4,
               planned_pp_safety_seconds=5, planned_trial_timeout_seconds=180,
               planned_long_prefix_workers=4, long_prefix_from_decision=32,
               trial_seed=202609182)
    return rows, [rows[0]["state_id"], rows[3]["state_id"]], cfg


class MatchedRemainingTests(unittest.TestCase):
    def test_complete_complement_not_new_sampling(self):
        rows, excluded, cfg = complement_fixture()
        chosen = prep.select_complement(rows, excluded, cfg)
        self.assertEqual({r["state_id"] for r in chosen}, {r["state_id"] for r in rows} - set(excluded))
        self.assertEqual(len(chosen), 4)

    def test_reuses_historical_family_member_selection(self):
        rows, excluded, cfg = complement_fixture()
        chosen = prep.select_complement(rows, excluded, cfg)
        for row in chosen:
            source = next(r for r in rows if r["state_id"] == row["state_id"])
            old = select([source], cfg | dict(maps=1, roots_per_map=1))[0]
            self.assertEqual(row["pair_ids"], old["pair_ids"])
            self.assertEqual(row["family"], old["family"])
            self.assertEqual(row["anchor_id"], old["anchor_id"])
            self.assertEqual({c["candidate_id"] for c in row["candidates"]}, set(row["pair_ids"]))

    def test_outcomes_scores_features_and_order_cannot_select_pairs(self):
        rows, excluded, cfg = complement_fixture()
        expected = prep.select_complement(rows, excluded, cfg)
        for row in rows:
            row.update(values=[False]*8, label=9, features={"future": 99})
            row["pool"].reverse()
            for candidate in row["pool"]:
                candidate.update(score=-10000, trials=[True]*8, repair_time=123)
        self.assertEqual(prep.select_complement(list(reversed(rows)), list(reversed(excluded)), cfg), expected)

    def test_anchor_inside_pair_is_retained_once(self):
        rows, excluded, cfg = complement_fixture()
        for row in rows:
            row["anchor_id"] = "a"
        chosen = prep.select_complement(rows, excluded, cfg)
        self.assertTrue(all(r["anchor_id"] in r["pair_ids"] and len(r["candidates"]) == 2 for r in chosen))

    def test_duplicate_or_foreign_exclusions_rejected(self):
        rows, excluded, cfg = complement_fixture()
        for ids in ([excluded[0]]*2, [excluded[0], "unknown"], excluded[:1]):
            with self.assertRaises(ValueError):
                prep.select_complement(rows, ids, cfg)

    def test_repeated_episode_or_missing_root_rejected(self):
        rows, excluded, cfg = complement_fixture()
        with self.assertRaises(ValueError):
            prep.select_complement(rows[:-1], excluded, cfg)
        rows[1]["episode"] = rows[0]["episode"]
        with self.assertRaisesRegex(ValueError, "source episode"):
            prep.select_complement(rows, excluded, cfg)

    def test_no_family_fallback_and_actual_size_checked(self):
        for mutation in (
            lambda c: c.update(selection_families=["random:16", "target:16"]),
            lambda c: c.update(actual_size=15),
        ):
            rows, excluded, cfg = complement_fixture()
            mutation(rows[1]["pool"][1])
            with self.assertRaises(ValueError):
                prep.select_complement(rows, excluded, cfg)

    def test_staged_batches_cover_every_job_once(self):
        rows, excluded, cfg = complement_fixture()
        rows[1]["decision"] = 31
        rows[2]["decision"] = 32
        chosen = prep.select_complement(rows, excluded, cfg)
        jobs, phases = prep.planned_batches(chosen, cfg)
        flattened = [jid for phase in phases for batch in phase["batches"] for jid in batch]
        self.assertEqual(len(jobs), 4*2*8)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {j["job_id"] for j in jobs})
        self.assertEqual([p["workers"] for p in phases], [20, 4])
        self.assertEqual(phases[1]["roots"], 1)
        self.assertTrue(all(len(b) <= p["workers"] for p in phases for b in p["batches"]))
        for root in chosen:
            for trial in range(8):
                self.assertEqual(sum(j["root"] == root and j["trial"] == trial for j in jobs), 2)

    def test_budget_accounts_for_independent_reset_and_prefix(self):
        rows, excluded, cfg = complement_fixture()
        chosen = prep.select_complement(rows, excluded, cfg)
        budget = work_budget(chosen, cfg)
        self.assertEqual(budget["independent_resets"], 64)
        self.assertEqual(budget["maximum_prefix_replay_repairs"], 16*sum(r["decision"] for r in chosen))
        self.assertEqual(budget["maximum_rollout_repairs"], 64*32)
        self.assertEqual(budget["executed_jobs"], 0)

    def test_registered_stream_is_shared_and_not_old_stream(self):
        rows, excluded, cfg = complement_fixture()
        chosen = prep.select_complement(rows, excluded, cfg)
        audit = prep.audit_streams(chosen[0], {"seed": 20260919}, cfg)
        self.assertEqual(audit["overlap"], 0)
        self.assertEqual(len(set(audit["new_streams"])), 8)
        self.assertEqual(audit, prep.audit_streams(chosen[0], {"seed": 20260919}, cfg))
        self.assertTrue(audit["candidates_share_stream_within_trial"])
        with self.assertRaisesRegex(ValueError, "stream collision"):
            prep.audit_streams(chosen[0], {"seed": cfg["trial_seed"]}, cfg)

    def test_prefix_checked_without_search(self):
        events = [dict(decision=i, pool=["saved"], action={"agents":[1, 2]}) for i in range(3)]
        root = dict(source=dict(decision=2), control_event=events[2])
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(e) for e in events), encoding="utf8")
            self.assertEqual(prep.prefix_event(trace, root), 3)
            bad = copy.deepcopy(root)
            bad["control_event"]["pool"] = ["changed"]
            with self.assertRaisesRegex(ValueError, "pool/action"):
                prep.prefix_event(trace, bad)
            trace.write_text(json.dumps(events[0]), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "missing"):
                prep.prefix_event(trace, root)
            trace.write_text(json.dumps(events[1]), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "gap"):
                prep.prefix_event(trace, root)

    def test_fixed_boundary_and_path(self):
        cfg = json.loads((prep.ROOT / prep.CONFIG).read_text())
        prep.check_config(cfg)
        for changes in (dict(collection_ready=True), dict(formal_ttf_allowed=True), dict(training_allowed=True),
                        dict(remaining_roots=30), dict(planned_trials=16), dict(extra_anchor_jobs=1),
                        dict(planned_long_prefix_workers=20), dict(trial_seed=0),
                        dict(role="test_ood"), dict(output="../elsewhere"), dict(output="build")):
            with self.assertRaises(ValueError):
                prep.check_config(cfg | changes)

    def test_repeat_outputs_and_tampering_are_rejected(self):
        rows, excluded, cfg = complement_fixture()
        chosen = prep.select_complement(rows, excluded, cfg)
        _, phases = prep.planned_batches(chosen, cfg)
        plan = dict(config=cfg | dict(output="build/preparation"), roots=chosen, binding="binding", inputs={},
                    stream_audit=[prep.audit_streams(r, {"seed": 20260919}, cfg) for r in chosen],
                    coverage={}, budget={}, disk_estimate={}, phases=phases, collection_ready=False,
                    solver_calls=0, model_fits=0, native_replay_performed=False, blockers=["not collected"])
        with tempfile.TemporaryDirectory() as directory, patch.object(prep, "ROOT", Path(directory)), patch.object(prep, "build", return_value=plan):
            result = prep.run("prepare")
            self.assertEqual(result, prep.run("prepare"))
            self.assertEqual(result, prep.run("verify"))
            manifest = Path(directory) / "build/preparation/job_manifest.json"
            manifest.write_text("[]", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "output changed"):
                prep.run("verify")


if __name__ == "__main__":
    unittest.main()
