import copy
import unittest
from scripts import probe_sa_original_terminal_support as s


class SupportTests(unittest.TestCase):
    def test_scope(self):
        c = s.config()
        self.assertIsNone(c["max_decisions"])
        self.assertEqual(c["workers"], 20)
        self.assertFalse(c["training"])
        self.assertFalse(c["formal_ttf"])
        self.assertEqual(c["trials"], 4)

    def test_first_crossing_not_minimum(self):
        rows = [dict(conflicts=c, generated=i, decision=i) for i, c in enumerate([25, 10, 2, 9, 0])]
        self.assertEqual(s.first_crossing(rows, 10, 100), rows[1])
        self.assertIsNone(s.first_crossing([dict(conflicts=0, generated=0)], 10, 100))

    def test_budget_not_replenished(self):
        self.assertIsNone(s.first_crossing([dict(conflicts=2, generated=100)], 10, 100))
        c = dict(max_decisions=None, node_budget=100)
        self.assertIsNone(s.runtime.work_stop(False, 20000, 99, c))
        self.assertEqual(s.runtime.work_stop(False, 20000, 101, c), "node_budget")
        self.assertEqual(s.runtime.work_stop(True, 20000, 101, c), "feasible")

    def test_selection_ignores_future_outcome(self):
        rows = [dict(episode_id=str(i), map_id="m", split="train", pair_id="p", replica=i,
                     root=None if i == 0 else dict(decision=300), source_success=i == 2) for i in range(4)]
        before = copy.deepcopy(rows)
        selected = s.select_roots(rows, ["m", "missing"])
        self.assertEqual(selected[0]["episode_id"], "1")
        self.assertEqual(rows, before)
        for r in rows:
            r["source_success"] = not r["source_success"]
        self.assertEqual(s.select_roots(list(reversed(rows)), ["m"])[0]["episode_id"], "1")

    def test_reject_heldout_and_duplicate(self):
        r = dict(episode_id="id", map_id="m", split="train", pair_id="p", replica=0, root={})
        with self.assertRaises(ValueError):
            s.select_roots([r, r], ["m"])
        with self.assertRaises(ValueError):
            s.select_roots([dict(r, split="validation")], ["m"])
        with self.assertRaises(ValueError):
            s.select_roots([r], ["other"])

    def test_new_streams_and_full_root_identity(self):
        r = dict(episode_id="source", source_job=dict(replica=2))
        reg = dict(binding="b", config=s.config(), roots=[r])
        jobs = s.jobs_for(reg, "branches")
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j["job_id"] for j in jobs}), 4)
        self.assertEqual({j["trial"] for j in jobs}, {0, 1, 2, 3})
        self.assertTrue(all(j["root"] == r for j in jobs))
        self.assertEqual(len(s.jobs_for(reg, "control")), 1)

    def test_summary_not_end_to_end_or_training(self):
        roots = [dict(episode_id="root", map_id="m", source_success=False, root=dict(decision=300))]
        rows = [dict(job_id=str(i), root_episode="root", trial=i, status="ok",
                     success=i == 2, stop="feasible" if i == 2 else "node_budget") for i in range(4)]
        report = s.support_summary(roots, rows, 4)
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["mixed_roots"], 1)
        self.assertEqual(report["recovered_failed_sources"], 1)
        self.assertFalse(report["automatic_promotion"])
        self.assertTrue(report["no_training"])
        with self.assertRaises(ValueError):
            s.support_summary(roots, rows[:-1], 4)
        rows[0].update(status="censored", stop="wall_safety")
        with self.assertRaises(ValueError):
            s.support_summary(roots, rows, 4)

    def test_history_retains_long_prefix_and_ages(self):
        h = s.History(dict(conflict_edges=[[0, 1]], num_of_colliding_pairs=1))
        a = s.history_signature(h)
        h.decision = 350
        h.ages[(0, 1)] = 200
        self.assertNotEqual(s.history_signature(h), a)
        self.assertEqual(h.decision, 350)

    def test_trial_stream_separated_from_original(self):
        plan = dict(config=dict(stream_seed=17))
        old = s.run.stream_draw(plan, "train-1-new-conditions", "pair", 0, 350, "pp")
        new = [s.run.stream_draw(plan, s.config()["phase"], "root", i, 350, "pp") for i in range(4)]
        self.assertEqual(len(set(new + [old])), 5)


if __name__ == "__main__":
    unittest.main()
