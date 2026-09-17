import importlib.util
from pathlib import Path
import tempfile
import unittest

from experiments.sa_policy_aligned_update import (
    split_sources, work_stop, budget_features, aggregate, training_matrix, fit_once,
)
from scripts.run_sa_policy_aligned_update import jobs_for, check_result, record, failure


def cfg():
    return dict(seed=2, source_arm="gbdt", train_maps=2, validation_maps=1, trials=2,
                continuations=["frozen", "gbdt"], reference_weight=.5, max_decisions=128, node_budget=1000)


def root():
    return dict(id="r", map_id="a", candidates=[dict(candidate_id="x", agents=[0]), dict(candidate_id="y", agents=[1])],
                features=[{"f":1.}, {"f":0.}])


def outcomes():
    return [dict(root_id="r", candidate_id=c, arm=a, trial=t, status="ok",
                 stop="feasible" if c == "x" else "decision_budget", success=c == "x", first_successor=c+str(t))
            for c in ("x", "y") for a in ("frozen", "gbdt") for t in range(2)]


class PolicyUpdateTests(unittest.TestCase):
    def test_splits_ignore_results_and_keep_maps_together(self):
        rows = [dict(job_id=f"m{m}-{i}", pair_id=f"m{m}-{i}", map_id=str(m), arm="gbdt", success=True)
                for m in range(3) for i in range(4)]
        before = split_sources(rows, cfg())
        for r in rows:
            r.update(success=False, decisions=9999, final_conflicts=200)
        self.assertEqual(before, split_sources(rows[::-1], cfg()))
        self.assertFalse(set(before["train_maps"]) & set(before["validation_maps"]))
        self.assertEqual(len(before["training_episode_ids"]), 2)
        self.assertEqual(len(before["validation_pair_ids"]), 4)

    def test_budget_is_episode_relative_not_restarted_at_root(self):
        c = cfg()
        self.assertEqual(budget_features({}, 16, 400, c),
                         {"budget.remaining_decision_fraction":112/128, "budget.remaining_node_fraction":.6})
        self.assertEqual(work_stop(False, 128, 400, c), "decision_budget")
        self.assertEqual(work_stop(False, 17, 1001, c), "node_budget")
        self.assertEqual(work_stop(True, 128, 1001, c), "feasible")
        self.assertIsNone(work_stop(False, 127, 999, c))

    def test_two_continuations_are_aggregated_not_independent_states(self):
        r = aggregate(root(), outcomes(), cfg())
        self.assertEqual(r["target"], {"x":1., "y":0.})
        rows = outcomes()
        rows[0].update(stop="decision_budget", success=False)
        self.assertEqual(aggregate(root(), rows, cfg())["target"]["x"], .75)
        rows[1].update(stop="wall_safety", success=False)
        r = aggregate(root(), rows, cfg())
        self.assertFalse(r["complete"])
        self.assertIsNone(r["target"])

    def test_missing_duplicate_and_unpaired_first_action_rejected(self):
        for change in (lambda r:r.pop(), lambda r:r.append(r[0]), lambda r:r[0].update(first_successor="wrong")):
            rows = outcomes()
            change(rows)
            with self.assertRaises(ValueError):
                aggregate(root(), rows, cfg())

    def test_training_map_isolation_weights_and_censoring(self):
        a, b, c = root(), dict(root(), id="b"), dict(root(), id="c", map_id="b")
        roots = [a, b, c]
        labels = [dict(aggregate(root(), outcomes(), cfg()), root_id=r["id"], map_id=r["map_id"]) for r in roots]
        m = training_matrix(roots, labels, cfg(), ["a", "b"], ["held"])
        self.assertEqual(m["y"], [1., -1.]*3)
        self.assertEqual(m["weights"], [.375, .375, .75, .75, .375, .375])
        with self.assertRaises(ValueError):
            training_matrix(roots, labels, cfg(), ["a", "b"], ["b"])
        labels[0]["complete"] = False
        with self.assertRaises(ValueError):
            training_matrix(roots, labels, cfg(), ["a", "b"], ["held"])

    def test_resume_hash_change_and_error_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            record(folder/"result.json", dict(binding="x", status="ok", files={}))
            self.assertEqual(check_result(folder, dict(binding="x"))["status"], "ok")
            with self.assertRaises(ValueError):
                check_result(folder, dict(binding="wrong"))
        self.assertEqual(failure(dict(job_id="x"), "error", "cause")["error"], "cause")

    def test_schedule_never_labels_validation_maps(self):
        plan = dict(config=cfg(), roots=[dict(id="r", candidates=["x", "y"])],
                    split=dict(validation_pair_ids=["task-s7"]), cases=[dict(task_id="task")])
        jobs = jobs_for(plan, "labels")
        self.assertEqual(len(jobs), 8)
        self.assertEqual({j["root"]["id"] for j in jobs}, {"r"})
        ev = jobs_for(plan, "evaluation")
        self.assertEqual(len(ev), 3)
        self.assertEqual({j["arm"] for j in ev}, {"frozen", "gbdt", "updated"})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Windows sklearn only")
    def test_fixed_fit_and_portable(self):
        from experiments.sa_paired_closed_loop import portable_payload, portable_model
        m = dict(x=[[i, -i] for i in range(-15,16)], y=[float(i>0) for i in range(-15,16)],
                 weights=[1.]*31, names=["f"])
        model = fit_once(m)
        portable = portable_model(portable_payload(model), ["f"], native=False)
        s = dict(anchor_id="x", agent_ids=[0,1], candidates=[dict(c, features=f) for c,f in zip(root()["candidates"], root()["features"])])
        self.assertEqual(model.rank(s), portable.rank(s))


if __name__ == "__main__":
    unittest.main()
