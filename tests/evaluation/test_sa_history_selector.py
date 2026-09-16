import copy
import unittest
import os
from pathlib import Path
import sys
import tempfile
import time
import importlib.util

from experiments.sa_history_selector import History, choose_candidates, targets, aggregate_trials, select_prediction, pareto_indices


def state():
    return dict(num_of_colliding_pairs=1, conflict_edges=[[2, 7]], feasible=False,
                agents=[dict(id=2, path=[0, 1]), dict(id=7, path=[2, 1]), dict(id=19, path=[3, 3])])


def event(decision=0):
    return dict(decision=decision, metrics=dict(neighborhood=[2, 7], conflicts_before=1, conflicts_after=1,
                                               pp_rolled_back=False, acceptance_evaluated=True, pp_failure_reason="none"))


def orphan_parent(connection):
    from scripts.train_sa_history_selector import die_with_parent
    parent = os.getpid()
    child = os.fork()
    if child == 0:
        die_with_parent(parent)
        connection.send(os.getpid())
        time.sleep(60)
        os._exit(0)
    time.sleep(60)


class HistoryTests(unittest.TestCase):
    def test_no_future_and_noncontinuous_ids(self):
        s = state(); h = History(s); c = dict(agents=[7, 2])
        before = h.features(s, c, 1000)
        self.assertEqual(before["history.same_set_count"], 0)
        self.assertEqual(before["history.unknown_context"], 1)
        h.observe(s, event(), s)
        after = h.features(s, c, 990)
        self.assertGreater(after["history.same_set_count"], 0)
        self.assertEqual(after["history.unknown_context"], 0)
        self.assertEqual(before["history.same_set_count"], 0)

    def test_outside_change_invalidates_not_erases_history(self):
        s = state(); h = History(s); h.observe(s, event(), s)
        changed = copy.deepcopy(s); changed["agents"][2]["path"] = [3, 4]
        f = h.features(changed, dict(agents=[2, 7]), 990)
        self.assertEqual(f["history.valid_context_count"], 0)
        self.assertGreater(f["history.stale_context_count"], 0)

    def test_input_order_invariant(self):
        s = state(); h = History(s)
        a = h.features(s, dict(agents=[2, 7]), 1000)
        s["agents"].reverse(); s["conflict_edges"] = [[7, 2]]
        self.assertEqual(a, h.features(s, dict(agents=[7, 2]), 1000))

    def test_repetition_not_deduplicated(self):
        s = state(); h = History(s)
        for i in range(40): h.observe(s, event(i), s)
        f = h.features(s, dict(agents=[2, 7]), 1)
        self.assertEqual(f["history.same_set_count"], 1)
        self.assertEqual(f["history.incident_age_mean"], 1)

    def test_unknown_agent_and_bad_prefix_rejected(self):
        s = state(); h = History(s)
        with self.assertRaises(ValueError): h.features(s, dict(agents=[99]), 1)
        with self.assertRaises(ValueError): h.observe(s, event(1), s)

    def test_labels_not_features(self):
        s = state(); h = History(s)
        x = h.features(s, dict(agents=[2, 7]), 1)
        s["runtime"] = 999; s["trial_outcome"] = dict(success=True)
        self.assertEqual(x, h.features(s, dict(agents=[2, 7]), 1))

    def test_trial_aggregation(self):
        labels = dict(conflicts=.5, persistence=.25, feasible=0)
        rows = [dict(trial=i, members=[2, 7], target=labels) for i in range(4)]
        self.assertEqual(aggregate_trials(rows, 4), labels)
        with self.assertRaises(ValueError): aggregate_trials(rows[:3], 4)
        rows[-1]["members"] = [2, 19]
        with self.assertRaises(ValueError): aggregate_trials(rows, 4)

    def test_censor_not_failure(self):
        s = state(); m = event()["metrics"]; m["pp_failure_reason"] = "time_limit"
        self.assertIsNone(targets(s, s, [([2, 7], 1)], m))
        rows = [dict(trial=i, members=[2], target=None) for i in range(4)]
        self.assertIsNone(aggregate_trials(rows, 4))

    def test_persistence_not_feasibility(self):
        s = state(); a = copy.deepcopy(s); a["conflict_edges"] = [[2, 19]]
        t = targets(s, a, [([2, 7], 1)], event()["metrics"])
        self.assertEqual(t, dict(conflicts=1, persistence=0, feasible=0))

    def test_candidate_sampling(self):
        pool = [dict(candidate_id=str(i), agents=list(range(n)), score=-i) for i,n in enumerate([4,8,16,5,6,7,9])]
        choice = choose_candidates(pool, 2, 4)
        self.assertEqual(len(choice), 6)
        self.assertEqual(choice[0], 2)
        self.assertEqual(choice, choose_candidates(pool, 2, 4))
        self.assertTrue({4,8,16} <= {len(pool[i]["agents"]) for i in choice})

    def test_quality_guard(self):
        pred = [dict(conflicts=.1, persistence=1, feasible=0), dict(conflicts=.5, persistence=0, feasible=0)]
        self.assertEqual(select_prediction(pred, ["b","a"], 10), 0)
        pred[1]["conflicts"] = .15
        self.assertEqual(select_prediction(pred, ["b","a"], 10), 1)
        self.assertEqual(select_prediction(pred, ["b","a"], 10, risk=False), 0)

    def test_ties_and_pareto(self):
        pred = [dict(conflicts=1, persistence=1, feasible=0)] * 2
        self.assertEqual(select_prediction(pred, ["b","a"], 10), 1)
        self.assertEqual(pareto_indices(pred), [0,1])
        with self.assertRaises(ValueError): select_prediction([dict(conflicts=float("nan"))], ["a"], 1)

    def test_receipt_cannot_omit_control(self):
        from scripts.train_sa_history_selector import validate_receipt
        from experiments._common import write_json, sha256_file
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            root = dict(binding="x", source={"id":"s"}, candidates=[dict(candidate_id="c", agents=[2,7])],
                        feature_rows=[{}], old_selected_id="c")
            write_json(folder/"root.json", root)
            for files in ({}, {"root.json":sha256_file(folder/"root.json")}):
                write_json(folder/"receipt.json", dict(binding="x", files=files))
                with self.assertRaisesRegex(ValueError, "incomplete receipt"):
                    validate_receipt(folder, dict(binding="x", config=dict(max_candidates=6, trials=4)), {"id":"s"})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "training uses Windows sklearn")
    def test_heldout_labels_do_not_change_predictions(self):
        from scripts.train_sa_history_selector import train_fold
        rows = [dict(state_id=f"s{s}", map_id=str(s//2), candidate_id=str(c),
                     features={"x":float(c), "history.age":float(s)},
                     labels=dict(conflicts=float(c), persistence=float(c), feasible=float(1-c)))
                for s in range(6) for c in range(2)]
        config = dict(model=dict(max_iter=2, max_leaf_nodes=3, min_samples_leaf=1, early_stopping=False, random_state=4))
        first = train_fold((rows, "2", config))
        for row in rows:
            if row["map_id"] == "2": row["labels"] = dict(conflicts=99., persistence=99., feasible=99.)
        self.assertEqual(first, train_fold((rows, "2", config)))
        self.assertEqual({r["state_id"] for r in first}, {"s4", "s5"})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "training uses Windows scipy")
    def test_bootstrap_point_matches_state_summary(self):
        from scripts.train_sa_history_selector import evaluate
        rows, predictions = [], []
        for s in range(3):
            for c in range(2):
                label = dict(conflicts=float(c), persistence=float(c), feasible=0.)
                rows.append(dict(state_id=str(s), map_id=str(s//2), decision=16, candidate_id=str(c), size=4+c*4,
                                 labels=label, trial_labels=[label]*4, initial_conflicts=10, old_selected=c==0))
                for profile in ("dynamic", "history"):
                    predicted = 1-c if profile == "history" and s < 2 else c
                    predictions.append(dict(profile=profile, state_id=str(s), candidate_id=str(c),
                                            prediction=dict(conflicts=float(predicted), persistence=float(predicted), feasible=0.)))
        report = evaluate(rows, predictions, dict(seed=4, bootstrap_samples=20, quality_tolerance_pairs=1.))
        difference = report["summary"]["history"]["conflict_regret"]-report["summary"]["dynamic"]["conflict_regret"]
        self.assertAlmostEqual(difference, report["comparisons"]["dynamic"]["conflict_regret"]["delta"])

    @unittest.skipUnless(sys.platform == "linux", "Linux process death guard")
    def test_children_die_when_root_is_killed(self):
        import multiprocessing
        import signal
        ctx = multiprocessing.get_context("fork")
        receiver, sender = ctx.Pipe(duplex=False)
        parent = ctx.Process(target=orphan_parent, args=(sender,))
        parent.start()
        child = None
        try:
            self.assertTrue(receiver.poll(10))
            child = receiver.recv()
            parent.kill(); parent.join(5)
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                path = Path(f"/proc/{child}/stat")
                if not path.exists() or path.read_text().split()[2] == "Z":
                    break
                time.sleep(.05)
            else:
                self.fail("orphan trial survived root death")
        finally:
            if parent.is_alive(): parent.kill()
            parent.join(5)
            if child is not None and Path(f"/proc/{child}").exists():
                try: os.kill(child, signal.SIGKILL)
                except ProcessLookupError: pass
            receiver.close(); sender.close()


if __name__ == "__main__":
    unittest.main()
