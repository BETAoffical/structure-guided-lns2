import copy
import json
from pathlib import Path
import unittest

from scripts.audit_sa_history_objective_factorial import (
    MODELS, analyze, empirical_choice, rates, split_reference, training_rows,
)


def labels():
    return {c:[dict(completion=v, sustained_progress=1) for v in values]
            for c, values in {"a":[0]*8, "b":[1]*4 + [0]*4, "c":[0]*4 + [1]*4}.items()}


def example():
    cfg = dict(profiles=["dynamic", "ordered", "temporal_bag"],
               targets=["sustained_progress", "completion"], bootstrap=20, seed=3)
    ids, ps = ["a", "b", "c", "unobserved"], [.1, .2, .3, .9]
    root = dict(target=dict(id="root", map_id="map", stratum="progress"),
                old_selected_id="a", selected=ids[:3], rows=[dict(candidate_id=c) for c in ids],
                predictions={name:ps for name in MODELS})
    source = dict(states=[dict(id="root", map_id="map", stratum="progress", censored=[], labels=labels())])
    folds = [dict(root_id="root", map_id="map", profile=p, target=t, candidate_ids=ids, probabilities=ps)
             for p in cfg["profiles"] for t in cfg["targets"]]
    return dict(roots=[root]), source, folds, cfg


class ObjectiveFactorialTests(unittest.TestCase):
    def test_fixed_grid_and_no_runtime(self):
        path = Path(__file__).resolve().parents[2] / "configs/sa_history_objective_factorial.json"
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["profiles"], ["dynamic", "ordered", "temporal_bag"])
        self.assertEqual(cfg["targets"], ["sustained_progress", "completion"])
        self.assertFalse(cfg["runtime_integration_allowed"])
        self.assertEqual(cfg["new_solver_calls"], 0)

    def test_held_map_excluded_and_episode_identity_checked(self):
        root = dict(target=dict(map_id="held", item=dict(job_id="episode")))
        rows = [dict(map_id="held", episode="episode"), dict(map_id="train", episode="other")]
        self.assertEqual(training_rows(rows, root), rows[1:])
        with self.assertRaisesRegex(ValueError, "episode leakage"):
            training_rows([dict(map_id="train", episode="episode")], root)

    def test_half_selection_never_reads_evaluation_outcomes(self):
        before = labels()
        after = copy.deepcopy(before)
        for c in after:
            after[c][4:] = [dict(completion=1, sustained_progress=0)]*4
        self.assertEqual(split_reference(before, "completion", "a")["halves"][0]["selected"],
                         split_reference(after, "completion", "a")["halves"][0]["selected"])
        result = split_reference(before, "completion", "a")
        self.assertEqual([h["selected"] for h in result["halves"]], ["b", "c"])
        self.assertEqual(result["rates"]["completion"], 0)
        self.assertTrue(all(not set(h["selection_trials"]) & set(h["evaluation_trials"]) for h in result["halves"]))

    def test_ties_prefer_baseline_otherwise_candidate_id(self):
        data = labels()
        self.assertEqual(empirical_choice(data, "sustained_progress", range(8), "c"), "c")
        self.assertEqual(empirical_choice(data, "completion", range(8), "a"), "b")

    def test_missing_trials_are_not_failures(self):
        data = labels()
        data["a"].pop()
        with self.assertRaisesRegex(ValueError, "trial grid"):
            split_reference(data, "completion", "a")
        with self.assertRaisesRegex(ValueError, "unobserved"):
            rates(data, "missing")

    def test_unobserved_full_pool_choice_remains_unknown(self):
        report = analyze(*example())
        choice = report["states"][0]["policies"]["ordered/completion"]
        self.assertEqual(choice["selected"], "c")
        self.assertFalse(choice["full_pool_observed"])
        self.assertIsNone(choice["full_pool_rates"])
        self.assertTrue(report["no_promotion"])

    def test_existing_probability_mismatch_rejected(self):
        pre, report, folds, cfg = example()
        folds[0] = dict(folds[0], probabilities=[.2, .3, .4, .9])
        with self.assertRaisesRegex(ValueError, "prediction parity"):
            analyze(pre, report, folds, cfg)

    def test_complete_report_is_deterministic_and_label_order_independent(self):
        inputs = example()
        first = analyze(*inputs)
        inputs[1]["states"][0]["labels"] = dict(reversed(list(inputs[1]["states"][0]["labels"].items())))
        self.assertEqual(first, analyze(*inputs))

    def test_censoring_and_duplicate_folds_rejected(self):
        pre, report, folds, cfg = example()
        report["states"][0]["censored"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "censored"):
            analyze(pre, report, folds, cfg)
        pre, report, folds, cfg = example()
        folds[0] = folds[1]
        with self.assertRaisesRegex(ValueError, "duplicate fold"):
            analyze(pre, report, folds, cfg)


if __name__ == "__main__":
    unittest.main()
