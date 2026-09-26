import copy
import importlib.util
from collections import Counter, defaultdict
import json
from pathlib import Path
import unittest

from experiments._common import json_fingerprint
from experiments.sa_uncertainty_audit import (
    map_draws, training_matrix, inference_state, fit_member, ensemble_prediction, state_result, summarize,
)
from tests.evaluation.test_sa_low_complexity_ranker import fixture

ROOT = Path(__file__).resolve().parents[2]
CFG = dict(members=20, seed=37, high_agreement=.75, bootstrap=100, states=4, maps=3)


def prediction(s, scores):
    selected = min(scores, key=lambda c: (-scores[c], c != s["anchor_id"], c))
    return dict(state_id=s["state_id"], scores=scores, selected=selected)


class UncertaintyTests(unittest.TestCase):
    def test_scope_and_no_new_solver(self):
        cfg = json.loads((ROOT / "configs/sa_uncertainty_audit.json").read_text())
        self.assertEqual((cfg["states"], cfg["maps"], cfg["members"], cfg["workers"]), (47, 8, 20, 20))
        self.assertEqual(cfg["new_solver_calls"], 0)
        self.assertFalse(cfg["automatic_promotion"])

    def test_map_bootstrap_and_holdout_exclusion(self):
        d = fixture()
        for member in range(-1, 20):
            draws = map_draws(d, "held", member, CFG)
            self.assertNotIn("held", draws)
            self.assertEqual(len(draws), 2)
            a = training_matrix(d, "held", member, CFG)
            self.assertEqual(a, training_matrix(d, "held", member, CFG))
            self.assertNotIn("s3", a["train_ids"])
            totals = defaultdict(float)
            for sid, w in zip(a["owners"], a["weights"]):
                totals[sid] += w
            for sid, w in a["state_weights"].items():
                self.assertAlmostEqual(totals[sid], w)
            counts = Counter(draws)
            maps = {s["state_id"]: s["map_id"] for s in d["states"]}
            for m, count in counts.items():
                self.assertAlmostEqual(sum(w for sid, w in totals.items() if maps[sid] == m), 3*count/2)

    def test_held_labels_and_features_do_not_change_training(self):
        d = fixture()
        before = training_matrix(d, "held", 3, CFG)
        for c in d["states"][-1]["candidates"]:
            c["features"]["realized.signal"] = 999
            c["trials"] = copy.deepcopy(d["states"][0]["candidates"][0]["trials"])
        self.assertEqual(before, training_matrix(d, "held", 3, CFG))

    def test_input_candidate_order_invariant(self):
        d = fixture()
        before = training_matrix(d, "held", 2, CFG)
        d["states"].reverse()
        for s in d["states"]:
            s["candidates"].reverse()
        self.assertEqual(before, training_matrix(d, "held", 2, CFG))

    def test_no_outcome_in_inference(self):
        s = fixture()["states"][0]
        clean = inference_state(s)
        self.assertNotIn("trials", clean["candidates"][0])
        for c in s["candidates"]:
            c["trials"] = []
            c["future_paths"] = [99]
        self.assertEqual(clean, inference_state(s))

    def test_vote_expectation_not_calibrated_probability(self):
        s = fixture()["states"][0]
        a = prediction(s, {"0": 3., "1": 2., "2": 1., "3": 0.})
        b = prediction(s, {"0": 0., "1": 2., "2": 1., "3": 0.})
        e = ensemble_prediction(s, [a, a, a, b])
        self.assertEqual(e["vote_probabilities"], {"0": .75, "1": .25, "2": 0., "3": 0.})
        r = state_result(s, [a, a, a, b], b, CFG)
        self.assertAlmostEqual(r["methods"]["vote_expectation"], .75*7/8+.25*4/8)
        self.assertFalse(e["calibrated_confidence"])
        self.assertTrue(r["high_agreement"])

    def test_ties_keep_anchor_and_not_model_confidence(self):
        s = fixture()["states"][0]
        p = prediction(s, {str(i): 0. for i in range(4)})
        e = ensemble_prediction(s, [p]*20)
        self.assertEqual(e["selected"], s["anchor_id"])
        self.assertEqual(e["agreement"], 1.)
        self.assertFalse(e["calibrated_confidence"])

    def test_corrupt_predictions_and_missing_trials_rejected(self):
        s = fixture()["states"][0]
        p = prediction(s, {str(i): float(i) for i in range(4)})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ensemble_prediction(s, [dict(p, selected="0")])
        s["candidates"][0]["trials"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            state_result(s, [p], p, CFG)

    def test_held_unknown_member_and_censored_rejected(self):
        d = fixture()
        with self.assertRaises(ValueError):
            training_matrix(d, "absent", 0, CFG)
        with self.assertRaises(ValueError):
            training_matrix(d, "held", 20, CFG)
        d["states"][0]["candidates"][0]["trials"][0].update(completed=None, stop="wall_safety", steps=0, final_conflicts=1)
        with self.assertRaisesRegex(ValueError, "censored"):
            training_matrix(d, "held", -1, CFG)

    def test_no_strict_gate_or_automatic_promotion(self):
        data = fixture()
        rows = []
        for s in data["states"]:
            p = prediction(s, {str(i): float(-i) for i in range(4)})
            rows.append(state_result(s, [p]*20, p, CFG))
        report = summarize(rows, CFG)
        self.assertEqual(report, summarize(rows, CFG))
        self.assertFalse(report["confidence_calibrated"])
        self.assertFalse(report["closed_loop_tested"])
        self.assertEqual(report["positive_point_estimates"], ["ensemble_mean", "vote_expectation"])
        self.assertIsNone(report["agreement_groups"]["low"]["anchor_gain"])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'registered sklearn environment required')
    def test_fit_deterministic(self):
        import sklearn
        if sklearn.__version__ != "1.5.0":
            self.skipTest("registered sklearn only")
        d = fixture()
        a = fit_member(d, "held", 0, CFG)
        self.assertEqual(a, fit_member(d, "held", 0, CFG))


if __name__ == "__main__":
    unittest.main()
