import copy
import unittest

from experiments.sa_selector_contract_audit import feature_support, contrast_diagnostic, label_curves
from experiments.sa_linear_closed_loop import LinearRanker
from tests.evaluation.test_sa_low_complexity_ranker import fixture, model


class SelectorContractTests(unittest.TestCase):
    def test_shared_features_are_not_candidate_evidence(self):
        d = fixture()
        support = feature_support(d, model())
        self.assertEqual(support["never_varying"], ["sa.log_temperature"])
        self.assertEqual(support["max_abs_contrast"]["realized.signal"], 3.)
        self.assertEqual(support["history_features"], [])
        m = LinearRanker(dict(model(), schema="lns2.sa.linear_h32_runtime_probe.v1"))
        s = copy.deepcopy(d["states"][0])
        original = m.rank(s)
        for c in s["candidates"]:
            c["features"]["sa.log_temperature"] = 100
        self.assertEqual(original, m.rank(s))

    def test_absolute_range_does_not_imply_contrast_support(self):
        d = fixture()
        for i, s in enumerate(d["states"]):
            for j, c in enumerate(s["candidates"]):
                c["features"]["realized.signal"] = 10*i + j
        support = feature_support(d, model())
        fs = [{"realized.signal": 0., "sa.log_temperature": 2.},
              {"realized.signal": 30., "sa.log_temperature": 2.}]
        r = contrast_diagnostic(fs, ["a", "b"], "a", "b", support)
        self.assertEqual(r["absolute_outside_fraction"], 0)
        self.assertEqual(r["contrast_outside"], ["realized.signal"])
        self.assertEqual(contrast_diagnostic(fs, ["a", "b"], "a", "a", support)["contrast_outside"], [])

    def test_unseen_variation_and_schema_errors(self):
        support = feature_support(fixture(), model())
        fs = [{"realized.signal": 1., "sa.log_temperature": 2.},
              {"realized.signal": 2., "sa.log_temperature": 3.}]
        self.assertEqual(contrast_diagnostic(fs, ["a", "b"], "a", "b", support)["previously_unvarying"],
                         ["sa.log_temperature"])
        with self.assertRaises(ValueError):
            contrast_diagnostic(fs, ["a", "a"], "a", "a", support)
        with self.assertRaises(ValueError):
            contrast_diagnostic([fs[0], {}], ["a", "b"], "a", "b", support)

    def test_tied_horizon_can_hide_earlier_completion(self):
        d = fixture()
        for s in d["states"]:
            for j, c in enumerate(s["candidates"]):
                for t in c["trials"]:
                    t.update(completed=True, stop="feasible", final_conflicts=0, steps=1+8*j)
        pred = {s["state_id"]: {"linear_difference": "3", "gbdt": "0"} for s in d["states"]}
        r = label_curves(d, pred)
        self.assertEqual(r["map_macro"]["linear_difference"], {"8": 0., "16": 0., "32": 1.})
        self.assertEqual(r["map_macro"]["gbdt"], {"8": 1., "16": 1., "32": 1.})
        self.assertEqual(r["tied_h32_pairs_with_different_completion_curves"], 24)
        self.assertFalse(r["horizon_selected"])
        self.assertFalse(r["new_labels_for_training"])
        self.assertEqual(r["all_success_states"][0]["max_mean_steps"], 25.)

    def test_unknowns_and_missing_predictions_are_not_silently_dropped(self):
        d = fixture()
        pred = {s["state_id"]: {"gbdt": "0"} for s in d["states"]}
        with self.assertRaises(ValueError):
            label_curves(d, {})
        d["states"][0]["candidates"][0]["trials"][0].update(completed=None, stop="wall_safety")
        with self.assertRaises(ValueError):
            label_curves(d, pred)

    def test_map_macro_not_trial_pseudoreplication(self):
        d = fixture()
        pred = {s["state_id"]: {"gbdt": "0" if s["map_id"] == "m0" else "3"} for s in d["states"]}
        r = label_curves(d, pred)
        self.assertAlmostEqual(r["map_macro"]["gbdt"]["32"], (7/8)/3)
        d["states"].reverse()
        for s in d["states"]:
            s["candidates"].reverse()
        self.assertEqual(r["map_macro"], label_curves(d, pred)["map_macro"])


if __name__ == "__main__":
    unittest.main()
