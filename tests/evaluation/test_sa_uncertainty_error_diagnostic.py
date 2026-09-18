import unittest

from experiments.sa_uncertainty_error_diagnostic import weighted_auc, describe


class ErrorDiagnosticTests(unittest.TestCase):
    def test_auc_direction_ties_and_censor_free_sign(self):
        rows = [dict(map_id="a", x=2, gain=-1), dict(map_id="b", x=1, gain=1), dict(map_id="c", x=99, gain=0)]
        self.assertEqual(weighted_auc(rows, "x", "gain"), 1.)
        rows[0]["x"] = 0
        self.assertEqual(weighted_auc(rows, "x", "gain"), 0.)
        rows[0]["x"] = 1
        self.assertEqual(weighted_auc(rows, "x", "gain"), .5)

    def test_map_replication_does_not_multiply_independent_evidence(self):
        rows = [dict(map_id="a", x=2, gain=-1), dict(map_id="b", x=1, gain=1), dict(map_id="c", x=3, gain=1)]
        self.assertAlmostEqual(weighted_auc(rows, "x", "gain"), weighted_auc(rows+[rows[0]]*5, "x", "gain"))
        self.assertIsNone(weighted_auc(rows, "x", "gain", {"b": 1}))

    def test_half_labels_separate_and_bootstrap_undefined_reported(self):
        rows = [dict(map_id="a", disagreement_sd=2, small_margin=-2, negative_votes=.8,
                     gain=-.25, first_half=-.5, second_half=0),
                dict(map_id="b", disagreement_sd=1, small_margin=-1, negative_votes=.2,
                     gain=.25, first_half=0, second_half=.5)]
        a = describe(rows, 1, 100)
        self.assertEqual(a, describe(rows, 1, 100))
        self.assertGreater(a["undefined_bootstrap"], 0)
        self.assertEqual(a["auc"]["disagreement_sd"]["gain"], 1.)
        self.assertIsNone(a["auc"]["disagreement_sd"]["first_half"])
        self.assertIsNone(a["auc"]["disagreement_sd"]["second_half"])
        self.assertFalse(a["calibrated_confidence"])
        self.assertTrue(a["posthoc"])


if __name__ == "__main__":
    unittest.main()
