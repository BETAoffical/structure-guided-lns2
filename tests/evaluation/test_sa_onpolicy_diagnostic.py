from types import SimpleNamespace
import unittest
import numpy as np

from scripts.diagnose_sa_onpolicy_batch import hidden_metrics


class DiagnosticTests(unittest.TestCase):
    def test_shared_large_input_can_erase_candidate_differences(self):
        actor = SimpleNamespace(bundle={"feature_names": ["candidate", "shared"]},
                                mean=np.zeros(2), scale=np.ones(2), w1=np.ones((2, 2)), b1=np.zeros(2))
        varied = hidden_metrics([{"candidate": -1., "shared": 0.}, {"candidate": 1., "shared": 0.}], actor)
        self.assertFalse(varied["nearly_action_invariant"])
        same = hidden_metrics([{"candidate": -1., "shared": 100.}, {"candidate": 1., "shared": 100.}], actor)
        self.assertTrue(same["nearly_action_invariant"])
        self.assertEqual(same["saturated_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
