"""Descriptive analysis cannot turn a sample oracle into held-out evidence."""
import unittest

from scripts.summarize_sa_policy_aligned_update import describe_root


class DescriptionTest(unittest.TestCase):
    def test_half_reversal_is_not_stability(self):
        root = dict(id="r", map_id="m", decision=0, state=dict(num_of_colliding_pairs=2),
                    anchor_id="a", candidates=[dict(candidate_id=c) for c in "ab"],
                    source_event=dict(ranking=dict(selected="b")))
        rows = [dict(candidate_id=c, trial=t, arm=arm, success=(c == "a") == (t < 4))
                for c in "ab" for t in range(8) for arm in ("frozen", "gbdt")]
        label = dict(complete=True, target=dict(a=.5, b=.5), rates={})
        d = describe_root(root, rows, label, "b")
        self.assertTrue(d["all_tied"])
        self.assertEqual(d["half_informative_pairs"], 1)
        self.assertEqual(d["half_repeated_strict_pairs"], 0)
        self.assertEqual(d["cross_half_gain_vs_anchor"], -.5)
        self.assertEqual(d["training_completion"]["sample_oracle"], .5)


if __name__ == "__main__":
    unittest.main()
