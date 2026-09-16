import unittest

from scripts.audit_sa_history_bridge_support import directions, support


class ActionSupportTests(unittest.TestCase):
    def test_support_never_uses_held_out_map(self):
        def row(m, c, size):
            return dict(map_id=m, episode=m, current_conflicts=c,
                        base={"proposal.actual_size":size})
        result = support([row("held", 20, 4), row("train", 1, 4), row("train", 20, 16)], 4, 20, "held")
        self.assertEqual(result["train_rows"], 2)
        self.assertEqual(result["same_size_rows"], 1)
        self.assertEqual(result["same_size_comparable_conflict_rows"], 0)

    def test_direction_requires_both_halves_same_sign(self):
        labels = {"a":[{"y":v} for v in [1]*4 + [0]*4],
                  "b":[{"y":v} for v in [0]*4 + [1]*4]}
        pair = directions({"a":.9, "b":.1}, labels, "y")[0]
        self.assertFalse(pair["stable_strict"])

    def test_wrong_and_tied_predictions_are_not_correct(self):
        labels = {"a":[{"y":1}]*8, "b":[{"y":0}]*8}
        for predictions in ({"a":.1, "b":.9}, {"a":.5, "b":.5}):
            pair = directions(predictions, labels, "y")[0]
            self.assertTrue(pair["stable_strict"])
            self.assertFalse(pair["agrees"])


if __name__ == "__main__":
    unittest.main()
