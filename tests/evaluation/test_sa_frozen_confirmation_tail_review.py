import unittest
from scripts.review_sa_frozen_confirmation_tails import selected_pairs, ARMS


class TailSelectionTests(unittest.TestCase):
    def test_preserves_successful_controls_for_any_failure(self):
        rows = [dict(pair_id=p,comparison_arm=a,success_within_budget=not (p=='hard' and a=='parent'))
                for p in ('easy','hard') for a in ARMS]
        group = selected_pairs(rows)
        self.assertEqual(set(group),{'hard'})
        self.assertEqual(set(group['hard']),ARMS)
        self.assertEqual(group,selected_pairs(reversed(rows)))

    def test_missing_and_duplicate_pairs_rejected(self):
        rows = [dict(pair_id='p',comparison_arm=a,success_within_budget=False) for a in ARMS]
        for bad in (rows[:-1],rows+rows[:1]):
            with self.assertRaises(ValueError): selected_pairs(bad)


if __name__ == '__main__':
    unittest.main()
