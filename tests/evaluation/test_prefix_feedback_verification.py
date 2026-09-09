"""Saved-path attribution must not confuse rollback or missing paths with relief."""
import unittest

from scripts.verify_prefix_budget_feedback import target_pair_attribution


def record(agent, path=None):
    return dict(agent=agent, search=dict(status='path' if path is not None else 'empty', path=path))


class TargetPairAttributionTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(records=[record(7, [0, 1, 2]), record(19, [2, 1, 0])])
        self.option = dict(blocker=7, victim=19)

    def attempt(self, records):
        return dict(option=self.option, result=dict(status='rolled_back', records=records,
                    paths=[[99], [99]]))

    def test_uses_attempt_records_instead_of_restored_final_paths(self):
        result = target_pair_attribution(self.base, self.attempt(self.base['records']))
        self.assertEqual(result['classification'], 'pair_persists')
        self.assertGreater(result['attempt_events'], 0)

    def test_missing_victim_is_not_pair_removal(self):
        result = target_pair_attribution(self.base, self.attempt([record(7, [0, 3, 2])]))
        self.assertEqual(result['classification'], 'victim_path_unavailable')
        result = target_pair_attribution(self.base, self.attempt([record(7)]))
        self.assertEqual(result['classification'], 'blocker_path_unavailable')

    def test_noncontinuous_ids_and_removed_pair(self):
        result = target_pair_attribution(self.base, self.attempt([
            record(7, [0, 3, 2]), record(19, [2, 1, 0])]))
        self.assertEqual(result['classification'], 'pair_removed')
        self.assertEqual(result['attempt_events'], 0)
        self.assertGreater(result['baseline_events'], 0)
        self.assertIsNone(target_pair_attribution(self.base, dict(option=None)))


if __name__ == '__main__':
    unittest.main()
