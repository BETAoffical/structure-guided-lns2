import unittest

from scripts.audit_preentry_trace import exit_availability, pool_record


class PreentryTraceTests(unittest.TestCase):
    def test_no_exit_is_not_recovery(self):
        self.assertIsNone(exit_availability([], None))
        self.assertIsNone(exit_availability([], {'selection': {'candidate_id': 'x'}}))

    def test_previously_available_rank_preserves_ties(self):
        rows = [{'step': 7, 'selection': {'score': 3, 'alternatives': [
            {'candidate_id': 'x', 'score': 2}, {'candidate_id': 'y', 'score': 2}]}}]
        result = exit_availability(rows, {'selection': {'candidate_id': 'x'}})
        self.assertEqual(result, {'unselected_appearances': 1,
            'first': [{'step': 7, 'rank': 2}], 'best_rank': 2})

    def test_absence_is_not_low_rank(self):
        result = exit_availability([{'selection': {'score': 3, 'alternatives': []}}],
                                   {'selection': {'candidate_id': 'x'}})
        self.assertEqual(result['unselected_appearances'], 0)
        self.assertIsNone(result['best_rank'])

    def test_wrong_selected_members_fail(self):
        event = {'controller': {'selected_candidate_id': 'x', 'candidate_pool': [
            {'candidate_id': 'x', 'agents': [2, 5], 'selection_families': ['random:4'], 'score': 1}]},
            'metrics': {'neighborhood': [2, 6]}}
        with self.assertRaises(AssertionError):
            pool_record(event)


if __name__ == '__main__':
    unittest.main()
