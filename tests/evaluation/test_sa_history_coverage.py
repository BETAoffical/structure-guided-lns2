import copy
import unittest
from scripts.audit_sa_history_pilot_results import scan_events


class CoverageTests(unittest.TestCase):
    def rows(self, count):
        return [dict(decision=i, pool=[dict(agents=[2,7]), dict(agents=[7,19])], selected_index=0,
                     metrics=dict(neighborhood=[2,7], conflicts_before=1, conflicts_after=1)) for i in range(count)]

    def test_current_result_not_counted_as_past(self):
        rows = self.rows(40)
        baseline = list(scan_events(rows))
        changed = copy.deepcopy(rows)
        changed[32]["metrics"]["neighborhood"] = [7,19]
        changed[32]["metrics"]["conflicts_after"] = 0
        other = list(scan_events(changed))
        self.assertEqual(baseline[:33], other[:33])
        self.assertFalse(baseline[0]["pool_has_seen"])
        self.assertTrue(baseline[1]["pool_has_seen"])
        self.assertFalse(baseline[1]["pool_has_twice_tried"])
        self.assertTrue(baseline[2]["pool_has_twice_tried"])
        self.assertTrue(baseline[32]["prospective_history_opportunity"])
        self.assertEqual(other[33]["since_best"], 0)

    def test_window_forgets_old_membership(self):
        rows = self.rows(5)
        for row in rows[1:]: row["metrics"]["neighborhood"] = [7,19]
        values = list(scan_events(rows, window=2))
        self.assertTrue(values[2]["chosen_seen"])
        self.assertFalse(values[3]["chosen_seen"])


if __name__ == "__main__":
    unittest.main()
