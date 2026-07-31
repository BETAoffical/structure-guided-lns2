import unittest

from lns2_selector.compatibility.metrics import fixed_budget_conflict_auc
from lns2_selector.runtime.metrics import wall_clock_conflict_auc


class SelectorMetricTests(unittest.TestCase):
    def test_wall_clock_auc_uses_the_complete_deadline_window(self) -> None:
        self.assertEqual(
            wall_clock_conflict_auc([10, 6, 0], [2.0, 7.0], 10.0), 50.0
        )

    def test_wall_clock_auc_excludes_repairs_finishing_after_deadline(self) -> None:
        self.assertEqual(
            wall_clock_conflict_auc([10, 6, 0], [2.0, 7.0], 5.0), 38.0
        )

    def test_wall_clock_auc_rejects_unordered_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and ordered"):
            wall_clock_conflict_auc([10, 8, 6], [2.0, 1.0], 5.0)

    def test_fixed_budget_metric_is_read_only_window_compatibility(self) -> None:
        self.assertEqual(
            fixed_budget_conflict_auc([4, 3, 2, 1, 0], 2, success=True),
            6.0,
        )
