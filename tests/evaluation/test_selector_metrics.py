import pytest

from lns2_selector.compatibility.metrics import fixed_budget_conflict_auc
from lns2_selector.runtime.metrics import wall_clock_conflict_auc


def test_wall_clock_auc_uses_the_complete_deadline_window() -> None:
    assert wall_clock_conflict_auc([10, 6, 0], [2.0, 7.0], 10.0) == 50.0


def test_wall_clock_auc_excludes_repairs_finishing_after_deadline() -> None:
    assert wall_clock_conflict_auc([10, 6, 0], [2.0, 7.0], 5.0) == 38.0


def test_wall_clock_auc_rejects_unordered_times() -> None:
    with pytest.raises(ValueError, match="finite and ordered"):
        wall_clock_conflict_auc([10, 8, 6], [2.0, 1.0], 5.0)


def test_fixed_budget_metric_is_read_only_window_compatibility() -> None:
    assert fixed_budget_conflict_auc([4, 3, 2, 1, 0], 2, success=True) == 6.0
