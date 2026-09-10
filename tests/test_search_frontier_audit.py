from scripts.audit_search_frontier import convert_and_filter
from experiments.search_frontier_observation import profiles
import pytest


def test_independent_profile_reconstruction_and_exact_boundaries():
    rows = [[kind, 0, 0, 1, 1, 1, h, c, c-1, 1, 0, 0]
            for kind in (1, 2) for h in (8, 9, 10) for c in (1, 2, 3)]
    expected = profiles(dict(events=rows, goal_conflicts=1, goal_cost=10))
    for name, values in expected.items():
        assert convert_and_filter(rows, 1, 10, name) == values
    assert len(expected['queued_envelope']) == 4


def test_invalid_collision_delta_is_rejected():
    with pytest.raises(ValueError):
        convert_and_filter([[1, 0, 0, 1, 1, 1, 0, 4, 0, 1, 0, 0]], 1, 10, 'queued_all')
