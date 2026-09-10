import pytest
from scripts.audit_search_occupancy import reconstruct_contacts, independent_ranking


def test_independent_swap_vertex_goal_and_duplicate_ownership():
    events = [[0, 1, 2, 1, 2, 1, 2, 2, 1, 1],
              [2, 1, 1, 9, 10, 9, 10, 10, 1, 0]]
    fixed = {7: [2, 1], 19: [3, 2, 1]}
    summary = reconstruct_contacts(42, fixed, [0, 0], events * 2, {7, 19})
    assert not summary['unmatched']
    assert summary['flagged'] == 6
    assert independent_ranking([summary, summary]) == [
        dict(agent=7, planner_count=1, contact_count=2),
        dict(agent=19, planner_count=1, contact_count=2)]


def test_returned_path_and_internal_owners_not_counted():
    events = [[0, 0, 1, 1, 2, 1, 2, 2, 1, 0]]
    assert not reconstruct_contacts(42, {7: [2, 1]}, [0, 1], events, {7})['contacts']
    assert not reconstruct_contacts(42, {7: [2, 1]}, [0, 0], events, set())['contacts']


def test_unmapped_and_invalid_flags_are_not_silently_ignored():
    event = [0, 0, 9, 1, 2, 1, 2, 2, 1, 0]
    assert len(reconstruct_contacts(42, {7: [2, 1]}, [0, 0], [event], {7})['unmatched']) == 1
    event[8] = 2
    with pytest.raises(ValueError, match='invalid flags'):
        reconstruct_contacts(42, {7: [2, 1]}, [0, 0], [event], {7})
