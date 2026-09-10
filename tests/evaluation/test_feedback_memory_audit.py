from copy import deepcopy

import pytest

from scripts.audit_feedback_memory import condition_keys, summarize, total_stats


def state():
    return {"agents": [
        {"id": 8, "start": 1, "goal": 3, "path": [1, 2, 3]},
        {"id": 21, "start": 4, "goal": 2, "path": [4, 3, 2]},
    ], "conflict_edges": [[8, 21]], "iteration": 2}


def keys(s, agents=(8,), budget=5):
    return condition_keys(s, list(agents), budget, {"mode": "explicit_neighborhood"}, "map-task")


def row(s, before=1, after=1, censored=False):
    return {**keys(s), "before_conflicts": before, "after_conflicts": after,
            "path_unchanged": True, "censored": censored}


def test_external_path_changes_invalidate_full_condition_not_coarse():
    a, b = state(), state()
    b["agents"][1]["path"] = [4, 4, 3, 2]
    assert keys(a)["coarse"] == keys(b)["coarse"]
    assert keys(a)["full_paths"] != keys(b)["full_paths"]
    assert keys(a)["repair_problem"] != keys(b)["repair_problem"]


def test_membership_and_agent_order_canonical_noncontiguous_ids():
    a, b = state(), state()
    b["agents"].reverse()
    b["conflict_edges"][0].reverse()
    assert keys(a, (8, 21)) == keys(b, (21, 8))


def test_counters_excluded_budget_separated_and_input_untouched():
    a = state()
    original = deepcopy(a)
    b = deepcopy(a)
    b["iteration"] = 99
    assert keys(a) == keys(b)
    assert keys(a)["full_paths"] == keys(a, budget=4)["full_paths"]
    assert keys(a)["full_paths_budget"] != keys(a, budget=4)["full_paths_budget"]
    assert a == original


@pytest.mark.parametrize("members", [(), (7,), (8, 8)])
def test_invalid_members_rejected(members):
    with pytest.raises(ValueError):
        keys(state(), members)


def test_duplicate_ids_rejected():
    s = state()
    s["agents"][1]["id"] = 8
    with pytest.raises(ValueError):
        keys(s)


def test_changed_context_not_counted_as_exact_reuse():
    a, b = state(), state()
    b["agents"][0]["path"] = [1, 1, 2, 3]
    stats = summarize([row(a), row(b, after=0)])
    assert stats["coarse_repeat_steps"] == 1
    assert stats["full_paths_repeat_steps"] == 0
    assert stats["coarse_repeat_without_any_full_path_match"] == 1
    assert stats["coarse_success_after_prior_nondrop"] == 1
    assert stats["full_paths_success_after_prior_nondrop"] == 0


def test_no_future_feedback_and_censored_observation_not_negative_evidence():
    s = state()
    assert summarize([row(s, after=0), row(s)])["full_paths_success_after_prior_nondrop"] == 0
    assert summarize([row(s), row(s, after=0)])["full_paths_success_after_prior_nondrop"] == 1
    assert summarize([row(s, censored=True), row(s, after=0)])[
        "full_paths_success_after_prior_nondrop"] == 0


def test_empty_episode_and_state_scoped_history():
    assert summarize([])["steps"] == 0
    assert summarize([row(state())])["full_paths_repeat_steps"] == 0


def test_selected_old_path_not_part_of_repair_problem():
    a, b = state(), state()
    b["agents"][0]["path"] = [1, 1, 2, 3]
    assert keys(a)["repair_problem"] == keys(b)["repair_problem"]
    assert keys(a)["full_paths"] != keys(b)["full_paths"]
    assert summarize([row(a), row(b)])["repair_problem_repeat_steps"] == 1


def test_acceptance_bound_and_selected_goal_change_repair_problem():
    a, b = state(), state()
    b["conflict_edges"] = []
    assert keys(a)["repair_problem"] != keys(b)["repair_problem"]
    b = state()
    b["agents"][0]["goal"] = 5
    assert keys(a)["repair_problem"] != keys(b)["repair_problem"]


def test_totals_preserve_zero_categories():
    assert total_stats([{"a": 0, "b": 1}, {"a": 0, "b": 2}]) == {"a": 0, "b": 3}
