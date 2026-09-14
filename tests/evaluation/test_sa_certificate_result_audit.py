import pytest

from scripts.audit_sa_certificate_results import diagnosis_decision,checked_feasible,checked_window,recurrence_after_clear,repetition_metrics


def test_decision_does_not_promote_local_recovery_or_control_only_gain():
    control=[dict(kind="control",before=6,after=5)]
    certificate=[dict(kind="certificate_single",before=6,after=5)]
    assert diagnosis_decision([],certificate,[])=="no_membership_obstruction_proved"
    assert diagnosis_decision([1],control,[])=="proved_membership_obstruction_without_observed_native_recovery"
    assert diagnosis_decision([1],certificate,[dict(arm="frozen",feasible=False)])=="proved_membership_obstruction_and_local_recovery_not_full_escape"
    assert diagnosis_decision([1],certificate,[dict(arm="add_certified",feasible=True)])=="bounded_development_completion_not_controller_promotion"


def test_feasible_must_mean_zero_conflicts():
    assert checked_feasible(dict(num_of_colliding_pairs=0,feasible=True))
    with pytest.raises(ValueError,match="feasible flag"):
        checked_feasible(dict(num_of_colliding_pairs=6,feasible=True))


def test_recurrence_detects_clear_after_first_step():
    assert recurrence_after_clear([True,False,False]) is False
    assert recurrence_after_clear([True,False,True]) is True
    assert recurrence_after_clear([True,True]) is None


def test_window_rejects_short_horizon_and_post_feasible_actions():
    events=[dict(metrics=dict(pp_failure_reason="none"))]
    plan=dict(horizon=2,node_budget=100,seconds=10.)
    row=dict(events=events,stop="horizon",feasible=False,full_window=True)
    with pytest.raises(ValueError,match="incomplete horizon"): checked_window(row,plan,[6,5])
    row.update(stop="feasible",feasible=True)
    assert checked_window(row,plan,[6,0])
    with pytest.raises(ValueError,match="post-feasible"): checked_window(row,plan,[0,0])
    row.update(stop="pp_safety",feasible=False,full_window=False)
    with pytest.raises(ValueError,match="missing PP censor"): checked_window(row,plan,[6,5])


def test_window_rejects_actions_after_first_timeout():
    events=[dict(metrics=dict(pp_failure_reason=s)) for s in ("time_limit","none","time_limit")]
    row=dict(events=events,stop="pp_safety",feasible=False,full_window=False)
    with pytest.raises(ValueError,match="continued after censored"):
        checked_window(row,dict(horizon=3,node_budget=100,seconds=10.),[6,6,5,5])


def test_repeated_paths_are_not_inferred_from_equal_conflict_counts():
    assert repetition_metrics(["a","b","b","a"])==dict(unique_physical_states=2,revisited_states=2,unchanged_steps=1)
    assert repetition_metrics(["a","b","c"])["unchanged_steps"]==0
