import pytest

from scripts import analyze_sa_rejection_branches as analysis


def test_budget_unknown_is_not_horizon_failure():
    yes = dict(feasible=True, stop="feasible")
    assert analysis.classify(yes, dict(feasible=False, stop="node_budget")).endswith("resource_unknown")
    assert analysis.classify(yes, dict(feasible=False, stop="horizon")).endswith("horizon")
    assert analysis.classify(yes, yes) == "both_feasible"


def test_reconvergence_requires_full_state_and_future_controls(monkeypatch):
    monkeypatch.setattr(analysis.probe.pilot, "state_fingerprint", lambda s: tuple(s.items()))
    monkeypatch.setattr(analysis.probe.pilot, "apply_state_delta", lambda s, d: d)
    def row(start, end, action):
        event = dict(delta=end, decision=1, action=action, temperature=1, uniform=.5,
                     pool=[], selected_index=0)
        return dict(target_after=start, events=[event], final=end)
    a = row(dict(conflicts=1, paths="a"), dict(conflicts=0, paths="c"), "a")
    b = row(dict(conflicts=1, paths="b"), dict(conflicts=0, paths="c"), "b")
    result = analysis.reconvergence(a, b)
    assert result["first_equal_after_followup_steps"] == 1
    assert result["equal_final_fingerprint"]
    assert result["subsequent_controls_equal"]  # No decisions after terminal merge.
    b["events"][0]["delta"] = b["final"] = dict(conflicts=0, paths="d")
    assert analysis.reconvergence(a, b)["first_equal_after_followup_steps"] is None
    b["final"] = dict(conflicts=0, paths="bad")
    with pytest.raises(ValueError, match="final state"):
        analysis.reconvergence(a, b)
