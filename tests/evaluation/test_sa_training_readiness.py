from copy import deepcopy
import pytest
from scripts import audit_sa_training_readiness as audit


def state():
    return dict(rows=2, cols=3, obstacles=[False]*6, conflict_edges=[[2, 8]], agents=[
        dict(id=2, start=0, goal=2, path=[0, 1, 2]),
        dict(id=8, start=2, goal=0, path=[2, 1, 0]),
        dict(id=31, start=3, goal=5, path=[3, 4, 5])])


def test_visible_key_ignores_selected_old_path_but_not_external_path():
    a = state(); b = deepcopy(a); b["agents"][0]["path"] = [0, 0, 1, 2]
    def key(s, agents): return audit.RepairConditions(s, "instance").problem(s, agents, {"mode": "explicit"})
    assert key(a, [2, 8]) == key(b, [8, 2])
    b["agents"][2]["path"] = [3, 3, 4, 5]
    assert key(a, [2, 8]) != key(b, [2, 8])
    b = deepcopy(a); b["conflict_edges"] = []
    assert key(a, [2, 8]) != key(b, [2, 8])
    with pytest.raises(ValueError): key(a, [2, 2])
    with pytest.raises(ValueError): key(a, [999])


def test_cached_path_update_matches_fresh_key():
    a = state(); b = deepcopy(a); b["agents"][2]["path"] = [3, 3, 4, 5]
    obj = audit.RepairConditions(a, "instance")
    obj.update(dict(agents=dict(mode="patch", patches=[dict(id=31, set={"path": b["agents"][2]["path"]})])), b)
    assert obj.problem(b, [2, 8], {}) == audit.RepairConditions(b, "instance").problem(b, [2, 8], {})


def test_history_is_before_current_action_and_censor_is_not_negative():
    h = {}; before = audit.before_features(h, "a")
    audit.record(h, "a", True, True)
    assert before["drop"] == 0 and before["attempts"] == 0
    assert audit.before_features(h, "a")["drop"] == 1
    audit.record(h, "a", False, False)
    assert audit.before_features(h, "a")["nondrop"] == 0
    audit.record(h, "a", True, False)
    assert audit.before_features(h, "a") == dict(attempts=3, complete=2, drop=1, nondrop=1, censored=1)
    assert audit.before_features({}, "a")["drop"] == 0


def candidate(key, values, selected=False):
    return dict(candidate_id=key, conflicts=values, selected=selected, censored=0, feasible=sum(x==0 for x in values))


def test_candidate_stability_does_not_pool_trials_or_claim_completion():
    s = dict(task_id="map_task", solver_seed=1, job_id="root", candidates=[
        candidate("a", [4]*4, True), candidate("b", [3, 4, 4, 3]), candidate("c", [3, 3, 4, 4])])
    r = audit.alternative_summary(s)
    assert r["trials"] == 12 and r["candidates"] == 3
    assert r["candidates_both_halves_better"] == 1
    assert r["candidates_all_four_better"] == 0 and r["feasible_trials"] == 0
    s["candidates"][1]["conflicts"].pop()
    with pytest.raises(ValueError): audit.alternative_summary(s)


def test_zero_step_history_and_missing_selected():
    assert audit.before_features({}, "empty")["attempts"] == 0
    with pytest.raises(ValueError):
        audit.alternative_summary(dict(candidates=[candidate("x", [1]*4)]))


def test_same_agents_different_instance_not_pooled():
    s = state()
    assert audit.RepairConditions(s, "a").problem(s, [2, 8], {}) != audit.RepairConditions(s, "b").problem(s, [2, 8], {})


def test_sa_can_revisit_a_pre_decrease_condition_without_future_leakage():
    a = state(); b = deepcopy(a); b["conflict_edges"] = []
    ka = audit.RepairConditions(a, "i").problem(a, [2, 8], {})
    kb = audit.RepairConditions(b, "i").problem(b, [2, 8], {})
    history = {}
    audit.record(history, ka, True, True)
    assert audit.before_features(history, kb)["drop"] == 0
    # A later increase can return to the original visible condition.
    assert audit.before_features(history, ka)["drop"] == 1


def test_saved_audit_hash_and_binding():
    r = dict(status="ok", binding="a", rows=[])
    r["integrity_sha256"] = audit.q.json_fingerprint(r)
    audit.check_saved(r, "a")
    with pytest.raises(ValueError): audit.check_saved(r, "b")
    r["rows"].append({})
    with pytest.raises(ValueError): audit.check_saved(r, "a")
