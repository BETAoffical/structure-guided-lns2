from copy import deepcopy

import pytest

from experiments.feedback_exploration_diagnostic import FeedbackSelection, classify_rounds, padded_auc, paired_summary
from scripts import run_feedback_exploration_diagnostics as diagnostic


@pytest.mark.parametrize("arm", FeedbackSelection.ARMS)
def test_no_feedback_uses_original(arm):
    selector = FeedbackSelection(arm)
    assert selector.select(["a", "b"], ["1", "2"], 1, 4)[0] == 1


@pytest.mark.parametrize("arm", ["uniform", "feedback"])
def test_permutation_and_full_pool(arm):
    selector = FeedbackSelection(arm)
    selector.observe("a", 2, 2, False)
    choices = set()
    for seed in range(200):
        i, _ = selector.select(["a", "b", "c"], ["1", "2", "3"], 0, seed)
        j, _ = selector.select(["c", "a", "b"], ["3", "1", "2"], 1, seed)
        assert ["1", "2", "3"][i] == ["3", "1", "2"][j]
        choices.add(i)
    assert choices == {0, 1, 2}


def test_censor_ignored_context_reset_recovery_retained():
    s = FeedbackSelection("feedback")
    s.observe("a", 3, 3, True)
    assert not s.observations
    s.observe("a", 3, 3, False)
    s.observe("a", 3, 2, False)
    assert s.observations["a"] == [1, 1]
    assert not s.select(["new"], ["1"], 0, 1)[1]["active"]
    assert not FeedbackSelection("feedback").observations


def test_auc_and_pairing():
    assert padded_auc([3, 0], 4) == 1.5
    assert padded_auc([3, 2], 4) == 8.5
    with pytest.raises(ValueError):
        padded_auc([1, 0, 0], 1)
    rows = [{"case_id": "a", "trial": 0, "arm": arm, "feasible": True, "auc": 1,
             "conflicts": [2, 0], "censored": False, "transitions": []} for arm in FeedbackSelection.ARMS]
    assert paired_summary(rows)["feedback"]["paired_feasibility_losses"] == 0
    with pytest.raises(ValueError):
        paired_summary(rows + [rows[0]])
    with pytest.raises(ValueError):
        paired_summary(rows[:-1])


def test_predefined_decision():
    base = {a: {"feasible": 4, "paired_feasibility_gains": 0,
                 "paired_feasibility_losses": 0, "censored": 0} for a in FeedbackSelection.ARMS}
    first, third = deepcopy(base), deepcopy(base)
    assert classify_rounds(first, third) == "no_consistent_recovery_signal"
    first["feedback"].update(feasible=6, paired_feasibility_gains=2)
    assert classify_rounds(first, third) == "feedback_incremental_development_signal"
    first["uniform"].update(feasible=6, paired_feasibility_gains=2)
    assert classify_rounds(first, third) == "exploration_signal_without_feedback_increment"
    third["feedback"]["censored"] = 1
    assert classify_rounds(first, third) == "inconclusive_resource"


def test_job_pairing_seeds_and_no_explicit_order(monkeypatch):
    monkeypatch.setattr(diagnostic, "sha256_file", lambda _: "sha")
    jobs = diagnostic.stage_jobs({"config": {}}, [{"case_id": "case"}], "round1")
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 12
    assert diagnostic.seed("round1", "a", 0, 0, "pp") != diagnostic.seed("round1", "a", 1, 0, "pp")
    assert diagnostic.seed("round1", "a", 0, 0, "pp") != diagnostic.seed("round3", "a", 0, 0, "pp")


def test_final_path_validation():
    state = {"rows": 2, "cols": 2, "obstacles": [0]*4, "agents": [
        {"id": 3, "start": 0, "goal": 1, "path": [0, 1]},
        {"id": 7, "start": 2, "goal": 3, "path": [2, 3]}],
        "conflict_edges": [], "num_of_colliding_pairs": 0, "feasible": True}
    diagnostic.validate_final(state)
    state["agents"][0]["path"] = [0, 3, 1]
    with pytest.raises(ValueError):
        diagnostic.validate_final(state)


def test_aggregate_gains_do_not_hide_paired_harm():
    stage = {a: {"feasible": 10, "paired_feasibility_gains": 0,
                 "paired_feasibility_losses": 0, "censored": 0} for a in FeedbackSelection.ARMS}
    stage["feedback"].update(feasible=15, paired_feasibility_gains=6, paired_feasibility_losses=1)
    stage["uniform"].update(feasible=15, paired_feasibility_gains=6, paired_feasibility_losses=1)
    assert classify_rounds(stage, stage) == "no_consistent_recovery_signal"


def test_positive_feedback_is_not_carried_across_acceptance_bound_change():
    from scripts.audit_feedback_memory import condition_keys
    before = {"agents": [{"id": 3, "start": 0, "goal": 1, "path": [0, 1]},
                         {"id": 7, "start": 1, "goal": 0, "path": [1, 0]}],
              "conflict_edges": [[3, 7]]}
    after = deepcopy(before)
    after["agents"][0]["path"] = [0, 2, 3, 1]
    after["conflict_edges"] = []
    old_key = condition_keys(before, [3], 5, {"mode": "explicit_neighborhood"}, "task")["repair_problem_budget"]
    new_key = condition_keys(after, [3], 5, {"mode": "explicit_neighborhood"}, "task")["repair_problem_budget"]
    assert old_key != new_key
    selector = FeedbackSelection("feedback")
    selector.observe(old_key, 1, 0, False)
    assert selector.observations[old_key] == [1, 0]
    assert selector.select([new_key], ["same-agents"], 0, 1)[1]["prior"] == [[0, 0]]


def test_result_feedback_auditor_rejects_seed_or_outcome_change():
    from scripts.analyze_feedback_exploration import verify_rollout
    state = {"rows": 1, "cols": 2, "obstacles": [0, 0], "agents": [
        {"id": 3, "start": 0, "goal": 1, "path": [0, 1]}],
        "conflict_edges": [], "num_of_colliding_pairs": 0, "feasible": True}
    job = {"case": {"case_id": "case", "state": {"num_of_colliding_pairs": 1}}, "arm": "frozen", "phase": "round1", "trial": 0}
    pp = diagnostic.seed("round1", "case", 0, 0, "pp")
    metrics = {"requested_random_seed": pp, "requested_pp_time_limit_seconds": 5.0,
               "requested_repair_order": [], "requested_pp_random_seed": -1,
               "conflicts_before": 1, "conflicts_after": 0, "action_valid": True,
               "step_applied": True, "neighborhood": [3], "pp_failure_reason": "none",
               "native_replan_seconds": .1, "pp_rolled_back": False}
    event = {"candidate_pool": [{"candidate_id": "c", "score": .4, "agents": [3]}],
             "keys": ["key"], "base_id": "c", "candidate_id": "c", "selected_key": "key",
             "feedback": {"active": False, "prior": [[0, 0]], "samples": None},
             "action": {"mode": "explicit_neighborhood", "agents": [3], "random_seed": pp}, "metrics": metrics}
    result = {"conflicts": [1, 0], "transitions": [event], "final_state": state}
    assert verify_rollout(result, job)["strict_drops"] == 1
    for field, value in (("requested_random_seed", pp + 1), ("conflicts_after", 1), ("neighborhood", [99])):
        changed = deepcopy(result)
        changed["transitions"][0]["metrics"][field] = value
        with pytest.raises(ValueError):
            verify_rollout(changed, job)
