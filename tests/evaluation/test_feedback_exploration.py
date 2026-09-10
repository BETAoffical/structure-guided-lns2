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
