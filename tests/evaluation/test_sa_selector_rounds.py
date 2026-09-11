import pytest

from scripts import diagnose_sa_selector_rounds as rounds


def test_acceptance_factorial_and_unknown_arm():
    assert [rounds.acceptance_arm(a) for a in rounds.ARMS] == [
        "standard", "complete_greedy", "annealed", "complete_greedy", "annealed"]
    with pytest.raises(ValueError):
        rounds.acceptance_arm("typo")


def test_uniform_is_order_invariant_and_shared_across_acceptance_modes():
    pool = [{"candidate_id": str(i)} for i in range(19)]
    index = rounds.choose_index("uniform_sa", "case", 7, 0, pool)
    reverse = list(reversed(pool))
    other = rounds.choose_index("uniform_greedy", "case", 7, 0, reverse)
    assert pool[index]["candidate_id"] == reverse[other]["candidate_id"]
    assert rounds.choose_index("rank_sa", "case", 7, 8, pool) == 8
    with pytest.raises(ValueError):
        rounds.choose_index("uniform_sa", "case", 0, 0, pool + [pool[0]])


def test_fixed_schedule_and_seed_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(rounds, "OUT", tmp_path)
    (tmp_path / "plan.json").write_text("{}")
    cases = [{"case_id": f"c{i}-{s}", "phase": p} for p, seeds in rounds.SEEDS.items()
             for i in range(5) for s in seeds]
    plan = {"config": {}, "legacy_config": {"old": True}}
    a, b = rounds.jobs(plan, cases, "round2"), rounds.jobs(plan, cases, "round3")
    assert len(a) == len(b) == 50
    assert not {j["case"]["case_id"] for j in a} & {j["case"]["case_id"] for j in b}
    smoke = rounds.jobs(plan, cases, "smoke")
    assert len(smoke) == 12 and all(j["horizon"] == 2 for j in smoke)
    assert all(j["config"] == {"old": True} for j in smoke if j["arm"] == "legacy")


def row(steps, feasible=True):
    return {"transitions": [{}] * steps, "feasible": feasible}


def test_portfolio_charges_both_lanes_and_does_not_read_future():
    assert rounds.portfolio(row(200), row(180), 256)["feasible"] is False
    result = rounds.portfolio(row(200), row(180), 512)
    assert result == {"feasible": True, "repair_calls": 360, "selected_lane": "rank_sa"}
    assert rounds.portfolio(row(3), row(1), 256)["repair_calls"] == 2
    assert rounds.portfolio(row(0), row(0), 256)["repair_calls"] == 0
    assert not rounds.portfolio(row(1, False), row(256, False), 512)["feasible"]


def records():
    return [{**row(5), "case_id": "case", "map_id": "map", "initial_fingerprint": "same", "arm": a,
             "censored": False, "generated": 100, "auc": 20, "accepted_increases": 0} for a in rounds.ARMS]


def test_summary_is_paired_and_deterministic_without_promoting():
    data = records()
    a = rounds.summarize(data)
    assert a == rounds.summarize(list(reversed(data)))
    assert a["no_ttf_or_promotion"]
    assert a["map_bootstrap_auc_reduction_ci95_descriptive"]["rank_sa"] == [0, 0]
    with pytest.raises(ValueError):
        rounds.summarize(data[:-1])
    with pytest.raises(ValueError):
        rounds.summarize(data + [data[0]])
    data[1]["initial_fingerprint"] = "other"
    with pytest.raises(ValueError):
        rounds.summarize(data)


def test_audit_acceptance_and_action_integrity():
    event = {"candidate_pool": [{"candidate_id": "a", "score": 1., "agents": [0]}],
             "selected_id": "a", "action": {"agents": [0]}, "temperature": 1000., "uniform": 0.1,
             "metrics": {"action_valid": True, "conflicts_before": 1, "conflicts_after": 2,
                         "pp_attempt_conflict_pair_count": 2, "pp_old_conflict_pair_count": 1,
                         "acceptance_evaluated": True, "acceptance_probability": rounds.probability(1, 1000),
                         "replan_success": True}}
    records = [{"arm": "annealed", "transitions": [event]}]
    assert rounds.audit_rows(records)["annealed"]["worse_accepted"] == 1
    event["metrics"]["acceptance_probability"] = .1
    with pytest.raises(ValueError):
        rounds.audit_rows(records)


def test_smoke_missing_rows_rejected():
    with pytest.raises(ValueError):
        rounds.smoke_gate([])
