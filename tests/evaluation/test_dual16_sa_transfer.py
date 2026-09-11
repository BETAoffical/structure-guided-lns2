import json

import pytest

from scripts import diagnose_dual16_sa_transfer as transfer


def test_initial_import_uses_exact_id_order_and_rejects_invalid_paths():
    event = {"agents": [{"id": 1, "path": [2, 3]}, {"id": 0, "path": [0, 1]}]}
    assert transfer.path_array(event) == [[0, 1], [2, 3]]
    event["agents"][0]["id"] = 3
    with pytest.raises(ValueError):
        transfer.path_array(event)
    with pytest.raises(ValueError):
        transfer.path_array({"agents": [{"id": 0, "path": []}]})


def test_source_is_first_init_not_final_or_late_repair(tmp_path):
    p = tmp_path / "events.jsonl"
    events = [{"event": "initial_pp", "conflicts": -1}, {"event": "init", "conflicts": 20},
              {"event": "step", "conflicts": 1}, {"event": "init", "conflicts": 50}]
    p.write_text("\n".join(json.dumps(e) for e in events))
    assert transfer.initial_event(p)["conflicts"] == 20


def test_20_imports_60_rollouts_and_20_smokes(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "OUT", tmp_path)
    (tmp_path / "plan.json").write_text("{}")
    cases = [{"case_id": f"c{i}-{s}", "source_seed": s} for i in range(5) for s in (0, 2, 3, 5)]
    plan = {"config": {"native": "experimental"}, "legacy_config": {"native": "old"}}
    jobs = transfer.schedule(plan, cases)
    smoke = transfer.schedule(plan, cases, True)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 60
    assert len(smoke) == 20 and all(j["horizon"] == 2 for j in smoke)
    assert all(j["horizon"] == 256 and j["trial"] == 0 for j in jobs)
    assert all(j["config"]["native"] == ("old" if j["arm"] == "legacy" else "experimental") for j in smoke)


def rows():
    return [{"case_id": "a", "trial": 0, "map_id": "map", "arm": arm, "feasible": True,
             "censored": False, "generated": 100, "conflicts": [2, 0], "auc": 1,
             "accepted_increases": 0, "transitions": [{}] * count}
            for arm, count in (("standard", 10), ("complete_greedy", 8), ("annealed", 5))]


def test_effort_improvement_does_not_silently_relax_success_gain_gate():
    result = transfer.summarize(rows())
    assert result["decision"] == "no_go_this_pilot"
    assert result["common_success_comparisons"]["standard"]["fewer_repairs"] == 1
    assert result["effort"]["annealed"]["mean_executed_repairs"] == 5


def test_failed_short_rollout_cannot_win_common_success_effort():
    data = rows()
    data[-1].update(feasible=False, conflicts=[2, 2], transitions=[{}])
    result = transfer.summarize(data)
    assert result["common_success_comparisons"]["standard"]["common_success"] == 0
    assert result["annealed_vs"]["standard"]["losses"] == [("a", 0)]


def test_smoke_censor_or_missing_arm_is_rejected():
    with pytest.raises(ValueError):
        transfer.smoke_gate(rows())
    data = rows() + [{**rows()[0], "arm": "legacy", "censored": True}]
    with pytest.raises(ValueError):
        transfer.smoke_gate(data)
