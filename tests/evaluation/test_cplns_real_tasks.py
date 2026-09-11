from copy import deepcopy
import json

import pytest

from scripts import diagnose_cplns_real_tasks as audit


def case():
    return {"case_id": "x", "state": {"rows": 2, "cols": 2, "obstacles": [False] * 4,
            "agents": [{"id": 0, "start": 0, "goal": 1, "path": [0, 2, 3, 1]},
                       {"id": 1, "start": 1, "goal": 0, "path": [1, 3, 2, 0]}]}}


def test_task_conversion_preserves_od_not_historical_paths():
    original = case()
    grid, scen, count = audit.task_files(original)
    assert count == 2 and "height 2\nwidth 2" in grid
    assert scen.splitlines()[1].split()[4:8] == ["0", "0", "1", "0"]
    original["state"]["agents"][0]["path"] = [0, 1]
    assert audit.task_files(original) == (grid, scen, count)


@pytest.mark.parametrize("change", ["obstacle", "duplicate", "id", "dimensions"])
def test_task_rejects_invalid_inputs(change):
    item = case()
    if change == "obstacle":
        item["state"]["obstacles"][0] = True
    elif change == "duplicate":
        item["state"]["agents"][1]["start"] = 0
    elif change == "id":
        item["state"]["agents"][1]["id"] = 3
    else:
        item["state"]["rows"] = 3
    with pytest.raises(ValueError):
        audit.task_files(item)


def test_four_arm_schedule_only_changes_registered_options():
    config = json.loads(audit.CONFIG.read_text())
    ref = json.loads((audit.ROOT / "configs/cplns_sequential_reference_v1.json").read_text())
    cases = [{"case_id": name, "map_id": name, "agent_count": 300} for name in config["cases"]]
    jobs = audit.schedule(config, cases, ref)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 80
    for start in range(0, len(jobs), 4):
        options = [dict(zip(j["argv"][1::2], j["argv"][2::2])) for j in jobs[start:start + 4]]
        clean = [{k: v for k, v in row.items() if k not in ("--sa", "--sa_max_con_fail")} for row in options]
        assert all(row == clean[0] for row in clean)
    assert config["no_ttf_or_promotion"] and config["seeds"] == [0, 2, 3, 5]


def test_qualification_stops_when_tasks_are_easy_and_rejects_duplicates():
    config = json.loads(audit.CONFIG.read_text())
    rows = [{"case_id": c, "map_id": c, "seed": s, "profile": audit.BASE,
             "initial_conflicts": int(i < 2)} for i, c in enumerate(config["cases"]) for s in config["seeds"]]
    assert not audit.qualification(rows, config)["passed"]
    for row in rows[:12]:
        row["initial_conflicts"] = 1
    assert audit.qualification(rows, config)["passed"]
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        audit.qualification(rows, config)


def event(kind, conflicts, **extra):
    return {"event": kind, "conflicts": conflicts, "cost": 20, "agents": [[0, [0, 1]]],
            "restart": 0, "decisions_seen": 0, **extra}


def test_worsening_then_breakthrough_requires_new_best_not_just_recovery():
    events = [event("initial_pp", -1), event("init", 3),
              event("step", 5, accepted=1, old_pairs=3, new_pairs=5),
              event("step", 3, accepted=1, old_pairs=5, new_pairs=3),
              event("step", 2, accepted=1, old_pairs=3, new_pairs=2), event("final", 2, decisions_seen=3)]
    result = audit.event_summary(events)
    assert result["accepted_worse"] == 1 and result["worse_then_new_best_same_restart"] == 1
    assert result["observed_steps"] == 3 and result["observed_best_conflicts"] == 2
    changed = deepcopy(events)
    changed.insert(3, event("init", 2, restart=1))
    assert audit.event_summary(changed)["worse_then_new_best_same_restart"] == 0


def test_terminal_after_truncated_observation_is_not_prefix_success():
    events = [event("initial_pp", -1), event("init", 3), event("final", 0, decisions_seen=1000)]
    result = audit.event_summary(events)
    assert result["prefix_truncated"] and not result["feasible_within_observed_steps"]
    other = deepcopy(events)
    other[1]["seconds"] = 200
    assert audit.event_summary(other)["initial_digest"] == result["initial_digest"]
    with pytest.raises(ValueError):
        audit.event_summary([event("initial_pp", -1), event("final", 1)])
