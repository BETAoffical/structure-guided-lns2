from copy import deepcopy
import json

import pytest

from scripts import verify_cplns_observation as bridge
from scripts import diagnose_cplns_observation_divergence as divergence


def test_patch_refuses_missing_and_ambiguous_anchors():
    assert bridge.insert_once("abc", "b", "X") == "abXc"
    for text in ("ac", "abbc"):
        with pytest.raises(ValueError):
            bridge.insert_once(text, "b", "X")
    assert bridge.instrument("src/defines.cpp", "random()") == "random()"


def test_all_anchors_on_pinned_source_when_available():
    config = json.loads(bridge.CONFIG.read_text())
    ref = json.loads((bridge.ROOT / config["reference_config"]).read_text())
    root = bridge.ROOT / ref["source"]
    if not root.exists():
        pytest.skip("optional pinned upstream source unavailable")
    for name in ("src/InitLNS.cpp", "src/LNS.cpp"):
        text = (root / name).read_text(encoding="utf-8")
        patched = bridge.instrument(name, text)
        assert patched.count('#include "cplns_observer.h"') == 1
        assert len(patched) > len(text)


@pytest.mark.parametrize("name", ["open", "order_reversal_corridor", "dense_open", "two_door"])
def test_fixed_fixtures_unique_and_deterministic(name):
    grid, scen, count = bridge.fixture(name, 917)
    assert (grid, scen, count) == bridge.fixture(name, 917)
    rows = [line.split() for line in scen.splitlines()[1:]]
    assert len(rows) == count
    assert len({tuple(row[4:6]) for row in rows}) == count
    assert len({tuple(row[6:8]) for row in rows}) == count


def agent(aid, path):
    return {"id": aid, "start": path[0], "goal": path[-1], "path": path}


def test_conflicts_include_swaps_and_terminal_wait_noncontiguous_ids():
    agents = [agent(2, [0, 1]), agent(9, [1, 0])]
    assert bridge.path_metrics(agents, 3, ["..."], {2: (0, 1), 9: (1, 0)})[:2] == (1, 2)
    agents = [agent(2, [0]), agent(9, [2, 1, 0, 1])]
    assert bridge.path_metrics(agents, 3, ["..."], {2: (0, 0), 9: (2, 1)})[0] == 1


@pytest.mark.parametrize("agents,rows", [([agent(2, [0, 2])], ["..."]), ([agent(2, [0, 1])], [".@."]),
                                         ([agent(2, [0]), agent(2, [0])], ["..."])])
def test_illegal_paths_rejected(agents, rows):
    with pytest.raises(ValueError):
        bridge.path_metrics(agents, 3, rows, {2: (0, agents[0]["goal"])})


def example_events():
    grid = "type octile\nheight 2\nwidth 2\nmap\n..\n..\n"
    scen = "version 1\n0 a.map 2 2 0 0 1 0 1\n0 a.map 2 2 1 0 0 0 1\n"
    state = dict(seconds=0, restart=0, iteration=1, conflicts=1, cost=2,
                 agents=[agent(0, [0, 1]), agent(1, [1, 0])], selected=[0, 1], decisions_seen=0)
    initial = {**state, "event": "initial_pp", "conflicts": -1}
    init = {**state, "event": "init"}
    order = dict(event="pp_order", seconds=1, restart=0, iteration=2, order=[1, 0])
    step = {**state, "event": "step", "seconds": 2, "iteration": 2, "accepted": 0, "old_pairs": 1, "new_pairs": 1}
    final = {**state, "event": "final", "seconds": 3}
    return [initial, init, order, step, final], grid, scen


def test_transition_rollback_order_and_delta_checks():
    events, grid, scen = example_events()
    assert bridge.validate_events(events, grid, scen)["steps_checked"] == 1
    for change in ("order", "rollback", "delta", "clock"):
        broken = deepcopy(events)
        if change == "order":
            broken[2]["order"] = [0]
        elif change == "rollback":
            broken[3]["agents"][0]["path"] = [0, 2, 3, 1]
            broken[3]["cost"] = 4
            broken[3]["conflicts"] = 0
        elif change == "delta":
            broken[3].update(accepted=1, new_pairs=2)
        else:
            broken[3]["seconds"] = -1
        with pytest.raises(ValueError):
            bridge.validate_events(broken, grid, scen)


def test_scheduled_three_lanes_same_options_and_no_training():
    config = json.loads(bridge.CONFIG.read_text())
    ref = json.loads((bridge.ROOT / config["reference_config"]).read_text())
    jobs = bridge.schedule(config, ref)
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 96
    for i in range(0, len(jobs), 3):
        assert jobs[i]["argv"][1:] == jobs[i+1]["argv"][1:] == jobs[i+2]["argv"][1:]
    assert config["no_ttf_or_promotion"]


def test_log_comparison_ignores_only_wall_time():
    text = "Neighbors: 2, 1,\nAfter agent 2: Remaining agents = 1, colliding pairs = 0, LL nodes = 4, remaining time = 3\n"
    assert bridge.scientific_log(text) == bridge.scientific_log(text.replace("time = 3", "time = 2"))
    assert bridge.scientific_log(text) != bridge.scientific_log(text.replace("nodes = 4", "nodes = 5"))


def test_result_file_tamper_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    path = tmp_path / "out/jobs"
    path.mkdir(parents=True)
    log = path / "x.log"
    log.write_text("original")
    job = dict(job_id="x", output="out", argv=["plns"])
    row = dict(job_id="x", argv=["plns"], log_sha256=bridge.sha256_file(log), events_sha256=None)
    bridge.verify_job(row, job)
    log.write_text("changed")
    with pytest.raises(ValueError):
        bridge.verify_job(row, job)


def test_divergence_jobs_only_change_rule_and_keep_original_binary_repeat():
    config = json.loads(bridge.CONFIG.read_text())
    ref = json.loads((bridge.ROOT / config["reference_config"]).read_text())
    jobs = divergence.jobs(config, ref)
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 32
    for job in jobs:
        if job["lane"] == "upstream_repeat":
            original = next(j for j in jobs if j["lane"] == "upstream" and j["rule"] == job["rule"] and j["fixture"] == job["fixture"])
            assert job["argv"] == original["argv"]
    assert not divergence.first_difference([], ["a"])["equal_prefix"]
    assert divergence.first_difference(["a", "b"], ["a", "c"])["first_difference"] == 1
