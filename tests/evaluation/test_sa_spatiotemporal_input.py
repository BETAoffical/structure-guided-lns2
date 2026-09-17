import copy
import gzip
import json
import random

import pytest

from experiments.sa_spatiotemporal_input import (
    candidate_diagnostics, candidate_input, decode_paths, encode_state,
)
from experiments.state_analysis import reconstruct_conflicts
from scripts.audit_sa_spatiotemporal_input import frozen_read, summarize, verify, write_pack


def state(paths=None):
    paths = paths or {3:[3,4,5], 8:[1,4,7]}
    agents = [dict(id=i, path=p, start=p[0], goal=p[-1], path_cost=len(p)-1) for i,p in paths.items()]
    pairs = sorted({(e.left,e.right) for e in reconstruct_conflicts(agents)})
    return dict(rows=3,cols=3,obstacles=[0]*9,agents=agents,
                conflict_edges=[list(e) for e in pairs], num_of_colliding_pairs=len(pairs))


def test_vertex_roundtrip_and_nonconsecutive_ids():
    s = state()
    encoded = encode_state(s)
    assert [a["path"] for a in decode_paths(encoded)] == [[3,4,5],[1,4,7]]
    assert encoded["events"] == [[1,"vertex",3,8,[4]]]
    assert candidate_input(encoded,[8]) == {"selected_mask":[False,True]}
    assert candidate_diagnostics(encoded,[3,8])["internal_events"] == 1


def test_edge_swap_and_persistent_goal_occupancy():
    assert encode_state(state({3:[3,4],8:[4,3]}))["events"] == [[1,"edge",3,8,[3,4]]]
    encoded = encode_state(state({3:[3,4],8:[1,2,5,4,7]}))
    assert encoded["agents"][0]["occupancy"][-1] == [4,1,None]
    assert encoded["events"] == [[3,"vertex",3,8,[4]]]
    assert candidate_diagnostics(encoded,[8])["outsider_goal_tail_events"] == 1


def test_no_movement_and_final_waits_roundtrip():
    encoded = encode_state(state({3:[3,4,4,4],8:[7]}))
    assert decode_paths(encoded)[0]["path"] == [3,4,4,4]
    assert encoded["agents"][0]["occupancy"][-1] == [4,1,None]
    assert encoded["agents"][1]["occupancy"] == [[7,0,None]]


def test_current_conflict_edges_do_not_preserve_timing():
    early = encode_state(state({3:[3,4,5,5],8:[1,4,7,7]}))
    late = encode_state(state({3:[3,3,4,5],8:[1,1,4,7]}))
    assert [e[1:] for e in early["events"]] == [e[1:] for e in late["events"]]
    assert [a["path_cost"] for a in early["agents"]] == [a["path_cost"] for a in late["agents"]]
    assert early["events"][0][0] == 1 and late["events"][0][0] == 2
    assert early != late


def test_external_conflicts_do_not_become_candidate_conflicts():
    encoded = encode_state(state({3:[3,4,5],8:[1,4,7],25:[6]}))
    row = candidate_diagnostics(encoded,[25])
    assert row["internal_events"] == row["boundary_events"] == 0
    assert row["external_events"] == 1
    assert row["nearby_noncolliding_outsiders"] > 0


def test_whitelist_and_order_invariance():
    original = state()
    reference = encode_state(original)
    changed = copy.deepcopy(original)
    changed.update(runtime=12345, low_level={"generated":99999}, context={"map_id":"LEAK"},
                   after_paths=[[0]], outcome={"completion":True})
    for a in changed["agents"]:
        a.update(delay=123, generated=456, after_path=[0])
    changed["agents"].reverse()
    assert encode_state(changed) == reference
    assert original == state()
    assert candidate_input(reference,[8,3]) == candidate_input(reference,[3,8])


@pytest.mark.parametrize("mutation", [
    lambda s:s.update(obstacles=[0]*8),
    lambda s:s["obstacles"].__setitem__(4,1),
    lambda s:s["obstacles"].__setitem__(0,2),
    lambda s:s["agents"][0].update(id=True),
    lambda s:s["agents"][0].update(id=8),
    lambda s:s["agents"][0].update(path=[]),
    lambda s:s["agents"][0].update(path=[3,2,5]),
    lambda s:s["agents"][0].update(start=0),
    lambda s:s["agents"][0].update(goal=0),
    lambda s:s["agents"][0].update(path_cost=55),
    lambda s:s.update(conflict_edges=[[3,99]]),
    lambda s:s.update(conflict_edges=[[3,3]]),
    lambda s:s.update(conflict_edges=[]),
    lambda s:s.update(conflict_edges=[[3,8],[8,3]]),
    lambda s:s.update(num_of_colliding_pairs=2),
])
def test_invalid_states_rejected(mutation):
    s = state()
    mutation(s)
    with pytest.raises(ValueError):
        encode_state(s)


@pytest.mark.parametrize("membership", [[],[3,3],[99],[True]])
def test_invalid_candidate_rejected(membership):
    with pytest.raises(ValueError,match="membership"):
        candidate_input(encode_state(state()),membership)


def test_independent_interval_scan_matches_random_walks():
    rng = random.Random(371)
    checked = 0
    for _ in range(150):
        paths = {}
        for i,start in enumerate(rng.sample(range(9),4)):
            p = [start]
            for _ in range(rng.randint(0,12)):
                row,col = divmod(p[-1],3)
                possible = [r*3+c for r,c in ((row,col),(row-1,col),(row+1,col),(row,col-1),(row,col+1))
                            if 0 <= r < 3 and 0 <= c < 3]
                p.append(rng.choice(possible))
            paths[10+3*i] = p
        if len({p[-1] for p in paths.values()}) != 4:
            continue
        encode_state(state(paths))
        checked += 1
    assert checked >= 20


def test_pack_deterministic_and_readable(tmp_path):
    value = encode_state(state())
    first, second = tmp_path/"one.gz", tmp_path/"two.gz"
    write_pack(first,value)
    write_pack(second,value)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first,"rt") as stream:
        assert json.load(stream) == value


def test_changed_source_and_completed_output_rejected(tmp_path,monkeypatch):
    from scripts import audit_sa_spatiotemporal_input as audit
    from experiments._common import sha256_file, write_json, json_fingerprint
    monkeypatch.setattr(audit,"ROOT",tmp_path)
    config = "config.json"
    cfg = {"output":"build/result"}
    write_json(tmp_path/config,cfg)
    inputs = {config:sha256_file(tmp_path/config)}
    binding = json_fingerprint(dict(config=cfg,inputs=inputs))
    out = tmp_path/cfg["output"]
    write_json(out/"run_config.json",dict(config=cfg,inputs=inputs,binding=binding))
    write_json(out/"run_status.json",dict(status="completed",binding=binding))
    write_json(out/"complete.json",dict(inputs=inputs,binding=binding,
        files={"run_config.json":sha256_file(out/"run_config.json")}))
    assert verify(config)["status"] == "verified"
    with pytest.raises(ValueError,match="SHA mismatch"):
        frozen_read(config,"0"*64)
    write_json(out/"run_config.json",{})
    with pytest.raises(ValueError,match="output changed"):
        verify(config)


def test_no_training_or_solver_promotion_from_input_counts():
    row = dict(state_id="x",map_id="map",source="old16",candidate_diagnostics=[
        candidate_diagnostics(encode_state(state()),[3])], conflict_pairs=1,conflict_events=1,
        path_points=6,occupancy_intervals=6,bytes=100,max_path_cost=2)
    report = summarize([row])
    assert report["states"] == 1 and report["existing_label_trials"] == 8
    assert report["input_integrity_passed"]
    assert not report["learnability_established"]
    assert not report["automatic_training_allowed"]
    assert not report["runtime_integration_allowed"]
