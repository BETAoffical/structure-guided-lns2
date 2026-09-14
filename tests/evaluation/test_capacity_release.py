from copy import deepcopy
from experiments.capacity_release import propose, domain_at_tick
from experiments.joint_slot_capacity import hall_witness
import pytest


def state():
    return dict(rows=1, cols=4, obstacles=[False]*4, agents=[
        dict(id=2,start=0,goal=3,path=[0,1,2,3]),
        dict(id=8,start=2,goal=1,path=[2,1])],
        conflict_edges=[[2,8]], num_of_colliding_pairs=1)


def test_release_domains_are_monotonic_and_external_only():
    s = state()
    old = deepcopy(s)
    blocked = domain_at_tick(s, [2], [2], 1)
    released = domain_at_tick(s, [2], [2], 1, [8])
    assert blocked[2] <= released[2]
    assert released[2] and not blocked[2]
    assert s == old
    with pytest.raises(ValueError):
        domain_at_tick(s, [2], [2], 1, [2])


def test_domain_matching_distinguishes_necessary_condition_from_joint_path():
    assert hall_witness({2:{0,1},8:{0,1}}) is None
    assert hall_witness({2:{0},8:{0}}) is not None


def test_input_only_proposals_are_deterministic_and_never_select_internal_agents():
    s = dict(rows=3, cols=4, obstacles=[False]*12, agents=[
        dict(id=2,start=0,goal=3,path=[0,1,2,3]),
        dict(id=8,start=6,goal=5,path=[6,5]),
        dict(id=9,start=11,goal=11,path=[11])], conflict_edges=[], num_of_colliding_pairs=0)
    a = propose(s, [2], [2], 1, "test", limit=1)
    assert a == propose(deepcopy(s), [2], [2], 1, "test", limit=1)
    assert len(a["candidates"]) == 3
    assert all(2 not in c["members"] for c in a["candidates"])
