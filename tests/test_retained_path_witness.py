import itertools
import random

import pytest

from scripts.audit_retained_path_witness import (
    check_splice, closure, dependency_row, edges, task_signature,
)


def two_car():
    return dict(rows=2, cols=3, obstacles=[False]*6,
                agents=[dict(id=4, start=0, goal=2, path=[0, 1, 2]),
                        dict(id=9, start=2, goal=0, path=[2, 1, 0])],
                conflict_edges=[[4, 9]], num_of_colliding_pairs=1)


def test_external_path_replacement_is_actual_not_membership():
    s = two_car()
    r = check_splice(s, {4:[0, 3, 4, 5, 2]}, [9])
    assert r['conflicts_after']==0 and r['global_feasible']
    assert r['changed_agents']==r['external_changed_agents']==[4]
    assert s['agents'][0]['path']==[0, 1, 2]


def test_task_identity_ignores_path_but_not_endpoints():
    s = two_car(); t = two_car(); t['agents'][0]['path']=[0, 3, 4, 5, 2]
    assert task_signature(s)==task_signature(t)
    t['agents'][0]['goal']=5
    assert task_signature(s)!=task_signature(t)


def test_dependency_direction_and_noncontiguous_ids():
    s = two_car(); ref = {4:[0, 3, 4, 5, 2], 9:[2, 1, 0]}
    assert dependency_row(s, ref, 4)==[]
    assert dependency_row(s, ref, 9)==[4]
    assert closure({4:[], 9:[4]}, [9])==[4, 9]


def test_cycle_is_not_dropped():
    assert closure({1:[2], 2:[3], 3:[1], 4:[]}, [1])==[1, 2, 3]


def test_unknown_dependency_rejected():
    with pytest.raises(ValueError, match='endpoint'):
        closure({4:[9]}, [4])
    with pytest.raises(ValueError, match='root'):
        closure({4:[]}, [9])


def test_swap_and_permanent_goal_conflicts():
    assert edges([dict(id=0,path=[0,1]), dict(id=1,path=[1,0])])=={(0,1)}
    assert edges([dict(id=0,path=[0]), dict(id=1,path=[2,1,0])])=={(0,1)}


def test_no_conflict_count_shortcut():
    with pytest.raises(ValueError, match='witness'):
        check_splice(two_car(), {4:[0,1,2]}, [4])


def test_same_path_not_counted_as_changed():
    r = check_splice(two_car(), {4:[0,3,4,5,2], 9:[2,1,0]}, [4])
    assert r['changed_agents']==[4]
    assert r['external_changed_agents']==[]


def test_invalid_witness_geometry_rejected():
    with pytest.raises(ValueError, match='move'):
        check_splice(two_car(), {4:[0,5,2]}, [4])


@pytest.mark.parametrize('seed', range(12))
def test_closed_splices_against_all_binary_assignments(seed):
    rng = random.Random(seed); starts=[0,2,6,8]; agents=[]
    for i, start in enumerate(starts):
        walk=[start]
        for _ in range(5):
            pos=walk[-1]; r,c=divmod(pos,3)
            choices=[pos]+[rr*3+cc for rr,cc in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)) if 0<=rr<3 and 0<=cc<3]
            walk.append(rng.choice(choices))
        path=walk+walk[-2::-1]
        agents.append(dict(id=i,start=start,goal=start,path=path))
    before=edges(agents)
    state=dict(rows=3,cols=3,obstacles=[False]*9,agents=agents,
               conflict_edges=sorted(before),num_of_colliding_pairs=len(before))
    ref={i:[start] for i,start in enumerate(starts)}
    graph={i:dependency_row(state,ref,i) for i in range(4)}
    for bits in itertools.product((False,True),repeat=4):
        selected={i for i,b in enumerate(bits) if b}
        if not selected or set(closure(graph,selected))!=selected:
            continue
        r=check_splice(state,{i:ref[i] for i in selected},[])
        assert r['conflicts_after']==len({e for e in before if not selected.intersection(e)})
    for root in range(4):
        minimal=set(closure(graph,[root]))
        for bits in itertools.product((False,True),repeat=4):
            selected={i for i,b in enumerate(bits) if b}
            if root in selected and set(closure(graph,selected))==selected:
                assert minimal<=selected


def test_policy_boundaries_are_fixed():
    from scripts.audit_retained_path_witness import CONFIG, read_json
    c=read_json(CONFIG)
    assert not c['solver_search_allowed'] and not c['timing_experiment']
    assert c['path_capture']=='first_feasible.json'
    assert c['reference_controller']=='official_adaptive'
    assert c['workers']==20
