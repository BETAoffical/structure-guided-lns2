import pytest

from scripts.diagnose_reservation_mediation import (
    compare_contexts, context_before, query_seed, released_pp,
)


def state():
    return dict(agents=[dict(id=0,path=[0,1],start=0,goal=1),
                        dict(id=1,path=[2,2],start=2,goal=2)],
                num_of_colliding_pairs=0, conflict_edges=[],sum_of_costs=2,
                rows=1,cols=3,obstacles=[False]*3)


def test_query_seed_frozen_and_distinct():
    assert query_seed(0,0)==query_seed(0,0)
    assert len({query_seed(i,j) for i in range(8) for j in range(3)})==24


def test_context_preserves_external_and_prior_new_paths():
    s=dict(agents=[dict(id=4,path=[0]),dict(id=7,path=[1]),dict(id=9,path=[2])])
    assert context_before(s,[7,4],[dict(agent=7,path=[3])],4)=={9:[2],7:[3]}


def test_single_reservation_intervention_guard():
    compare_contexts({1:[0],2:[3]},{2:[3]},1)
    with pytest.raises(ValueError,match='non-target'):
        compare_contexts({1:[0],2:[3]},{2:[4]},1)


def test_ignored_cat_path_still_counts_for_rollback():
    class Probe:
        def seed_rng(self,seed): pass
        def plan(self,agent,fixed,overrides,hard,seconds):
            assert fixed==[]
            return dict(status='path',path=[0,1,2,1])
    r=released_pp(Probe(),state(),[0],12,1)
    assert r['rolled_back'] and r['paths']==[[0,1],[2,2]]
    assert r['attempted_pairs']==1


def test_query_unknown_is_not_infeasible():
    class Probe:
        def seed_rng(self,seed): pass
        def plan(self,*args,**kwargs): return dict(status='unknown',path=[])
    assert released_pp(Probe(),state(),[0],12,1)['status']=='unknown'


def test_cannot_release_selected_agent():
    with pytest.raises(ValueError,match='external'):
        released_pp(None,state(),[0],12,0)
