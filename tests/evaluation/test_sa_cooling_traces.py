from copy import deepcopy
import pytest
from scripts import audit_sa_cooling_traces as audit


@pytest.mark.parametrize("decision,expected",[(0,"0-99"),(99,"0-99"),(100,"100-499"),(499,"100-499"),(500,"500-999"),(999,"500-999"),(1000,"1000+")])
def test_iteration_boundaries(decision,expected):
    assert audit.iteration_bin(decision)==expected


def test_physical_identity_ignores_counters_but_not_paths():
    a=dict(rows=1,cols=3,obstacles=[0,0,0],conflict_edges=[[3,8]],
           agents=[dict(id=3,path=[0,1,2]),dict(id=8,path=[2,1,0])],iteration=0,low_level=dict(generated=3))
    key=audit.PhysicalIdentity(a).key(a)
    b=deepcopy(a)
    b.update(iteration=100,runtime=20,low_level=dict(generated=1000))
    assert audit.PhysicalIdentity(b).key(b)==key
    b["agents"][0]["path"]=[0,0,1,2]
    assert audit.PhysicalIdentity(b).key(b)!=key


def test_incremental_physical_key_matches_full_rebuild():
    a=dict(rows=1,cols=3,obstacles=[0,0,0],conflict_edges=[[3,8]],
           agents=[dict(id=3,path=[0,1,2]),dict(id=8,path=[2,1,0])])
    b=deepcopy(a)
    b["agents"][0]["path"]=[0,0,1,2]
    physical=audit.PhysicalIdentity(a)
    delta=dict(agents=dict(mode="patch",patches=[dict(id=3,set=dict(path=[0,0,1,2]),remove=[])]))
    assert physical.update(delta,b)==audit.PhysicalIdentity(b).key(b)


def test_conflict_bins_and_terminal_exclusion():
    assert [audit.conflict_bin(n) for n in (1,5,6,20,21)]==["1-5","1-5","6-20","6-20","21+"]
    with pytest.raises(ValueError): audit.conflict_bin(0)


def test_followup_is_censored_not_assumed_failure():
    assert audit.recovery_window([4,4],3,4,False)=="right_censored"
    assert audit.recovery_window([4,2],3,4,False)=="below_pre_increase"


def test_zero_conflict_episode_has_no_decision_statistics(monkeypatch):
    monkeypatch.setattr(audit,"REPORTS",dict(zero=(None,None,1)))
    r=dict(cohort="zero",success=True,bins={},attempts=[])
    out=audit.aggregate([r])["zero"]
    assert out["success"]["episodes"]==1
    assert out["all"]["bins"]=={}
    assert out["all"]["probability_median"] is None
