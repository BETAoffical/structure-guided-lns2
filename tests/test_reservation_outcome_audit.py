import pytest
from scripts.audit_reservation_outcomes import verify_branch


def test_final_external_occupancy_cannot_be_omitted_from_acceptance():
    s=dict(agents=[dict(id=0,path=[0,1],start=0,goal=1),dict(id=1,path=[2],start=2,goal=2)],
           rows=1,cols=3,obstacles=[False]*3,num_of_colliding_pairs=0,conflict_edges=[],sum_of_costs=1)
    raw=dict(diagnostics=[dict(agent=0,status='path',path=[0,1,2,1])],
             rolled_back=False,paths=[[0,1,2,1],[2]],attempted_pairs=0)
    with pytest.raises(ValueError,match='rollback versus true'):
        verify_branch(s,[0],raw)


def test_terminal_occupancy_accounted_after_external_path_ends():
    s=dict(agents=[dict(id=0,path=[0,1],start=0,goal=1),dict(id=1,path=[2],start=2,goal=2)],
           rows=1,cols=3,obstacles=[False]*3,num_of_colliding_pairs=0,conflict_edges=[],sum_of_costs=1)
    raw=dict(diagnostics=[dict(agent=0,status='path',path=[0,1,2,1])],
             rolled_back=True,paths=[[0,1],[2]],attempted_pairs=1)
    assert verify_branch(s,[0],raw)['pairs']=={(0,1)}
