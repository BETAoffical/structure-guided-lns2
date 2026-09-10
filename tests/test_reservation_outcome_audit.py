import pytest
from scripts.audit_reservation_outcomes import verify_branch, canonical_role, corrected_gate


@pytest.mark.parametrize('name',['failed_long','failed_long_unchanged'])
def test_failure_role_aliases(name):
    assert canonical_role(name)=='failed_tail'


def test_unknown_role_is_rejected_not_silently_excluded():
    with pytest.raises(ValueError,match='unknown case role'):
        canonical_role('failure_new_unknown')


def test_short_failure_alias_can_satisfy_failure_source_gate():
    rows=[]
    for i in range(3):
        for _ in range(2):
            rows.append(dict(job=dict(case=dict(case_id=str(i),map_id=str(i),role='failed_long')),
                triggered=True,branches=dict(
                    directed_release=dict(applicable=True,recovered=True,final_conflicts=0),
                    random_release=dict(applicable=True,recovered=False,final_conflicts=1),
                    fresh_retry=dict(applicable=True,recovered=False,final_conflicts=1))))
    assert corrected_gate(rows)['passed']


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
