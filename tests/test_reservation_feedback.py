from copy import deepcopy
from scripts.diagnose_reservation_feedback import choices, feedback_ranking, gate


def test_rank_uses_only_external_prefix_events():
    base=dict(records=[dict(incident_events=[dict(left=4,right=9,time=8),dict(left=4,right=9,time=9),
        dict(left=4,right=7,time=1),dict(left=4,right=10,time=10),dict(left=7,right=10,time=11)])])
    ranked=feedback_ranking(base,[4,7])
    assert [r['agent'] for r in ranked]==[10,9]
    assert ranked[0]['victims']==[4,7]


def test_feedback_does_not_read_final_outcomes():
    b=dict(records=[dict(incident_events=[dict(left=4,right=9,time=1)])])
    altered=deepcopy(b); altered.update(paths=[[999]],conflicts=-1000,future_success=True)
    assert choices(b,[4],'state/seed')==choices(altered,[4],'state/seed')


def test_single_feedback_random_equals_directed():
    b=dict(records=[dict(incident_events=[dict(left=4,right=9,time=1)])])
    s=choices(b,[4],'state/seed'); assert s['random']==s['directed']==9


def test_empty_external_feedback_is_not_applicable():
    assert choices(dict(records=[]),[4],'x')['directed'] is None


def test_single_seed_and_control_ties_cannot_pass_gate():
    rows=[]
    for i in range(4):
        for s in range(2):
            b=dict(applicable=True,recovered=True,final_conflicts=1)
            rows.append(dict(job=dict(case=dict(case_id=str(i),map_id=str(i),role='failed_long_unchanged')),
                             triggered=True,branches={k:dict(b) for k in ('directed_release','random_release','fresh_retry')}))
    g=gate(rows)
    assert len(g['stable_states'])==4 and not g['passed'] and not g['unique_states']
    assert not gate(rows[:1])['passed']
