"""Pure preparation checks: no native module, selector inference or repair."""
from collections import Counter
from copy import deepcopy
import pytest

from experiments import sa_direct_confirmation as rt
from scripts import prepare_sa_direct_confirmation as prep


def sample():
    cfg = dict(maps=12,densities=[.2,.25],solver_seeds=[359,367,373],
               comparison_arms=list(rt.ARMS),phase='test-preparation',output='build/test-unused',
               budget_seconds=120.,timing_mode='deferred_scientific_audit_v1')
    cases = [dict(task_id=f'm{m:02d}-d{d}',map_id=f'm{m:02d}',density=d)
             for m in range(12) for d in (.2,.25)]
    models = dict(direct=dict(path='direct.json',sha256='d'*64),
                  a=dict(path='a.json',sha256='a'*64))
    return rt.make_schedule(cases,cfg,models,{'frozen':True},'b'*64),cfg


def test_williams_balance_and_pairing():
    orders = [rt.order(i) for i in range(10)]
    assert len(set(orders))==10
    for a in rt.ARMS:
        assert Counter(row.index(a) for row in orders)=={p:2 for p in range(5)}
    transitions = Counter((a,b) for row in orders for a,b in zip(row,row[1:]))
    assert set(transitions.values())=={2} and len(transitions)==20
    jobs,cfg = sample()
    counts = rt.check_schedule(jobs,cfg)
    assert counts['conditions']==72 and counts['episodes']==360
    assert all(j['execution_authorized'] is False and j['expected_initial'] is None for j in jobs)
    assert all(j['iteration']==2 for j in jobs if j['comparison_arm']=='a')


def test_execution_guard_before_native(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('native/timing worker reached during preparation')
    monkeypatch.setattr(rt.direct.lean,'lean_worker',forbidden)
    monkeypatch.setattr(rt.direct.lean,'WORKERS',{a:forbidden for a in rt.ARMS})
    jobs,_ = sample()
    for j in jobs[:5]:
        with pytest.raises(ValueError,match='timing not authorized'):
            rt.timed_worker(j)
    j = deepcopy(jobs[0])
    j['execution_authorized'] = True
    with pytest.raises(ValueError,match='initial qualification'):
        rt.timed_worker(j)


def test_modified_schedule_rejected():
    jobs,cfg = sample()
    jobs[0]['solver_seed'] = 999
    with pytest.raises(ValueError,match='unpaired solver_seed'):
        rt.check_schedule(jobs,cfg)
    jobs,cfg = sample()
    jobs[0]['execution_authorized'] = True
    with pytest.raises(ValueError,match='must not authorize'):
        rt.check_schedule(jobs,cfg)


def test_cli_has_no_timing_phase(monkeypatch):
    for phase in ('collect','evaluate','preflight','qualify','resume'):
        monkeypatch.setattr(prep.sys,'argv',['prepare_sa_direct_confirmation.py',phase])
        with pytest.raises(SystemExit) as caught:
            prep.main()
        assert caught.value.code==2
