from copy import deepcopy
import os

import pytest

from scripts import prepare_official_sa_pressure as s
from tests.evaluation.test_sa_pressure_first_feasible import cases
from tests.evaluation.test_sa_path_quality import template
from tests.evaluation.test_path_quality_execution import _cases


def test_schedule_preserves_all_conditions_and_no_iteration_limit():
    rows = s.schedule(cases())
    assert len(rows) == len({r['job_id'] for r in rows}) == 96
    assert rows == s.schedule(list(reversed(cases())))
    assert {r['controller'] for r in rows} == {'official_sa'}
    assert {r['protocol'] for r in rows} == {'first_feasible'}
    assert {r['budget_seconds'] for r in rows} == {120.}
    assert {r['repair_iteration_cap'] for r in rows} == {None}
    assert {r['timed_workers'] for r in rows} == {1}
    old = {s.q.admission_key(i): i for i in s.historical.schedule(cases()) if i['controller']=='official_adaptive'}
    for i in rows:
        a = old[s.q.admission_key(i)]
        for k in ('task_id', 'solver_seed', 'stage2_seed', 'map_id', 'budget_seconds'):
            assert i[k] == a[k]


def test_official_sa_does_not_request_learned_runtime(tmp_path):
    c = _cases(1)[0]
    item = dict(s.q.schedule([c])[0], controller='official_sa')
    t = template()
    original = deepcopy(t)
    job = s.q.worker_job(c, item, t, tmp_path, 'test')
    assert job['controller'] == 'official_adaptive'
    assert job['controller_runtime'] == 'reference'
    assert job['sa_controller'] == 'official_sa'
    assert job['environment']['max_repair_iterations'] == 0
    assert s.q.action_for('official_sa', {'case_id':'case'}, 0, [], None) == {'mode':'official'}
    assert original == t


def test_zero_conflict_official_sa_native(tmp_path, monkeypatch):
    if os.environ.get('LNS2_SA_PATH_NATIVE_TESTS') != '1':
        pytest.skip('explicit frozen native check required')
    pytest.importorskip('lns2_env')
    monkeypatch.setattr(s.q, 'ROOT', tmp_path)
    (tmp_path/'tiny.map').write_text('type octile\nheight 3\nwidth 3\nmap\n...\n...\n...\n')
    (tmp_path/'tiny.scen').write_text('version 1\n0\ttiny.map\t3\t3\t0\t0\t2\t2\t4\n')
    c=dict(task_id='tiny',map_id='tiny',family='warehouse',status='static_ready_runtime_unverified',
           solver_seeds=[51,52],static_audit=dict(agent_count=1),files=dict(map_file='tiny.map',scenario_file='tiny.scen'))
    item=dict(s.q.schedule([c])[0],controller='official_sa',budget_seconds=.5)
    r=dict(cases=[c],template=template(),fingerprint='test',inputs={})
    job=s.q.worker_job(c,item,r['template'],tmp_path/'out','test')
    env=s.q._make_environment(job['dataset_root'],job['row'],job['environment'],'Adaptive')
    state=s.q._plain(env.reset(seed=item['solver_seed']))
    anchor=dict(state_fingerprint=s.q.state_fingerprint(state))
    spec=s.q.spec_for(r,item,anchor,tmp_path/'out')
    s.q.child(spec)
    result=s.q.execution.read_artifact(tmp_path/'out/result.json',spec['binding'])
    assert result['status']=='completed'
    phase=s.q.execution.read_artifact(tmp_path/'out/first_phase_result.json',spec['binding'])
    assert phase['summary']['repair_iterations']==0
    s.q.audit_trace(spec)


def test_timing_authorization_precedes_any_runtime(monkeypatch):
    monkeypatch.setattr(s, 'ready', lambda: pytest.fail('must not inspect/start timing'))
    with pytest.raises(PermissionError):
        s.collect()
    with pytest.raises(ValueError, match='explicit resume'):
        s.collect(authorized=True, clear_stop=True)


def setup_collect(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'OUT', tmp_path)
    r = dict(fingerprint='test', schedule=s.schedule(cases()))
    monkeypatch.setattr(s, 'ready', lambda: r)
    monkeypatch.setattr(s, 'old_anchors', lambda r: {})
    monkeypatch.setattr(s.q, 'runtime_environment', lambda: {})
    return r


def test_stop_before_new_episode(tmp_path, monkeypatch):
    setup_collect(tmp_path, monkeypatch)
    s.q.write_json(tmp_path/'STOP_AFTER_EPISODE.json', {'requested':True})
    monkeypatch.setattr(s, 'spec_for', lambda *a: pytest.fail('no new episode'))
    assert s.collect(authorized=True) == dict(stopped=True, completed=0)
    assert not (tmp_path/'episodes').exists()


def test_partial_episode_never_automatically_retried(tmp_path, monkeypatch):
    setup_collect(tmp_path, monkeypatch)
    folder = tmp_path/'partial'
    folder.mkdir()
    monkeypatch.setattr(s, 'spec_for', lambda *a:dict(output=str(folder)))
    with pytest.raises(ValueError, match='interruption'):
        s.collect(authorized=True)
    assert s.q.read_json(tmp_path/'status.json')['status']=='failed_or_interrupted'


def test_resume_rejects_changed_completed_file(tmp_path, monkeypatch):
    r = setup_collect(tmp_path, monkeypatch)
    folder = tmp_path/'episode'
    folder.mkdir()
    (folder/'result.json').write_text('{}')
    s.q.write_json(tmp_path/'manifest.json', dict(binding='test', jobs={r['schedule'][0]['job_id']:
                    dict(files={'result.json':'wrong'})}))
    monkeypatch.setattr(s, 'spec_for', lambda *a:dict(output=str(folder)))
    with pytest.raises(ValueError, match='artifact changed'):
        s.collect(authorized=True, resume=True)


def test_stop_arriving_during_episode_commits_then_stops(tmp_path, monkeypatch):
    setup_collect(tmp_path, monkeypatch)
    folder=tmp_path/'episode'
    monkeypatch.setattr(s, 'spec_for', lambda *a:dict(output=str(folder)))
    def run(*args, **kwargs):
        folder.mkdir()
        s.q.write_json(folder/'result.json', dict(status='completed'))
        s.q.write_json(tmp_path/'STOP_AFTER_EPISODE.json', dict(requested=True))
        return dict(error=False, status='completed')
    monkeypatch.setattr(s.q.execution, 'supervise_episode', run)
    monkeypatch.setattr(s, 'inspect', lambda *a:dict(status='completed', success=True, ttf_seconds=1.))
    monkeypatch.setattr(s.q, 'audit_trace', lambda *a:None)
    assert s.collect(authorized=True) == dict(stopped=True, completed=1)
    assert len(s.q.read_json(tmp_path/'manifest.json')['jobs'])==1


def test_outer_timeout_stops_without_committing_or_retrying(tmp_path, monkeypatch):
    setup_collect(tmp_path, monkeypatch)
    monkeypatch.setattr(s, 'spec_for', lambda *a:dict(output=str(tmp_path/'episode')))
    monkeypatch.setattr(s.q.execution, 'supervise_episode', lambda *a,**k:dict(error=False,status='external_timeout'))
    with pytest.raises(ValueError, match='outer fuse'):
        s.collect(authorized=True)
    assert not (tmp_path/'manifest.json').exists()


def test_fingerprint_and_path_guards(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        s.contained(tmp_path, '../escape.json')
    with pytest.raises(ValueError, match='fingerprint'):
        s.check_fingerprint(dict(fingerprint='wrong'))


def test_first_phase_uses_sa_without_selector(tmp_path, monkeypatch):
    state = dict(initial_solution_complete=True, feasible=False)
    metrics = dict(action_valid=True, step_applied=True, native_replan_seconds=0., pp_failure_reason='none')
    calls = []
    class Env:
        def reset(self, seed):
            return state
        def step_experimental_pp(self, action, remaining, acceptance, temperature, draw):
            calls.append((action, acceptance, temperature, draw))
            return dict(observation=dict(state,feasible=True),metrics=metrics)
        def step_with_time_limit(self, *args):
            pytest.fail('Official+SA must not use standard acceptance')
    monkeypatch.setattr(s.q, '_make_environment', lambda *a:Env())
    monkeypatch.setattr(s.q, '_plain', lambda x:x)
    monkeypatch.setattr(s.q, 'state_fingerprint', lambda x:'fake')
    monkeypatch.setattr(s.q, 'encode_state_delta', lambda *a:{})
    monkeypatch.setattr(s.q, 'SingleFullCheckPool', lambda *a:pytest.fail('no model'))
    job=dict(dataset_root='',row=dict(task_id='task'),environment={},sa_case_id='case',solver_seed=61,
             sa_proposal={},sa_controller='official_sa',output_root=str(tmp_path),wall_time_budget_seconds=120)
    result=s.q.first_phase(job,path_observer=lambda *a:None)
    assert result['summary']['repair_iterations']==1
    assert len(calls)==1 and calls[0][:3]==({'mode':'official'},'annealed',1000.)
