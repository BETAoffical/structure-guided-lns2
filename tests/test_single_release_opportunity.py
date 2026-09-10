import copy

import pytest

from experiments.local_path_compatibility import digest
from scripts.diagnose_single_release_opportunity import (
    compact, make_jobs, reconstruct, select_cases, summarize,
)


def state():
    return dict(agents=[dict(id=0, path=[0, 1]), dict(id=1, path=[2, 2])])


def record(path):
    return dict(agent=0, status='path', path=path, generated=5, expanded=3,
                cost=len(path)-1, low_level_collisions=0)


def test_all_external_ids_not_feedback_subset():
    c = dict(id='c', case=dict(case_id='s'), order=[0])
    jobs = make_jobs([c], {'s': state()})
    assert [j['released'] for j in jobs] == [1]
    assert jobs == make_jobs([c], {'s': state()})


def test_noncontiguous_native_ids_rejected():
    s = state(); s['agents'][1]['id'] = 5
    with pytest.raises(ValueError, match='scenario-indexed'):
        make_jobs([dict(id='c', case=dict(case_id='s'), order=[0])], {'s': s})


def test_duplicate_order_rejected():
    with pytest.raises(ValueError, match='order'):
        make_jobs([dict(id='c', case=dict(case_id='s'), order=[0, 0])], {'s': state()})


def test_selection_ignores_outcomes_seed_and_input_order():
    cases = [dict(case_id=str(i), map_id=str(i//3), role='failed_long') for i in range(6)]
    jobs = [dict(case=c, outcome=999) for c in cases]
    selected = select_cases(jobs, 'frozen/')
    assert len(selected) == 2
    assert selected == select_cases(list(reversed(jobs))*2, 'frozen/')
    assert select_cases([dict(case=dict(case_id='fast', map_id='0', role='fast'))]+jobs, 'frozen/') == selected


def test_compact_roundtrip_and_original_immutable():
    s = state(); before = copy.deepcopy(s)
    raw = dict(status='ok', rolled_back=False, paths=[[0, 1], [2, 2]],
               diagnostics=[record([0, 1])], attempted_pairs=0)
    packed = compact(raw)
    assert 'paths' not in packed and 'paths' in raw
    assert reconstruct(s, [0], packed) == raw['paths']
    assert digest(raw['paths']) == packed['paths_sha256']
    assert s == before


def test_released_external_still_forces_rollback():
    raw = dict(status='ok', rolled_back=True, diagnostics=[record([0, 1, 2, 1])], attempted_pairs=1)
    assert reconstruct(state(), [0], raw) == [[0, 1], [2, 2]]
    raw['rolled_back'] = False
    with pytest.raises(ValueError, match='rollback'):
        reconstruct(state(), [0], raw)


def test_false_rollback_rejected():
    with pytest.raises(ValueError, match='spurious'):
        reconstruct(state(), [0], dict(status='ok', rolled_back=True,
                    diagnostics=[record([0, 1])], attempted_pairs=0))


def test_bad_diagnostic_order_rejected():
    r = record([0, 1]); r['agent'] = 1
    with pytest.raises(ValueError, match='order/prefix'):
        reconstruct(state(), [0], dict(status='ok', rolled_back=False, diagnostics=[r], attempted_pairs=0))


def test_wrong_pair_ledger_rejected():
    with pytest.raises(ValueError, match='ledger'):
        reconstruct(state(), [0], dict(status='ok', rolled_back=False,
                    diagnostics=[record([0, 1])], attempted_pairs=9))


def test_unknown_not_reconstructed_as_no_solution():
    with pytest.raises(ValueError, match='completed'):
        reconstruct(state(), [0], dict(status='unknown', diagnostics=[]))


def setup_summary():
    cases = [dict(case_id=str(i), map_id=str(i)) for i in range(3)]
    conditions = [dict(id=f'{i}/{s}', case=c, seed=s) for i, c in enumerate(cases) for s in (1, 2)]
    rows = [dict(job=dict(condition=c['id'], released=a), status='ok', path_check=dict(conflicts=9))
            for c in conditions for a in (0, 1)]
    p = dict(cases=cases, conditions=conditions, jobs=[r['job'] for r in rows],
             config=dict(minimum_mean_relative_reduction=.05, minimum_stable_states=2))
    return p, rows, {c['id']: 10 for c in conditions}


def test_same_external_required_across_seeds():
    p, rows, base = setup_summary()
    for r in rows:
        seed = int(r['job']['condition'][-1])
        r['path_check']['conflicts'] = 9 if r['job']['released']==seed-1 else 10
    report = summarize(p, rows, base)
    assert not report['passed'] and report['stable_states_at_five_percent'] == 0
    assert all(any(s['improving_agents'] for s in c['seeds']) for c in report['cases'])


def test_opportunity_does_not_mean_runtime_admission():
    p, rows, base = setup_summary()
    result = summarize(p, rows, base)
    assert result['passed']
    assert result['decision'] == 'opportunity_only_selector_not_validated'
    assert all(c['best_fixed_agent']==0 for c in result['cases'])


@pytest.mark.parametrize('status', ['unknown', 'not_found'])
def test_censored_result_prevents_complete_gate(status):
    p, rows, base = setup_summary(); rows[0]['status'] = status
    result = summarize(p, rows, base)
    assert not result['passed'] and not result['complete']
    assert result['decision'] == 'incomplete_or_unknown'


def test_missing_result_prevents_gate():
    p, rows, base = setup_summary()
    assert not summarize(p, rows[1:], base)['passed']


def test_duplicate_agent_rejected():
    p, rows, base = setup_summary()
    with pytest.raises(ValueError, match='duplicate'):
        summarize(p, rows+rows[:1], base)


def test_exact_gate_not_rounded_up():
    p, rows, base = setup_summary()
    for r in rows:
        r['path_check']['conflicts'] = 9.500001
    assert not summarize(p, rows, base)['passed']


def test_configuration_fixed_scope():
    from scripts.diagnose_single_release_opportunity import CONFIG, read_json
    c = read_json(CONFIG)
    assert c['workers'] == 20 and c['expected_jobs'] == 2708
    assert c['seeds'] == [20260912, 20260913]
    assert c['session_seconds'] == 3600
    assert len(c['expected_cases']) == 3
