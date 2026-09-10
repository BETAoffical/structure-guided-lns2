import gzip
import json

from experiments.native_path_compatibility import seal
from scripts import audit_single_release_outcomes as audit


def make_fixture(tmp_path, monkeypatch, raw, row):
    state = dict(agents=[dict(id=0, path=[0, 1]), dict(id=1, path=[2, 2])])
    base = dict(paths=[[0, 1], [2, 2]], diagnostics=[dict(agent=0, path=[0, 1])])
    monkeypatch.setattr(audit, '_DATA', ({}, tmp_path, {'c': (dict(order=[0]), state, base)}))
    monkeypatch.setattr(audit, 'read_result', lambda *args: row)
    (tmp_path/'raw').mkdir()
    with gzip.open(tmp_path/'raw/j.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(seal(dict(raw=raw)), f)
    return dict(id='j', condition='c', released=1)


def test_failure_classification_counts_real_external_pair(tmp_path, monkeypatch):
    raw = dict(diagnostics=[dict(agent=0, path=[0, 1, 2, 1])], rolled_back=True, attempted_pairs=1)
    job = make_fixture(tmp_path, monkeypatch, raw, dict(status='ok'))
    _, counts = audit.describe(job)
    assert counts['rolled_back'] == 1
    assert counts['rollback_with_new_pair_touching_released_agent'] == 1
    assert counts['different_attempted_path_or_stop'] == 1


def test_same_path_tie_not_improvement(tmp_path, monkeypatch):
    raw = dict(diagnostics=[dict(agent=0, path=[0, 1])], rolled_back=False)
    job = make_fixture(tmp_path, monkeypatch, raw, dict(status='ok', path_check=dict(conflicts=0)))
    _, counts = audit.describe(job)
    assert counts['accepted_tie'] == 1 and counts['same_attempted_paths_and_stop'] == 1
    assert not counts.get('accepted_better')


def test_unknown_not_counted_as_mechanism_failure(tmp_path, monkeypatch):
    job = make_fixture(tmp_path, monkeypatch, {}, dict(status='unknown'))
    _, counts = audit.describe(job)
    assert counts == dict(total=1, unknown_or_not_found=1)
