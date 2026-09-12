from types import SimpleNamespace

import pytest

from scripts import audit_sa_engineering_speed as audit


def test_variant_changes_only_redundant_check_flag(monkeypatch):
    calls = []
    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return "candidates"
    monkeypatch.setattr(audit.online_selection, "generate_online_candidates", generate)
    def base_select(self, env, state, decision):
        return audit.online_selection.generate_online_candidates(env, state, decision_index=decision,
            verify_full_state=True, proposal_backend="optimized", state_hash="unchanged")
    monkeypatch.setattr(audit.source.pilot.TimedPool, "select", base_select)
    selector = object.__new__(audit.SingleFullCheckPool)
    assert selector.select("env", "state", 9) == "candidates"
    assert calls == [(('env', 'state'), dict(decision_index=9, verify_full_state=False,
                                           proposal_backend="optimized", state_hash="unchanged"))]
    assert audit.online_selection.generate_online_candidates is generate


def test_patch_is_restored_after_failure(monkeypatch):
    original = audit.online_selection.generate_online_candidates
    def fail(*args):
        raise ValueError("guard failed")
    monkeypatch.setattr(audit.source.pilot.TimedPool, "select", fail)
    with pytest.raises(ValueError, match="guard failed"):
        object.__new__(audit.SingleFullCheckPool).select(None, {}, 0)
    assert audit.online_selection.generate_online_candidates is original


def test_counting_proxy_forwards_revision_and_fault_is_observation_only():
    native = SimpleNamespace(get_state=lambda: dict(iteration=7), get_state_revision=lambda: 3)
    proxy = audit.CountingEnvironment(native)
    assert proxy.get_state_revision() == 3
    assert proxy.get_state() == dict(iteration=7)
    proxy.corrupt_snapshot = True
    assert proxy.get_state() == dict(iteration=8)
    assert native.get_state() == dict(iteration=7)
    assert proxy.snapshots == 2
