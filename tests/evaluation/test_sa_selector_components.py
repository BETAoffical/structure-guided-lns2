import pytest

from scripts import profile_sa_selector_components as profile


def test_positions_are_fixed_and_deduplicated():
    assert profile.positions(195) == [0,97,194]
    assert profile.positions(6) == [0,2,5]
    assert profile.positions(1) == [0]
    for n in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            profile.positions(n)


def test_private_bindings_reuse_frozen_bytecode(monkeypatch):
    monkeypatch.setattr(profile.SingleFullCheckPool,"__init__",lambda self,case:None)
    original = profile.SingleFullCheckPool.select
    before = dict(original.__globals__)
    measured = profile.ProfilePool({}, True)
    reference = profile.ProfilePool({}, False)
    assert measured.frozen_call.__code__ is original.__code__
    assert reference.frozen_call.__code__ is original.__code__
    assert original.__globals__ == before
    assert measured.frozen_call.__globals__ is not original.__globals__
    assert reference.frozen_call.__globals__["generate_online_candidates"] is before["generate_online_candidates"]
    assert measured.frozen_call.__globals__["generate_online_candidates"] is not before["generate_online_candidates"]


def test_proxy_preserves_attributes_and_return_values(monkeypatch):
    monkeypatch.setattr(profile.SingleFullCheckPool,"__init__",lambda self,case:None)
    owner = profile.ProfilePool({},True)
    class Target:
        revision=7
        def prepare(self,value): return value
    target=Target()
    proxy=profile.MethodProxy(target,owner,{"prepare":"prepare"})
    obj={"state":[1,2,3]}
    assert proxy.revision==7
    assert proxy.prepare(obj) is obj
    assert owner.parts["prepare"]>=0
    assert target.revision==7


def test_nested_metadata_not_added_to_disjoint_timers(monkeypatch):
    monkeypatch.setattr(profile.SingleFullCheckPool,"__init__",lambda self,case:None)
    owner=profile.ProfilePool({},True)
    value=([],{"internal_seconds":999.,"not_a_timer":4})
    assert owner.wrap("outer",lambda:value)() is value
    assert owner.details["outer"]=={"internal_seconds":999.}
    assert "internal_seconds" not in owner.parts
