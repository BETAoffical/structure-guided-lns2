"""Explain the retained v1 shadow overhead without changing frozen runtime."""

from experiments import online_feature_engine as engine


def test_native_shadow_constructs_but_does_not_read_python_index(monkeypatch):
    constructed=[]
    class Index:
        def __init__(self,paths): constructed.append(paths)
        def all_events(self): raise AssertionError("native shadow must not use this index")
    monkeypatch.setattr(engine,"TemporalConflictIndex",Index)
    cache=engine.TopologyAnalysisCache.__new__(engine.TopologyAnalysisCache)
    cache.native_function=object()
    events=["native events"]
    cache._native_events=lambda state:events
    cache._analysis_from_events=lambda state,values:values
    paths={7:[0,1]}
    assert cache._shadow_analysis({},paths) is events
    assert constructed==[paths]


def test_python_shadow_still_requires_index(monkeypatch):
    calls=[]
    events=["python events"]
    class Index:
        def __init__(self,paths): pass
        def all_events(self): calls.append(True); return events
    monkeypatch.setattr(engine,"TemporalConflictIndex",Index)
    cache=engine.TopologyAnalysisCache.__new__(engine.TopologyAnalysisCache)
    cache.native_function=None
    cache._analysis_from_events=lambda state,values:values
    assert cache._shadow_analysis({},{7:[0,1]}) is events
    assert calls==[True]
