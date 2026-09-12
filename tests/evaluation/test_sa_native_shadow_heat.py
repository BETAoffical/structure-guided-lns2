from copy import deepcopy
from unittest.mock import patch

import pytest

from experiments import online_feature_engine as engine
from experiments.sa_incremental_heat_runtime import ExactChangedTopology, IncrementalHeatPool
from experiments.sa_native_shadow_heat_runtime import NativeShadowTopology, NativeShadowHeatPool
from tests.evaluation.test_sa_incremental_heat import state
from tests.runtime.test_controller_v2 import _refresh_conflicts
from scripts import audit_sa_incremental_heat as base
from scripts import audit_sa_native_shadow_heat as runner


@pytest.mark.parametrize("backend",["native","python"])
def test_shadow_equal_and_python_index_only_when_needed(backend):
    s=state()
    old=ExactChangedTopology(s,backend=backend,shadow_interval=1)
    new=NativeShadowTopology(s,backend=backend,shadow_interval=1)
    expected=old._shadow_analysis(s,old.paths)
    with patch.object(engine,"TemporalConflictIndex",wraps=engine.TemporalConflictIndex) as index:
        assert new._shadow_analysis(s,new.paths)==expected
        assert index.call_count==(0 if backend=="native" else 1)
    for updated in [s,dict(s,agents=[dict(id=7,path=[0,4,5,6,2,3]),s["agents"][1]]),s]:
        updated=deepcopy(updated)
        _refresh_conflicts(updated)
        new.prepare(updated)
        old.prepare(updated)
        assert new.analysis==old.analysis
        assert new.visit_heat==old.visit_heat
        assert new.agent_heat==old.agent_heat
        assert new.last_shadow_validation


@pytest.mark.parametrize("backend",["native","python"])
def test_shadow_still_rejects_corruption_and_empty_paths(backend):
    cache=NativeShadowTopology(state(),backend=backend,shadow_interval=1)
    for paths in ({},{7:[]}):
        with pytest.raises(ValueError,match="non-empty"):
            cache._shadow_analysis(state(),paths)
    cache.visit_heat[15]+=8
    with pytest.raises(ValueError,match="shadow"):
        cache.prepare(state())


def test_private_runtime_binding_and_runner_reuse():
    old,new=IncrementalHeatPool.select,NativeShadowHeatPool.select
    assert old.__code__ is new.__code__
    assert old.__globals__["TopologyAnalysisCache"] is ExactChangedTopology
    assert new.__globals__["TopologyAnalysisCache"] is NativeShadowTopology
    assert {k:v for k,v in old.__globals__.items() if k!="TopologyAnalysisCache"}=={
        k:v for k,v in new.__globals__.items() if k!="TopologyAnalysisCache"}
    worker=runner.bound(base.worker,IncrementalHeatPool=NativeShadowHeatPool)
    assert worker.__code__ is base.worker.__code__
    assert base.worker.__globals__["IncrementalHeatPool"] is IncrementalHeatPool
    assert worker.__globals__["IncrementalHeatPool"] is NativeShadowHeatPool
    assert runner.OUT != base.OUT
