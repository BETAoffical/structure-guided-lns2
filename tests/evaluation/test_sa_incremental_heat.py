from copy import deepcopy

import pytest

from experiments.online_feature_engine import TopologyAnalysisCache
from experiments.sa_incremental_heat_runtime import ExactChangedTopology, IncrementalHeatPool
from experiments.sa_single_check_runtime import SingleFullCheckPool
from tests.runtime.test_controller_v2 import _refresh_conflicts


def state():
    value=dict(rows=4,cols=4,obstacles=[0]*16,
               agents=[dict(id=7,path=[0,1,2,3]),dict(id=42,path=[3,2,1,0])])
    _refresh_conflicts(value)
    return value


@pytest.mark.parametrize("backend",["python","native"])
def test_changes_rollback_waits_and_skipped_states(backend):
    first=state()
    incremental=ExactChangedTopology(first,backend=backend,shadow_interval=2)
    reference=TopologyAnalysisCache(first,backend=backend)
    changed=deepcopy(first)
    changed["agents"][0]["path"]=[0,4,5,6,2,3]
    waited=deepcopy(changed)
    waited["agents"][1]["path"]=[3,3,2,1,0]
    for i,current in enumerate([changed,changed,waited,first],1):
        current=deepcopy(current)
        _refresh_conflicts(current)
        expected={a["id"] for a in current["agents"] if incremental.paths[a["id"]]!=a["path"]}
        incremental.prepare(current)
        reference.prepare(current)
        assert set(incremental.heat_changed_agents)==expected
        assert incremental.analysis==reference.analysis
        assert incremental.visit_heat==reference.visit_heat
        assert incremental.agent_heat==reference.agent_heat
        assert incremental.last_shadow_validation==(i%2==0)
    # Caller mutation cannot silently alter the cache's saved old paths.
    current["agents"][0]["path"].append(3)
    assert incremental.paths[7]!=current["agents"][0]["path"]


def test_cache_identity_and_hint_fail_closed():
    first=state()
    cache=ExactChangedTopology(first,backend="python")
    invalid=deepcopy(first)
    invalid["agents"][1]["id"]=7
    with pytest.raises(ValueError,match="duplicate"):
        cache.prepare(invalid)
    invalid["agents"][1]["id"]=43
    with pytest.raises(ValueError,match="identity"):
        cache.prepare(invalid)
    with pytest.raises(ValueError,match="external hints"):
        cache.prepare(first,changed_agents=[])
    invalid=deepcopy(first)
    invalid["obstacles"][8]=1
    with pytest.raises(ValueError,match="grid changed"):
        cache.prepare(invalid)


@pytest.mark.parametrize("backend",["python","native"])
def test_periodic_shadow_detects_corrupt_heat(backend):
    cache=ExactChangedTopology(state(),backend=backend,shadow_interval=1)
    cache.visit_heat[15]+=9
    with pytest.raises(ValueError,match="shadow"):
        cache.prepare(state())


def test_only_private_topology_binding_changes():
    a,b=SingleFullCheckPool.select,IncrementalHeatPool.select
    assert a.__code__ is b.__code__
    assert a.__globals__["TopologyAnalysisCache"] is TopologyAnalysisCache
    assert b.__globals__["TopologyAnalysisCache"] is ExactChangedTopology
    assert {k:v for k,v in a.__globals__.items() if k!="TopologyAnalysisCache"}=={
        k:v for k,v in b.__globals__.items() if k!="TopologyAnalysisCache"}
