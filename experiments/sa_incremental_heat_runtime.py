"""Isolated exact heat update; frozen selector operations and native stay intact."""

from types import FunctionType

from experiments.online_feature_engine import TopologyAnalysisCache
from experiments.sa_single_check_runtime import SingleFullCheckPool


class ExactChangedTopology(TopologyAnalysisCache):
    def prepare(self, state, *, changed_agents=None):
        if changed_agents is not None:
            raise ValueError("derive changed paths from cache; external hints are not accepted")
        paths = {int(a["id"]): a["path"] for a in state["agents"]}
        if len(paths) != len(state["agents"]):
            raise ValueError("duplicate agent ID")
        if self.paths is None:
            self.heat_changed_agents = None
            return super().prepare(state)
        if paths.keys() != self.paths.keys():
            raise ValueError("topology cache agent identity changed")
        changed = [agent for agent, path in paths.items() if path != self.paths[agent]]
        self.heat_changed_agents = tuple(changed)
        return super().prepare(state, changed_agents=changed)


class IncrementalHeatPool(SingleFullCheckPool):
    # Bind only the cache class in a private namespace. Original code stays frozen.
    select = FunctionType(
        SingleFullCheckPool.select.__code__,
        dict(SingleFullCheckPool.select.__globals__, TopologyAnalysisCache=ExactChangedTopology),
        SingleFullCheckPool.select.__name__,
        SingleFullCheckPool.select.__defaults__,
        SingleFullCheckPool.select.__closure__,
    )
