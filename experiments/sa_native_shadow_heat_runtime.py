"""Keep full shadow validation while skipping its unused native-side Python index."""

from types import FunctionType

from experiments.sa_incremental_heat_runtime import ExactChangedTopology, IncrementalHeatPool


class NativeShadowTopology(ExactChangedTopology):
    def _shadow_analysis(self, state, paths):
        if self.native_function is None:
            return super()._shadow_analysis(state, paths)
        # Preserve the old index constructor's valid-path precondition.
        if not paths or any(not path for path in paths.values()):
            raise ValueError("temporal conflict index requires non-empty paths")
        # This remains a fresh full extraction, not the incremental prepared cache.
        return self._analysis_from_events(state, self._native_events(state))


class NativeShadowHeatPool(IncrementalHeatPool):
    select = FunctionType(
        IncrementalHeatPool.select.__code__,
        dict(IncrementalHeatPool.select.__globals__, TopologyAnalysisCache=NativeShadowTopology),
        IncrementalHeatPool.select.__name__,
        IncrementalHeatPool.select.__defaults__,
        IncrementalHeatPool.select.__closure__,
    )
