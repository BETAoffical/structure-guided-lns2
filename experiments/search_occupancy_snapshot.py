"""Snapshot-only correction for the frozen v1 diagnostic recorder."""
from experiments.search_occupancy_observation import ObservingProbe


class SnapshotObservingProbe(ObservingProbe):
    def plan(self, agent, fixed, overrides, hard, *args, **kwargs):
        snapshot = list(fixed)
        result = super().plan(agent, fixed, overrides, hard, *args, **kwargs)
        if self.enabled:
            self.captures[-1]['fixed'] = snapshot
        return result
