"""Positive queued transitions and a fixed post-search competitiveness envelope."""
from experiments.search_occupancy_observation import summarize_contacts


def interval_event(row):
    if len(row) != 12 or row[0] not in (1, 2) or row[11]:
        raise ValueError('invalid frontier row')
    _, src, parent_tick, dst, tick, _, _, collisions, parent_collisions, vertex, _, _ = row
    delta = collisions - parent_collisions
    edge = delta - vertex
    if tick <= parent_tick or vertex not in (0, 1) or edge not in (0, 1) or delta <= 0:
        raise ValueError('unexpected transition collision increment')
    return [0, src, dst, tick, tick+1, tick, tick+1, tick+1, vertex, edge]


def within_envelope(row, capture):
    return row[7] <= capture['goal_conflicts'] + 1 and row[5] + row[6] <= capture['goal_cost']


def profiles(capture):
    rows = capture['events']
    return {
        'queued_all': [interval_event(r) for r in rows if r[0] == 1],
        'queued_envelope': [interval_event(r) for r in rows if r[0] == 1 and within_envelope(r, capture)],
        'popped_envelope': [interval_event(r) for r in rows if r[0] == 2 and within_envelope(r, capture)],
    }


class FrontierProbe:
    def __init__(self, module, probe, paths, external, enabled, limit):
        self.module, self.probe, self.paths = module, probe, paths
        self.external, self.enabled, self.limit = set(external), enabled, limit
        self.captures = []
        self.summaries = {name: [] for name in ('queued_all', 'queued_envelope', 'popped_envelope')}

    def seed_rng(self, seed):
        self.probe.seed_rng(seed)

    def plan(self, agent, fixed, overrides, hard, *args, **kwargs):
        if not self.enabled:
            return self.probe.plan(agent, fixed, overrides, hard, *args, **kwargs)
        snapshot = list(fixed)
        self.module.begin_observation(self.limit)
        try:
            result = self.probe.plan(agent, fixed, overrides, hard, *args, **kwargs)
        finally:
            capture = self.module.end_observation()
        self.captures.append(dict(agent=agent, fixed=snapshot, capture=capture))
        if result['status'] == 'path':
            if capture['goal_conflicts'] != result['low_level_collisions'] or capture['goal_cost'] != result['cost']:
                raise ValueError('goal observation mismatch')
            if capture['popped'] != result['expanded']:
                raise ValueError('expansion observation mismatch')
            visible = {i: overrides.get(i, self.paths[i]) for i in snapshot}
            for name, events in profiles(capture).items():
                self.summaries[name].append(summarize_contacts(agent, visible, result['path'], events, self.external))
        return result
