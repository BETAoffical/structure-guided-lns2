"""Opt-in, same-state feature sharing; no policy, native or timer changes."""
from copy import deepcopy

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.sa_history_selector import History
from experiments.sa_raw_selection_fast import FastPolicy, _bind
from experiments.sa_single_check_runtime import SingleFullCheckPool
from lns2_selector.runtime.contracts import require


class SharedFeatureEngine(OnlineFeatureEngine):
    """Compute the full actor schema once, return the original GBDT projection."""

    def __init__(self, state, **kwargs):
        require(kwargs.get('dense_output') is True and kwargs.get('backend') == 'native',
                'shared features require native dense output')
        schema = tuple(PROFILE_FEATURE_NAMES['realized_dynamic'])
        requested = kwargs.get('required_features', {}).get('realized_dynamic', schema)
        require(len(requested) == len(set(requested)) and set(requested) <= set(schema),
                'invalid scoring projection')
        self.score_names = tuple(n for n in schema if n in requested)
        self.snapshot = None
        super().__init__(state, **(kwargs | {'required_features': {'realized_dynamic': schema}}))

    def realized_rows(self, candidates, *, state_hash):
        self.snapshot = None
        rows, metrics = super().realized_rows(candidates, state_hash=state_hash)
        ids = [c['candidate_id'] for c in candidates]
        require(ids and len(ids) == len(set(ids)), 'duplicate or empty shared candidates')
        require([r['candidate_id'] for r in rows] == ids, 'shared row alignment')
        full = {}
        projected = []
        for candidate, row in zip(candidates, rows, strict=True):
            values = dict(zip(row['feature_names'], row['feature_values'], strict=True))
            full[candidate['candidate_id']] = (deepcopy(candidate), values)
            projected.append(row | {'feature_names': self.score_names,
                                    'feature_values': tuple(values[n] for n in self.score_names)})
        self.snapshot = (self.state, state_hash, full)
        return projected, metrics

    def actor_rows(self, state, candidates, state_hash):
        require(self.snapshot is not None, 'missing shared feature snapshot')
        original_state, original_hash, full = self.snapshot
        require(state is original_state and state_hash == original_hash,
                'shared features belong to another state')
        ids = [c['candidate_id'] for c in candidates]
        require(len(ids) == len(set(ids)) and set(ids) == set(full), 'shared candidate coverage')
        rows = []
        for candidate in candidates:
            saved, values = full[candidate['candidate_id']]
            require({k: v for k, v in candidate.items() if k != 'score'} == saved,
                    'shared candidate members or provenance changed')
            rows.append({'candidate_id': candidate['candidate_id'],
                         'features': {'realized_dynamic': dict(values)}})
        return rows, {'shared_snapshot': True}


class SharedFeaturePool(SingleFullCheckPool):
    select = _bind(SingleFullCheckPool.select, OnlineFeatureEngine=SharedFeatureEngine)


class ActorFeatureView:
    """Read only this decision's selector snapshot; never cache across decisions."""

    def __init__(self, selector):
        self.selector = selector
        self.state = None

    def prepare(self, state):
        self.state = state

    def realized_rows(self, candidates, *, state_hash):
        engine = self.selector.engine
        require(isinstance(engine, SharedFeatureEngine), 'shared selector engine not ready')
        return engine.actor_rows(self.state, candidates, state_hash)


class SharedFeaturePolicy(FastPolicy):
    def __init__(self, job, q, ctx):
        super().__init__(job, q, ctx)
        if self.actor:
            self.selector = SharedFeaturePool(ctx)

    def start(self, state):
        if self.actor:
            self.history = History(state)
            self.engine = ActorFeatureView(self.selector)
            self.ctx = dict(self.ctx, initial_nodes=state['low_level']['generated'])
