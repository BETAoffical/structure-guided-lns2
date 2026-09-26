from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.sa_raw_selection_fast import FastPolicy
from experiments.sa_shared_features import ActorFeatureView, SharedFeatureEngine, SharedFeaturePool, SharedFeaturePolicy
from experiments.sa_single_check_runtime import SingleFullCheckPool
from tests.native_support import isolated_sa_native


class SharedFeaturesTests(unittest.TestCase):
    def fixture(self):
        state = {'agents': [{'id': 4}, {'id': 9}]}
        candidates = [{'candidate_id': 'b', 'agents': [9, 4], 'sources': ['target']},
                      {'candidate_id': 'a', 'agents': [4], 'sources': ['collision']}]
        engine = object.__new__(SharedFeatureEngine)
        engine.state, engine.score_names, engine.snapshot = state, ('y',), None
        rows = [dict(candidate_id=c['candidate_id'], candidate_key=c['candidate_id'], state_id='fp',
                     feature_names=('x', 'y'), feature_values=(i + .1, i + .2), feature_profile='realized_dynamic')
                for i, c in enumerate(candidates)]
        with patch.object(OnlineFeatureEngine, 'realized_rows', return_value=(rows, {})):
            projected, _ = engine.realized_rows(candidates, state_hash='fp')
        return engine, state, candidates, projected

    def test_projection_and_actor_reordering_without_recompute(self):
        engine, state, candidates, projected = self.fixture()
        self.assertEqual(projected[0]['feature_names'], ('y',))
        self.assertEqual(projected[0]['feature_values'], (.2,))
        scored = [c | {'score': .5} for c in reversed(candidates)]
        with patch.object(OnlineFeatureEngine, 'realized_rows', side_effect=AssertionError('second extraction')):
            rows, _ = engine.actor_rows(state, scored, 'fp')
        self.assertEqual([r['candidate_id'] for r in rows], ['a', 'b'])
        self.assertEqual(rows[0]['features']['realized_dynamic'], {'x': 1.1, 'y': 1.2})
        rows[0]['features']['realized_dynamic']['x'] = 99.
        self.assertEqual(engine.actor_rows(state, scored, 'fp')[0][0]['features']['realized_dynamic']['x'], 1.1)

    def test_reject_stale_state_hash_missing_duplicate_and_changed_candidates(self):
        engine, state, candidates, _ = self.fixture()
        for s, c, h in [(deepcopy(state), candidates, 'fp'), (state, candidates, 'other'),
                        (state, candidates[:1], 'fp'), (state, candidates * 2, 'fp')]:
            with self.assertRaises(ValueError):
                engine.actor_rows(s, c, h)
        for field, value in [('agents', [4]), ('sources', ['random'])]:
            changed = deepcopy(candidates)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                engine.actor_rows(state, changed, 'fp')
        candidates[0]['agents'].reverse()
        with self.assertRaises(ValueError):
            engine.actor_rows(state, candidates, 'fp')

    def test_failed_extraction_invalidates_previous_snapshot(self):
        engine, state, candidates, _ = self.fixture()
        with patch.object(OnlineFeatureEngine, 'realized_rows', side_effect=ValueError('bad native')):
            with self.assertRaises(ValueError):
                engine.realized_rows(candidates, state_hash='fp')
        with self.assertRaises(ValueError):
            engine.actor_rows(state, candidates, 'fp')

    def test_view_requires_ready_selector_and_current_state(self):
        engine, state, candidates, _ = self.fixture()
        selector = SimpleNamespace(engine=None)
        view = ActorFeatureView(selector)
        view.prepare(state)
        with self.assertRaises(ValueError):
            view.realized_rows(candidates, state_hash='fp')
        selector.engine = engine
        self.assertEqual(len(view.realized_rows(candidates, state_hash='fp')[0]), 2)
        view.prepare(deepcopy(state))
        with self.assertRaises(ValueError):
            view.realized_rows(candidates, state_hash='fp')

    def test_original_functions_unchanged_and_no_baseline_switch(self):
        self.assertIs(SharedFeaturePool.select.__code__, SingleFullCheckPool.select.__code__)
        self.assertIs(SingleFullCheckPool.select.__globals__['OnlineFeatureEngine'], OnlineFeatureEngine)
        self.assertIs(SharedFeaturePool.select.__globals__['OnlineFeatureEngine'], SharedFeatureEngine)
        self.assertIs(SharedFeaturePolicy.choose, FastPolicy.choose)
        sentinel = object()
        def init(policy, *args):
            policy.actor, policy.selector = None, sentinel
        with patch.object(FastPolicy, '__init__', init):
            policy = SharedFeaturePolicy({}, None, {})
            policy.start({})
        self.assertIs(policy.selector, sentinel)

    @isolated_sa_native
    def test_native_full_rows_equal_dictionary_features_and_original_projection(self):
        from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
        from experiments.online_feature_engine import TopologyAnalysisCache
        # Non-contiguous IDs and a swap conflict with persistent goal occupancy.
        state = dict(rows=3, cols=4, obstacles=[0] * 12, num_of_colliding_pairs=1,
                     agents=[dict(id=4, path=[0, 1, 2, 3], start=0, goal=3, delay=0,
                                  path_cost=3, shortest_path_cost=3, conflict_degree=1),
                             dict(id=9, path=[3, 2, 1, 0], start=3, goal=0, delay=0,
                                  path_cost=3, shortest_path_cost=3, conflict_degree=1)],
                     conflict_edges=[[4, 9]], iteration=0, sum_of_costs=6,
                     low_level=dict(generated=20, expanded=10, runs=2))
        # This test uses feature APIs only, no solver/environment construction.
        candidates = [dict(candidate_id='b', agents=[9, 4], sources=[]),
                      dict(candidate_id='a', agents=[4], sources=[])]
        names = PROFILE_FEATURE_NAMES['realized_dynamic'][::2]
        shared = SharedFeatureEngine(state, backend='native', dense_output=True,
                                     required_features={'realized_dynamic': names})
        reference = OnlineFeatureEngine(state, backend='native', dense_output=True,
                                        required_features={'realized_dynamic': names})
        actor = OnlineFeatureEngine(state, backend='native')
        topology = TopologyAnalysisCache(state, static_grid=shared.static_grid, backend='native')
        for engine in (shared, reference):
            engine.prepare(state, prepared_native_analysis=topology.last_native_prepared)
        actual, _ = shared.realized_rows(candidates, state_hash='fp')
        expected, _ = reference.realized_rows(candidates, state_hash='fp')
        self.assertEqual(actual, expected)
        ar, _ = shared.actor_rows(state, candidates, 'fp')
        er, _ = actor.realized_rows(candidates, state_hash='fp')
        self.assertEqual([x['features'] for x in ar], [x['features'] for x in er])


if __name__ == '__main__':
    unittest.main()
