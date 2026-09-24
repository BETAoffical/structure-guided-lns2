import copy
import math
import unittest
import numpy as np

from experiments.sa_onpolicy_actor import NumpyActor, SCHEMA
from experiments.sa_onpolicy_contract import anchor_distribution
from scripts import audit_sa_policy_expressivity as audit


class ExpressivityTests(unittest.TestCase):
    def distribution(self, count, all_challengers=False):
        ids = [str(i) for i in range(count)]
        residual = {c: -2. for c in ids}
        for c in ids[1:] if all_challengers else ids[1:2]:
            residual[c] = 2.
        return anchor_distribution(ids, "0", residual)

    def test_global_modal_boundary(self):
        for count in range(2, 33):
            bounds = audit.family_bounds(count)
            p = self.distribution(count)
            self.assertEqual(bounds['challenger_can_outrank'], count <= 5)
            self.assertEqual(p['1'] > p['0'], count <= 5)
            self.assertAlmostEqual(math.log(p['0']/p['1']), bounds['minimum_anchor_log_odds'])

    def test_single_challenger_supremum(self):
        for count in (2, 5, 6, 18, 19, 100):
            self.assertAlmostEqual(self.distribution(count)['1'], audit.family_bounds(count)['challenger_probability_max'])

    def test_total_exploration_is_not_single_action_concentration(self):
        p = self.distribution(19, all_challengers=True)
        self.assertLess(p['0'], .5)
        self.assertTrue(all(p['0'] > v for k,v in p.items() if k != '0'))
        self.assertAlmostEqual(p['0'], audit.family_bounds(19)['anchor_probability_min'])

    def test_singleton_and_invalid_count(self):
        self.assertIsNone(audit.family_bounds(1)['challenger_probability_max'])
        self.assertFalse(audit.family_bounds(1)['nontrivial_restriction'])
        for value in (True, 0, -1, 2.5):
            with self.assertRaises(ValueError): audit.family_bounds(value)

    def actor_event(self):
        bundle = dict(schema=SCHEMA, feature_names=['x'], mean=[0.], scale=[1.],
            w1=np.ones((32,1)).tolist(), b1=[0.]*32, w2=[[0.]*32], b2=[0.], epsilon=.1, residual_bound=2.)
        actor = NumpyActor(bundle)
        ids = [str(i) for i in range(7)]
        fs = [dict(x=float(i)) for i in range(7)]
        p = actor.probabilities(ids, '0', fs)
        draw = .99
        event = dict(candidate_ids=ids, anchor_id='0', features=fs, probabilities=p,
            policy_sha256=actor.sha, selected_id=audit.select_with_draw(p, draw), selection_draw=draw, decision=0)
        return actor, event

    def test_zero_actor_reproduces_prior_but_can_sample_nonanchor(self):
        actor, event = self.actor_event()
        row = audit.measure_event(actor, event)
        self.assertTrue(row['anchor_strict_mode'] and row['locked'] and row['selected_nonanchor'])
        self.assertEqual(row['residual_span'], 0.)
        self.assertEqual(row['prior_total_variation'], 0.)
        self.assertFalse(row['same_draw_changed'])

    def test_frozen_probabilities_and_action_cannot_be_changed(self):
        actor, event = self.actor_event()
        for field, value in [('selected_id','0'), ('policy_sha256','bad'), ('candidate_ids',list(reversed(event['candidate_ids'])))]:
            bad = dict(event, **{field:value})
            with self.assertRaises(ValueError): audit.measure_event(actor, bad)
        bad = copy.deepcopy(event)
        bad['probabilities']['0'] += .001
        with self.assertRaises(ValueError): audit.measure_event(actor, bad)

    def test_no_feature_or_model_mutation(self):
        actor, event = self.actor_event()
        before = copy.deepcopy((actor.bundle, event))
        audit.measure_event(actor, event)
        self.assertEqual(before, (actor.bundle, event))

    def test_replayed_nonzero_actor_log_odds_identity(self):
        actor, event = self.actor_event()
        bundle = dict(actor.bundle, w2=[[.01]*32])
        actor = NumpyActor(bundle)
        event['policy_sha256'] = actor.sha
        event['probabilities'] = actor.probabilities(event['candidate_ids'], '0', event['features'])
        event['selected_id'] = audit.select_with_draw(event['probabilities'], event['selection_draw'])
        row = audit.measure_event(actor, event)
        self.assertGreater(row['residual_span'], 0)
        self.assertGreater(row['prior_total_variation'], 0)

    def test_validation_or_censored_data_rejected(self):
        job = dict(split='train', arm=audit.source.ARMS[0], result=dict(status='ok', stop='node_budget'))
        audit.checked_job(job)
        for field,value in [('split','validation'), ('arm','official')]:
            with self.assertRaises(ValueError): audit.checked_job(dict(job, **{field:value}))
        with self.assertRaises(ValueError):
            audit.checked_job(dict(job, result=dict(status='censored',stop='wall_safety')))

    def test_summary_keeps_episode_and_step_counts_separate(self):
        actor, event = self.actor_event()
        row = dict(map_id='m', success=True, steps=[audit.measure_event(actor, event)]*3)
        result = audit.summarize([row])
        self.assertEqual((result['episodes'],result['maps'],result['steps']), (1,1,3))
        self.assertEqual(result['counts']['anchor_strict_mode'], 3)


if __name__ == '__main__':
    unittest.main()
