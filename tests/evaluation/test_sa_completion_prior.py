from collections import Counter
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from experiments import sa_completion_prior as rt
from scripts import run_sa_completion_prior as cli
from tests.evaluation.test_sa_raw_timing_metrics import fixture, fail


def jobs():
    return [dict(job_id=f'{m}-{d}-{a}',pair_id=f'm{m:02d}-d{d}-s281',replica=0,
                 comparison_arm=a,prior_mode=a,arm='trained_actor',model={'sha256':rt.MODEL_SHA},
                 solver_seed=281,expected_initial=f'{m}-{d}',phase='source',plan={'config':{'stream_seed':8}},
                 case={'map_id':f'm{m:02d}','task_variant':f'bottleneck_d{d}'},source_folder='source',
                 source_files={'trace':'fixed'},budget_seconds=120.,timing_mode=rt.TIMING_MODE,
                 max_decisions=None,execution_node_budget=None)
            for m in range(12) for d in (20,25) for a in rt.ARMS]


def rows():
    names = dict(raw_parent='uniform_prior',raw_updated='original',dual16_sa='random_prior')
    # Fixture arm names are shared with the historical raw confirmation schema.
    available = list(dict.fromkeys(r['comparison_arm'] for r in fixture()))
    names = dict(zip(available[:3],rt.ARMS))
    return [dict(r,comparison_arm=names[r['comparison_arm']],policy_sha256='same',timing_mode=rt.TIMING_MODE,
                 selection_seconds=1.,native_pp_seconds=2.,bookkeeping_seconds=.1,reset_seconds=.2)
            for r in fixture() if r['comparison_arm'] in names]


class PriorTests(unittest.TestCase):
    def test_prior_mass_and_identity(self):
        ids = [f'{i:02d}' for i in range(16)]
        p = rt.prior(ids,'03')
        self.assertEqual(p['03'],.90625)
        self.assertEqual(p['02'],.00625)
        self.assertEqual(rt.replace_prior(p,'03','03'),p)
        u = rt.replace_prior(p,'03',None)
        self.assertTrue(all(abs(v-1/16) < 1e-14 for v in u.values()))
        q = rt.replace_prior(p,'03','07')
        self.assertTrue(all(abs(q[c]-rt.prior(ids,'07')[c]) < 1e-14 for c in ids))

    def test_residual_preferences_survive_prior_removal(self):
        ids = ['a','b','c']
        q = rt.prior(ids,'a')
        residual = dict(a=-.4,b=1.2,c=.1)
        weights = {c:q[c]*math.exp(residual[c]) for c in ids}
        p = {c:w/sum(weights.values()) for c,w in weights.items()}
        u = rt.replace_prior(p,'a',None)
        expected = {c:math.exp(residual[c])/sum(math.exp(v) for v in residual.values()) for c in ids}
        self.assertTrue(all(abs(u[c]-expected[c]) < 1e-14 for c in ids))
        self.assertGreater(u['b'],u['a'])
        self.assertNotAlmostEqual(u['b'],1/3)

    def test_direct_formula_includes_both_residuals(self):
        base = SimpleNamespace(mean=np.zeros(1),scale=np.ones(1),w1=np.array([[.2]]),b1=np.array([.1]),
                               w2=np.array([[.7]]),b2=np.array([.3]))
        actor = SimpleNamespace(base=base,bundle={'base':{'feature_names':['source.score']}},
                                w1=np.array([[-.5]]),b1=np.array([.2]),w2=np.array([[1.3]]),b2=np.array([-.1]))
        ids, fs = ['a','b'],[{'source.score':.1},{'source.score':2.}]
        p = rt.direct_probabilities(actor,ids,fs,'a')
        for anchor in (None,'a','b'):
            computed = rt.replace_prior(p,'a',anchor)
            direct = rt.direct_probabilities(actor,ids,fs,anchor)
            self.assertTrue(all(abs(computed[c]-direct[c]) < 1e-14 for c in ids))

    def test_invalid_inputs_fail_closed(self):
        for ids, anchor in (([],None),(['b','a'],'a'),(['a','a'],'a'),(['a'],'b')):
            with self.assertRaises(ValueError):rt.prior(ids,anchor)
        for draw in (-.1,1.,float('nan')):
            with self.assertRaises(ValueError):rt.random_anchor(['a'],draw)
        with self.assertRaises(ValueError):rt.replace_prior({'a':0.,'b':1.},'a',None)
        self.assertEqual(rt.random_anchor(['a','b'],0.),'a')
        self.assertEqual(rt.random_anchor(['a','b'],.5),'b')

    def test_wrapper_preserves_inputs_and_uses_independent_stream(self):
        fs = [{'source.score':4.},{'source.score':1.}]
        before = deepcopy(fs)
        original = {'a':.95,'b':.05}
        actor = SimpleNamespace(sha='frozen',bundle={'base':{'feature_names':['source.score']}},
            probabilities=lambda ids,anchor,features:dict(original),out_of_range_fraction=lambda f:0.)
        j = jobs()[0]
        with patch.object(rt.run,'stream_draw',return_value=.75) as draw:
            for mode in rt.ARMS:
                wrapped = rt.PriorActor(actor,j|{'prior_mode':mode})
                actual = wrapped.probabilities(['a','b'],'a',fs)
                replacement = 'a' if mode=='original' else (None if mode=='uniform_prior' else 'b')
                self.assertEqual(actual,rt.replace_prior(original,'a',replacement))
                self.assertEqual(wrapped.decision,1)
                self.assertEqual(wrapped.sha,'frozen')
                self.assertEqual(draw.call_args.args[-1],rt.PURPOSE)
        self.assertEqual(fs,before)

    def test_fixed_scope_and_balanced_schedule(self):
        self.assertEqual(cli.config()['audit_workers'],4)
        original = jobs()
        snapshot = deepcopy(original)
        ordered = cli.schedule(original)
        self.assertEqual(original,snapshot)
        self.assertEqual(ordered,cli.schedule(reversed(original)))
        self.assertEqual(ordered,cli.schedule(ordered))
        self.assertEqual(len(ordered),72)
        for density in (20,25):
            for position in range(3):
                chosen = [j['comparison_arm'] for j in ordered[position::3]
                          if j['case']['task_variant']==f'bottleneck_d{density}']
                self.assertEqual(Counter(chosen),dict.fromkeys(rt.ARMS,4))

    def test_identity_and_schedule_reject_changes(self):
        for field,value in (('prior_mode','wrong'),('arm','official_sa'),('model',{'sha256':'bad'}),
                            ('max_decisions',100)):
            with self.assertRaises(ValueError):rt.identity(jobs()[0]|{field:value})
        for bad in (jobs()[:-1],jobs()+jobs()[:1]):
            with self.assertRaises(ValueError):cli.schedule(bad)
        data = jobs()
        data[0]['expected_initial']='changed'
        with self.assertRaises(ValueError):cli.schedule(data)

    def test_summary_denominators_and_empty_cohorts(self):
        data = rows()
        fail(next(r for r in data if r['comparison_arm']=='uniform_prior'))
        result = rt.summarize(data,50)
        self.assertEqual(result,rt.summarize(reversed(data),50))
        c = result['contrasts']['original_vs_uniform_prior']
        self.assertEqual(c['success']['wins'],1)
        self.assertEqual(c['common_success']['ttf_seconds']['count'],1)
        for r in data:fail(r)
        result = rt.summarize(data,50)
        self.assertIsNone(result['contrasts']['original_vs_uniform_prior']['common_success']['ttf_seconds']['completion']['mean'])

    def test_summary_rejects_unpaired_stream_or_model(self):
        for field in ('policy_sha256','rng_stream_id','initial_fingerprint','timing_mode'):
            data=rows()
            data[0][field]='wrong'
            with self.assertRaises(ValueError):rt.summarize(data,20)

    def test_frozen_loop_bytecode_and_private_bindings(self):
        self.assertIs(rt._timed.__code__,rt.lean.lean_worker.__code__)
        self.assertIs(rt._timed.__globals__['Policy'],rt.PriorPolicy)
        self.assertIs(rt._audit.__globals__['Policy'],rt.AuditPolicy)
        self.assertIs(cli._collect.__code__,cli.first.collect.__code__)
        self.assertIs(cli._collect.__globals__['rt'],rt)


if __name__ == '__main__':
    unittest.main()
