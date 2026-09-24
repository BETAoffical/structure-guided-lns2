from copy import deepcopy
import importlib.util
import math
import unittest
import numpy as np

from experiments import sa_raw_residual_actor as raw
from experiments.sa_onpolicy_actor import NumpyActor, SCHEMA, select_with_draw, validate_bundle


def fixture(count=20):
    rng = np.random.default_rng(71)
    names = [f"f{i:03d}" for i in range(129)]
    base = dict(schema=SCHEMA, feature_names=names, mean=[0.]*129, scale=[1.]*129,
        w1=rng.normal(0,.1,(32,129)).tolist(), b1=rng.normal(0,.1,32).tolist(),
        w2=rng.normal(0,.05,(1,32)).tolist(), b2=[.02], epsilon=.1, residual_bound=2.)
    bundle = raw.initial_bundle(base, "contract-test")
    features = [dict(zip(names, row.tolist())) for row in np.eye(129)[:count]]
    return bundle, [f"c{i:02d}" for i in range(count)], features


def child(reference, target=1, raw_logit=8.):
    proposed = deepcopy(reference)
    proposed.update(iteration=reference['iteration']+1, parent_policy=raw.validate_bundle(reference))
    w1, w2 = np.zeros((32,129)), np.zeros((1,32))
    w1[0,target] = 1.
    w2[0,0] = raw_logit/math.tanh(1.)
    proposed['correction'] = dict(w1=w1.tolist(), b1=[0.]*32, w2=w2.tolist(), b2=[0.])
    return proposed


def episode(bundle, ids, features, repeats=1):
    p = raw.RawResidualActor(bundle).probabilities(ids, ids[0], features)
    state = dict(ids=ids, anchor=ids[0], features=features, probabilities=p, draw=.97,
                 selected=select_with_draw(p,.97))
    return dict(episode_id='toy', split='train', policy_sha256=raw.validate_bundle(bundle), weight=1.,
                states=[deepcopy(state) for _ in range(repeats)])


class RawContractTests(unittest.TestCase):
    def test_initialization_is_exact_parent_not_doubled_residual_or_fixed_prior(self):
        bundle, ids, fs = fixture()
        base = NumpyActor(bundle['base']).probabilities(ids, ids[0], fs)
        actual = raw.RawResidualActor(bundle).probabilities(ids, ids[0], fs)
        self.assertEqual(actual, base)
        self.assertGreater(max(abs(base[c]-(.1/len(ids)+(.9 if c==ids[0] else 0))) for c in ids), 1e-5)
        draws = [0., .5, math.nextafter(1.,0.)]
        cumulative = 0.
        for c in ids[:-1]:
            cumulative += base[c]
            draws.extend([math.nextafter(cumulative,0.), cumulative, math.nextafter(cumulative,1.)])
        for draw in draws:
            self.assertEqual(select_with_draw(base,draw), select_with_draw(actual,draw))

    def test_each_distinguishable_candidate_can_become_mode(self):
        bundle, ids, fs = fixture()
        for i,c in enumerate(ids):
            p = raw.RawResidualActor(child(bundle,i,12.)).probabilities(ids, ids[0], fs)
            self.assertEqual(max(p,key=p.get), c)
            self.assertGreater(p[c], .99)

    def test_identical_inputs_are_not_given_fake_identity_based_scores(self):
        bundle, ids, fs = fixture()
        model = raw.RawResidualActor(child(bundle))
        fs = [fs[0]]*len(fs)
        self.assertEqual(model.probabilities(ids, ids[0], fs), model.base.probabilities(ids, ids[0], fs))

    def test_permutation_and_common_shift(self):
        bundle, ids, fs = fixture()
        bundle = child(bundle)
        model = raw.RawResidualActor(bundle)
        p = model.probabilities(ids, ids[0], fs)
        self.assertEqual(p, model.probabilities(list(reversed(ids)),ids[0],list(reversed(fs))))
        shifted = deepcopy(bundle)
        shifted['correction']['b2'] = [5.]
        q = raw.RawResidualActor(shifted).probabilities(ids,ids[0],fs)
        np.testing.assert_allclose(list(p.values()),list(q.values()),atol=1e-14,rtol=0)

    def test_singleton(self):
        bundle, ids, fs = fixture(1)
        self.assertEqual(raw.RawResidualActor(bundle).probabilities(ids, ids[0],fs), {ids[0]:1.})

    def test_parent_snapshot_not_mutated_or_aliased(self):
        bundle, ids, fs = fixture()
        before = deepcopy(bundle)
        model = raw.RawResidualActor(bundle)
        model.probabilities(ids,ids[0],fs)
        self.assertEqual(bundle,before)
        bundle['base']['w2'][0][0] += 1
        self.assertEqual(model.bundle,before)
        with self.assertRaises(ValueError): raw.validate_bundle(bundle)

    def test_old_loader_rejects_new_schema_and_new_rejects_old(self):
        bundle, _, _ = fixture()
        with self.assertRaises(ValueError): validate_bundle(bundle)
        with self.assertRaises(ValueError): raw.validate_bundle(bundle['base'])

    def test_invalid_features_ids_and_head_are_rejected(self):
        bundle,ids,fs = fixture()
        actor = raw.RawResidualActor(bundle)
        for bad_ids,anchor,features in [(ids+ids,ids[0],fs+fs), (ids,'missing',fs), (ids,ids[0],fs[:-1])]:
            with self.assertRaises(ValueError): actor.probabilities(bad_ids,anchor,features)
        bad = deepcopy(fs)
        bad[0]['outcome'] = 0.
        with self.assertRaises(ValueError): actor.probabilities(ids,ids[0],bad)
        bad = deepcopy(bundle)
        bad['correction']['w2'][0][0] = float('nan')
        with self.assertRaises(ValueError): raw.RawResidualActor(bad)

    def test_numeric_underflow_is_an_error_not_fallback(self):
        with self.assertRaises(ValueError):
            raw.corrected_probabilities({'a':.9,'b':.1}, {'a':1000.,'b':-1000.})
        with self.assertRaises(ValueError):
            raw.corrected_probabilities({'a':.9,'b':.1}, {'a':float('inf'),'b':0.})

    def test_large_step_is_rejected_small_step_allowed(self):
        bundle,ids,fs = fixture()
        samples = [episode(bundle,ids,fs)]
        self.assertFalse(raw.evaluate_step(bundle,child(bundle),samples)['within_step_budget'])
        self.assertTrue(raw.evaluate_step(bundle,child(bundle,raw_logit=.1),samples)['within_step_budget'])

    def test_local_guard_does_not_impose_permanent_base_limit(self):
        current,ids,fs = fixture()
        base = deepcopy(current)
        for i in range(1,61):
            following = child(current,raw_logit=.1*i)
            self.assertTrue(raw.evaluate_step(current,following,[episode(current,ids,fs)])['within_step_budget'])
            current = following
        p = raw.RawResidualActor(current).probabilities(ids,ids[0],fs)
        self.assertEqual(max(p,key=p.get),ids[1])
        wrongly_rebased = dict(current, iteration=1, parent_policy=raw.validate_bundle(base))
        self.assertFalse(raw.evaluate_step(base,wrongly_rebased,[episode(base,ids,fs)])['within_step_budget'])

    def test_guard_checks_lineage_behavior_scope_and_trajectory_sum(self):
        bundle,ids,fs = fixture()
        proposed = child(bundle,raw_logit=.5)
        short = raw.evaluate_step(bundle,proposed,[episode(bundle,ids,fs)])
        long = raw.evaluate_step(bundle,proposed,[episode(bundle,ids,fs,1000)])
        self.assertTrue(short['within_step_budget'])
        self.assertFalse(long['within_step_budget'])
        for field,value in [('split','validation'),('policy_sha256','bad')]:
            with self.assertRaises(ValueError):
                raw.evaluate_step(bundle,proposed,[dict(episode(bundle,ids,fs),**{field:value})])
        with self.assertRaises(ValueError):
            raw.evaluate_step(bundle,dict(proposed,parent_policy='a'*64),[episode(bundle,ids,fs)])
        wrong = episode(bundle,ids,fs)
        wrong['states'][0]['probabilities'][ids[0]] += .001
        with self.assertRaises(ValueError): raw.evaluate_step(bundle,proposed,[wrong])

    def test_zero_action_episode_retained_but_not_enough_for_update(self):
        bundle,ids,fs = fixture()
        empty = dict(episode(bundle,ids,fs),states=[],initially_feasible=True)
        self.assertFalse(raw.evaluate_step(bundle,child(bundle),[empty])['within_step_budget'])
        populated = dict(episode(bundle,ids,fs), weight=.5)
        empty.update(episode_id='empty',weight=.5)
        small = child(bundle,raw_logit=.1)
        both = raw.evaluate_step(bundle,small,[populated,empty])
        only = raw.evaluate_step(bundle,small,[dict(populated,weight=1.)])
        self.assertAlmostEqual(both['stats']['mean_kl'],.5*only['stats']['mean_kl'])
        self.assertEqual(both['stats']['max_state_kl'],only['stats']['max_state_kl'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'existing Windows Torch environment')
class RawAutogradTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_zero_head_gradient_is_not_shortcircuited(self):
        import torch
        bundle,ids,fs = fixture()
        model = raw.torch_correction(bundle)
        logs = raw.torch_log_distribution(model,bundle,ids,ids[0],fs)
        grad = torch.autograd.grad(-logs[1],tuple(model.parameters()))
        self.assertGreater(float(grad[2].abs().sum()),0)
        self.assertEqual(float(grad[0].abs().sum()),0)
        self.assertLess(abs(float(grad[3].sum())),1e-12)

    def test_numpy_torch_and_export_nonzero(self):
        import torch
        for changed in (False,True):
            bundle,ids,fs = fixture()
            if changed: bundle=child(bundle,raw_logit=7.)
            model = raw.torch_correction(bundle)
            p = raw.RawResidualActor(bundle).probabilities(ids,ids[0],fs)
            actual=raw.torch_log_distribution(model,bundle,ids,ids[0],fs).exp().detach().numpy()
            np.testing.assert_allclose(actual,list(p.values()),atol=1e-12,rtol=0)
            before=deepcopy(bundle)
            exported=raw.export_correction(model,bundle,'test')
            self.assertEqual(raw.RawResidualActor(exported).probabilities(ids,ids[0],fs),p)
            self.assertEqual(bundle,before)
            self.assertEqual(exported['parent_policy'],raw.validate_bundle(bundle))

    def test_finite_difference_and_no_base_gradients(self):
        import torch
        bundle,ids,fs=fixture(5)
        bundle=child(bundle,raw_logit=.4)
        model=raw.torch_correction(bundle)
        def loss(): return -raw.torch_log_distribution(model,bundle,ids,ids[0],fs)[1]
        gradients=torch.autograd.grad(loss(),tuple(model.parameters()))
        for parameter,analytic in zip(model.parameters(),gradients):
            i=int(analytic.abs().argmax())
            flat=parameter.view(-1)
            original=float(flat[i].detach())
            with torch.no_grad():
                flat[i]=original+1e-6
                plus=float(loss())
                flat[i]=original-1e-6
                minus=float(loss())
                flat[i]=original
            self.assertAlmostEqual((plus-minus)/2e-6,float(analytic.flatten()[i]),places=7)
        self.assertEqual(sum(p.numel() for p in model.parameters()),4193)

    def test_constructing_module_does_not_consume_global_rng(self):
        import torch
        bundle,_,_=fixture()
        state=torch.random.get_rng_state().clone()
        raw.torch_correction(bundle)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
