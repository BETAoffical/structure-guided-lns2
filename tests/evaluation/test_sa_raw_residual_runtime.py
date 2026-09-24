from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from threading import Thread
import unittest
import numpy as np

from experiments import sa_raw_residual_actor as raw
from experiments import sa_raw_residual_update as update
from experiments import sa_uncapped_training_contract as credit
from experiments.sa_raw_residual_runtime import actor_scope, RuntimeActor
from experiments.sa_onpolicy_actor import SCHEMA, NumpyActor, select_with_draw
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_residual as cli


def fixture():
    rng = np.random.default_rng(13)
    base = dict(schema=SCHEMA,feature_names=[f'f{i:03d}' for i in range(129)],mean=[0.]*129,scale=[1.]*129,
        w1=rng.normal(0,.1,(32,129)).tolist(),b1=[0.]*32,w2=[[.02]*32],b2=[.01],
        epsilon=.1,residual_bound=2.)
    bundle = raw.initial_bundle(base,'unit')
    ids = ['a','b','c']
    fs = [dict(zip(base['feature_names'],r)) for r in np.eye(129)[:3].tolist()]
    ps = raw.RawResidualActor(bundle).probabilities(ids,'a',fs)
    draw = .97
    event = dict(decision=0,policy_sha256=raw.validate_bundle(bundle),candidate_ids=ids,anchor_id='a',features=fs,
        probabilities=ps,selection_draw=draw,selected_id=select_with_draw(ps,draw))
    event['behavior_log_probability'] = math.log(ps[event['selected_id']])
    return bundle,event


class RuntimeTests(unittest.TestCase):
    def test_adapter_exact_parent_and_restores_even_on_exception(self):
        bundle,e = fixture()
        saved = run.NumpyActor,run.actor_load,run.validate_bundle
        p = dict(binding='unit')
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with actor_scope(bundle,Path('out'),p):
                self.assertEqual(run.actor_load(Path('out'),p,0),bundle)
                actor = run.NumpyActor(bundle)
                self.assertEqual(actor.probabilities(e['candidate_ids'],'a',e['features']),e['probabilities'])
                self.assertEqual(actor.out_of_range_fraction(e['features']),actor.base.out_of_range_fraction(e['features']))
                with self.assertRaises(ValueError):run.actor_load(Path('elsewhere'),p,0)
                with self.assertRaises(ValueError):run.actor_load(Path('out'),p,1)
                with self.assertRaises(ValueError):
                    with actor_scope(bundle,Path('out'),p):pass
                raise RuntimeError('deliberate')
        self.assertEqual(saved,(run.NumpyActor,run.actor_load,run.validate_bundle))
        with actor_scope(bundle,Path('out'),p):pass

    def test_threads_rejected(self):
        bundle,_ = fixture()
        errors = []
        def invoke():
            try:
                with actor_scope(bundle,Path('out'),{}):pass
            except ValueError as exc:errors.append(str(exc))
        t = Thread(target=invoke)
        t.start()
        t.join()
        self.assertEqual(len(errors),1)

    def test_pack_alignment_permutation_and_no_outcome_input(self):
        bundle,e = fixture()
        a = update.pack_episode(bundle,[e])
        changed = deepcopy(e)
        changed['candidate_ids'].reverse()
        changed['features'].reverse()
        b = update.pack_episode(bundle,[changed])
        np.testing.assert_array_equal(a['padded'][0],b['padded'][0])
        np.testing.assert_array_equal(a['selected'],b['selected'])
        changed['features'][0]['outcome'] = 1
        with self.assertRaises(ValueError):update.pack_episode(bundle,[changed])

    def test_stale_policy_probability_draw_or_sequence_rejected(self):
        bundle,e = fixture()
        for key,value in [('policy_sha256','bad'),('decision',1),('selection_draw',0.),('behavior_log_probability',0.)]:
            with self.assertRaises(ValueError):update.pack_episode(bundle,[dict(e,**{key:value})])
        altered = deepcopy(e)
        altered['probabilities']['a'] -= .01
        with self.assertRaises(ValueError):update.pack_episode(bundle,[altered])
        self.assertIsNone(update.pack_episode(bundle,[]))

    def test_schedule_uses_train_only_and_paired_fresh_streams(self):
        cfg = cli.configuration()
        roots = [dict(pair_id=str(i),split='train',case=dict(map_id=str(i//4))) for i in range(24)]
        initial = {str(i):'a'*64 for i in range(24)}
        train = cli.schedule(roots,initial,cfg['phase'],4,[0])
        comparison = cli.schedule(roots,initial,cfg['comparison_phase'],2,[0,1])
        self.assertEqual(len(train),96)
        self.assertEqual(len(comparison),96)
        self.assertEqual(len({j['job_id'] for j in train+comparison}),192)
        plan = dict(config=dict(stream_seed=7))
        for a,b in zip(comparison[::2],comparison[1::2]):
            self.assertEqual(run.stream_draw(plan,a['phase'],a['pair_id'],a['replica'],100000,'select'),
                             run.stream_draw(plan,b['phase'],b['pair_id'],b['replica'],100000,'select'))
        self.assertNotEqual(run.stream_draw(plan,cfg['phase'],'0',0,0,'pp'),
                            run.stream_draw(plan,cfg['comparison_phase'],'0',0,0,'pp'))
        roots[0]['split'] = 'test'
        with self.assertRaises(ValueError):cli.schedule(roots,initial,'x',4,[0])
        self.assertIsNone(cfg['max_decisions'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'existing Torch environment required')
class UpdateTests(unittest.TestCase):
    def test_vectorized_logs_match_scalar_and_nonzero_previous_policy(self):
        import torch
        torch.set_num_threads(1)
        bundle,e = fixture()
        for nonzero in (False,True):
            if nonzero:
                bundle['correction']['w2'][0][1] = .2
                e['policy_sha256'] = raw.validate_bundle(bundle)
                e['probabilities'] = raw.RawResidualActor(bundle).probabilities(e['candidate_ids'],'a',e['features'])
                e['selected_id'] = select_with_draw(e['probabilities'],e['selection_draw'])
                e['behavior_log_probability'] = math.log(e['probabilities'][e['selected_id']])
            model = raw.torch_correction(bundle)
            pack = update.pack_episode(bundle,[e])
            logs = update.log_distribution(model,pack)
            scalar = raw.torch_log_distribution(model,bundle,e['candidate_ids'],'a',e['features'])
            np.testing.assert_allclose(logs.detach().numpy()[0],scalar.detach().numpy(),atol=1e-14,rtol=0)

    def test_gradient_is_terminal_sum_not_mean_and_direction_improves_likelihood(self):
        import torch
        bundle,e = fixture()
        one = dict(update.pack_episode(bundle,[e]),weight=1.,coefficient=.25)
        doubled = dict(update.pack_episode(bundle,[e,dict(e,decision=1)]),weight=1.,coefficient=.25)
        a,b = update.terminal_gradient(bundle,[one]),update.terminal_gradient(bundle,[doubled])
        for x,y in zip(a,b):torch.testing.assert_close(y,2*x,rtol=1e-12,atol=1e-14)
        proposed,diag = update.guarded_update(bundle,[one],'update')
        self.assertIsNotNone(proposed)
        ps = raw.RawResidualActor(proposed).probabilities(e['candidate_ids'],'a',e['features'])
        self.assertGreater(ps[e['selected_id']],e['probabilities'][e['selected_id']])
        self.assertEqual(proposed['base'],bundle['base'])
        self.assertEqual(proposed['parent_policy'],raw.validate_bundle(bundle))
        self.assertEqual(diag['reference_policy_sha256'],raw.validate_bundle(bundle))

    def test_padding_zero_action_weights_and_guard_match_contract(self):
        import torch
        bundle,e = fixture()
        small = deepcopy(e)
        small.update(decision=1,candidate_ids=['a'],features=[e['features'][0]],probabilities={'a':1.},
                     selected_id='a',behavior_log_probability=0.)
        pack = dict(update.pack_episode(bundle,[e,small]),weight=.5,coefficient=.1)
        model = raw.torch_correction(bundle)
        with torch.no_grad():model[2].weight[0,0] += .1
        logs = update.log_distribution(model,pack)
        self.assertEqual(logs.exp()[1].tolist(),[1.,0.,0.])
        proposed = raw.export_correction(model,bundle,'test')
        states = [dict(ids=v['candidate_ids'],anchor=v['anchor_id'],features=v['features'],
            probabilities=v['probabilities'],draw=v['selection_draw'],selected=v['selected_id']) for v in (e,small)]
        episodes = [dict(episode_id='full',split='train',policy_sha256=raw.validate_bundle(bundle),weight=.5,states=states),
                    dict(episode_id='empty',split='train',policy_sha256=raw.validate_bundle(bundle),weight=.5,states=[],initially_feasible=True)]
        ref = raw.evaluate_step(bundle,proposed,episodes)['stats']
        result = update.step_statistics([pack],[logs.detach().exp().numpy()])
        for key in ('mean_kl','mean_trajectory_kl','max_state_kl','mean_tv'):
            self.assertAlmostEqual(ref[key],result[key],places=13)

    def test_zero_credit_does_not_update_or_train_again(self):
        bundle,e = fixture()
        pack = dict(update.pack_episode(bundle,[e]),weight=1.,coefficient=0.)
        model,result = update.guarded_update(bundle,[pack],'test')
        self.assertIsNone(model)
        self.assertEqual(result['gradient_norm'],0.)
        self.assertIsNone(update.guarded_update(bundle,[],'test')[0])


if __name__ == '__main__':unittest.main()
