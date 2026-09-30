from copy import deepcopy
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sa_completion_credit_transfer as d
from scripts import audit_sa_completion_credit_transfer as cli


def fixture():
    rng = np.random.default_rng(731)
    bundle = dict(correction=dict(w1=rng.normal(0,.1,(32,129)).tolist(),
        b1=rng.normal(0,.1,32).tolist(),w2=rng.normal(0,.2,(1,32)).tolist(),b2=[.3]))
    x = rng.normal(0,.3,(3,3,129))
    valid = np.array([[1,1,1],[1,1,0],[1,0,0]],dtype=bool)
    prior = np.array([[.2,.3,.5],[.7,.3,0],[1,0,0]])
    pack = dict(padded=(x,np.zeros_like(prior),valid),base=prior,selected=np.array([2,0,0]))
    probability = np.exp(d.log_probabilities(bundle,pack)[0])
    pack['padded'] = (x,probability,valid)
    return bundle,pack


def row(episode, m, vector, coefficient=1.):
    return dict(episode_id=episode,map_id=m,pair_id=m+'p',loss_gradient=vector,
                coefficient=coefficient,success=coefficient>0,decisions=3,task_variant='d25',
                surrogate_delta={'A2':.01,'A-wide':-.01})


class CreditTransferTests(unittest.TestCase):
    def test_nonzero_head_finite_difference_each_tensor(self):
        bundle,pack = fixture()
        score,_ = d.trajectory_score(bundle,pack)
        offset = 0
        selected = pack['selected']
        for key in d.ORDER:
            shape = np.asarray(bundle['correction'][key]).shape
            size = int(np.prod(shape))
            index = int(np.argmax(np.abs(score[offset:offset+size])))
            observed = []
            for sign in (-1,1):
                other = deepcopy(bundle)
                value = np.asarray(other['correction'][key])
                value.ravel()[index] += sign*1e-5
                other['correction'][key] = value.tolist()
                logs,_ = d.log_probabilities(other,pack)
                observed.append(logs[np.arange(len(selected)),selected].sum())
            self.assertAlmostEqual((observed[1]-observed[0])/2e-5,score[offset+index],places=8)
            offset += size

    def test_torch_nonzero_all_parameters(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch checked in existing graph environment')
        bundle,pack = fixture()
        c = bundle['correction']
        parameters = [torch.tensor(c[k],dtype=torch.float64,requires_grad=True) for k in d.ORDER]
        w1,b1,w2,b2 = parameters
        x,_,valid = pack['padded']
        h = torch.tanh(torch.tensor(x,dtype=torch.float64) @ w1.T+b1)
        prior = np.zeros_like(pack['base'])
        prior[valid] = np.log(pack['base'][valid])
        logits = torch.tensor(prior)+(h @ w2.T).squeeze(-1)+b2[0]
        logs = torch.log_softmax(logits.masked_fill(~torch.tensor(valid),-torch.inf),dim=1)
        chosen = logs[torch.arange(3),torch.tensor(pack['selected'])].sum()
        actual = np.concatenate([g.detach().numpy().ravel() for g in torch.autograd.grad(chosen,parameters)])
        np.testing.assert_allclose(d.trajectory_score(bundle,pack)[0],actual,atol=1e-13,rtol=1e-12)

    def test_padding_has_no_gradient_and_single_candidate_zero(self):
        bundle,pack = fixture()
        score,_ = d.trajectory_score(bundle,pack)
        altered = deepcopy(pack)
        altered['padded'][0][~pack['padded'][2]] = 1000
        np.testing.assert_array_equal(score,d.trajectory_score(bundle,altered)[0])
        single = dict(padded=tuple(a[2:3,:1] for a in pack['padded']),base=pack['base'][2:3,:1],selected=np.array([0]))
        np.testing.assert_array_equal(d.trajectory_score(bundle,single)[0],np.zeros_like(score))

    def test_entire_trajectory_sum_not_length_average(self):
        bundle,pack = fixture()
        duplicate = dict(pack,padded=tuple(np.concatenate([a,a],axis=0) for a in pack['padded']),
                         base=np.concatenate([pack['base'],pack['base']],axis=0),
                         selected=np.concatenate([pack['selected'],pack['selected']]))
        np.testing.assert_allclose(d.trajectory_score(bundle,duplicate)[0],2*d.trajectory_score(bundle,pack)[0],atol=1e-13)

    def test_stale_behavior_rejected(self):
        bundle,pack = fixture()
        pack['padded'][1][0,0] += .01
        with self.assertRaisesRegex(ValueError,'behavior replay'):
            d.trajectory_score(bundle,pack)

    def test_opposing_batches_and_zero_map_not_removed(self):
        a = [row('a0','a',[1.,0.]),row('a1','zero',[0.,0.],0.)]
        b = [row('b0','b',[-1.,0.])]
        report = d.transfer(a,b,samples=200,seed=4)
        self.assertEqual(report['direction_cosine'],-1.)
        self.assertLess(report['left_update_on_right_derivative'],0)
        self.assertEqual(report['right_maps_opposing_left'],1)
        self.assertGreater(report['bootstrap']['zero_gradient_draws'],0)
        self.assertEqual(report,d.transfer(a,b,samples=200,seed=4))
        with self.assertRaisesRegex(ValueError,'overlapping batch'):
            d.transfer(a,[row('x','a',[-1.,0.])],samples=2,seed=3)

    def test_saved_direction_reconstruction_and_leave_map_out(self):
        parent = dict(correction=dict(w1=[[0.]],b1=[0.],w2=[[0.]],b2=[0.]))
        g = np.array([1.,2.,0.,0.])
        proposed = dict(correction=dict(w1=[[-.25/np.sqrt(5)]],b1=[-.5/np.sqrt(5)],w2=[[0.]],b2=[0.]))
        rows = [row('x','a',[1.,0.,0.,0.]),row('y','b',[0.,2.,0.,0.]),row('z','c',[0.,0.,0.,0.],0.)]
        report = d.summarize(rows,parent,proposed,np.linalg.norm(g),other_direction=-g/np.linalg.norm(g))
        self.assertEqual(report['maps'],3)
        self.assertIsNone(report['by_map']['c']['alignment_with_batch'])
        self.assertAlmostEqual(report['by_map']['c']['leave_map_out_alignment'],1.)
        self.assertAlmostEqual(report['map_concentration']['effective_norm_groups'],1.8)
        bad = deepcopy(proposed)
        bad['correction']['w1'][0][0] *= -1
        with self.assertRaisesRegex(ValueError,'direction mismatch'):
            d.summarize(rows,parent,bad,np.linalg.norm(g))

    def test_no_duplicate_episodes(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            d.loss_gradient([row('x','m',[1.]),row('x','m',[2.])])

    def test_cli_help_does_no_input_or_output(self):
        with patch('sys.argv',['audit','--help']),patch.object(cli,'configuration',side_effect=AssertionError('I/O')),\
                contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code,0)

    def test_worker_rejects_heldout(self):
        item = dict(job={},coefficient=dict(episode_id='x'))
        with patch.object(cli.continuation.old.previous,'load_model',return_value={}),\
                patch.object(cli.continuation.old.previous,'extract_worker',return_value=({'split':'development'},{})):
            with self.assertRaisesRegex(ValueError,'non-train'):
                cli.worker(item)

    def test_shard_rejects_identity_and_missing_torch_proof(self):
        item = dict(binding='b',batch='A2',job={'job_id':'j'},coefficient={'episode_id':'e','coefficient':.1},torch_check=True)
        doc = dict(binding='b',batch='A2',job_id='j',episode_id='e',coefficient=.1,torch_check=True,torch_max_error=0.)
        cli.verify_shard(doc,item)
        doc['binding']='other'
        with self.assertRaisesRegex(ValueError,'stale shard'):
            cli.verify_shard(doc,item)
        doc['binding']='b'
        doc['torch_max_error']=.1
        with self.assertRaisesRegex(ValueError,'Torch proof'):
            cli.verify_shard(doc,item)


if __name__=='__main__':
    unittest.main()
