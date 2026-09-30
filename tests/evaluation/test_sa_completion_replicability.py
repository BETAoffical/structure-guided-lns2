from copy import deepcopy
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sa_completion_replicability as s
from scripts import run_sa_completion_replicability as cli


def rows(n,phase,all_success=False):
    return [dict(episode_id=phase+str(i),rng_stream_id=phase+'rng'+str(i),pair_id='p',
        initial_fingerprint='initial',policy_sha256='A',status='ok',split='train',replica=i,
        success=True if all_success else i%4!=2,decisions=773 if i%4==2 else 20+i,
        generated=25000001 if i%4==2 else 1000,score=[float(i+1),float(i%3)]) for i in range(n)]


class ReplicabilityTests(unittest.TestCase):
    def test_conditional_gradient_matches_independent_baseline(self):
        y=np.array([1.,1.,0.,1.])
        score=np.array([[1.,2.],[3.,4.],[8.,9.],[5.,6.]])
        actual,parts=s.conditional_loss_gradient(y,score)
        expected=np.sum([-1/4*(y[i]-np.mean(np.delete(y,i)))*score[i] for i in range(4)],axis=0)
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        np.testing.assert_allclose(parts.sum(axis=0),actual)

    def test_all_success_failure_no_credit_and_no_fabricated_direction(self):
        a,b=rows(4,'a',True),rows(16,'b',True)
        report,go,gn=s.condition_report(a,b,bootstrap=100,seed=3)
        np.testing.assert_array_equal(go,[0.,0.])
        np.testing.assert_array_equal(gn,[0.,0.])
        self.assertIsNone(report['old_fresh_direction_cosine'])
        self.assertIsNone(report['largest_episode_contribution']['norm_share'])
        self.assertEqual(report['bootstrap']['uninformative_draws'],100)
        y=[False]*16
        np.testing.assert_array_equal(s.conditional_loss_gradient(y,[r['score'] for r in b])[0],[0.,0.])

    def test_bootstrap_covariance_form_equals_recomputed_replica_baseline(self):
        original=rows(16,'r')
        # A bootstrap sample can repeat an episode; its baseline must be recomputed, not copied.
        indices=[0,0,0,1,2,2,3,4,5,6,7,8,9,10,14,15]
        y=np.array([original[i]['success'] for i in indices],dtype=float)
        score=np.array([original[i]['score'] for i in indices])
        actual,_=s.conditional_loss_gradient(y,score)
        covariance=-16/15*(np.mean(y[:,None]*score,axis=0)-y.mean()*score.mean(axis=0))
        np.testing.assert_allclose(actual,covariance,atol=1e-14)

    def test_independent_repetitions_bootstrap_deterministic_and_long_not_capped(self):
        a,b=rows(4,'a'),rows(16,'b')
        r,_,_=s.condition_report(a,b,bootstrap=100,seed=9)
        self.assertEqual(r,s.condition_report(a,b,bootstrap=100,seed=9)[0])
        self.assertEqual(r['fresh']['failed_ge_512'],4)
        self.assertEqual(r['fresh']['decisions']['max'],773)
        b[0]['decisions']=1024
        r,_,_=s.condition_report(a,b,bootstrap=100,seed=9)
        self.assertEqual(r['fresh']['successful_ge_512'],1)
        self.assertEqual(r['fresh']['decisions']['max'],1024)

    def test_unknown_heldout_changed_initial_and_reused_stream_rejected(self):
        a,b=rows(4,'a'),rows(16,'b')
        for key,value in [('status','censored'),('split','validation'),('initial_fingerprint','other'),
                          ('policy_sha256','A2'),('rng_stream_id',a[0]['rng_stream_id'])]:
            changed=deepcopy(b)
            changed[0][key]=value
            with self.assertRaises(ValueError):
                s.condition_report(a,changed,bootstrap=10,seed=1)

    def test_replica_order_not_gradient_information(self):
        a,b=rows(4,'a'),rows(16,'b')
        r,go,gn=s.condition_report(a,b,bootstrap=10,seed=1)
        other,go2,gn2=s.condition_report(a[::-1],b[::-1],bootstrap=10,seed=1)
        np.testing.assert_allclose(go,go2)
        np.testing.assert_allclose(gn,gn2)
        self.assertEqual(r['fresh'],other['fresh'])
        # Same seed assigns resampling counts to row order; no claim of bitwise bootstrap order invariance.

    def test_schedule_round_robin_and_only_frozen_parent(self):
        conditions=[dict(pair_id=str(i),case={'map_id':str(i)},split='train',solver_seed=233,
                         expected_initial='x') for i in range(4)]
        before=deepcopy(conditions)
        jobs=cli.schedule(conditions,'new',16)
        self.assertEqual(conditions,before)
        self.assertEqual(len(jobs),64)
        self.assertEqual(len({j['job_id'] for j in jobs}),64)
        self.assertEqual([j['pair_id'] for j in jobs[:8]],['0','1','2','3']*2)
        self.assertEqual({j['comparison_arm'] for j in jobs},{'parent'})
        self.assertTrue(all(j['split']=='train' for j in jobs))
        self.assertNotEqual(jobs[0]['job_id'],cli.schedule(conditions,'old',16)[0]['job_id'])
        with self.assertRaises(ValueError):
            cli.schedule(conditions[:3],'new',16)

    def test_fixed_uncapped_budget_and_no_training_entry(self):
        cfg=cli.configuration()
        self.assertIsNone(cfg['max_decisions'])
        self.assertEqual(cfg['replicas'],16)
        self.assertEqual(cfg['maximum_updates_per_arm'],0)
        self.assertFalse(cfg['formal_ttf'])
        self.assertEqual(len(cfg['pairs']),4)
        self.assertNotIn('train',cli.main.__code__.co_consts)
        self.assertEqual(cli.prior.CONFIG,'configs/sa_completion_continuation.json')

    def test_help_has_no_runtime_or_file_writes(self):
        with patch('sys.argv',['replication','--help']),patch.object(cli,'configuration',side_effect=AssertionError('I/O')),\
                contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code,0)

    def test_dry_run_exact_count_and_atomic_budget_boundary(self):
        r={'binding':'b','config':cli.configuration(),'train_jobs':[None]*64}
        summary=cli.dry_run_body(r)
        self.assertEqual(summary['total_node_boundary'],1600000000)
        self.assertEqual(summary['process_fuse_batch_upper_seconds'],3840.)
        self.assertTrue(summary['atomic_repair_may_overshoot'])
        self.assertEqual(summary['maximum_parameter_updates'],0)
        r['train_jobs'].pop()
        with self.assertRaises(ValueError):
            cli.dry_run_body(r)

    def test_wilson_keeps_uncertainty_at_extremes(self):
        self.assertGreater(s.wilson(0,16)[1],0.)
        self.assertLess(s.wilson(16,16)[0],1.)
        with self.assertRaises(ValueError):
            s.wilson(17,16)


if __name__=='__main__':
    unittest.main()
