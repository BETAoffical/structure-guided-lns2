import math
from types import SimpleNamespace
import unittest

from experiments import sa_budget_cooling as c
from scripts import probe_sa_budget_cooling as probe


class CoolingTests(unittest.TestCase):
    def test_decision_clock_unchanged(self):
        for d in (0,1,256,331,1000,10000):
            for work in (0,123456,24999999):
                self.assertEqual(c.temperature('decision',d,work),1000.*.99**d)

    def test_work_endpoint_and_monotonicity(self):
        values=[c.temperature('nodes',123,n) for n in (0,6250000,12500000,18750000,25000000)]
        self.assertEqual(values[0],1000.)
        self.assertEqual(values,sorted(values,reverse=True))
        self.assertAlmostEqual(math.exp(-1/values[-1]),.05,places=14)
        self.assertEqual(c.temperature('nodes',999,12500000),values[2])

    def test_budget_overshoot_clamped(self):
        self.assertEqual(c.temperature('nodes',20,27000000),c.temperature('nodes',20,25000000))

    def test_clock_subtracts_initial_pp_and_checks_order(self):
        clock=c.WorkClock('nodes',1000)
        self.assertEqual(clock(0),1000.)
        clock.observe(dict(low_level=dict(generated=1005)))
        self.assertEqual(clock.used,5)
        self.assertEqual(clock(1),c.temperature('nodes',1,5))
        with self.assertRaises(ValueError): clock(0)
        with self.assertRaises(ValueError): clock.observe(dict(low_level=dict(generated=1004)))

    def test_wrapper_forwards_same_object(self):
        result={'observation':{'low_level':{'generated':120}}}
        calls=[]
        env=SimpleNamespace(propose=lambda:'unchanged',step_experimental_pp=lambda *a,**kw:(calls.append((a,kw)) or result))
        clock=c.WorkClock('nodes',100)
        wrapper=c.ClockedEnvironment(env,clock,lambda x:x)
        self.assertIs(wrapper.step_experimental_pp('action',1,mode='x'),result)
        self.assertEqual(calls,[(('action',1),{'mode':'x'})])
        self.assertEqual(wrapper.propose(),'unchanged')
        self.assertEqual((clock.used,clock.decision),(20,1))

    def test_scope_restores_on_exception(self):
        original=lambda d:d
        q=SimpleNamespace(temperature=original)
        with self.assertRaises(RuntimeError):
            with c.clock_scope(q,lambda d:3):
                self.assertEqual(q.temperature(999),3)
                raise RuntimeError('test')
        self.assertIs(q.temperature,original)

    def test_nested_scope_restores(self):
        q=SimpleNamespace(temperature=lambda d:1)
        with c.clock_scope(q,lambda d:2):
            with c.clock_scope(q,lambda d:3): self.assertEqual(q.temperature(0),3)
            self.assertEqual(q.temperature(0),2)
        self.assertEqual(q.temperature(0),1)

    def test_no_step_cap(self):
        self.assertIsNone(probe.configuration()['max_decisions'])
        self.assertGreater(c.temperature('nodes',1000000,0),0)

    def test_schedule_full_factorial_train_only(self):
        cfg=probe.configuration()
        roots=[dict(split='train',pair_id=str(i)) for i in range(24)]
        jobs=probe.schedule(roots,cfg,{str(i):str(i) for i in range(24)})
        self.assertEqual(len({j['job_id'] for j in jobs}),192)
        self.assertEqual({j['phase'] for j in jobs},{cfg['phase']})
        self.assertEqual(len([j for j in jobs if j['pair_id']=='0']),8)
        roots[0]['split']='validation'
        with self.assertRaises(ValueError): probe.schedule(roots,cfg,{})

    def test_pairing_reports_gains_and_losses_not_all_win_gate(self):
        common=dict(map_id='m',policy_sha256='p',initial_fingerprint='s',rng_stream_id='r',replica=0,
                    generated=100,decisions=2,soc=10,makespan=5,wait_steps=1)
        rows=[dict(common,pair_id=pair,clock=clock,success=win) for pair,a,b in
              [('a',False,True),('b',True,False),('c',True,True)] for clock,win in [('decision',a),('nodes',b)]]
        result=probe.contrasts(rows)
        self.assertEqual((result['net_success'],result['common_success']),(0,1))
        self.assertEqual(result['gains'],[['a',0]])
        self.assertEqual(result['losses'],[['b',0]])
        self.assertEqual(result,probe.contrasts(list(reversed(rows))))
        with self.assertRaises(ValueError): probe.contrasts(rows[:-1])
        rows[-1]['rng_stream_id']='other'
        with self.assertRaises(ValueError): probe.contrasts(rows)

    def test_map_bootstrap_deterministic(self):
        values={'b':dict(net_success=-1,pairs=8),'a':dict(net_success=2,pairs=8)}
        self.assertEqual(probe.map_bootstrap(values),probe.map_bootstrap(dict(reversed(list(values.items())))))
        self.assertEqual(probe.map_bootstrap(values)['unit'],'map')


if __name__=='__main__': unittest.main()
