from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import sa_raw_fast_timing as rt
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as old_metrics
from experiments.sa_raw_selection_fast import FastPolicy
from scripts import run_sa_raw_fast_ttf as cli
from scripts import run_sa_onpolicy as run


def old_jobs():
    jobs=[]
    for m in range(6):
        for d in (.2,.25):
            for seed in (251,257):
                for replica in (0,1):
                    for arm in ('raw_parent','raw_updated','dual16_sa','official_sa'):
                        jobs.append(dict(pair_id=f'm{m}-{d}-s{seed}',solver_seed=seed,replica=replica,
                            comparison_arm=arm,job_id=f'{m}-{d}-{seed}-{replica}-{arm}',
                            case=dict(map_id=f'm{m}',density=d),model={'sha':'frozen'} if arm.startswith('raw_') else None,
                            phase='original-stream',budget_seconds=120.))
    return jobs


def rows():
    result=[]
    for m in range(6):
        for d in range(2):
            for arm in rt.ARMS:
                result.append(dict(job_id=f'{m}-{d}-{arm}',pair_id=f'{m}-{d}',replica=0,
                    comparison_arm=arm,map_id=str(m),initial_fingerprint='i',rng_stream_id='r',status='ok',
                    budget_seconds=120.,feasible=True,success_within_budget=True,delivered_within_budget=True,
                    ttf_seconds=8. if arm=='raw_fast' else 10.,delivery_seconds=11.,search_end_seconds=10.,
                    decisions=5,generated=20,soc=100,makespan=20,wait_steps=2,final_conflicts=0,
                    stop='feasible',last_pp_failure_reason='none',selection_seconds=1.,native_pp_seconds=2.,bookkeeping_seconds=1.))
    return result


class FastTimingTests(unittest.TestCase):
    def test_blind_selection_stream_model_identity_and_balanced_order(self):
        source=old_jobs()
        copy=deepcopy(source)
        jobs=cli.selected_jobs(source,cli.config())
        self.assertEqual(source,copy)
        self.assertEqual(len(jobs),36)
        self.assertEqual(jobs,cli.schedule(jobs))
        orders=Counter(tuple(j['comparison_arm'] for j in jobs[i:i+3]) for i in range(0,36,3))
        self.assertEqual(len(orders),6)
        self.assertEqual(set(orders.values()),{2})
        self.assertEqual(set(j['solver_seed'] for j in jobs),{251})
        self.assertEqual(set(j['replica'] for j in jobs),{0})
        for i in range(0,36,3):
            pair={j['comparison_arm']:j for j in jobs[i:i+3]}
            a,b=pair['raw_fast'],pair['raw_reference']
            for key in ('pair_id','replica','phase','model','case','source_timed_job_id'):
                self.assertEqual(a[key],b[key])
            self.assertNotEqual(a['job_id'],b['job_id'])
        with self.assertRaises(ValueError):cli.schedule(jobs[:-1])
        with self.assertRaises(ValueError):cli.schedule(jobs+[jobs[0]])
        with self.assertRaises(ValueError):cli.selected_jobs([],cli.config())

    def test_timer_body_unchanged_and_globals_private(self):
        self.assertIs(rt._fast_worker.__code__,reference.timed_worker.__code__)
        self.assertIs(rt._fast_preflight.__code__,reference.preflight_worker.__code__)
        self.assertIs(rt._fast_worker.__globals__['Policy'],FastPolicy)
        self.assertIs(reference.timed_worker.__globals__['Policy'],reference.Policy)
        self.assertIs(reference.audit_worker.__globals__['Policy'],reference.Policy)
        self.assertEqual(rt._validate.__globals__['ARMS'],rt.ARMS)
        self.assertEqual(len(old_metrics.ARMS),4)

    def test_dispatch(self):
        with patch.object(rt,'_fast_worker',return_value='fast') as fast, patch.object(reference,'timed_worker',return_value='old') as old:
            self.assertEqual(rt.timed_worker({'comparison_arm':'raw_fast'}),'fast')
            self.assertEqual(rt.timed_worker({'comparison_arm':'raw_reference'}),'old')
            self.assertEqual(rt.timed_worker({'comparison_arm':'dual16_sa'}),'old')
            self.assertEqual((fast.call_count,old.call_count),(1,2))
            with self.assertRaises(ValueError):rt.timed_worker({'comparison_arm':'raw_parent'})

    def test_three_arm_metrics_pure_and_deterministic(self):
        data=rows()
        frozen=deepcopy(data)
        report=rt.summarize(data,bootstrap=80)
        self.assertEqual(report,rt.summarize(reversed(data),bootstrap=80))
        self.assertEqual(data,frozen)
        self.assertEqual(set(report['arms']),set(rt.ARMS))
        c=report['contrasts']['raw_reference']['common_success']['ttf_seconds']
        self.assertEqual(c['count'],12)
        self.assertEqual(c['ci95'],[-2.,-2.])
        self.assertEqual(c['wins'],12)
        self.assertEqual(report['contrasts']['raw_reference']['modeled_completion']['1.0']['difference']['mean'],0.)
        self.assertEqual(rt.summarize(data,bootstrap=80)['by_map']['0']['pairs'],2)
        with self.assertRaises(ValueError):rt.summarize(data[:-1],bootstrap=80)
        with self.assertRaises(ValueError):rt.summarize(data+[data[0]],bootstrap=80)
        with self.assertRaises(ValueError):rt.summarize(data,bootstrap=0)

    def test_failure_is_not_zero_and_success_sign(self):
        data=rows()
        old=next(r for r in data if r['comparison_arm']=='raw_reference')
        old.update(feasible=False,success_within_budget=False,delivered_within_budget=False,
            ttf_seconds=None,final_conflicts=1,stop='deadline',search_end_seconds=120.,delivery_seconds=121.)
        c=rt.summarize(data,bootstrap=80)['contrasts']['raw_reference']
        self.assertEqual(c['common_success']['ttf_seconds']['count'],11)
        self.assertEqual(c['success']['wins'],1)
        self.assertEqual(c['success']['losses'],0)
        self.assertAlmostEqual(c['success']['difference']['mean'],1/12)
        self.assertEqual(c['common_delivered_count'],11)
        data[0]['rng_stream_id']='different'
        with self.assertRaises(ValueError):rt.summarize(data,bootstrap=80)

    def test_no_decision_or_node_cap(self):
        c=cli.config()
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertEqual(c['workers'],1)

    def test_pair_audit_rejects_drift_except_deadline_paths(self):
        from scripts import train_sa_history_selector as training
        fields=('decision','before','action','pool','proposal_order','anchor_id','selected_id',
                'selection_draw','features','probabilities','candidate_ids',
                'behavior_log_probability','temperature','uniform')
        event={k:'same' for k in fields}
        event.update(delta={'fp':'next'},metrics={'pp_failure_reason':'none'})
        other=deepcopy(event)
        q=SimpleNamespace(state_fingerprint=lambda s:s['fp'],apply_state_delta=lambda s,d:d)
        result=dict(feasible=True,decisions=1,final_fingerprint='next')
        def read(path):
            return {'fp':'initial'} if path.name=='initial.json' else run.sealed(result)
        job=dict(plan={},output='build/fake',runtime_jobs=['old','fast'],parent_pid=1,job_id='pair')
        with patch.object(training,'die_with_parent'),patch.object(run,'native_runtime',return_value=q), \
             patch.object(run,'read_json',side_effect=read),patch.object(run,'sha256_file',return_value='x'), \
             patch.object(run,'trace_read',side_effect=lambda f:iter([event if f.name=='old' else other])):
            self.assertEqual(rt.pair_worker(job)['common_decisions'],1)
            other['action']='bad'
            with self.assertRaisesRegex(ValueError,'action'):rt.pair_worker(job)
            other['action']='same'
            other['delta']={'fp':'different'}
            with self.assertRaisesRegex(ValueError,'path drift'):rt.pair_worker(job)
            other['metrics']['pp_failure_reason']='time_limit'
            self.assertTrue(rt.pair_worker(job)['deadline_branch'])

    def test_changed_config_rejected_before_result_read(self):
        c=cli.config()
        for key,value in (('workers',2),('max_decisions',100),('solver_seed',257),('automatic_promotion',True)):
            with patch.object(run,'read_json',return_value=c|{key:value}):
                with self.assertRaises(ValueError):cli.config()

    def test_safe_stop_between_episodes_and_explicit_resume(self):
        jobs=[dict(job_id='a',comparison_arm='raw_fast',pair_id='p'),dict(job_id='b',comparison_arm='raw_reference',pair_id='p')]
        r=dict(binding='x',jobs=jobs,config=cli.config())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def execute(*args):
                j=args[2][0]
                run.once(out/'collect'/(j['job_id']+'.json'),run.sealed(dict(status='ok',job_id=j['job_id'])))
                if j['job_id']=='a':run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
            with patch.object(cli,'verify',return_value=(r,out)), patch.object(cli,'os',SimpleNamespace(name='posix',environ={})), \
                 patch.object(cli.source,'check_complete'),patch.object(cli.source,'read_result'), \
                 patch.object(cli.source,'execute',side_effect=execute) as execution:
                result=cli.collect()
                self.assertTrue(result['paused'])
                self.assertEqual(result['completed'],1)
                self.assertEqual(execution.call_count,1)
                with self.assertRaises(ValueError):cli.collect()
                self.assertEqual(cli.collect(True)['collected'],2)
                self.assertEqual(execution.call_count,2)
                self.assertFalse((out/'STOP_AFTER_EPISODE').exists())

    def test_partial_attempt_refused(self):
        r=dict(binding='x',jobs=[dict(job_id='a',comparison_arm='raw_fast',pair_id='p')],config=cli.config())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'episodes'/'a').mkdir(parents=True)
            with patch.object(cli,'verify',return_value=(r,out)),patch.object(cli,'os',SimpleNamespace(name='posix',environ={})), \
                 patch.object(cli.source,'check_complete'),patch.object(cli.source,'execute') as execution:
                with self.assertRaisesRegex(ValueError,'partial attempt'):cli.collect()
                execution.assert_not_called()


if __name__=='__main__':unittest.main()
