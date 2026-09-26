from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.sa_raw_selection_fast import _bind
from scripts import run_sa_raw_fast_remaining as cli
from scripts import run_sa_raw_fast_ttf as first
from scripts import run_sa_onpolicy as run


def source_jobs():
    jobs=[]
    for m in range(6):
        for d in (20,25):
            for seed in (251,257):
                for replica in (0,1):
                    for arm in ('raw_parent','raw_updated','dual16_sa','official_sa'):
                        jobs.append(dict(pair_id=f'm{m}-d{d}-s{seed}',solver_seed=seed,replica=replica,
                            comparison_arm=arm,job_id=f'{m}-{d}-{seed}-{replica}-{arm}',
                            case=dict(map_id=f'm{m}',task_id=f'm{m}-d{d}',task_variant=f'bottleneck_d{d}'),
                            model={'sha':'frozen'} if arm.startswith('raw_') else None,
                            phase='frozen-stream',budget_seconds=120.,preflight_steps=3,
                            expected_initial='initial',plan={'native':'frozen'},source_folder='frozen'))
    return jobs


def result_rows():
    selected=cli.selected_jobs(source_jobs(),cli.config())
    prior=first.selected_jobs(source_jobs(),first.config())
    result=[]
    for j in prior+selected:
        if j['comparison_arm'] not in cli.ARMS:
            continue
        result.append(dict(j,map_id=j['case']['map_id'],task_id=j['case']['task_id'],
            initial_fingerprint='i',rng_stream_id='r',status='ok',feasible=True,
            success_within_budget=True,delivered_within_budget=True,
            ttf_seconds=8. if j['comparison_arm']=='raw_fast' else 10.,delivery_seconds=11.,
            search_end_seconds=10.,decisions=5,generated=20,soc=100,makespan=20,wait_steps=2,
            final_conflicts=0,stop='feasible',last_pp_failure_reason='none',
            selection_seconds=1.,native_pp_seconds=2.,bookkeeping_seconds=1.))
    return result[:24],result[24:]


class RemainingTimingTests(unittest.TestCase):
    def test_complete_blind_cohort_preserves_model_stream_and_source(self):
        old=source_jobs()
        frozen=deepcopy(old)
        jobs=cli.selected_jobs(old,cli.config())
        self.assertEqual(old,frozen)
        self.assertEqual(len(jobs),72)
        self.assertEqual(jobs,cli.selected_jobs(list(reversed(old)),cli.config()))
        orders=Counter(tuple(j['comparison_arm'] for j in jobs[i:i+2]) for i in range(0,72,2))
        self.assertEqual(orders,{cli.ARMS:18,cli.ARMS[::-1]:18})
        lookup={j['job_id']:j for j in old}
        for j in jobs:
            origin=lookup[j['source_timed_job_id']]
            self.assertNotEqual((j['solver_seed'],j['replica']),(251,0))
            for key in ('pair_id','replica','solver_seed','model','phase','case','plan','source_folder','expected_initial'):
                self.assertEqual(j[key],origin[key])
            self.assertNotEqual(j['job_id'],origin['job_id'])
        # There is no branch on a source outcome or runtime measurement.
        altered=[dict(j,feasible=False,ttf_seconds=9999.) for j in old]
        self.assertEqual([j['job_id'] for j in cli.selected_jobs(altered,cli.config())],
                         [j['job_id'] for j in jobs])
        with self.assertRaises(ValueError):cli.schedule(jobs[:-1])
        with self.assertRaises(ValueError):cli.schedule(jobs+[jobs[0]])
        with self.assertRaises(ValueError):cli.selected_jobs([],cli.config())

    def test_combination_exact_frozen_coverage(self):
        prior,new=result_rows()
        rows=cli.combine_rows(prior,new)
        self.assertEqual(len(rows),96)
        self.assertEqual(len({cli.result_key(r) for r in rows}),96)
        for bad in (new[:-1],new+[new[0]],new[:-1]+[prior[0]]):
            with self.assertRaises(ValueError):cli.combine_rows(prior,bad)
        wrong=deepcopy(new)
        wrong[0]['comparison_arm']='raw_reference'
        with self.assertRaises(ValueError):cli.combine_rows(prior,wrong)
        wrong=deepcopy(new)
        wrong[0]['solver_seed']=999
        with self.assertRaises(ValueError):cli.combine_rows(prior,wrong)
        wrong=deepcopy(new)
        wrong[0]['task_id']='unregistered'
        with self.assertRaises(ValueError):cli.combine_rows(prior,wrong)

    def test_two_arm_summary_is_private_and_deterministic(self):
        prior,new=result_rows()
        data=cli.combine_rows(prior,new)
        frozen=deepcopy(data)
        report=cli.summarize(data,bootstrap=80)
        self.assertEqual(report,cli.summarize(reversed(data),bootstrap=80))
        self.assertEqual(data,frozen)
        self.assertEqual(set(report['arms']),set(cli.ARMS))
        self.assertEqual(set(report['contrasts']),{'dual16_sa'})
        self.assertEqual(report['pairs'],48)
        self.assertEqual(report['by_map']['m0']['pairs'],8)
        c=report['contrasts']['dual16_sa']['common_success']['ttf_seconds']
        self.assertEqual((c['count'],c['wins'],c['ci95']),(48,48,[-2.,-2.]))
        self.assertEqual(len(first.rt.ARMS),3)
        self.assertEqual(len(first.rt._validate.__globals__['ARMS']),3)
        self.assertEqual(cli.summarize.__globals__['ARMS'],cli.ARMS)
        with self.assertRaises(ValueError):cli.summarize(data[:-1],bootstrap=80)
        data[0]['initial_fingerprint']='different'
        with self.assertRaises(ValueError):cli.summarize(data,bootstrap=80)

    def test_failure_never_becomes_zero_time(self):
        prior,new=result_rows()
        row=next(r for r in new if r['comparison_arm']=='dual16_sa')
        row.update(feasible=False,success_within_budget=False,delivered_within_budget=False,
            ttf_seconds=None,final_conflicts=1,stop='deadline',search_end_seconds=120.,delivery_seconds=121.)
        c=cli.summarize(new,bootstrap=80)['contrasts']['dual16_sa']
        self.assertEqual(c['common_success']['ttf_seconds']['count'],35)
        self.assertEqual(c['success']['wins'],1)
        self.assertEqual(c['success']['losses'],0)
        self.assertAlmostEqual(c['success']['difference']['mean'],1/36)

    def test_frozen_config_and_no_caps(self):
        c=cli.config()
        for key,value in (('workers',2),('audit_workers',4),('max_decisions',100),
                          ('execution_node_budget',25000000),('budget_seconds',180.),
                          ('conditions',[[251,0]]),('preflight_steps',0),
                          ('process_fuse_seconds',360.),('bootstrap',1),
                          ('no_training',False),('automatic_promotion',True)):
            with patch.object(run,'read_json',return_value=c|{key:value}):
                with self.assertRaises(ValueError):cli.config()

    def test_baseline_proof_reused_honestly_and_tampering_rejected(self):
        job=dict(job_id='new',source_timed_job_id='old',comparison_arm='dual16_sa',model=None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'proof.json'
            proof=dict(binding='old-binding',job_id='old',status='ok',prefix_steps=3,action_feature_path_equal=True)
            run.write_json(path,run.sealed(proof))
            r=dict(inputs={'proof.json':run.sha256_file(path)},
                   reusable_preflight={'new':dict(path='proof.json',binding='old-binding',job_id='old')})
            with patch.object(cli,'ROOT',root):
                result=cli.reused_preflight(job,r)
                self.assertTrue(result['reused'])
                self.assertEqual(result['native_steps_executed'],0)
                self.assertEqual(result['prefix_steps'],3)
                for invalid in (dict(job,model={}),dict(job,comparison_arm='raw_fast'),dict(job,source_timed_job_id='other')):
                    with self.assertRaises(ValueError):cli.reused_preflight(invalid,r)
                for key,value in (('binding','other'),('prefix_steps',2),('action_feature_path_equal',False),('status','error')):
                    run.write_json(path,run.sealed(proof|{key:value}))
                    r['inputs']['proof.json']=run.sha256_file(path)
                    with self.assertRaises(ValueError):cli.reused_preflight(job,r)
                run.write_json(path,run.sealed(proof))
                with self.assertRaisesRegex(ValueError,'changed prefix'):cli.reused_preflight(job,r)
                invalid=run.sealed(proof)
                invalid['status']='error'
                run.write_json(path,invalid)
                r['inputs']['proof.json']=run.sha256_file(path)
                with self.assertRaises(ValueError):cli.reused_preflight(job,r)

    def test_new_preflight_executes_only_fast_arm(self):
        with patch.object(cli.rt,'preflight_worker',return_value={'prefix_steps':3,'status':'ok'}) as worker:
            result=cli.preflight_worker({'comparison_arm':'raw_fast'})
            self.assertEqual(result['native_steps_executed'],3)
            self.assertFalse(result['reused'])
            with self.assertRaises(ValueError):cli.preflight_worker({'comparison_arm':'dual16_sa'})
            self.assertEqual(worker.call_count,1)

    def test_collector_body_clock_and_original_globals_unchanged(self):
        self.assertIs(cli.collect.__code__,first.collect.__code__)
        self.assertIs(cli.collect.__globals__['verify'],cli.verify)
        self.assertIs(first.collect.__globals__['verify'],first.verify)
        self.assertIs(cli.collect.__globals__['rt'].timed_worker,first.rt.timed_worker)

    def test_safe_stop_resume_without_rerunning_completed_episode(self):
        jobs=[dict(job_id='a',comparison_arm='raw_fast',pair_id='p'),dict(job_id='b',comparison_arm='dual16_sa',pair_id='p')]
        r=dict(binding='x',jobs=jobs,config=cli.config())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            collect=_bind(cli.collect,verify=lambda:(r,out),os=SimpleNamespace(name='posix',environ={}))
            def execute(*args):
                j=args[2][0]
                run.once(out/'collect'/(j['job_id']+'.json'),run.sealed(dict(status='ok',job_id=j['job_id'])))
                if j['job_id']=='a':run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
            with patch.object(cli.source,'check_complete'),patch.object(cli.source,'read_result'), \
                 patch.object(cli.source,'execute',side_effect=execute) as execution:
                result=collect()
                self.assertTrue(result['paused'])
                self.assertEqual(result['completed'],1)
                self.assertEqual(execution.call_count,1)
                with self.assertRaises(ValueError):collect()
                self.assertEqual(collect(True)['collected'],2)
                self.assertEqual(execution.call_count,2)
                self.assertFalse((out/'STOP_AFTER_EPISODE').exists())

    def test_partial_episode_refused_without_automatic_retry(self):
        r=dict(binding='x',jobs=[dict(job_id='a',comparison_arm='raw_fast',pair_id='p')],config=cli.config())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'episodes'/'a').mkdir(parents=True)
            collect=_bind(cli.collect,verify=lambda:(r,out),os=SimpleNamespace(name='posix',environ={}))
            with patch.object(cli.source,'check_complete'),patch.object(cli.source,'execute') as execution:
                with self.assertRaisesRegex(ValueError,'partial attempt'):collect()
                execution.assert_not_called()


if __name__=='__main__':unittest.main()
