from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from tests.native_support import isolated_sa_native
from unittest.mock import patch

from experiments.sa_raw_confirmation import ARMS
from experiments.sa_uncapped_runtime import feature_plan
from experiments import sa_raw_timed_runtime as rt
from scripts import run_sa_raw_ttf as cli
from scripts import run_sa_onpolicy as run


class TimedRuntimeTests(unittest.TestCase):
    def test_balanced_order_and_pair_identity(self):
        jobs=[dict(pair_id=f'p{i:02d}',replica=r,comparison_arm=a,job_id=f'{i}-{r}-{a}')
              for i in range(24) for r in range(2) for a in ARMS]
        result=cli.schedule(jobs)
        self.assertEqual(len(result),192)
        self.assertEqual(cli.schedule(result),result)
        orders=Counter(tuple(x['comparison_arm'] for x in result[i:i+4]) for i in range(0,192,4))
        self.assertEqual(len(orders),24)
        self.assertEqual(set(orders.values()),{2})
        for a in ARMS:
            positions=Counter(i%4 for i,j in enumerate(result) if j['comparison_arm']==a)
            self.assertEqual(set(positions.values()),{12})
        with self.assertRaises(ValueError):cli.schedule(jobs[:-1])
        with self.assertRaises(ValueError):cli.schedule(jobs+[jobs[0]])

    def test_no_execution_work_limit_and_frozen_feature_reference(self):
        c=cli.config()
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        p=dict(proposal=dict(max_decisions=None,decision_feature_reference=256,node_budget=25000000))
        original=deepcopy(p)
        fp=feature_plan(p)
        self.assertEqual(p,original)
        self.assertEqual(run.budget_features({},1000,30000000,fp['proposal']),{
            'budget.remaining_decision_fraction':0.,'budget.remaining_node_fraction':0.})

    @isolated_sa_native
    def test_four_frozen_arms_native_micro_and_artifact_audit(self):
        try:import lns2_env
        except ImportError:self.skipTest('frozen WSL native needed')
        from scripts import confirm_sa_raw_independent as source
        old=run.check_seal(run.read_json(run.ROOT/'build/sa-raw-residual-runtime-v1/execution_registration.json'))
        plan=source.runtime_plan(dict(config=source.config(),source_plan=old['source_plan']),'runs')
        q=run.native_runtime(plan)
        class ResetPaths:
            def __init__(self,env):self.env=env
            def reset(self,seed):return self.env.reset_paths([[0,1,2,3],[3,2,1,0]],seed=seed)
            def __getattr__(self,name):return getattr(self.env,name)
        with tempfile.TemporaryDirectory(dir=run.ROOT/'build',prefix='sa-timed-micro-') as tmp:
            root=Path(tmp)
            (root/'tiny.map').write_text('type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n',encoding='utf8')
            (root/'tiny.scen').write_text('version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n',encoding='utf8')
            for a in ARMS:
                env=ResetPaths(lns2_env.LNS2RepairEnv(str(root/'tiny.map'),str(root/'tiny.scen'),2))
                state=q._plain(env.reset(seed=11))
                ctx=dict(case_id='tiny-seed11',task_id='tiny',solver_seed=11,proposal=plan['template']['proposal'])
                i=int(a=='raw_updated')
                spec=source.prior.model_spec(old,i) if a.startswith('raw_') else None
                job=dict(plan=plan,arm='trained_actor' if spec else a,comparison_arm=a,iteration=i,model=spec,
                    phase='timed-micro',pair_id='tiny',replica=0,job_id=a,case=dict(map_id='tiny',task_id='tiny'),
                    output=root.relative_to(run.ROOT).as_posix(),timing_binding='x'*64,budget_seconds=5.,
                    solver_seed=11,expected_initial=q.state_fingerprint(state),parent_pid=os.getppid())
                with patch.object(rt,'prepare_environment',return_value=(q,env,ctx)):
                    rt.timed_worker(job)
                row=run.check_seal(run.read_json(root/'episodes'/a/'result.json'))
                self.assertTrue(row['success_within_budget'])
                self.assertGreaterEqual(row['delivery_seconds'],row['ttf_seconds'])
                self.assertIsNone(row['execution_node_budget'])
                self.assertEqual(rt.audit_worker(job)['status'],'ok')
                for changes in ({'task_id':'wrong'}, {'files':{}}, {'initial_conflicts':999},
                                {'final_conflicts':999}, {'native_pp_seconds':999.}):
                    bad={k:v for k,v in row.items() if k!='integrity'}
                    run.write_json(root/'episodes'/a/'result.json',run.sealed(dict(bad,**changes)))
                    with self.assertRaises(ValueError):rt.audit_worker(job)
                run.write_json(root/'episodes'/a/'result.json',row)
                deadline_job=dict(job,job_id=a+'-deadline',budget_seconds=1e-9)
                with patch.object(rt,'prepare_environment',return_value=(q,env,ctx)):
                    rt.timed_worker(deadline_job)
                deadline=run.check_seal(run.read_json(root/'episodes'/deadline_job['job_id']/'result.json'))
                self.assertFalse(deadline['success_within_budget'])
                self.assertIsNone(deadline['ttf_seconds'])
                self.assertEqual(deadline['decisions'],0)
                self.assertEqual(rt.audit_worker(deadline_job)['status'],'ok')
                with patch.object(rt,'prepare_environment',return_value=(q,env,ctx)):
                    with self.assertRaises(ValueError):rt.timed_worker(deadline_job)
                with (root/'episodes'/a/'final.json').open('a') as f:f.write(' ')
                with self.assertRaises(ValueError):rt.audit_worker(job)


if __name__=='__main__':unittest.main()
