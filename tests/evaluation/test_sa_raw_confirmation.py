from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from tests.native_support import isolated_sa_native
from unittest.mock import patch

from experiments.sa_raw_confirmation import ARMS, schedule, summarize
from experiments import sa_uncapped_runtime as runtime
from scripts import confirm_sa_raw_independent as cli
from scripts import run_sa_onpolicy as run


def fixture():
    conditions = [dict(pair_id=f'p{i}',split='independent_confirmation',solver_seed=251,
                       case=dict(map_id=f'm{i}',task_id=f't{i}')) for i in range(6)]
    rows = []
    for j in schedule(conditions,'phase',2):
        rows.append(dict(j,comparison_arm=j['comparison_arm'],map_id=j['case']['map_id'],status='ok',
            stop='feasible',success=True,final_conflicts=0,initial_fingerprint=j['pair_id'],
            rng_stream_id=f"{j['pair_id']}-{j['replica']}",generated=100,decisions=10,soc=20,makespan=10,wait_steps=2))
    return conditions,rows


class ConfirmationTests(unittest.TestCase):
    def test_four_arms_paired_streams_and_never_train(self):
        roots,_ = fixture()
        jobs = schedule(roots,'new',2)
        self.assertEqual(len({j['job_id'] for j in jobs}),48)
        for start in range(0,48,4):
            group=jobs[start:start+4]
            self.assertEqual([j['comparison_arm'] for j in group],list(ARMS))
            for purpose in ('select','pp','accept'):
                self.assertEqual(len({run.stream_draw(dict(config=dict(stream_seed=1)),j['phase'],j['pair_id'],
                    j['replica'],100000,purpose) for j in group}),1)
        for split in ('train','test','validation','test_ood'):
            with self.assertRaises(ValueError):schedule([dict(roots[0],split=split)],'x',2)
        with self.assertRaises(ValueError):schedule([roots[0],roots[0]],'x',2)

    def test_summary_all_denominators_and_common_success(self):
        roots,rows=fixture()
        rows[0].update(success=False,stop='node_budget',final_conflicts=2,generated=25000100)
        report=summarize(rows,roots,2,bootstrap=100)
        self.assertEqual(report['success']['raw_parent'],11)
        self.assertEqual(report['success']['raw_updated'],12)
        contrast=report['contrasts']['raw_parent']
        self.assertEqual(contrast['gains'],[['p0',0]])
        self.assertEqual(contrast['common_success'],11)
        self.assertEqual(contrast['common_totals']['raw_parent']['generated'],1100)
        self.assertTrue(report['no_training'] and report['no_ttf'] and report['no_promotion'])
        self.assertEqual(report,summarize(rows,roots,2,bootstrap=100))

    def test_missing_duplicate_censored_or_unpaired_refused(self):
        roots,rows=fixture()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):summarize(bad,roots,2,bootstrap=10)
        for field,value in [('initial_fingerprint','wrong'),('rng_stream_id','wrong'),('map_id','wrong'),
                            ('status','censored'),('stop','wall_safety'),('final_conflicts',1)]:
            altered=deepcopy(rows)
            altered[0][field]=value
            with self.assertRaises(ValueError):summarize(altered,roots,2,bootstrap=10)

    def test_baseline_bypasses_actor_scope_and_plan_unchanged(self):
        for arm in ('dual16_sa','official_sa'):
            with patch.object(cli,'actor_scope',side_effect=AssertionError('raw used')):
                with cli.scope(dict(comparison_arm=arm,arm=arm,model=None)):pass
            with self.assertRaises(ValueError):cli.scope(dict(comparison_arm=arm,arm='trained_actor',model=None))
        source=dict(config=dict(output='old'),proposal=dict(max_decisions=0),binding='old')
        frozen=deepcopy(source)
        result=cli.runtime_plan(dict(config=cli.config(),source_plan=source),'runs')
        self.assertEqual(source,frozen)
        self.assertIsNone(result['proposal']['max_decisions'])
        self.assertIsNone(runtime.work_stop(False,10**9,100,result['proposal']))

    @isolated_sa_native
    def test_frozen_native_four_arms_micro(self):
        try:import lns2_env
        except ImportError:self.skipTest('frozen WSL native required')
        old=run.check_seal(run.read_json(run.ROOT/'build/sa-raw-residual-runtime-v1/execution_registration.json'))
        cfg=cli.config()
        p=cli.runtime_plan(dict(config=cfg,source_plan=old['source_plan']),'runs')
        q=run.native_runtime(p)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'tiny.map').write_text('type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n',encoding='utf8')
            (root/'tiny.scen').write_text('version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n',encoding='utf8')
            for arm in ARMS:
                env=lns2_env.LNS2RepairEnv(str(root/'tiny.map'),str(root/'tiny.scen'),2)
                state=q._plain(env.reset_paths([[0,1,2,3],[3,2,1,0]],seed=11))
                i=int(arm=='raw_updated')
                spec=cli.prior.model_spec(old,i) if arm.startswith('raw_') else None
                job=dict(plan=p,arm='trained_actor' if spec else arm,comparison_arm=arm,iteration=i,model=spec,
                    phase='micro',pair_id='tiny',replica=0,job_id=arm,split='independent_confirmation',
                    comparison_binding='x'*64,case=dict(map_id='tiny'))
                ctx=dict(case_id='tiny-seed11',task_id='tiny',solver_seed=11,proposal=p['template']['proposal'])
                with cli.scope(job):
                    bundle=cli.prior.load_model(job) if spec else None
                    row=runtime.episode_loop(job,q,env,state,ctx,root/arm,bundle)
                self.assertTrue(row['success'])
                self.assertEqual(row['stop'],'feasible')


if __name__=='__main__':unittest.main()
