from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import sa_completion_independent_timing as rt
from scripts import run_sa_completion_independent_ttf as cli
from tests.evaluation.test_sa_raw_timing_metrics import fixture, fail


def jobs():
    return [dict(job_id=f'{m}-{d}-{s}-{arm}', pair_id=f'm{m:02d}-d{d}-s{s}',replica=0,
        comparison_arm=arm,solver_seed=s,expected_initial=f'{m}-{d}-{s}',phase='fixed',
        plan=dict(config=dict(stream_seed=5)),case=dict(map_id=f'm{m:02d}',task_variant=f'bottleneck_d{d}'))
        for m in range(12) for d in (20,25) for s in (271,277) for arm in rt.ARMS]


def rows():
    names = dict(raw_parent='parent',raw_updated='completion',official_sa='official_sa')
    return [dict(r,comparison_arm=names[r['comparison_arm']],timing_mode=rt.TIMING_MODE,
                 selection_seconds=1.,native_pp_seconds=2.,bookkeeping_seconds=.1,reset_seconds=.2)
            for r in fixture() if r['comparison_arm'] in names]


class IndependentCompletionTests(unittest.TestCase):
    def test_frozen_scope_and_same_lean_engine(self):
        c=cli.config()
        self.assertEqual((c['maps'],c['workers'],c['audit_workers']),(12,1,20))
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertFalse(c['replacement_permitted'])
        self.assertEqual(rt._actor_worker.__code__,rt.lean.lean_worker.__code__)
        self.assertIs(rt._actor_worker.__globals__['Policy'],rt.SharedFeaturePolicy)

    def test_balanced_schedule_each_density_and_seed(self):
        source=jobs()
        frozen=deepcopy(source)
        result=cli.schedule(source)
        self.assertEqual(source,frozen)
        self.assertEqual(result,cli.schedule(list(reversed(source))))
        self.assertEqual(len(result),144)
        triples=[result[i:i+3] for i in range(0,144,3)]
        self.assertEqual(sorted(Counter(tuple(j['comparison_arm'] for j in t) for t in triples).values()),[8]*6)
        for d in (20,25):
            for s in (271,277):
                selected=[t for t in triples if t[0]['solver_seed']==s and t[0]['case']['task_variant']==f'bottleneck_d{d}']
                self.assertEqual(sorted(Counter(tuple(j['comparison_arm'] for j in t) for t in selected).values()),[2]*6)

    def test_incomplete_mismatched_schedule_rejected(self):
        for data in (jobs()[:-1],jobs()+[jobs()[0]]):
            with self.assertRaises(ValueError):cli.schedule(data)
        for key in ('expected_initial','phase'):
            data=jobs()
            data[0][key]='changed'
            with self.assertRaisesRegex(ValueError,'unpaired'):cli.schedule(data)

    def test_official_never_uses_actor_or_standard_acceptance(self):
        j=dict(comparison_arm='official_sa',arm='official_sa',model=None)
        with patch.object(rt.lean,'lean_worker',return_value='official') as worker:
            self.assertEqual(rt.timed_worker(j),'official')
            worker.assert_called_once_with(j)
        for bad in (dict(j,model={'path':'actor'}),dict(j,arm='trained_actor'),dict(j,comparison_arm='official')):
            with self.assertRaises(ValueError):rt.timed_worker(bad)

    def test_three_contrasts_preserve_cohorts_and_no_penalties(self):
        data=rows()
        fail(next(r for r in data if r['comparison_arm']=='parent'))
        saved=deepcopy(data)
        report=rt.summarize(data,100)
        self.assertEqual(data,saved)
        self.assertEqual(report,rt.summarize(reversed(data),100))
        self.assertEqual(set(report['contrasts']),{'completion_vs_parent','completion_vs_official_sa','parent_vs_official_sa'})
        self.assertEqual(report['contrasts']['completion_vs_parent']['success']['wins'],1)
        self.assertEqual(report['contrasts']['completion_vs_parent']['common_success']['ttf_seconds']['count'],1)
        self.assertEqual(report['contrasts']['completion_vs_official_sa']['common_success']['ttf_seconds']['count'],2)
        self.assertFalse(report['failure_time_imputation'])
        self.assertFalse(report['cross_layout_claim'])

    def test_missing_arm_and_mixed_clock_rejected(self):
        with self.assertRaises(ValueError):rt.summarize(rows()[:-1],10)
        data=rows()
        data[0]['timing_mode']='old'
        with self.assertRaisesRegex(ValueError,'timing'):rt.summarize(data,10)

    def test_all_failed_means_remain_null(self):
        data=rows()
        for row in data:fail(row)
        result=rt.summarize(data,10)
        self.assertIsNone(result['contrasts']['completion_vs_parent']['common_success']['ttf_seconds']['completion']['mean'])

    def test_isolation_no_seed_or_geometry_replacement(self):
        data=[dict(map_id=f'm{m}',map_seed=1000+m,task_seed=10000+2*m+i,map_file=f'm{m}',
                   task_variant=('bottleneck_d20','bottleneck_d25')[i]) for m in range(12) for i in range(2)]
        with patch.object(cli.run,'sha256_file',side_effect=lambda p:p.name):
            self.assertEqual(len(cli.check_isolation(data,dict(seeds=[],map_hashes=[]))),12)
            for history in (dict(seeds=[1000],map_hashes=[]),dict(seeds=[],map_hashes=['m0'])):
                with self.assertRaisesRegex(ValueError,'historical overlap'):cli.check_isolation(data,history)

    def test_preserved_errors_and_wrong_receipt_block_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            reg=dict(binding='new',jobs=[dict(job_id='j')])
            cli.run.once(out/'collect/j.json',cli.run.sealed(dict(status='ok',binding='old',job_id='j')))
            with patch.object(cli,'verify',return_value=(reg,out)),patch.object(cli,'_collect') as collect:
                with self.assertRaisesRegex(ValueError,'receipt'):cli.collect(True)
                collect.assert_not_called()
            cli.run.once(out/'failures/collect-j.json',dict(status='error'))
            with patch.object(cli,'verify',return_value=(reg,out)):
                with self.assertRaisesRegex(ValueError,'preserved error'):cli.collect(True)


if __name__=='__main__':unittest.main()
