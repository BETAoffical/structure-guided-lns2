from collections import Counter
from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import sa_expanded_timing as rt
from scripts import run_sa_expanded_ttf as cli
from tests.evaluation.test_sa_raw_timing_metrics import fixture, fail


def jobs():
    return [dict(job_id=f'{m}-{d}-{s}-{a}', pair_id=f'm{m:02d}-d{d}-s{s}', replica=0,
        comparison_arm=a, solver_seed=s, expected_initial=f'{m}-{d}-{s}', phase='fixed',
        plan=dict(config=dict(stream_seed=5)), case=dict(map_id=f'm{m:02d}', task_variant=f'bottleneck_d{d}'))
        for m in range(12) for d in (20,25) for s in cli.SEEDS for a in rt.ARMS]


def rows():
    names = dict(raw_parent='parent', raw_updated='completion', dual16_sa='official', official_sa='official_sa')
    return [dict(r, comparison_arm=names[r['comparison_arm']], timing_mode=rt.TIMING_MODE,
                 selection_seconds=1., native_pp_seconds=2., bookkeeping_seconds=.1, reset_seconds=.2)
            for r in fixture()]


class ExpandedTimingTests(unittest.TestCase):
    def test_fixed_scope(self):
        c = cli.config()
        self.assertEqual((c['maps'], c['solver_seeds'], c['workers'], c['audit_workers']), (12, [281,283,293], 1, 20))
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertTrue(c['no_training'])
        self.assertFalse(c['replacement_permitted'])

    def test_schedule_balances_each_density_seed_and_position(self):
        source = jobs()
        saved = deepcopy(source)
        ordered = cli.schedule(source)
        self.assertEqual(saved, source)
        self.assertEqual(ordered, cli.schedule(reversed(source)))
        self.assertEqual(ordered, cli.schedule(ordered))
        self.assertEqual(len(ordered), 288)
        quartets = [ordered[i:i+4] for i in range(0,288,4)]
        self.assertEqual(sorted(Counter(tuple(j['comparison_arm'] for j in g) for g in quartets).values()), [18]*4)
        for d in (20,25):
            for s in cli.SEEDS:
                subset = [g for g in quartets if g[0]['solver_seed']==s and g[0]['case']['task_variant']==f'bottleneck_d{d}']
                for position in range(4):
                    self.assertEqual(Counter(g[position]['comparison_arm'] for g in subset), {a:3 for a in rt.ARMS})

    def test_invalid_or_unpaired_schedule_rejected(self):
        for bad in (jobs()[:-1], jobs()+[jobs()[0]]):
            with self.assertRaises(ValueError):
                cli.schedule(bad)
        for field in ('expected_initial', 'phase', 'solver_seed', 'replica'):
            bad = jobs()
            bad[0][field] = 'changed'
            with self.assertRaises(ValueError):
                cli.schedule(bad)

    def test_dispatch_reuses_frozen_worker_functions(self):
        j = dict(comparison_arm='official', arm='official', model=None)
        with patch.dict(rt.lean.WORKERS, {'official':lambda job:('standard', job)}):
            self.assertEqual(rt.timed_worker(j), ('standard', j))
        for arm in ('parent','completion'):
            j = dict(comparison_arm=arm, arm='trained_actor', model={'sha256':'fixed'})
            with patch.object(rt.old, '_actor_worker', return_value='actor') as worker:
                self.assertEqual(rt.timed_worker(j), 'actor')
                worker.assert_called_once_with(j)
        for j in (dict(comparison_arm='official',arm='official_sa',model=None),
                  dict(comparison_arm='completion',arm='trained_actor',model=None)):
            with self.assertRaises(ValueError):
                rt.identity(j)

    def test_six_contrasts_curves_and_no_failure_imputation(self):
        data = rows()
        fail(next(r for r in data if r['comparison_arm']=='parent'))
        saved = deepcopy(data)
        report = rt.summarize(data, 50)
        self.assertEqual(report, rt.summarize(reversed(data), 50))
        self.assertEqual(data, saved)
        self.assertEqual(len(report['contrasts']), 6)
        self.assertEqual(report['contrasts']['completion_vs_parent']['success']['wins'], 1)
        self.assertEqual(report['four_arm_common_success']['count'], 1)
        self.assertEqual(report['solved_by_seconds']['parent']['120'], 1)
        self.assertEqual(report['solved_by_seconds']['completion']['10'], 2)
        self.assertFalse(report['failure_time_imputation'])
        self.assertFalse(report['cross_layout_claim'])
        for row in data:
            fail(row)
        report = rt.summarize(data, 50)
        self.assertIsNone(report['four_arm_common_success']['arms']['completion']['ttf_seconds']['mean'])

    def test_mismatched_clock_and_missing_pair_rejected(self):
        with self.assertRaises(ValueError):
            rt.summarize(rows()[:-1], 20)
        data = rows()
        data[0]['initial_fingerprint'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            rt.summarize(data, 20)

    def test_reuses_safe_stop_and_receipt_checks(self):
        self.assertIs(cli.collect.__code__, cli.old.collect.__code__)
        self.assertIs(cli._collect.__code__, cli.first.collect.__code__)
        self.assertIs(cli._collect.__globals__['verify'], cli.verify)
        self.assertIs(cli._collect.__globals__['rt'], rt)

    def test_generation_fallback_preserves_identity_and_only_known_error(self):
        md, template = object(), {'frozen':'metadata'}
        tc = {'agent_density':.25}
        error = ValueError('unable to sample valid endpoints for agent 300; reduce agent_count or distance constraints')
        with patch('generators.task_flows.generate_tasks', side_effect=error), patch.object(cli.repair, 'repair_task', return_value='matched') as match:
            self.assertEqual(cli.generate_task(md, tc, 73, 'task', template), ('matched', True, str(error)))
            match.assert_called_once_with(md, tc, 73, 'task', template)
        for density, error in ((.2,error), (.25,ValueError('different error'))):
            with patch('generators.task_flows.generate_tasks', side_effect=error), patch.object(cli.repair, 'repair_task') as match:
                with self.assertRaises(ValueError):
                    cli.generate_task(md, {'agent_density':density}, 73, 'task', template)
                match.assert_not_called()
        with patch('generators.task_flows.generate_tasks', return_value='original'), patch.object(cli.repair, 'repair_task') as match:
            self.assertEqual(cli.generate_task(md, tc, 73, 'task', template), ('original',False,None))
            match.assert_not_called()


if __name__ == '__main__':
    unittest.main()
