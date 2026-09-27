from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import sa_terminal_efficiency_timing as rt
from scripts import run_sa_terminal_efficiency_ttf as cli
from tests.evaluation.test_sa_raw_timing_metrics import fixture, fail


def jobs():
    result = []
    for m in range(6):
        for density in (20, 25):
            for seed in (251, 257):
                for replica in (0, 1):
                    for arm in rt.ARMS:
                        pair = f'm{m}-d{density}-s{seed}'
                        result.append(dict(job_id=f'{pair}-{replica}-{arm}', pair_id=pair,
                            comparison_arm=arm, solver_seed=seed, replica=replica,
                            expected_initial=pair, phase='frozen',
                            plan=dict(config=dict(stream_seed=12)),
                            case=dict(map_id=f'm{m}', task_variant=f'bottleneck_d{density}')))
    return result


def rows():
    result = []
    for row in fixture():
        arm = {'raw_parent': 'parent', 'raw_updated': 'completion'}.get(row['comparison_arm'])
        if arm:
            result.append(dict(row, comparison_arm=arm, timing_mode=rt.TIMING_MODE,
                               selection_seconds=1., native_pp_seconds=2.,
                               bookkeeping_seconds=.1, reset_seconds=.2))
    return result


class TerminalEfficiencyTimingTests(unittest.TestCase):
    def test_scope_frozen_no_step_or_node_cap(self):
        c = cli.config()
        self.assertEqual(c['workers'], 1)
        self.assertEqual(c['audit_workers'], 20)
        self.assertEqual(c['budget_seconds'], 120.)
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertEqual(c['comparison_arms'], list(rt.ARMS))
        self.assertFalse(c['automatic_promotion'])
        self.assertEqual(rt._timed.__code__, rt.lean.lean_worker.__code__)
        self.assertIs(rt._timed.__globals__['Policy'], rt.SharedFeaturePolicy)

    def test_balanced_complete_schedule_pure_and_deterministic(self):
        source = jobs()
        copy = deepcopy(source)
        ordered = cli.schedule(source)
        self.assertEqual(source, copy)
        self.assertEqual(ordered, cli.schedule(list(reversed(source))))
        self.assertEqual(len(ordered), 96)
        self.assertEqual(Counter(j['comparison_arm'] for j in ordered[::2]), {'parent':24, 'completion':24})
        for m in range(6):
            first = [j['comparison_arm'] for j in ordered[::2] if j['case']['map_id'] == f'm{m}']
            self.assertEqual(Counter(first), {'parent':4, 'completion':4})

    def test_schedule_rejects_missing_duplicate_or_unpaired(self):
        with self.assertRaises(ValueError):
            cli.schedule(jobs()[:-1])
        with self.assertRaises(ValueError):
            cli.schedule(jobs()+[jobs()[0]])
        for field, value in (('expected_initial', 'other'), ('phase', 'other')):
            source = jobs()
            source[0][field] = value
            with self.assertRaises(ValueError):
                cli.schedule(source)
        source = jobs()
        source[0]['plan']['config']['stream_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'stream'):
            cli.schedule(source)

    def test_runtime_rejects_unknown_arm(self):
        for worker in (rt.timed_worker, rt.preflight_worker, rt.audit_worker):
            with self.assertRaisesRegex(ValueError, 'unknown'):
                worker({'comparison_arm':'completion_work'})

    def test_admission_does_not_require_irrelevant_pair_worker(self):
        self.assertFalse(hasattr(rt, 'pair_worker'))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            reg = {'binding':'test', 'config':cli.config(), 'jobs':[{'job_id':'one'}]}
            def execute(r, output, js, worker, phase, workers, timeout):
                self.assertIs(worker, rt.preflight_worker)
                self.assertEqual((phase, workers), ('preflight', 20))
                result = {'status':'ok', 'job_id':'one', 'prefix_steps':3}
                cli.run.once(output/phase/'one.json', cli.run.sealed(result))
                return [result]
            with patch.object(cli, 'verify', return_value=(reg,out)), \
                 patch.object(cli.first.source, 'execute', side_effect=execute):
                self.assertEqual(cli.phase('preflight')['verified'], 1)
            self.assertTrue((out/'preflight.complete.json').is_file())
        self.assertTrue(cli.phase('pair')['skipped'])

    def test_model_loading_requires_frozen_sha_and_iteration(self):
        from scripts.run_sa_raw_residual import load_model
        from scripts import run_sa_raw_residual as raw
        spec = {'path':'dummy', 'sha256':'sha', 'policy_sha256':'policy'}
        job = {'model':spec, 'iteration':2}
        with patch.object(raw.run, 'contained_file', return_value='dummy'), \
             patch.object(raw.run, 'sha256_file', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'bytes'):
                load_model(job)
        with patch.object(raw.run, 'contained_file', return_value='dummy'), \
             patch.object(raw.run, 'sha256_file', return_value='sha'), \
             patch.object(raw.run, 'read_json', return_value={'iteration':1}), \
             patch.object(raw.raw, 'validate_bundle', return_value='policy'):
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_model(job)

    def test_raw_ttf_and_modeled_completion_are_distinct(self):
        data = rows()
        for row in data:
            if row['comparison_arm'] == 'completion':
                row.update(ttf_seconds=8., search_end_seconds=8.5, delivery_seconds=10., makespan=25)
        saved = deepcopy(data)
        result = rt.summarize(data, 100)
        self.assertEqual(data, saved)
        self.assertEqual(result, rt.summarize(reversed(data), 100))
        contrast = result['contrast']
        self.assertEqual(contrast['common_success']['ttf_seconds']['difference']['mean'], -2.)
        self.assertEqual(contrast['modeled_completion']['1.0']['difference']['mean'], 3.)
        self.assertEqual(result['bootstrap_unit'], 'map')
        self.assertFalse(result['failure_time_imputation'])

    def test_failure_not_zero_ttf_or_executable_quality(self):
        data = rows()
        fail(next(r for r in data if r['comparison_arm'] == 'completion'))
        result = rt.summarize(data, 100)
        self.assertEqual(result['arms']['completion']['success_count'], 1)
        contrast = result['contrast']
        self.assertEqual(contrast['success']['losses'], 1)
        self.assertEqual(contrast['common_success']['ttf_seconds']['count'], 1)
        self.assertEqual(contrast['common_success']['soc']['completion']['mean'], 100)
        self.assertEqual(contrast['common_delivered_count'], 1)

    def test_missing_pair_or_mixed_timing_rejected(self):
        with self.assertRaises(ValueError):
            rt.summarize(rows()[:-1])
        data = rows()
        data[0]['timing_mode'] = 'online_audit'
        with self.assertRaisesRegex(ValueError, 'timing'):
            rt.summarize(data)
        data = rows()
        data[0]['rng_stream_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            rt.summarize(data)

    def test_empty_common_success_stays_null(self):
        data = rows()
        for row in data:
            fail(row)
        result = rt.summarize(data, 10)
        self.assertIsNone(result['contrast']['common_success']['ttf_seconds']['completion']['mean'])
        self.assertEqual(result['contrast']['common_success']['ttf_seconds']['ci95'], [None, None])


if __name__ == '__main__':
    unittest.main()
