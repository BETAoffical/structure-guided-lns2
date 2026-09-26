from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import sa_shared_feature_timing as rt
from experiments import sa_raw_timed_runtime as reference
from experiments.sa_raw_selection_fast import FastPolicy
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_shared_feature_ttf as cli
from scripts import run_sa_raw_fast_ttf as first
from tests.evaluation.test_sa_raw_fast_timing import old_jobs, rows as old_rows


def rows():
    data = old_rows()
    for row in list(data):
        if row['comparison_arm'] == 'dual16_sa':
            for name in ('official_sa', 'official'):
                data.append(dict(row, comparison_arm=name, job_id=row['job_id']+'-'+name))
    return data


class SharedTimingTests(unittest.TestCase):
    def test_balanced_frozen_cohort_and_identity(self):
        source = old_jobs()
        saved = deepcopy(source)
        jobs = cli.selected_jobs(source, cli.config())
        self.assertEqual(source, saved)
        self.assertEqual(len(jobs), 240)
        orders = Counter(tuple(j['comparison_arm'] for j in jobs[i:i+5]) for i in range(0, 240, 5))
        self.assertEqual(len(orders), 10)
        self.assertEqual(set(orders.values()), {4, 5})
        self.assertEqual({j['runtime_variant'] for j in jobs}, {'existing_fast', 'shared_features', 'reference', 'standard', 'annealed'})
        prior_ids = {j['job_id'] for j in first.selected_jobs(source, first.config())}
        self.assertTrue(prior_ids.isdisjoint(j['job_id'] for j in jobs))
        for i in range(0, 240, 5):
            g = {j['comparison_arm']: j for j in jobs[i:i+5]}
            for key in ('model', 'case', 'phase', 'pair_id', 'solver_seed', 'replica', 'source_timed_job_id'):
                self.assertEqual(g['raw_reference'][key], g['raw_fast'][key])
            self.assertEqual(g['official']['arm'], 'official')
            self.assertIsNone(g['official']['model'])
        registration = dict(jobs=jobs, binding='test-binding')
        self.assertEqual(len(cli.phase_jobs(registration, 'pair')), 48)
        with self.assertRaises(ValueError):
            cli.selected_jobs(source[:-4], cli.config())

    def test_private_policy_binding_preserves_clock_and_baseline(self):
        for arm, policy in (('raw_reference', FastPolicy), ('raw_fast', SharedFeaturePolicy)):
            self.assertIs(rt.WORKERS[arm].__code__, reference.timed_worker.__code__)
            self.assertIs(rt.PREFLIGHTS[arm].__code__, reference.preflight_worker.__code__)
            self.assertIs(rt.WORKERS[arm].__globals__['Policy'], policy)
            self.assertIs(rt.PREFLIGHTS[arm].__globals__['Policy'], policy)
        self.assertIs(rt.WORKERS['dual16_sa'], reference.timed_worker)
        self.assertIs(reference.timed_worker.__globals__['Policy'], reference.Policy)
        self.assertIs(cli.collect.__code__, first.collect.__code__)
        self.assertIs(cli.collect.__globals__['rt'], rt)
        self.assertIs(cli.phase.__globals__['rt'], rt)
        self.assertIs(rt._standard_audit.__globals__['Policy'], rt.OfficialPolicy)
        self.assertIs(rt._standard_audit.__globals__['run'].native_runtime, rt.standard_native)
        self.assertIs(cli.main.__globals__['collect'], cli.collect)

    def test_dispatch_and_no_unknown_arm(self):
        for arm in rt.ARMS:
            with patch.dict(rt.WORKERS, {arm: lambda j: ('worker', j['comparison_arm'])}), \
                 patch.dict(rt.PREFLIGHTS, {arm: lambda j: ('prefix', j['comparison_arm'])}), \
                 patch.object(rt, 'official_preflight', return_value=('prefix', 'official')):
                self.assertEqual(rt.timed_worker({'comparison_arm': arm}), ('worker', arm))
                self.assertEqual(rt.preflight_worker({'comparison_arm': arm}), ('prefix', arm))
        with self.assertRaises(ValueError):
            rt.timed_worker({'comparison_arm': 'raw_updated'})

    def test_labels_and_statistics_remain_explicit(self):
        report = rt.summarize(rows(), 50, 123)
        self.assertEqual(report, rt.summarize(reversed(rows()), 50, 123))
        self.assertEqual(report['runtime_labels'], rt.LABELS)
        self.assertTrue(report['reference_is_existing_fast_policy'])
        self.assertEqual(report['contrasts']['raw_reference']['common_success']['ttf_seconds']['count'], 12)
        self.assertEqual(report['contrasts']['raw_reference']['common_success']['ttf_seconds']['ci95'], [-2., -2.])
        self.assertEqual(set(report['arms']), set(rt.ARMS))
        self.assertEqual(report['official_sa_vs_official']['target'], 'official_sa')
        self.assertEqual(report['official_sa_vs_official']['success']['difference']['mean'], 0.)

    def test_config_rejects_caps_parallel_timer_and_changed_scope(self):
        c = cli.config()
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        for key, value in (('workers', 20), ('max_decisions', 100), ('solver_seeds', [257]),
                           ('budget_seconds', 600.), ('automatic_promotion', True)):
            with patch.object(cli.run, 'read_json', return_value=c | {key: value}):
                with self.assertRaises(ValueError):
                    cli.config()

    def test_changed_registered_file_refused(self):
        c = cli.config()
        body = dict(config=c, runtime_labels=rt.LABELS, inputs={'some.py': 'frozen'}, jobs=[])
        body['binding'] = cli.run.json_fingerprint(body)
        with patch.object(cli, 'config', return_value=c), \
             patch.object(cli.run, 'read_json', return_value=cli.run.sealed(body)), \
             patch.object(cli.run, 'contained_file', return_value='some.py'), \
             patch.object(cli.run, 'sha256_file', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed shared timing input'):
                cli.verify()

    def test_standard_baseline_never_calls_sa_or_changes_action(self):
        from unittest.mock import Mock
        env = Mock()
        env.step_with_time_limit.return_value = {'observation': 'after', 'metrics': {}}
        q = SimpleNamespace(_plain=lambda x: x, temperature=lambda d: 7.)
        job = dict(plan={}, phase='p', pair_id='id', replica=0)
        with patch.object(rt.run, 'stream_draw', return_value=.2):
            self.assertEqual(rt.standard_transition(job, q, env, {}, {'action': {'mode': 'official'}}, 2, 9.),
                             ('after', {}, 7., .2))
        env.step_with_time_limit.assert_called_once_with({'mode': 'official'}, 9.)
        env.step_experimental_pp.assert_not_called()
        native = Mock()
        wrapper = rt.StandardRuntime(native)
        wrapper.validate_transition('before', 'after', {}, [], 'annealed', 7., .2)
        native.validate_transition.assert_called_once_with('before', 'after', {}, [], 'standard', 7., .2)
        with self.assertRaisesRegex(ValueError, 'SA leaked'):
            wrapper.validate_transition({}, {}, {'experimental_acceptance': 'annealed'}, [], 'annealed', 7., .2)

    def test_same_timer_and_safe_stop_collector_body_for_all_arms(self):
        for name, worker in rt.WORKERS.items():
            self.assertIs(worker.__code__, reference.timed_worker.__code__, name)
        self.assertIs(rt.WORKERS['official'].__globals__['transition'], rt.standard_transition)
        self.assertIs(rt.WORKERS['official_sa'].__globals__['transition'], reference.transition)
        self.assertIs(cli.collect.__globals__['rt'], rt)
        self.assertIs(cli.main.__globals__['phase'], cli.phase)


if __name__ == '__main__':
    unittest.main()
