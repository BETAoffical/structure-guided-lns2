from copy import deepcopy
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sa_terminal_control_variate_confirmation as s
from scripts import confirm_sa_terminal_control_variate as cli


def rows(prefix='fit'):
    return [dict(episode_id=prefix+str(i), job_id=str(i), rng_stream_id=prefix+'rng'+str(i), replica=i,
        pair_id='p', initial_fingerprint='initial', policy_sha256='A', split='train', status='ok',
        success=i % 3 != 0, score=[float(i+1), float((-1)**i)], decisions=20+i,
        generated=1000+i, stop='feasible' if i % 3 != 0 else 'node_budget') for i in range(16)]


def condition(reference=10., weighted=8.):
    return dict(episodes=16, successes=10, failures=6, mixed=True,
        methods={m: dict(second_moment=v, second_moment_by_replica=[v]*16)
                 for m, v in [('mean', reference), ('score_squared', weighted)]})


class ConfirmationTests(unittest.TestCase):
    def test_prior_batch_only_fit_and_complete_identity(self):
        data = rows()
        frozen = s.freeze_baselines(data)
        y = np.array([r['success'] for r in data], dtype=float)
        energy = np.sum(np.array([r['score'] for r in data])**2, axis=1)
        self.assertEqual(frozen['fit_episodes'], 16)
        self.assertEqual(frozen['baselines']['mean'], float(y.mean()))
        self.assertAlmostEqual(frozen['baselines']['score_squared'], float(y@energy/energy.sum()))
        self.assertEqual(frozen['parameter_count'], 2)

    def test_evaluation_changes_never_refit_the_baselines(self):
        frozen = s.freeze_baselines(rows())
        data = rows('new')
        before = deepcopy(frozen)
        for r in data:
            r['success'] = True
            r['score'] = [1000., -2000.]
        report = s.condition_report(data, frozen, frozen['fit_rng_streams'])
        self.assertEqual(frozen, before)
        for method in ('mean', 'score_squared'):
            self.assertEqual(report['methods'][method]['baseline'], before['baselines'][method])
            self.assertFalse(report['methods'][method]['fits_evaluation_data'])

    def test_dense_contribution_matches_second_moment_dispersion_and_gradient(self):
        frozen, data = s.freeze_baselines(rows()), rows('new')
        report = s.condition_report(data, frozen, [])
        score = np.array([r['score'] for r in data])
        y = np.array([r['success'] for r in data])
        for method, b in frozen['baselines'].items():
            g = -(y-b)[:, None]*score
            expected = report['methods'][method]
            self.assertAlmostEqual(expected['second_moment'], np.mean(np.sum(g*g, axis=1)))
            self.assertAlmostEqual(expected['within_replica_dispersion'],
                                   np.mean(np.sum((g-g.mean(axis=0))**2, axis=1)))
            self.assertAlmostEqual(expected['mean_gradient_norm'], np.linalg.norm(g.mean(axis=0)))

    def test_independent_constant_baselines_preserve_expected_score_gradient(self):
        probabilities = np.array([.25, .5, .25])
        y, score = np.array([0., 1., 1.]), np.array([-1., -1., 3.])
        for b in (0., .030015768859559373, .1875, .6404500494469754, 1.):
            self.assertAlmostEqual(float(probabilities@(-(y-b)*score)), -.25, places=12)

    def test_zero_energy_fit_and_new_zeros_do_not_invent_signal(self):
        data = rows()
        for r in data:
            r['score'] = [0., 0.]
        frozen = s.freeze_baselines(data)
        self.assertEqual(frozen['baselines']['mean'], frozen['baselines']['score_squared'])
        fresh = rows('new')
        for r in fresh:
            r['score'] = [0., 0.]
        report = s.condition_report(fresh, frozen, [])
        self.assertIsNone(report['relative_change'])
        self.assertIsNone(report['gradient_cosine'])
        self.assertIsNone(report['methods']['mean']['largest_episode_moment_share'])

    def test_constant_fit_can_still_detect_new_failures_without_rebaselining(self):
        fit = rows()
        for r in fit:
            r['success'] = True
        frozen = s.freeze_baselines(fit)
        fresh = rows('new')
        report = s.condition_report(fresh, frozen, [])
        self.assertGreater(report['methods']['mean']['second_moment'], 0)
        self.assertEqual(frozen['baselines']['mean'], 1.)
        self.assertEqual(report['failures'], 6)

    def test_partial_heldout_censored_or_mismatched_identity_rejected(self):
        frozen = s.freeze_baselines(rows())
        for key, value in [('split', 'validation'), ('status', 'censored'), ('policy_sha256', 'B'),
                           ('initial_fingerprint', 'bad'), ('pair_id', 'other'), ('episode_id', 'new1'),
                           ('rng_stream_id', 'newrng1'), ('replica', 1), ('score', [float('nan'), 0.]),
                           ('success', .5)]:
            data = rows('new')
            data[0][key] = value
            with self.assertRaises(ValueError):
                s.condition_report(data, frozen, [])
        with self.assertRaises(ValueError):
            s.condition_report(rows('new')[:-1], frozen, [])

    def test_even_homogeneous_new_identity_must_match_frozen_condition(self):
        frozen = s.freeze_baselines(rows())
        for key in ('pair_id', 'initial_fingerprint', 'policy_sha256'):
            data = rows('new')
            for r in data:
                r[key] = 'different'
            with self.assertRaises(ValueError):
                s.condition_report(data, frozen, [])

    def test_fitted_episode_or_random_stream_overlap_rejected(self):
        frozen = s.freeze_baselines(rows())
        for key, value in [('episode_id', 'fit0'), ('rng_stream_id', 'fitrng0')]:
            data = rows('new')
            data[0][key] = value
            with self.assertRaises(ValueError):
                s.condition_report(data, frozen, [])
        with self.assertRaises(ValueError):
            s.condition_report(rows('new'), frozen, ['newrng5'])

    def test_frozen_schema_dimension_and_baseline_checks(self):
        frozen = s.freeze_baselines(rows())
        for key, value in [('fit_episodes', 15), ('parameter_count', 3),
                           ('baselines', {'mean': .1}),
                           ('baselines', {'mean': .1, 'score_squared': float('inf')}),
                           ('baselines', {'mean': .1, 'score_squared': -1.})]:
            changed = deepcopy(frozen)
            changed[key] = value
            with self.assertRaises(ValueError):
                s.condition_report(rows('new'), changed, [])

    def test_zero_reference_is_not_silently_dropped(self):
        self.assertTrue(s.condition_safe(condition(0., 0.), .1))
        self.assertFalse(s.condition_safe(condition(0., 1e-25), .1))
        report = s.summarize({str(i): condition(0., 0.) for i in range(4)})
        self.assertFalse(report['confirmation_passed'])
        self.assertIsNone(report['condition_equal_second_moment']['relative_change'])
        self.assertEqual(report['conditional_replica_bootstrap']['undefined_replicates'], 2000)

    def test_condition_guard_applies_even_when_new_returns_are_constant(self):
        conditions = {str(i): condition() for i in range(4)}
        conditions['0'] = condition(1., 1.2)
        conditions['0']['mixed'] = False
        result = s.summarize(conditions)
        self.assertFalse(result['confirmation_passed'])
        self.assertFalse(result['condition_safety']['0'])

    def test_exact_threshold_no_rounding_and_no_promotion(self):
        cases = {str(i): condition(10., 9.) for i in range(4)}
        # Floating point evaluates 9/10 - 1 above -.1; do not round a near miss into a pass.
        report = s.summarize(cases)
        self.assertEqual(report['confirmation_passed'], report['condition_equal_second_moment']['relative_change'] <= -.1)
        for c in cases.values():
            c['methods']['score_squared'].update(second_moment=8.9, second_moment_by_replica=[8.9]*16)
        report = s.summarize(cases)
        self.assertTrue(report['confirmation_passed'])
        self.assertTrue(report['no_training'] and report['no_ttf'] and report['no_promotion'] and report['no_refit'])
        self.assertTrue(report['not_solver_performance'])

    def test_bootstrap_paired_within_fixed_condition_and_deterministic(self):
        cases = {str(i): condition(10.+i, 8.+i) for i in range(4)}
        a, b = s.summarize(cases), s.summarize(cases)
        self.assertEqual(a, b)
        ci = a['conditional_replica_bootstrap']['relative_change_ci95']
        self.assertAlmostEqual(ci[0], a['condition_equal_second_moment']['relative_change'])
        self.assertAlmostEqual(ci[1], a['condition_equal_second_moment']['relative_change'])

    def test_iteration_runtime_and_work_never_weight_scientific_moment(self):
        frozen, data = s.freeze_baselines(rows()), rows('new')
        before = s.condition_report(data, frozen, [])
        for r in data:
            r.update(decisions=10000, runtime=100000., generated=50000000)
        after = s.condition_report(data, frozen, [])
        self.assertEqual(before['methods'], after['methods'])
        self.assertEqual(before['relative_change'], after['relative_change'])

    def test_platform_tolerance_never_relaxes_identity_or_decision(self):
        s.compare_derived({'x': 1., 'decision': 'keep', 'flag': True},
                          {'x': 1.+1e-12, 'decision': 'keep', 'flag': True})
        for a, b in [({'x': 2.}, {'x': 1.}), ({'flag': 1}, {'flag': True}),
                     ({'decision': 'pass'}, {'decision': 'keep'}), ([1.], [1., 2.])]:
            with self.assertRaises(ValueError):
                s.compare_derived(a, b)

    def test_round_robin_phase_changes_all_job_ids(self):
        conditions = [dict(pair_id=str(i)) for i in range(4)]
        a = cli.source.schedule(conditions, 'fit', 16)
        b = cli.source.schedule(conditions, 'new', 16)
        self.assertEqual([j['pair_id'] for j in b[:4]], ['0', '1', '2', '3'])
        self.assertEqual(len({j['job_id'] for j in b}), 64)
        self.assertFalse({j['job_id'] for j in a} & {j['job_id'] for j in b})

    def test_fixed_scope_uncapped_budget_and_no_heldout_lane(self):
        cfg = cli.configuration()
        self.assertIsNone(cfg['max_decisions'])
        self.assertEqual(cfg['workers'], 20)
        self.assertEqual(cfg['maximum_updates_per_arm'], 0)
        self.assertFalse(cfg['formal_ttf'] or cfg['automatic_promotion'])
        with self.assertRaises(ValueError):
            cli.jobs_for({}, 'validation')
        with self.assertRaises(ValueError):
            cli.model_spec({}, 'challenger')
        report = cli.dry_run_body(dict(binding='test', train_jobs=[{}]*64,
            baselines={str(i): {'baselines': {'mean': .5, 'score_squared': .4}} for i in range(4)}))
        self.assertEqual(report['total_node_boundary'], 1600000000)
        self.assertEqual(report['process_fuse_batch_upper_seconds'], 3840.)

    def test_source_functions_not_patched_by_binding(self):
        old = cli.source.prior.old.jobs_for.__globals__['configuration']
        reg = dict(config=cli.configuration(), parent={'iteration': 2}, train_jobs=[], plans={})
        self.assertEqual(cli.jobs_for(reg), [])
        self.assertIs(cli.source.prior.old.jobs_for.__globals__['configuration'], old)

    def test_help_requires_no_files_or_native(self):
        with patch('sys.argv', ['confirm', '--help']), patch.object(cli, 'configuration', side_effect=AssertionError('I/O')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
