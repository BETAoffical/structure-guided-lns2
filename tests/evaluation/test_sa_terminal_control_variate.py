from copy import deepcopy
import contextlib
import io
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sa_terminal_control_variate as s
from scripts import audit_sa_terminal_control_variate as cli


def rows(count=4):
    return [dict(episode_id=str(i), rng_stream_id='rng' + str(i), replica=i, pair_id='p',
                 initial_fingerprint='initial', policy_sha256='A', split='train', status='ok',
                 success=i % 3 != 0, score=[float(i + 1), float((-1)**i)], decisions=20+i)
            for i in range(count)]


class ControlVariateTests(unittest.TestCase):
    def test_all_partitions_complete_equal_and_complement_unique(self):
        masks = s.balanced_partitions(16)
        self.assertEqual(masks.shape, (6435, 16))
        self.assertTrue(masks[:, 0].all())
        self.assertTrue((masks.sum(axis=1) == 8).all())
        self.assertEqual(len({tuple(m) for m in masks}), 6435)
        self.assertEqual(s.balanced_partitions(4).shape, (3, 4))
        with self.assertRaises(ValueError):
            s.balanced_partitions(3)

    def test_weighted_baseline_formula_zero_energy_fallback_and_invalid_energy(self):
        self.assertEqual(s.estimate_baseline([1, 0], [1, 9], 'score_squared'), .1)
        self.assertEqual(s.estimate_baseline([1, 0], [1, 9], 'mean'), .5)
        self.assertEqual(s.estimate_baseline([1, 0], [0, 0], 'score_squared'), .5)
        for e in ([1, -1], [1, float('inf')]):
            with self.assertRaises(ValueError):
                s.estimate_baseline([1, 0], e, 'score_squared')

    def test_evaluated_half_cannot_fit_its_own_baseline(self):
        masks = np.array([[True, True, False, False]])
        first, _ = s.crossfit_baselines([1, 0, 1, 0], [1, 3, 2, 8], masks, 'score_squared')
        changed, _ = s.crossfit_baselines([0, 1, 1, 0], [100, 9, 2, 8], masks, 'score_squared')
        np.testing.assert_array_equal(first[:, :2], changed[:, :2])
        self.assertFalse(np.array_equal(first[:, 2:], changed[:, 2:]))
        np.testing.assert_allclose(first, [[.2, .2, .25, .25]])

    def test_crossfit_unbiased_on_exhaustive_independent_toy(self):
        # Each draw has E[score]=0 and E[return*score]=.25; the other half is independent.
        probabilities = np.array([.25, .5, .25])
        y, score = np.array([0., 1., 1.]), np.array([-1., -1., 3.])
        masks = s.balanced_partitions(4)
        for method in ('mean', 'score_squared'):
            estimate = 0.
            for indices in product(range(3), repeat=4):
                ix = np.array(indices)
                baseline, _ = s.crossfit_baselines(y[ix], score[ix]**2, masks, method)
                gradient = np.mean(-(y[ix][None, :] - baseline)*score[ix][None, :])
                estimate += float(np.prod(probabilities[ix]))*gradient
            self.assertAlmostEqual(estimate, -.25, places=12)

    def test_gram_metrics_equal_dense_whole_trajectory_computation(self):
        score = np.array([[1., 2.], [-2., 3.], [4., 5.], [1., -2.]])
        a = np.array([[1., -.5, .2, -.4], [.8, -.2, .5, -.1]])
        ref = -a[0]/4
        report = s.coefficient_metrics(a, score@score.T, ref)
        contributions = -a[:, :, None]*score[None, :, :]
        full = contributions.mean(axis=1)
        second = np.mean(np.sum(contributions**2, axis=2), axis=1)
        dispersion = np.mean(np.sum((contributions-full[:, None, :])**2, axis=2), axis=1)
        self.assertAlmostEqual(report['second_moment_mean'], float(second.mean()), places=12)
        self.assertAlmostEqual(report['within_replica_dispersion_mean'], float(dispersion.mean()), places=12)
        self.assertAlmostEqual(report['partition_gradient_dispersion'],
                               float(np.mean(np.sum((full-full.mean(axis=0))**2, axis=1))), places=12)

    def test_constant_terminal_returns_and_zero_scores_do_not_fabricate_direction(self):
        data = rows(16)
        for outcome in (True, False):
            for r in data:
                r['success'] = outcome
            report = s.condition_report(data)
            self.assertFalse(report['mixed'])
            self.assertEqual(report['crossfit']['score_squared']['second_moment_mean'], 0.)
            self.assertIsNone(report['weighted_vs_mean_second_moment_change'])
            self.assertEqual(report['crossfit']['score_squared']['uninformative_directions'], 6435)
        data = rows()
        for r in data:
            r['score'] = [0., 0.]
        report = s.condition_report(data)
        self.assertEqual(report['crossfit']['score_squared']['zero_energy_fits'], 6)
        self.assertEqual(report['loro']['mean_partition_gradient_norm'], 0.)

    def test_oracle_only_an_in_sample_lower_bound_and_not_deployable(self):
        report = s.condition_report(rows())
        self.assertTrue(report['oracle']['uses_evaluation_outcomes'])
        self.assertTrue(report['oracle']['not_deployable'])
        self.assertLessEqual(report['oracle']['second_moment_mean'],
                             report['crossfit']['mean']['second_moment_mean'])
        y, score = s.validate_rows(rows())
        for b in np.linspace(0, 1, 101):
            value = float(np.mean((y-b)**2*np.sum(score**2, axis=1)))
            self.assertGreaterEqual(value+1e-12, report['oracle']['second_moment_mean'])

    def test_partition_mean_control_reproduces_reference_loro(self):
        y = np.array([r['success'] for r in rows(16)], dtype=float)
        energy = np.arange(1., 17.)
        baseline, _ = s.crossfit_baselines(y, energy, s.balanced_partitions(16), 'mean')
        np.testing.assert_allclose(baseline.mean(axis=0), (y.sum()-y)/15, atol=1e-12)

    def test_complete_score_not_length_cost_or_runtime_weighted(self):
        data = rows()
        reference = s.condition_report(data)
        for i, r in enumerate(data):
            r.update(decisions=773+i, generated=99999999, runtime=9999., makespan=10000)
        self.assertEqual(reference, s.condition_report(data))

    def test_train_identity_completeness_and_score_sanity(self):
        for key, value in [('split', 'validation'), ('status', 'censored'),
                           ('policy_sha256', 'other'), ('initial_fingerprint', 'changed'),
                           ('episode_id', '1'), ('rng_stream_id', 'rng1'), ('replica', 1),
                           ('score', [float('nan'), 1.])]:
            changed = rows()
            changed[0][key] = value
            with self.assertRaises(ValueError):
                s.condition_report(changed)
        with self.assertRaises(ValueError):
            s.condition_report(rows()[:3])

    def test_partition_enumeration_order_independence_with_numeric_tolerance(self):
        a, b = s.condition_report(rows()), s.condition_report(rows()[::-1])
        for method in ('mean', 'score_squared'):
            for metric in ('second_moment_mean', 'within_replica_dispersion_mean',
                           'partition_gradient_dispersion', 'mean_partition_gradient_norm'):
                self.assertAlmostEqual(a['crossfit'][method][metric], b['crossfit'][method][metric], places=12)

    def test_thresholds_are_descriptive_and_oracle_does_not_gate(self):
        condition = dict(episodes=16, successes=12, mixed=True,
            crossfit={'mean': {'second_moment_mean': 10.}, 'score_squared': {'second_moment_mean': 9.5}},
            weighted_vs_mean_second_moment_change=-.05, oracle={'second_moment_mean': 0.})
        conditions = {str(i): {'fresh': deepcopy(condition)} for i in range(4)}
        report = s.summarize(conditions, .1, .1)
        self.assertFalse(report['descriptive_signal'])
        self.assertTrue(report['no_training'] and report['no_promotion'])
        for c in conditions.values():
            c['fresh']['crossfit']['score_squared']['second_moment_mean'] = 8.
            c['fresh']['weighted_vs_mean_second_moment_change'] = -.2
        report = s.summarize(conditions, .1, .1)
        self.assertTrue(report['descriptive_signal'])
        self.assertEqual(report['decision'], 'descriptive_signal_needs_fresh_confirmation')

    def test_fixed_read_only_scope_and_help_without_inputs(self):
        cfg = cli.configuration()
        self.assertEqual(cfg['maximum_parameter_updates'], 0)
        self.assertEqual(cfg['new_solver_calls'], 0)
        self.assertFalse(cfg['formal_ttf'])
        with patch('sys.argv', ['audit', '--help']), patch.object(cli, 'configuration', side_effect=AssertionError('I/O')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
