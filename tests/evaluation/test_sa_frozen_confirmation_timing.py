from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments import sa_frozen_confirmation_timing as rt
from scripts import run_sa_frozen_confirmation_ttf as cli
from tests.evaluation.test_sa_expanded_timing import jobs, rows


def new_jobs():
    mapping = dict(zip(cli.old.SEEDS, cli.SEEDS))
    return [dict(j, solver_seed=mapping[j['solver_seed']]) for j in jobs()]


class FrozenConfirmationTests(unittest.TestCase):
    def test_scope_and_model_identity(self):
        c = cli.config()
        self.assertEqual(c['solver_seeds'], [307,311,313])
        self.assertEqual((c['maps'],c['workers'],c['audit_workers']), (12,1,20))
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertFalse(c['automatic_promotion'])
        self.assertEqual(c['primary_contrast'], 'parent_vs_official_sa')
        self.assertIn('Frozen A + SA', rt.LABELS['parent'])
        self.assertIn('Frozen A2 + SA', rt.LABELS['completion'])
        for key, value in (('master_seed',1), ('parent_sha256','bad'), ('max_decisions',100)):
            bad = dict(c, **{key:value})
            with patch.object(cli.run,'read_json',return_value=bad):
                with self.assertRaises(ValueError): cli.config()

    def test_schedule_and_isolation_from_historical_module(self):
        js = new_jobs()
        saved = deepcopy(js)
        ordered = cli.schedule(js)
        self.assertEqual(ordered, cli.schedule(reversed(js)))
        self.assertEqual(js,saved)
        self.assertEqual(len(ordered),288)
        self.assertEqual(cli.old.SEEDS,(281,283,293))
        self.assertIn('parent',cli.old.rt.LABELS)
        self.assertNotEqual(cli.old.rt.LABELS['parent'],rt.LABELS['parent'])
        for seed in cli.SEEDS:
            for d in ('bottleneck_d20','bottleneck_d25'):
                groups=[ordered[i:i+4] for i in range(0,288,4)
                        if ordered[i]['solver_seed']==seed and ordered[i]['case']['task_variant']==d]
                for position in range(4):
                    self.assertEqual(sorted([g[position]['comparison_arm'] for g in groups]),sorted(list(rt.ARMS)*3))

    def test_missing_pairs_and_old_seeds_rejected(self):
        with self.assertRaises(ValueError): cli.schedule(new_jobs()[:-1])
        with self.assertRaises(ValueError): cli.schedule(jobs())

    def test_reuses_timer_stop_resume_and_deferred_audit(self):
        self.assertIs(rt.timed_worker,rt.old.timed_worker)
        self.assertIs(rt.audit_worker,rt.old.audit_worker)
        self.assertIs(cli.collect.__code__,cli.old.collect.__code__)
        self.assertIs(cli._collect.__code__,cli.first.collect.__code__)
        self.assertIs(cli._collect.__globals__['verify'],cli.verify)
        self.assertIs(cli._collect.__globals__['rt'],rt)
        self.assertIs(cli.generate.__globals__['generate_worker'],cli.generate_worker)
        self.assertEqual(cli.generate.__globals__['SEEDS'],cli.SEEDS)
        self.assertIs(cli.register.__globals__['schedule'],cli.schedule)

    def test_report_labels_and_scientific_metrics_unchanged(self):
        actual = rt.summarize(rows(),50,19)
        previous = rt.old.summarize(rows(),50,19)
        self.assertEqual(actual['contrasts'],previous['contrasts'])
        self.assertEqual(actual['arms'],previous['arms'])
        self.assertEqual(actual['runtime_labels'],rt.LABELS)
        self.assertFalse(actual['failure_time_imputation'])
        self.assertTrue(actual['scientific_audit_outside_ttf'])
        self.assertTrue(actual['no_promotion'])


if __name__ == '__main__':
    unittest.main()
