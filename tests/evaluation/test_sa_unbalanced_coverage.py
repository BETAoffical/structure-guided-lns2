import copy
import importlib.util
import json
from pathlib import Path
import unittest

from experiments.sa_unbalanced_coverage import state_weights,fit_models,summarize,validate_cohort
from experiments.sa_state_coverage import predict,fit_models as old_fit,signal_gate
from tests.evaluation.test_sa_paired_completion_pilot import training_fixture,target


class UnbalancedCoverageTests(unittest.TestCase):
    def test_fixed_31_state_config(self):
        cfg=json.loads((Path(__file__).resolve().parents[2]/'configs/sa_unbalanced_coverage.json').read_text())
        self.assertEqual(cfg['states'],31)
        self.assertEqual(sorted(cfg['map_counts'].values()),[3,4,4,4,4,4,4,4])
        self.assertEqual(cfg['primary_aggregation'],'map_equal')
        self.assertFalse(cfg['promotion_allowed'])

    def test_weights_map_equal_total_state_count(self):
        states=[dict(state_id=str(i),map_id=m) for i,m in enumerate(['a']*3+['b']*4+['held']*4)]
        w=state_weights(states,'held')
        self.assertAlmostEqual(sum(w.values()),7)
        self.assertAlmostEqual(sum(w[s['state_id']] for s in states if s['map_id']=='a'),3.5)
        self.assertAlmostEqual(sum(w[s['state_id']] for s in states if s['map_id']=='b'),3.5)
        self.assertFalse(set(w)&{s['state_id'] for s in states if s['map_id']=='held'})
        self.assertEqual(w,state_weights(states+[dict(state_id='extra',map_id='held')],'held'))

    def test_summary_is_map_not_trial_or_state_weighted(self):
        rows=[dict(state_id=str(i),map_id='a' if i<3 else 'b',rates=dict(method=1 if i<3 else 0)) for i in range(7)]
        report=summarize(rows)
        self.assertEqual(report['map_equal']['method'],.5)
        self.assertAlmostEqual(report['state_equal']['method'],3/7)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            summarize(rows+[rows[0]])

    def test_cohort_is_exact_not_resampled_or_filled(self):
        roots=[target(f'{m}-{i}',m,'early' if i==0 else 'continuing') for m,n in [('a',3),('b',4)] for i in range(n)]
        validate_cohort(roots,dict(a=3,b=4),set())
        with self.assertRaisesRegex(ValueError,'counts'):
            validate_cohort(roots,dict(a=4,b=4),set())
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_cohort(roots,dict(a=3,b=4),{roots[0]['item']['job_id']})
        modified=copy.deepcopy(roots)
        modified[0]['phase']='continuing'
        with self.assertRaisesRegex(ValueError,'phase'):
            validate_cohort(modified,dict(a=3,b=4),set())

    def test_performance_gate_is_not_lowered(self):
        good=dict(delta=.06,ci95=[.001,.1],wins=5)
        contrasts=dict(frozen=good.copy(),uniform=good.copy(),old_same_model=good.copy())
        self.assertTrue(signal_gate(contrasts))
        contrasts['frozen']['delta']=.04999999
        self.assertFalse(signal_gate(contrasts))

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'registered sklearn unavailable')
    def test_balanced_old_models_equal_existing_and_held_labels_unused(self):
        import sklearn
        if sklearn.__version__!='1.5.0': self.skipTest('registered sklearn version required')
        data=training_fixture()
        new=fit_models(data,'2')
        original=old_fit(data,'2')
        for s in data['states']:
            if s['map_id']=='2': self.assertEqual(predict(new,s),predict(original,s))
        changed=copy.deepcopy(data)
        changed['states']=[s for s in changed['states'] if s['state_id']!='0']
        fitted=fit_models(changed,'2')
        hidden=copy.deepcopy(changed)
        for s in hidden['states']:
            if s['map_id']=='2':
                for c in s['candidates']:
                    for t in c['trials']:
                        v=not t['completed']
                        t.update(completed=v,stop='feasible' if v else 'horizon',steps=1 if v else 32,final_conflicts=0 if v else 1)
        second=fit_models(hidden,'2')
        for s in changed['states']:
            if s['map_id']=='2':
                self.assertEqual(predict(fitted,s),predict(second,s))
                features_only=copy.deepcopy(s)
                for c in features_only['candidates']: c.pop('trials')
                self.assertEqual(predict(fitted,s),predict(fitted,features_only))


if __name__=='__main__':
    unittest.main()
