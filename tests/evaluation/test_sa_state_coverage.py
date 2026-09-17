import copy
import importlib.util
import json
from pathlib import Path
import unittest

from experiments.sa_state_coverage import choose_new_roots, merge_data, fit_models, predict, signal_gate
from tests.evaluation.test_sa_paired_completion_pilot import target, training_fixture


class StateCoverageTests(unittest.TestCase):
    def test_registered_budget_no_new_objective(self):
        cfg=json.loads((Path(__file__).resolve().parents[2]/'configs/sa_state_coverage.json').read_text())
        self.assertEqual((cfg['states'],cfg['max_candidates'],cfg['trials'],cfg['horizon'],cfg['workers']),(32,4,8,32,20))
        self.assertTrue(cfg['no_ttf'])
        self.assertFalse(cfg['runtime_integration_allowed'])

    def test_episode_first_blind_sampling_and_isolation(self):
        available=[target(f'{i}-{p}',str(i//8),p,str(i)) for i in range(16) for p in ('early','continuing')]
        roots,coverage=choose_new_roots(available,{'0','1'},{'0','8'},15)
        self.assertEqual(len(roots),8)
        self.assertEqual(len({r['item']['job_id'] for r in roots}),8)
        self.assertFalse({r['item']['job_id'] for r in roots}&{'0','8'})
        self.assertTrue(all(c['admitted'] for c in coverage))
        changed=copy.deepcopy(list(reversed(available)))
        for r in changed:
            r['future_completion']=1
        again,_=choose_new_roots(changed,{'0','1'},{'0','8'},15)
        self.assertEqual([r['id'] for r in roots],[r['id'] for r in again])

    def test_missing_phase_does_not_pass_by_filling(self):
        roots,coverage=choose_new_roots([target(i,'a','early') for i in range(9)],{'a'},set(),1)
        self.assertFalse(coverage[0]['admitted'])
        self.assertLess(len(roots),4)

    def test_duplicates_rejected(self):
        r=target('a','a','early')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            choose_new_roots([r,r],{'a'},set(),1)

    def test_merge_preserves_trials_and_disallows_episode_overlap(self):
        old=training_fixture()
        new=copy.deepcopy(old)
        with self.assertRaisesRegex(ValueError,'episode overlap'):
            merge_data(old,new,'d'*64)
        for s in new['states']:
            s['state_id']='new-'+s['state_id']
            s['episode']='new-'+s['episode']
        merged=merge_data(old,new,'d'*64)
        self.assertEqual(len(merged['states']),48)
        self.assertEqual(merged['states'][:24],old['states'])
        new['feature_names']=['changed']
        with self.assertRaises(ValueError):
            merge_data(old,new,'d'*64)

    def test_gate_includes_random_and_same_model(self):
        passed=dict(delta=.06,ci95=[.01,.1],wins=5)
        c=dict(frozen=passed.copy(),uniform=passed.copy(),old_same_model=passed.copy())
        self.assertTrue(signal_gate(c))
        for field in c:
            changed=copy.deepcopy(c)
            changed[field]['ci95'][0]=-.001
            self.assertFalse(signal_gate(changed))
        c['frozen']['delta']=.04999999
        self.assertFalse(signal_gate(c))

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'registered sklearn unavailable')
    def test_models_equal_prior_implementation_and_ignore_held_labels(self):
        import sklearn
        if sklearn.__version__!='1.5.0': self.skipTest('registered version required')
        from scripts.run_sa_paired_completion_pilot import fit_fold
        data=training_fixture()
        current=fit_models(data,'2')
        previous=fit_fold(dict(data=data,map_id='2'))
        for p in previous['predictions']:
            s=next(s for s in data['states'] if s['state_id']==p['state_id'])
            result=predict(current,s)
            self.assertEqual(result['paired'],p['paired'])
            self.assertEqual(result['direct'],p['pointwise'])
            changed=copy.deepcopy(s)
            for c in changed['candidates']: c.pop('trials')
            self.assertEqual(result,predict(current,changed))
        changed=copy.deepcopy(data)
        for s in changed['states']:
            if s['map_id']=='2':
                for c in s['candidates']:
                    for t in c['trials']:
                        value=not t['completed']
                        t.update(completed=value,stop='feasible' if value else 'horizon',steps=1 if value else 32,final_conflicts=0 if value else 1)
        other=fit_models(changed,'2')
        for s in data['states']:
            if s['map_id']=='2': self.assertEqual(predict(current,s),predict(other,s))
        self.assertTrue(all(next(s for s in data['states'] if s['state_id']==sid)['map_id']!='2' for sid in current['train_ids']))

    def test_wrong_map_prediction_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError,'held-map leakage'):
            predict(dict(held='x',train_maps=['y']),dict(map_id='y'))


if __name__=='__main__':
    unittest.main()
