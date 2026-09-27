from copy import deepcopy
import math
import unittest

from experiments import sa_terminal_efficiency as credit
from experiments.sa_uncapped_training_contract import gradient_coefficients
from scripts import run_sa_terminal_efficiency as cli


def episode(replica=0, success=True, generated=10, pair='p', map_id='m'):
    return dict(episode_id=f'{pair}-{replica}', pair_id=pair, map_id=map_id, replica=replica,
                split='train', policy_sha256='a'*64, initial_fingerprint=cli.run.json_fingerprint(pair),
                rng_stream_id=cli.run.json_fingerprint([pair,replica]), status='ok',
                stop='feasible' if success else 'node_budget', success=success,
                final_conflicts=0 if success else 1, generated=generated, decisions=1,
                steps=[dict(decision=0,policy_sha256='a'*64,probabilities={'a':.5,'b':.5},
                            selected_id='a',behavior_log_probability=math.log(.5))])


def coefficients(rows, objective, groups=None):
    return credit.coefficients(rows,objective=objective,policy_sha256='a'*64,
        expected_groups=groups or {'p':'m'},replicas=2,node_budget=100)


class TerminalEfficiencyTests(unittest.TestCase):
    def test_completion_control_exactly_matches_old_credit(self):
        rows = [episode(),episode(1,False,101)]
        self.assertEqual(coefficients(rows,'completion'),gradient_coefficients(rows,policy_sha256='a'*64,
            expected_groups={'p':'m'},replicas=2,max_decisions=None,node_budget=100))

    def test_whole_episode_work_distinguishes_two_successes(self):
        rows = [episode(generated=10),episode(1,generated=90)]
        self.assertEqual([c['coefficient'] for c in coefficients(rows,'completion')],[0.,0.])
        result = coefficients(rows,'completion_work')
        self.assertAlmostEqual(result[0]['coefficient'],.04)
        self.assertAlmostEqual(result[1]['coefficient'],-.04)
        changed = deepcopy(rows)
        for row in changed:
            row['runtime'] = 100000
            row['single_step_cost'] = -100
            row['final_conflicts'] = 0
        self.assertEqual(result,coefficients(changed,'completion_work'))

    def test_failures_receive_no_partial_progress_or_cost_reward(self):
        rows = [episode(0,False,100),episode(1,False,1000)]
        rows[1]['final_conflicts'] = 50
        self.assertEqual([c['coefficient'] for c in coefficients(rows,'completion_work')],[0.,0.])

    def test_bonus_is_bounded_and_success_after_atomic_overshoot_valid(self):
        for generated,expected in ((0,1.1),(50,1.05),(100,1.),(101,1.)):
            self.assertAlmostEqual(credit.episode_return(episode(generated=generated),
                objective='completion_work',max_decisions=None,node_budget=100),expected)

    def test_unknown_and_heldout_never_become_training_labels(self):
        rows = [episode(),episode(1,False,101)]
        for change in (dict(split='development_holdout'),dict(split='test'),dict(policy_sha256='b'*64),
                       dict(status='censored',stop='incomplete_pp'),dict(stop='decision_budget')):
            bad = deepcopy(rows)
            bad[1].update(change)
            with self.assertRaises(ValueError):coefficients(bad,'completion_work')
        with self.assertRaises(ValueError):coefficients(rows,'one_step')

    def test_map_weights_and_order_preserved(self):
        groups = {'p1':'a','p2':'a','p3':'b'}
        rows = [episode(r,generated=10+80*r,pair=p,map_id=m) for p,m in groups.items() for r in range(2)]
        result = coefficients(rows,'completion_work',groups)
        self.assertEqual(result,coefficients(rows[::-1],'completion_work',groups))
        self.assertAlmostEqual(sum(r['episode_weight'] for r in result),1.)
        self.assertAlmostEqual(sum(r['episode_weight'] for r in result if r['episode_id'].startswith('p3')), .5)

    def test_no_decision_cap_or_trajectory_length_normalization(self):
        a = episode()
        b = episode(1,generated=90)
        expected = coefficients([a,b],'completion_work')
        b['steps'] = [dict(b['steps'][0],decision=d) for d in range(1001)]
        b['decisions'] = 1001
        self.assertEqual(expected,coefficients([a,b],'completion_work'))

    def test_schedule_ids_pair_streams_and_uncapped_config(self):
        roots = [dict(pair_id=str(i),split='train',case=dict(map_id=str(i//4))) for i in range(24)]
        fps = {str(i):'a'*64 for i in range(24)}
        train = cli.schedule(roots,fps,'train',4,['parent'])
        comp = cli.schedule(roots,fps,'eval',2,cli.ARMS)
        self.assertEqual((len(train),len(comp)),(96,144))
        self.assertEqual(len({j['job_id'] for j in train+comp}),240)
        p = dict(config=dict(stream_seed=4))
        for a,b,c in zip(comp[::3],comp[1::3],comp[2::3]):
            self.assertEqual(len({cli.run.stream_draw(p,j['phase'],j['pair_id'],j['replica'],900,'pp') for j in (a,b,c)}),1)
        self.assertIsNone(cli.configuration()['max_decisions'])
        self.assertEqual(cli.configuration()['work_bonus'],credit.BONUS)
        with self.assertRaises(ValueError):cli.schedule(roots[:-1],fps,'x',4,['parent'])

    def test_pair_summary_bootstrap_and_integrity(self):
        rows = []
        for m in ('m0','m1'):
            for arm,factor in (('parent',1.),('completion',.8)):
                row = episode(map_id=m,pair=m)
                row.update(comparison_arm=arm,generated=100*factor,soc=100,makespan=10)
                rows.append(row)
        result = credit.contrast(rows,'parent','completion',bootstrap=100)
        self.assertAlmostEqual(result['metrics']['generated']['change_percent'],-20.)
        self.assertEqual(result['common_generated_difference_ci95'],[-20.,-20.])
        self.assertEqual(result,credit.contrast(rows[::-1],'parent','completion',bootstrap=100))
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):credit.contrast(bad,'parent','completion',bootstrap=10)
        bad = deepcopy(rows)
        bad[1]['rng_stream_id'] = 'bad'
        with self.assertRaises(ValueError):credit.contrast(bad,'parent','completion',bootstrap=10)

    def test_failures_not_counted_as_zero_work_success(self):
        a = dict(episode(),comparison_arm='parent',soc=10,makespan=3)
        b = dict(episode(success=False,generated=101),comparison_arm='completion',soc=10,makespan=3)
        result = credit.contrast([a,b],'parent','completion',bootstrap=10)
        self.assertEqual((result['gains'],result['losses'],result['common_success']),(0,1,0))
        self.assertIsNone(result['metrics']['generated']['baseline'])
        self.assertIsNone(result['common_generated_difference_ci95'])


if __name__ == '__main__':
    unittest.main()
