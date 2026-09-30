from copy import deepcopy
import unittest

from scripts import run_sa_completion_map_coverage as cli
from scripts import run_sa_terminal_efficiency as old


class MapCoverageTests(unittest.TestCase):
    def test_frozen_scope_and_parent_objective_unchanged(self):
        cfg = cli.configuration()
        self.assertEqual((cfg['train_maps'],cfg['development_maps'],cfg['workers']),(12,6,20))
        self.assertIsNone(cfg['max_decisions'])
        self.assertEqual(cfg['maximum_updates_per_arm'],1)
        self.assertEqual(cli.objective.OBJECTIVES,('completion',))
        self.assertEqual(old.objective.OBJECTIVES,('completion','completion_work'))

    def test_split_fixed_before_labels_and_map_isolated(self):
        rows = [dict(map_id=f'm{i:02d}',task_variant=v) for i in range(18) for v in ('d20','d25')]
        result = cli.split_cases(rows,cli.configuration())
        train = {r['map_id'] for r in result if r['split']=='train'}
        dev = {r['map_id'] for r in result if r['split']=='development_holdout'}
        self.assertEqual((len(train),len(dev)),(12,6))
        self.assertFalse(train&dev)
        self.assertEqual(cli.split_cases(rows[::-1],cli.configuration())[::-1],result)

    def test_uncapped_schedule_and_three_paired_models(self):
        rows = [dict(pair_id=f'p{i}',split='train',case=dict(map_id=f'm{i//4}')) for i in range(48)]
        initial = {r['pair_id']:'a'*64 for r in rows}
        train = cli.schedule(rows,initial,'train',4,['parent'])
        dev = cli.schedule(rows[:24],{r['pair_id']:initial[r['pair_id']] for r in rows[:24]},'dev',2,cli.ARMS)
        self.assertEqual((len(train),len(dev)),(192,144))
        self.assertEqual(len({j['job_id'] for j in train+dev}),336)
        for group in (dev[i:i+3] for i in range(0,len(dev),3)):
            draws = [cli.run.stream_draw(dict(config=dict(stream_seed=4)),j['phase'],j['pair_id'],j['replica'],10000,'pp') for j in group]
            self.assertEqual(len(set(draws)),1)
        with self.assertRaises(ValueError): cli.schedule(rows+rows[:1],initial,'train',4,['parent'])

    def test_credit_coverage_gate_is_not_success_count_gate(self):
        cfg = cli.configuration()
        rows = [dict(pair_id=f'p{i}',map_id=f'm{i%6}',split='train',success=r==0,decisions=1)
                for i in range(8) for r in range(4)]
        coeff = [dict(coefficient=1) for _ in rows]
        result = cli.credit_summary(rows,coeff,cfg)
        self.assertTrue(result['update_allowed'])
        self.assertEqual((len(result['mixed_conditions']),len(result['credited_maps'])),(8,6))
        collapsed = deepcopy(rows)
        for row in collapsed: row['map_id']='one'
        self.assertFalse(cli.credit_summary(collapsed,coeff,cfg)['update_allowed'])
        for row in collapsed: row['success']=True
        self.assertEqual(cli.credit_summary(collapsed,coeff,cfg)['credited_maps'],[])
        rows[0]['split']='development_holdout'
        with self.assertRaises(ValueError): cli.credit_summary(rows,coeff,cfg)

    def test_frozen_A2_reference_identity(self):
        cfg = cli.configuration()
        from unittest.mock import patch
        with patch.object(cli.raw,'validate_bundle',return_value='policy'):
            spec = cli.model_spec(dict(config=cfg,reference={}), 'reference')
        self.assertEqual(spec['sha256'],cfg['reference_sha256'])
        self.assertEqual(spec['path'],cfg['reference_model'])


if __name__=='__main__':
    unittest.main()
