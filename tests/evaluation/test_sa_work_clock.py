from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_sa_work_clock as audit


def step(before, after, conflicts=2, decision=0):
    return dict(nodes_before=before, nodes_after=after, after=conflicts,
                decision=decision, temperature=1000 * .99**decision)


class WorkClockTests(unittest.TestCase):
    def test_exact_boundary(self):
        result = audit.milestones([step(0,25), step(25,100)], False, 100, 100)
        self.assertEqual(result['25']['decision'], 0)
        self.assertEqual(result['25']['overshoot'], 0)
        self.assertEqual(result['100']['status'], 'crossed')

    def test_crosses_multiple_markers_without_interpolation(self):
        result = audit.milestones([step(0,120,decision=3)], False, 120, 100)
        self.assertEqual(result['25']['temperature'], 1000*.99**3)
        self.assertEqual(result['25']['overshoot'], 95)
        self.assertEqual(result['100']['nodes_before'], 0)

    def test_absorbed_success_not_zero_temperature(self):
        result = audit.milestones([step(0,10,0)], True, 10, 100)
        self.assertTrue(all(v['status']=='already_feasible' and v['temperature'] is None for v in result.values()))

    def test_success_on_crossing(self):
        result = audit.milestones([step(0,80,0)], True, 80, 100)
        self.assertEqual(result['75']['status'], 'success_on_crossing')
        self.assertEqual(result['90']['status'], 'already_feasible')

    def test_initially_feasible(self):
        self.assertEqual(audit.milestones([],True,0,100)['25']['status'], 'already_feasible')

    def test_incomplete_trace_rejected(self):
        with self.assertRaises(ValueError):
            audit.milestones([step(0,10)],False,10,100)

    def test_bins_action_start_and_stop(self):
        self.assertEqual(audit.work_bin(24,100),'25')
        self.assertEqual(audit.work_bin(25,100),'50')
        for nodes in (-1,100):
            with self.assertRaises(ValueError): audit.work_bin(nodes,100)

    def make_pair(self):
        a = dict(split='train', arm=audit.ARMS[0],pair_id='task-s1',replica=0,
                 map_id='m',task_id='t',solver_seed=1,initial_fingerprint='f',rng_stream_id='r')
        return [a,dict(a,arm=audit.ARMS[1])]

    def test_pairing_order(self):
        rows = self.make_pair()
        self.assertEqual(audit.paired(rows),audit.paired(list(reversed(rows))))

    def test_pairing_mismatch_and_missing(self):
        for field in ('map_id','task_id','solver_seed','initial_fingerprint','rng_stream_id'):
            rows = self.make_pair()
            rows[1][field]='bad'
            with self.assertRaises(ValueError): audit.paired(rows)
        with self.assertRaises(ValueError): audit.paired(self.make_pair()[:1])
        with self.assertRaises(ValueError): audit.paired(self.make_pair()*2)

    def test_evaluation_data_forbidden(self):
        rows = self.make_pair()
        rows[0]['split']='validation'
        with self.assertRaises(ValueError): audit.paired(rows)

    def test_summary_keeps_absorptions_in_denominator(self):
        rows=[dict(map_id='m',success=success,generated=nodes,decisions=1,
            milestones=audit.milestones([step(0,nodes,0 if success else 2)],success,nodes,100),bins={},sizes={})
            for success,nodes in ((True,10),(False,100))]
        summary=audit.summarize(rows)
        self.assertEqual(summary['episodes'],2)
        self.assertEqual(summary['milestones']['50']['temperature']['n'],1)
        self.assertEqual(summary['milestones']['50']['statuses']['already_feasible'],1)
        self.assertEqual(summary,audit.summarize(list(reversed(rows))))

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'source.json'
            p.write_text('{}',encoding='utf8')
            plan = dict(inputs={'source.json':audit.io.sha256_file(p)})
            with patch.object(audit,'ROOT',Path(temp)):
                audit.verify_inputs(plan)
                p.write_text('{"changed":true}',encoding='utf8')
                with self.assertRaises(ValueError): audit.verify_inputs(plan)

    def event(self):
        return dict(decision=0,temperature=1000.,uniform=.99999,action=dict(agents=[2,9]),
            metrics=dict(action_valid=True,step_applied=True,neighborhood=[2,9],
                conflicts_before=5,conflicts_after=5,acceptance_temperature=1000.,acceptance_uniform=.99999,
                pp_attempt_conflict_pair_count=2,pp_old_conflict_pair_count=1,acceptance_evaluated=True,
                acceptance_probability=__import__('math').exp(-1/1000.),replan_success=False,pp_rolled_back=True))

    def test_attempted_conflicts_not_retained_or_local_count(self):
        e=self.event()
        delta=audit.check_acceptance(e,5,5)
        self.assertEqual(5+delta,6)
        self.assertNotEqual(5+delta,e['metrics']['pp_attempt_conflict_pair_count'])

    def test_rollback_cannot_change_conflicts(self):
        e=self.event()
        e['metrics']['conflicts_after']=4
        with self.assertRaises(ValueError): audit.check_acceptance(e,5,4)

    def test_wrong_temperature_and_probability_rejected(self):
        for field in ('acceptance_probability','acceptance_temperature'):
            e=self.event()
            e['metrics'][field]=0.
            with self.assertRaises(ValueError): audit.check_acceptance(e,5,5)

    def test_noop_is_not_worse_attempt(self):
        e=self.event()
        e['metrics'].update(acceptance_evaluated=False,pp_attempt_conflict_pair_count=0,
            pp_old_conflict_pair_count=0,pp_rolled_back=False)
        self.assertEqual(audit.check_acceptance(e,5,5),0)


if __name__ == '__main__':
    unittest.main()
