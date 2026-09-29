from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import run_sa_completion_continuation as cli
from scripts import run_sa_terminal_efficiency as old
from tests.evaluation.test_sa_terminal_efficiency import episode


def jobs(split='train'):
    return [dict(pair_id=str(i),split=split,solver_seed=i%2,expected_initial=str(i),
        case=dict(map_id=f'm{i//4}',task_id=f't{i//2}',files=dict(map_file=f'm{i//4}',scenario_file=f't{i//2}')))
        for i in range(24)]


class CompletionContinuationTests(unittest.TestCase):
    def test_only_completion_and_old_module_unchanged(self):
        self.assertEqual(cli.objective.OBJECTIVES,('completion',))
        self.assertEqual(old.objective.OBJECTIVES,('completion','completion_work'))
        self.assertEqual(cli.ARMS,('parent','completion'))
        self.assertIsNone(cli.configuration()['max_decisions'])
        self.assertEqual(cli.configuration()['maximum_updates_per_arm'],1)

    def test_conditions_complete_deterministic_and_split_guarded(self):
        a=jobs()
        self.assertEqual(cli.conditions(a,'train'),cli.conditions((a+a)[::-1],'train'))
        for bad in (a[:-1],[dict(a[0],split='test')]+a[1:],[dict(a[0],expected_initial='bad')]+a):
            with self.assertRaises(ValueError):cli.conditions(bad,'train')

    def test_content_isolation_not_just_names(self):
        a=jobs()
        b=deepcopy(a)
        for row in b:
            for field in ('map_id','task_id'):row['case'][field]+='b'
            row['case']['files']={k:v+'b' for k,v in row['case']['files'].items()}
        names={n for row in a+b for n in row['case']['files'].values()}
        hashes={n:n for n in names}
        cli.validate_separation(a,b,hashes)
        hashes['m0b']=hashes['m0']
        with self.assertRaises(ValueError):cli.validate_separation(a,b,hashes)

    def test_job_model_iteration_follows_actual_parent(self):
        fake=[dict(comparison_arm=arm,iteration=i) for arm,i in zip(cli.ARMS,(1,2))]
        with patch.object(cli,'_call',return_value=fake):
            result=cli.jobs_for(dict(parent=dict(iteration=2)),'comparison')
        self.assertEqual([r['iteration'] for r in result],[2,3])
        with self.assertRaises(ValueError):cli.jobs_for({},'test')

    def test_fresh_paired_streams_and_counts(self):
        a,initial=cli.conditions(jobs(),'train')
        cfg=cli.configuration()
        train=old.schedule(a,initial,cfg['phase'],4,['parent'])
        comparison=old.schedule(a,initial,cfg['comparison_phase'],2,cli.ARMS)
        self.assertEqual((len(train),len(comparison)),(96,96))
        self.assertEqual(len({j['job_id'] for j in train+comparison}),192)
        p=dict(config=dict(stream_seed=4))
        for left,right in zip(comparison[::2],comparison[1::2]):
            draws=[cli.run.stream_draw(p,j['phase'],j['pair_id'],j['replica'],10001,'pp') for j in (left,right)]
            self.assertEqual(draws[0],draws[1])

    def test_identical_completion_credit_and_no_evaluation_labels(self):
        rows=[episode(),episode(1,False,101)]
        kw=dict(objective='completion',policy_sha256='a'*64,expected_groups={'p':'m'},replicas=2,node_budget=100)
        self.assertEqual(cli.objective.coefficients(rows,**kw),old.objective.coefficients(rows,**kw))
        rows[1]['split']='development_holdout'
        with self.assertRaises(ValueError):cli.objective.coefficients(rows,**kw)

    def test_development_signal_not_every_case_must_win(self):
        c=dict(baseline_success=41,challenger_success=42,metrics=dict(generated=dict(change_percent=10)))
        self.assertTrue(cli.signal(c))
        c['challenger_success']=40
        c['metrics']['generated']['change_percent']=-50
        self.assertFalse(cli.signal(c))
        c['challenger_success']=41
        c['metrics']['generated']['change_percent']=-4.9993
        self.assertFalse(cli.signal(c))
        c['metrics']['generated']['change_percent']=-5
        self.assertTrue(cli.signal(c))

    def test_reuse_has_private_bindings(self):
        original=old.verify.__globals__['configuration']
        def f():return configuration()['objective']
        self.assertEqual(cli._call(f),'completion')
        self.assertIs(old.verify.__globals__['configuration'],original)

    def test_no_timing_sources_in_training_entry(self):
        text=Path(cli.__file__).read_text(encoding='utf8')
        for name in ('sa-expanded-four-arm-ttf-v1','sa-completion-acceptance-v1'):
            self.assertNotIn(name,text)


if __name__ == '__main__':
    unittest.main()
