from collections import Counter
from copy import deepcopy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import sa_completion_acceptance as rt
from scripts import run_sa_completion_acceptance as cli
from tests.evaluation.test_sa_raw_timing_metrics import fixture, fail


def jobs():
    return [dict(job_id=f'{m}-{d}-{a}', pair_id=f'm{m:02d}-d{d}-s281', replica=0,
        comparison_arm=a, acceptance_mode=rt.MODES[a], arm='trained_actor',
        model={'sha256':rt.MODEL_SHA}, solver_seed=281, expected_initial=f'{m}-{d}', phase='source',
        plan={'config':{'stream_seed':8}}, case={'map_id':f'm{m:02d}', 'task_variant':f'bottleneck_d{d}'},
        source_folder='source', source_files={'trace':'fixed'}, budget_seconds=120., timing_mode=rt.TIMING_MODE,
        max_decisions=None, execution_node_budget=None)
        for m in range(12) for d in (20,25) for a in rt.ARMS]


def rows():
    names = {'raw_parent':'completion_greedy', 'raw_updated':'completion_sa'}
    return [dict(r, comparison_arm=names[r['comparison_arm']], policy_sha256='same',
                 timing_mode=rt.TIMING_MODE, selection_seconds=1., native_pp_seconds=2.,
                 bookkeeping_seconds=.1, reset_seconds=.2)
            for r in fixture() if r['comparison_arm'] in names]


class AcceptanceTests(unittest.TestCase):
    def test_scope_is_small_paired_and_uncapped(self):
        c = cli.config()
        self.assertEqual((c['maps'], c['solver_seeds'], c['workers'], c['audit_workers']), (12,[281],1,20))
        self.assertIsNone(c['max_decisions'])
        self.assertIsNone(c['execution_node_budget'])
        self.assertTrue(c['no_training'] and c['actor_temperature_input_unchanged'])

    def test_schedule_balanced_deterministic_and_input_untouched(self):
        js = jobs()
        saved = deepcopy(js)
        ordered = cli.schedule(js)
        self.assertEqual(js, saved)
        self.assertEqual(ordered, cli.schedule(reversed(js)))
        self.assertEqual(ordered, cli.schedule(ordered))
        self.assertEqual(len(ordered),48)
        for d in (20,25):
            firsts = [j['comparison_arm'] for j in ordered[::2] if j['case']['task_variant']==f'bottleneck_d{d}']
            self.assertEqual(Counter(firsts),dict.fromkeys(rt.ARMS,6))

    def test_missing_duplicate_mismatched_cases_rejected(self):
        for bad in (jobs()[:-1], jobs()+jobs()[:1]):
            with self.assertRaises(ValueError):
                cli.schedule(bad)
        for key in ('expected_initial','model','plan','phase','solver_seed','source_files','budget_seconds'):
            bad = jobs()
            bad[0][key] = None
            with self.assertRaises((ValueError,TypeError)):
                cli.schedule(bad)

    def test_mode_and_model_identity(self):
        for key,value in (('acceptance_mode','standard'),('arm','official_sa'),
                          ('model',{'sha256':'wrong'}),('max_decisions',100),('execution_node_budget',100)):
            with self.assertRaises(ValueError):
                rt.identity(jobs()[0] | {key:value})

    def test_transition_changes_mode_only_not_feature_temperature_or_rng(self):
        q = SimpleNamespace(temperature=lambda d:55.,_plain=lambda x:x)
        env = SimpleNamespace(step_experimental_pp=Mock(return_value={'observation':{},'metrics':{}}))
        event = {'action':{'mode':'explicit_neighborhood','agents':[2,7],'random_seed':9}}
        with patch.object(rt.run,'stream_draw',return_value=.37) as draw:
            for job in jobs()[:2]:
                result = rt.transition(job,q,env,{},event,3,12.)
                env.step_experimental_pp.assert_called_with(event['action'],12.,rt.MODES[job['comparison_arm']],55.,.37)
                self.assertEqual(result, ({},{},55.,.37))
            self.assertEqual(draw.call_args_list[0],draw.call_args_list[1])

    def test_auditor_routes_only_registered_acceptance(self):
        native = SimpleNamespace(validate_transition=Mock(return_value='ok'), temperature=lambda d:77.)
        wrapped = rt.AcceptanceRuntime(native,'complete_greedy')
        self.assertEqual(wrapped.temperature(1),77.)
        self.assertEqual(wrapped.validate_transition(1,2,3,4,'annealed',77.,.5),'ok')
        native.validate_transition.assert_called_once_with(1,2,3,4,'complete_greedy',77.,.5)
        with self.assertRaises(ValueError):
            wrapped.validate_transition(1,2,3,4,'standard',77.,.5)

    def test_unchanged_timer_and_safe_stop_resume(self):
        self.assertIs(rt._timed.__code__,rt.lean.lean_worker.__code__)
        self.assertIs(rt._timed.__globals__['Policy'],rt.SharedFeaturePolicy)
        self.assertIs(rt._timed.__globals__['transition'],rt.transition)
        self.assertIs(cli.collect.__code__,cli.old.collect.__code__)
        self.assertIs(cli._collect.__code__,cli.first.collect.__code__)
        self.assertIs(cli._collect.__globals__['verify'],cli.verify)
        self.assertIs(cli._collect.__globals__['rt'],rt)

    def test_summary_denominators_failures_sign_and_determinism(self):
        data = rows()
        fail(next(r for r in data if r['comparison_arm']=='completion_greedy'))
        result = rt.summarize(data,50)
        self.assertEqual(result,rt.summarize(reversed(data),50))
        self.assertEqual(result['contrast']['success']['wins'],1)
        self.assertEqual(result['contrast']['common_success']['ttf_seconds']['count'],1)
        self.assertEqual(result['contrast']['target'],'completion_sa')
        self.assertFalse(result['failure_time_imputation'])
        self.assertTrue(result['reused_maps_development_only'])
        for row in data:
            fail(row)
        result = rt.summarize(data,50)
        self.assertIsNone(result['contrast']['common_success']['ttf_seconds']['completion']['mean'])

    def test_summary_rejects_mixed_model_missing_arm_and_stream(self):
        for key in ('initial_fingerprint','rng_stream_id','policy_sha256','timing_mode'):
            data = rows()
            data[0][key] = 'changed'
            with self.assertRaises(ValueError):
                rt.summarize(data,20)
        with self.assertRaises(ValueError):
            rt.summarize(rows()[:-1],20)


class NativeAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native = importlib.import_module('lns2_env')
        except ImportError:
            raise unittest.SkipTest('requires frozen WSL native')
        from scripts.run_sa_path_quality import NATIVE, NATIVE_SHA
        path = cli.ROOT/NATIVE
        if Path(cls.native.__file__).resolve() != path.resolve() or rt.run.sha256_file(path) != NATIVE_SHA:
            raise unittest.SkipTest('requires registered frozen native, never another build')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        m, s = root/'tiny.map',root/'tiny.scen'
        m.write_text('type octile\nheight 2\nwidth 3\nmap\n...\n...\n',encoding='ascii')
        s.write_text('version 1\n0\ttiny.map\t3\t2\t0\t0\t2\t0\t2\n0\ttiny.map\t3\t2\t2\t0\t0\t0\t2\n',encoding='ascii')
        self.env = self.native.LNS2RepairEnv(str(m),str(s),2,time_limit=10)
        self.paths = [[0,1,2],[2,1,0]]
        self.action = {'mode':'explicit_neighborhood','agents':[0,1],'random_seed':19}

    def test_complete_greedy_equals_zero_temperature_sa(self):
        from experiments.repair_collection import state_fingerprint
        outputs = []
        for mode in ('complete_greedy','annealed'):
            self.env.reset_paths(self.paths,seed=19)
            out = self.env.step_experimental_pp(self.action,2.,mode,0.,.5)
            outputs.append(state_fingerprint(out['observation']))
            self.assertTrue(out['metrics']['acceptance_evaluated'])
            self.assertEqual(out['metrics']['pp_inserted_agent_count'],2)
        self.assertEqual(outputs[0],outputs[1])

    def test_zero_budget_rolls_back_and_mode_is_explicit(self):
        for mode in rt.MODES.values():
            before = self.env.reset_paths(self.paths,seed=19)
            out = self.env.step_experimental_pp(self.action,0.,mode,1000.,.4)
            self.assertEqual(out['observation']['agents'],before['agents'])
            self.assertFalse(out['metrics']['replan_success'])
            self.assertEqual(out['metrics']['experimental_acceptance'],mode)


if __name__ == '__main__':
    unittest.main()
